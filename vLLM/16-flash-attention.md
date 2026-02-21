# 16. FlashAttention 与 FlashInfer 实现

## 概述

vLLM V1 主要使用两个高性能注意力内核库：

1. **FlashAttention (FA2/FA3)**: 由 Dao-AILab 开发，是业界标准的高效注意力实现
2. **FlashInfer**: 专为 LLM serving 优化，支持更多特性（如 TRT-LLM kernel、FP4 量化）

本文详细分析这两个后端在 vLLM 中的实现。

## FlashAttention 后端

### 源码位置

```
vllm/v1/attention/backends/flash_attn.py      # 主实现
vllm/v1/attention/backends/fa_utils.py         # 工具函数
vllm/v1/attention/backends/flash_attn_diffkv.py # 不同 K/V head size
```

### FlashAttentionBackend

```python
# vllm/v1/attention/backends/flash_attn.py

class FlashAttentionBackend(AttentionBackend):
    """FlashAttention 2/3 后端
    
    核心特性:
    - 使用 FlashAttention 2 或 3 内核
    - 支持 Cascade Attention (共享前缀优化)
    - 支持 FP8 KV Cache 量化
    - 分离 KV Cache 更新和注意力计算
    """
    
    # 接受预分配的输出 buffer
    accept_output_buffer: bool = True
    
    # 支持的数据类型
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    
    # forward() 不包含 KV Cache 更新
    # KV Cache 更新由单独的 do_kv_cache_update() 方法处理
    forward_includes_kv_cache_update: bool = False
    
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """FA 要求 block size 是 16 的倍数"""
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        
        # 对于混合模型 + float32 Mamba 状态，使用固定大小
        # 避免 FA 的 NaN 传播问题
        if (model_config and model_config.is_hybrid and
            cache_config.mamba_cache_dtype == "float32"):
            return [16, 32, 64]
        
        return [MultipleOf(16)]
    
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"
    
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FA 支持所有注意力类型"""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )
    
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """FA 的 KV Cache 形状: [2, num_blocks, block_size, num_kv_heads, head_size]
        
        第一个维度 2 表示 K 和 V 两个 cache
        """
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)
    
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """物理布局可以是 NHD 或 HND"""
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # 物理: (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            return (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # 物理: (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout: {cache_layout}")
    
    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        """head_size 必须是 8 的倍数且 <= 256"""
        return head_size % 8 == 0 and head_size <= 256
    
    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        """FP8 支持需要特定的 FlashAttention 版本"""
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto", "bfloat16"]
    
    @classmethod
    def supports_sink(cls) -> bool:
        """Attention Sink 只在 FA3 中支持"""
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()
    
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """FA 需要 SM80+ (Ampere 及更新)"""
        return capability >= DeviceCapability(8, 0)
```

### FlashAttentionMetadata

```python
@dataclass
class FlashAttentionMetadata:
    """FlashAttention 专用元数据
    
    包含执行注意力计算所需的所有信息
    """
    
    # ============ 基础信息 ============
    num_actual_tokens: int        # 实际 token 数 (不含 padding)
    max_query_len: int            # 最大 query 长度
    query_start_loc: torch.Tensor # [batch+1] query 起始位置
    max_seq_len: int              # 最大序列长度
    seq_lens: torch.Tensor        # [batch] 序列长度
    block_table: torch.Tensor     # [batch, max_blocks] block 索引
    slot_mapping: torch.Tensor    # [num_tokens] slot 映射
    
    # ============ Cascade Attention ============
    use_cascade: bool
    """是否使用 cascade attention
    
    Cascade attention 将共享前缀和后缀分开计算:
    1. 所有 query 对共享前缀计算一次注意力
    2. 每个请求对自己的后缀计算注意力
    3. 合并两次计算的结果
    
    这对于长共享前缀场景（如系统提示）非常高效
    """
    
    common_prefix_len: int
    """共享前缀长度"""
    
    cu_prefix_query_lens: torch.Tensor | None
    """前缀计算的累积 query 长度"""
    
    prefix_kv_lens: torch.Tensor | None
    """前缀 KV 长度"""
    
    suffix_kv_lens: torch.Tensor | None
    """每个请求的后缀 KV 长度"""
    
    # ============ Decode Context Parallelism ============
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None
    """DCP 本地 KV 长度"""
    
    # ============ AOT Scheduling (FA3) ============
    scheduler_metadata: torch.Tensor | None = None
    """FA3 的 ahead-of-time 调度元数据
    
    FA3 支持预计算 tile 调度，避免运行时开销
    这对 CUDA Graph 兼容性很重要
    """
    
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0
    """最大分割数 (用于 CUDA Graph)"""
    
    causal: bool = True
    """是否使用因果 mask"""
```

### FlashAttentionMetadataBuilder

```python
class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    """FlashAttention 元数据构建器
    
    关键职责:
    1. 将通用元数据转换为 FA 特定格式
    2. 决定是否使用 cascade attention
    3. 处理 FA3 的 AOT 调度
    """
    
    # FA3 完全支持 CUDA Graph
    # FA2 只支持 uniform batch
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    
    # 支持更新 block table
    supports_update_block_table: bool = True
    
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        
        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        
        # FA3 使用 AOT 调度
        self.aot_schedule = get_flash_attn_version() == 3
        
        # DCP 支持
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            self.dcp_world_size = 1
            self.dcp_rank = 0
        
        # CUDA Graph 相关
        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        
        if self.use_full_cuda_graph and self.aot_schedule:
            # 预分配调度元数据 buffer
            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # 设置 num_splits 上限以预分配中间 buffer
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )
    
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """构建 FlashAttention 元数据
        
        Args:
            common_prefix_len: 共享前缀长度
            common_attn_metadata: 通用元数据
            fast_build: 快速模式 (禁用 AOT 调度)
        """
        # AOT 调度对于 spec-decode 等场景可能不值得
        aot_schedule = self.aot_schedule and not fast_build
        
        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        
        # 决定是否使用 cascade attention
        use_cascade = common_prefix_len > 0
        
        # 构建调度元数据
        def schedule(batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal):
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=self.max_num_splits,
                )
            return None
        
        # 处理 DCP
        if self.dcp_world_size > 1:
            dcp_context_kv_lens = get_dcp_local_seq_lens(
                seq_lens - query_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            scheduler_metadata = schedule(...)
        elif use_cascade:
            # Cascade attention 的两阶段调度
            prefix_scheduler_metadata = schedule(...)
            suffix_scheduler_metadata = schedule(...)
        else:
            # 标准调度
            scheduler_metadata = schedule(...)
        
        return FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            # ...
        )
```

### FlashAttentionImpl

```python
class FlashAttentionImpl(AttentionImpl):
    """FlashAttention 实现
    
    实际调用 FlashAttention kernel 的类
    """
    
    # 支持返回 LSE (用于 DCP)
    can_return_lse_for_decode: bool = True
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,  # Attention Sink
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        
        # ALiBi slopes
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        
        # Sliding window 格式: (left, right)
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap or 0
        self.sinks = sinks  # Attention Sink (FA3 only)
        
        self.vllm_flash_attn_version = get_flash_attn_version()
        
        # FP8 和 per-head scales 支持检查
        self.supports_quant_query_input = True
        self.supports_per_head_quant_scales = (
            self.vllm_flash_attn_version >= 3
        )
    
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,         # [num_tokens, num_heads, head_size]
        key: torch.Tensor,           # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,         # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,      # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FlashAttention 前向计算"""
        assert output is not None, "Output tensor must be provided."
        
        if attn_metadata is None:
            # Profiling run
            return output.fill_(0)
        
        num_actual_tokens = attn_metadata.num_actual_tokens
        
        # Encoder attention 不使用 KV Cache
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )
        
        # 分离 K 和 V cache
        key_cache, value_cache = kv_cache.unbind(0)
        
        # FP8 处理
        if self.kv_cache_dtype.startswith("fp8"):
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)
        
        if not attn_metadata.use_cascade:
            # 标准注意力计算
            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                self.num_kv_heads
            )
            
            if self.dcp_world_size > 1:
                # Decode Context Parallelism
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache, value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
            else:
                # 普通注意力
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=list(self.sliding_window),
                    block_table=attn_metadata.block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=attn_metadata.scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,  # Attention Sink
                )
        else:
            # Cascade Attention
            cascade_attention(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                key_cache, value_cache,
                cu_query_lens=attn_metadata.query_start_loc,
                cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
                prefix_kv_lens=attn_metadata.prefix_kv_lens,
                suffix_kv_lens=attn_metadata.suffix_kv_lens,
                common_prefix_len=attn_metadata.common_prefix_len,
                # ...
            )
        
        return output
    
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """更新 KV Cache (与 forward 分离)
        
        这种分离使得 piecewise CUDA Graph 可以捕获 forward
        而不需要包含 cache 更新操作
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        
        if self.kv_sharing_target_layer_name is not None or key is None:
            return
        
        key_cache, value_cache = kv_cache.unbind(0)
        
        reshape_and_cache_flash(
            key, value,
            key_cache, value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
```

### Cascade Attention

```python
def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    common_prefix_len: int,
    block_table: torch.Tensor,
    # ...
) -> torch.Tensor:
    """Cascade Attention 实现
    
    将注意力计算分为两部分:
    1. 前缀注意力: 所有 query 对共享前缀计算
    2. 后缀注意力: 每个请求对自己的后缀计算
    
    这对于共享系统提示的场景非常高效:
    - 共享前缀只计算一次 (而不是每个请求都算)
    - 使用 LSE (Log-Sum-Exp) 合并两部分结果
    """
    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    num_common_kv_blocks = common_prefix_len // block_size
    
    # Step 1: 计算共享前缀注意力
    # 所有 query token 对前缀 KV 计算，causal=False
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        causal=False,  # 前缀是完整的，不需要因果 mask
        block_table=block_table[:1],  # 只用第一个请求的前缀 blocks
        return_softmax_lse=True,
        s_aux=sinks,  # Attention Sink 合并到 prefix_lse
    )
    
    # Step 2: 计算每个请求的后缀注意力
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        causal=True,  # 后缀使用因果 mask
        block_table=block_table[:, num_common_kv_blocks:],
        return_softmax_lse=True,
    )
    
    # Step 3: 合并结果
    # 使用 LSE 进行加权平均
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
```

## FlashInfer 后端

### 源码位置

```
vllm/v1/attention/backends/flashinfer.py  # 主实现
```

### FlashInferBackend

```python
class FlashInferBackend(AttentionBackend):
    """FlashInfer 后端
    
    特性:
    - 支持 TRT-LLM attention kernel (SM100+/Blackwell)
    - 支持 FP4/FP8 量化
    - 专为 serving 优化的 wrapper API
    - 支持 Attention Sink (通过 TRT-LLM)
    """
    
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "bfloat16",
        "fp8", "fp8_e4m3", "fp8_e5m2",  # 比 FA 更丰富的 FP8 支持
    ]
    
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """FlashInfer 支持 16, 32, 64"""
        return [16, 32, 64]
    
    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"
    
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """FlashInfer 形状: [num_blocks, 2, block_size, num_kv_heads, head_size]
        
        注意: 2 (K/V) 在第二个维度，与 FA 不同
        """
        return (num_blocks, 2, block_size, num_kv_heads, head_size)
    
    @classmethod
    def supports_sink(cls) -> bool:
        """Sink 需要 TRT-LLM attention (Blackwell)"""
        from vllm.utils.flashinfer import (
            force_use_trtllm_attention,
            supports_trtllm_attention,
        )
        
        if force_use_trtllm_attention() is False:
            return False
        return supports_trtllm_attention()
    
    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        """Blackwell 需要 HND 布局"""
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 10:
            return "HND"
        return None
```

### FlashInferMetadata

```python
@dataclass
class FlashInferMetadata:
    """FlashInfer 元数据
    
    FlashInfer 使用更细粒度的 prefill/decode 分离
    """
    
    num_actual_tokens: int
    slot_mapping: torch.Tensor
    q_data_type: torch.dtype
    
    # Prefill/Decode 分离计数
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    
    # Prefill 元数据 (两种可能的类型)
    prefill: FIPrefill | TRTLLMPrefill | None
    """
    - FIPrefill: 使用 FlashInfer 原生 wrapper
    - TRTLLMPrefill: 使用 TRT-LLM kernel (Blackwell)
    """
    
    # Decode 元数据
    decode: FIDecode | TRTLLMDecode | None
    
    # Cascade Attention
    use_cascade: bool
    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None


@dataclass
class FIPrefill:
    """FlashInfer 原生 prefill 元数据"""
    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper


@dataclass
class FIDecode:
    """FlashInfer 原生 decode 元数据"""
    wrapper: BatchDecodeWithPagedKVCacheWrapper


@dataclass
class TRTLLMPrefill:
    """TRT-LLM prefill 元数据 (Blackwell)"""
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor
    max_q_len: int
    max_seq_len: int


@dataclass
class TRTLLMDecode:
    """TRT-LLM decode 元数据 (Blackwell)"""
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
```

### FlashInferMetadataBuilder

```python
class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    """FlashInfer 元数据构建器
    
    主要复杂性在于管理多个 wrapper 对象
    """
    
    reorder_batch_threshold: int = 1  # 分离 decode 和 prefill
    
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        
        # Wrappers (懒初始化)
        self._workspace_buffer = None
        self._prefill_wrapper = None
        self._decode_wrapper = None
        self._cascade_wrapper = None
        
        # CUDA Graph 模式: 每个 batch size 一个 decode wrapper
        if self.enable_cuda_graph:
            self._decode_wrappers_cudagraph: dict[int, BatchDecodeWithPagedKVCacheWrapper] = {}
        
        # TRT-LLM 支持检测
        can_use_trtllm = can_use_trtllm_attention(
            self.num_qo_heads, self.num_kv_heads
        )
        self.use_trtllm_decode_attention = can_use_trtllm
        
        # Q 量化
        if can_use_trtllm and not vllm_config.attention_config.disable_flashinfer_q_quantization:
            self.q_data_type = self.kv_cache_dtype  # FP8
        else:
            self.q_data_type = self.model_config.dtype
        
        # 持久化 buffer
        self.paged_kv_indptr = self._make_buffer(max_num_reqs + 1)
        self.paged_kv_indices = self._make_buffer(max_num_pages)
        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)
    
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        """构建 FlashInfer 元数据"""
        
        # 分离 decode 和 prefill
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        
        # 决定使用哪种 kernel
        prefill_use_trtllm = use_trtllm_attention(
            self.num_qo_heads, self.num_kv_heads,
            num_prefill_tokens, max_seq_len,
            self.dcp_world_size, self.cache_dtype, self.q_data_type,
            is_prefill=True, has_sinks=self.has_sinks,
        )
        decode_use_trtllm = self.use_trtllm_decode_attention
        
        # 初始化输出
        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            q_data_type=self.q_data_type,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=None,
            decode=None,
            use_cascade=use_cascade,
            cascade_wrapper=None,
        )
        
        # Cascade attention (共享前缀)
        if use_cascade:
            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()
            attn_metadata.cascade_wrapper.plan(
                [shared_qo_indptr_cpu, qo_indptr_cpu],
                [shared_kv_page_indptr_cpu, paged_kv_indptr_cpu],
                [shared_kv_page_indices_cpu, paged_kv_indices],
                [shared_kv_last_page_len_cpu, paged_kv_last_page_len_cpu],
                self.num_qo_heads, self.num_kv_heads, self.head_dim, self.page_size,
                causal=True, sm_scale=self.sm_scale,
            )
            return attn_metadata
        
        # Prefill 路径
        if num_prefills > 0:
            if prefill_use_trtllm:
                attn_metadata.prefill = TRTLLMPrefill(
                    block_tables=block_table_tensor[prefill_start:],
                    seq_lens=seq_lens[prefill_start:],
                    cum_seq_lens_q=qo_indptr_prefill_gpu,
                    cum_seq_lens_kv=paged_kv_indptr_prefill_gpu,
                    max_q_len=max_q_len_prefill,
                    max_seq_len=max_seq_len,
                )
            else:
                prefill_wrapper = self._get_prefill_wrapper()
                prefill_wrapper.plan(...)
                attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)
        
        # Decode 路径
        if num_decodes > 0:
            if decode_use_trtllm:
                attn_metadata.decode = TRTLLMDecode(
                    block_tables=block_table_tensor[:num_decodes],
                    seq_lens=seq_lens[:num_decodes],
                    max_seq_len=max_seq_len,
                )
            else:
                decode_wrapper = self._get_decode_wrapper(num_decode_tokens, use_cudagraph)
                fast_plan_decode(decode_wrapper, ...)
                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
        
        return attn_metadata
```

### FlashInferImpl

```python
class FlashInferImpl(AttentionImpl):
    """FlashInfer 实现"""
    
    can_return_lse_for_decode: bool = True
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        # ... 初始化代码 ...
        
        self.support_trtllm_attn = can_use_trtllm_attention(num_heads, num_kv_heads)
        
        # FP8 + TRT-LLM 时的 scale 计算
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
    
    def fused_output_quant_supported(self, quant_key: QuantKey):
        """检查是否支持融合输出量化"""
        return (
            self.support_trtllm_attn
            and self.kv_cache_dtype.startswith("fp8")
            and quant_key in (kFp8StaticTensorSym, kNvfp4Dynamic)
        )
    
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FlashInfer 前向计算"""
        
        if attn_metadata is None:
            return output.fill_(0)
        
        # 验证 query dtype
        assert attn_metadata.q_data_type == query.dtype
        
        # 初始化 scale (首次调用)
        if self.bmm1_scale is None:
            self.bmm1_scale = layer._q_scale_float * layer._k_scale_float * self.scale
        if self.bmm2_scale is None:
            self.bmm2_scale = layer._v_scale_float
        
        # 更新 KV Cache
        if self.kv_sharing_target_layer_name is None:
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key, value,
                kv_cache[:, 0], kv_cache[:, 1],
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale, layer._v_scale,
            )
        
        # 处理 FP8 视图
        if self.kv_cache_dtype.startswith("fp8"):
            kv_cache = kv_cache.view(
                FlashInferBackend.get_fp8_dtype_for_flashinfer(self.kv_cache_dtype)
            )
        
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        
        # 转置 KV Cache
        kv_cache_permute = kv_cache.permute(*FlashInferBackend.get_kv_cache_stride_order())
        
        # Cascade Attention
        if attn_metadata.use_cascade:
            output.copy_(attn_metadata.cascade_wrapper.run(query, kv_cache))
            return output
        
        # Prefill
        if num_prefill_tokens > 0:
            prefill_query = query[num_decode_tokens:]
            
            if isinstance(attn_metadata.prefill, FIPrefill):
                # FlashInfer 原生
                prefill_wrapper = attn_metadata.prefill.wrapper
                if use_dcp:
                    prefill_wrapper.run(
                        layer, prefill_query, kv_cache_permute,
                        key[num_decode_tokens:], value[num_decode_tokens:],
                        out=output[num_decode_tokens:],
                    )
                else:
                    prefill_wrapper.run(
                        prefill_query, kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output[num_decode_tokens:],
                    )
            else:
                # TRT-LLM
                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=kv_cache_permute,
                    workspace_buffer=workspace_buffer,
                    block_tables=attn_metadata.prefill.block_tables,
                    seq_lens=attn_metadata.prefill.seq_lens,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    sinks=self.sinks,
                    out=output[num_decode_tokens:],
                )
        
        # Decode
        if num_decode_tokens > 0:
            decode_query = query[:num_decode_tokens]
            
            if isinstance(attn_metadata.decode, FIDecode):
                # FlashInfer 原生
                decode_wrapper = attn_metadata.decode.wrapper
                decode_wrapper.run(
                    decode_query, kv_cache_permute,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    out=output[:num_decode_tokens],
                )
            else:
                # TRT-LLM
                trtllm_batch_decode_with_kv_cache(
                    query=decode_query,
                    kv_cache=kv_cache_permute,
                    block_tables=attn_metadata.decode.block_tables,
                    seq_lens=attn_metadata.decode.seq_lens,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    sinks=self.sinks,
                    out=output[:num_decode_tokens],
                )
        
        return output
```

## FA vs FlashInfer 对比

| 特性 | FlashAttention | FlashInfer |
|------|---------------|------------|
| **GPU 支持** | SM80+ (Ampere+) | SM75+ |
| **Blackwell 优化** | 有限 | TRT-LLM kernel |
| **KV Cache 布局** | NHD (默认) | HND (Blackwell) |
| **FP8 KV Cache** | FA2.5+/FA3 | 全面支持 |
| **FP4 量化** | 不支持 | 支持 |
| **Attention Sink** | FA3 only | TRT-LLM |
| **CUDA Graph** | FA3: ALWAYS, FA2: UNIFORM_BATCH | TRTLLM: UNIFORM_BATCH |
| **Cascade Attention** | 原生支持 | MultiLevelCascadeWrapper |
| **Encoder-Decoder** | 支持 | 不支持 |

## 性能优化建议

1. **GPU 选择**:
   - Ampere/Hopper: 优先 FlashAttention
   - Blackwell: 优先 FlashInfer (TRT-LLM kernel)

2. **CUDA Graph**:
   - FA3 提供最好的 CUDA Graph 兼容性
   - 使用 `--compilation-config.cudagraph_mode=FULL` 获得最佳性能

3. **FP8 量化**:
   - 两者都支持，FlashInfer 还支持 FP4
   - 使用 `--kv-cache-dtype fp8` 启用

4. **Cascade Attention**:
   - 共享前缀 > 256 tokens 时自动启用
   - 对长系统提示非常有效
