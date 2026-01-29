# SGLang ModelRunner 与 CUDA Graph 详解

> **默认场景**: Qwen/Qwen3-VL-235B-A22B-Thinking 多模态模型
>
> **启用特性**: PD 分离 + Chunked Prefill + ViT DP + Overlap Schedule + 多模态缓存

## 1. ModelRunner 概览

**核心文件**:
- `srt/model_executor/model_runner.py` - 模型执行器 (2454 行)
- `srt/model_executor/cuda_graph_runner.py` - CUDA Graph 管理 (948 行)
- `srt/model_executor/forward_batch_info.py` - ForwardBatch 定义 (1293 行)

### 1.1 职责分工

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           组件职责对比                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Scheduler                                                                   │
│  ├── 管理请求生命周期 (add_req, process_batch)                               │
│  ├── 调度决策 (哪些请求进入批次)                                              │
│  └── 输出: ModelWorkerBatch (高层批次信息)                                   │
│                                                                              │
│  ModelRunner                                                                 │
│  ├── 执行模型 forward pass                                                  │
│  ├── 管理 CUDA Graph                                                        │
│  ├── 采样下一个 token                                                        │
│  └── 输入: ForwardBatch (低层 GPU 张量)                                     │
│                                                                              │
│  CudaGraphRunner                                                             │
│  ├── 捕获和重放 CUDA Graph                                                  │
│  ├── 管理不同 batch size 的图                                                │
│  └── 优化 decode 阶段的 kernel launch 开销                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. ForwardMode 枚举

```python
class ForwardMode(IntEnum):
    EXTEND = auto()          # Prefill / 扩展模式
    DECODE = auto()          # Decode 模式 (逐 token 生成)
    MIXED = auto()           # 混合批次 (prefill + decode)
    IDLE = auto()            # 空闲模式 (无实际计算)
    TARGET_VERIFY = auto()   # 投机解码验证模式
    DRAFT_EXTEND = auto()    # Draft 模型扩展
    DRAFT_EXTEND_V2 = auto() # Draft 模型扩展 v2
    PREBUILT = auto()        # 预构建模式
    SPLIT_PREFILL = auto()   # 分层 Prefill (逐层执行)
    DLLM_EXTEND = auto()     # Diffusion LLM 扩展
```

### 2.1 模式判断方法

```python
# 常用判断
forward_mode.is_decode()           # DECODE
forward_mode.is_extend()           # EXTEND, DRAFT_EXTEND
forward_mode.is_prefill()          # EXTEND, MIXED, DRAFT_EXTEND
forward_mode.is_cuda_graph()       # DECODE, TARGET_VERIFY, DLLM_EXTEND
forward_mode.is_split_prefill()    # SPLIT_PREFILL
```

## 3. ForwardBatch 数据结构

```python
@dataclass
class ForwardBatch:
    """存储 forward pass 的所有输入"""
    
    # 基本信息
    forward_mode: ForwardMode
    batch_size: int
    input_ids: torch.Tensor          # [num_tokens]
    positions: torch.Tensor          # [num_tokens] 位置编码
    
    # 内存池引用
    req_pool_indices: torch.Tensor   # [batch_size] 请求槽位
    seq_lens: torch.Tensor           # [batch_size] 序列长度
    out_cache_loc: torch.Tensor      # [num_tokens] 输出 KV 位置
    
    # Extend 特有
    extend_num_tokens: int           # 本次扩展的 token 数
    extend_seq_lens: torch.Tensor    # [batch_size]
    extend_prefix_lens: torch.Tensor # [batch_size]
    extend_start_loc: torch.Tensor   # [batch_size] 每个请求的起始位置
    
    # Attention 后端
    attn_backend: AttentionBackend
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: BaseKVCache
    
    # Sampling
    sampling_info: SamplingBatchInfo
    return_logprob: bool
    top_logprobs_nums: List[int]
    
    # 投机解码
    spec_algorithm: SpeculativeAlgorithm
    spec_info: Optional[SpecInput]
    
    # LoRA
    lora_ids: Optional[List[str]]
```

## 4. ModelRunner 核心方法

### 4.1 forward 方法

```python
def forward(
    self,
    forward_batch: ForwardBatch,
    skip_attn_backend_init: bool = False,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> ModelRunnerOutput:
```

**执行流程**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ModelRunner.forward 流程                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. 检查是否可以使用 CUDA Graph                                              │
│     can_run_graph = forward_mode.is_cuda_graph()                            │
│                   and self.graph_runner                                     │
│                   and self.graph_runner.can_run(forward_batch)              │
│                                                                              │
│  2. 如果可以使用 CUDA Graph:                                                │
│     return self.graph_runner.replay(forward_batch)                          │
│                                                                              │
│  3. 否则根据 forward_mode 分发:                                             │
│     ├── is_decode()        → forward_decode()                               │
│     ├── is_split_prefill() → forward_split_prefill()                       │
│     ├── is_extend()        → forward_extend()                               │
│     └── is_idle()          → forward_idle()                                │
│                                                                              │
│  4. 返回 ModelRunnerOutput(logits_output, can_run_graph)                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 forward_decode

```python
def forward_decode(
    self,
    forward_batch: ForwardBatch,
    skip_attn_backend_init: bool = False,
) -> LogitsProcessorOutput:
    if not skip_attn_backend_init:
        self.attn_backend.init_forward_metadata(forward_batch)
    
    return self.model.forward(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
    )
```

### 4.3 forward_extend

```python
def forward_extend(
    self,
    forward_batch: ForwardBatch,
    skip_attn_backend_init: bool = False,
) -> LogitsProcessorOutput:
    # 尝试 Piecewise CUDA Graph (长序列优化)
    if self.piecewise_cuda_graph_runner and piecewise_runner.can_run(forward_batch):
        return self.piecewise_cuda_graph_runner.replay(forward_batch)
    
    if not skip_attn_backend_init:
        self.attn_backend.init_forward_metadata(forward_batch)
    
    return self.model.forward(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
    )
```

### 4.4 sample

```python
def sample(
    self,
    logits_output: LogitsProcessorOutput,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """采样下一个 token"""
    
    # 1. 预处理 logits (应用 bias, regex mask 等)
    self._preprocess_logits(logits_output, forward_batch.sampling_info)
    
    # 2. 调用 sampler
    next_token_ids = self.sampler(
        logits_output,
        forward_batch.sampling_info,
        forward_batch.return_logprob,
        forward_batch.top_logprobs_nums,
    )
    
    return next_token_ids
```

## 5. CUDA Graph 详解

### 5.1 为什么需要 CUDA Graph?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CUDA Graph 解决的问题                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Decode 阶段特点:                                                            │
│  ├── 每次只生成 1 个 token                                                  │
│  ├── 计算量小 (相对于 Prefill)                                              │
│  └── Kernel launch 开销占比大                                               │
│                                                                              │
│  传统执行:                                                                   │
│    CPU ─┬─launch kernel 1─┬─launch kernel 2─┬─launch kernel 3─┬─...        │
│    GPU  └────compute 1────└────compute 2────└────compute 3────└─...        │
│         ^                 ^                 ^                               │
│         └─ launch 开销 ───┴─ launch 开销 ───┘                               │
│                                                                              │
│  CUDA Graph 执行:                                                            │
│    CPU ─┬─replay graph─┬─(空闲)                                             │
│    GPU  └──────────────┴─compute 1→2→3→...                                 │
│         ^                                                                    │
│         └─ 单次 launch，所有 kernel 连续执行                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 CudaGraphRunner 架构

```python
class CudaGraphRunner:
    def __init__(self, model_runner: ModelRunner):
        # 获取需要捕获的 batch size 列表
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        # 例: capture_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
        # 存储捕获的图
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.output_buffers: Dict[int, LogitsProcessorOutput] = {}
        
        # 输入 buffer (预分配，复用)
        self.buffers: GraphInputBuffers = GraphInputBuffers.create(...)
        
        # 捕获所有 batch size
        self.capture()
```

### 5.3 捕获流程 (capture)

```python
def capture(self):
    # 从大到小捕获，利用内存共享
    for bs in reversed(self.capture_bs):
        with patch_model(self.model, bs in self.compile_bs, ...):
            graph, output = self.capture_one_batch_size(bs, forward)
            self.graphs[bs] = graph
            self.output_buffers[bs] = output
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       capture_one_batch_size 流程                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. 准备输入 buffers (预填充的固定大小张量)                                   │
│     input_ids = buffers.input_ids[:num_tokens]                              │
│     seq_lens = buffers.seq_lens[:bs]                                        │
│     out_cache_loc = buffers.out_cache_loc[:num_tokens]                      │
│                                                                              │
│  2. 构造 ForwardBatch (使用 buffer 张量)                                    │
│                                                                              │
│  3. 初始化 attention backend metadata                                        │
│     attn_backend.init_forward_metadata_capture_cuda_graph(...)              │
│                                                                              │
│  4. Warmup 运行两次 (确保所有 lazy init 完成)                                │
│     for _ in range(2):                                                      │
│         run_once()                                                           │
│                                                                              │
│  5. 捕获 CUDA Graph                                                         │
│     with torch.cuda.graph(graph, pool=memory_pool):                        │
│         output = run_once()                                                 │
│                                                                              │
│  6. 返回 (graph, output)                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 重放流程 (replay)

```python
def replay(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
    # 1. 找到合适的 batch size (向上取整)
    index = bisect.bisect_left(self.capture_bs, forward_batch.batch_size)
    bs = self.capture_bs[index]
    
    # 2. 将实际数据复制到 buffer
    self.buffers.populate_from_forward_batch(forward_batch, ...)
    
    # 3. 更新 attention backend metadata
    attn_backend.init_forward_metadata_replay_cuda_graph(...)
    
    # 4. 重放图
    self.graphs[bs].replay()
    
    # 5. 返回预分配的输出 buffer
    return self.output_buffers[bs]
```

### 5.5 can_run 判断

```python
def can_run(self, forward_batch: ForwardBatch) -> bool:
    bs = forward_batch.batch_size
    
    # 检查条件
    is_bs_supported = bs <= self.max_bs                    # batch size 在捕获范围内
    is_encoder_lens_supported = all(encoder_lens > 0)     # Encoder-Decoder 模型约束
    capture_hidden_mode_matches = ...                      # Hidden state 捕获模式匹配
    is_tbo_supported = forward_batch.can_run_tbo          # Two-Batch Overlap 兼容
    
    return is_bs_supported and is_encoder_lens_supported and ...
```

## 6. Batch Size 填充策略

### 6.1 get_batch_sizes_to_capture

```python
def get_batch_sizes_to_capture(model_runner):
    # 默认捕获列表 (2 的幂次 + 中间值)
    capture_bs = server_args.cuda_graph_bs
    # 例: [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    
    # 添加最大 running requests
    capture_bs += [model_runner.req_to_token_pool.size]
    
    # 过滤超出范围的值
    capture_bs = [bs for bs in capture_bs if bs <= req_to_token_pool.size]
    
    return sorted(set(capture_bs))
```

### 6.2 Batch Size Padding

```
实际 batch_size = 5
捕获的 bs 列表 = [1, 2, 4, 8, 16, ...]

使用 bisect_left 找到 >= 5 的最小值: 8

填充策略:
├── input_ids[:5] 是实际数据
├── input_ids[5:8] 填充 padding
├── seq_lens[:5] 是实际长度
└── seq_lens[5:8] 填充 seq_len_fill_value
```

## 7. torch.compile 集成

### 7.1 配置

```python
def set_torch_compile_config():
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True
    torch._dynamo.config.accumulated_cache_size_limit = 1024
```

### 7.2 patch_model

```python
@contextmanager
def patch_model(model, enable_compile, num_tokens, tp_group):
    if enable_compile:
        # 进入 torch.compile 模式
        _to_torch(model, reverse=False, num_tokens=num_tokens)
        yield torch.compile(
            torch.no_grad()(model.forward),
            mode="max-autotune-no-cudagraphs",
        )
        # 退出 torch.compile 模式
        _to_torch(model, reverse=True, num_tokens=num_tokens)
    else:
        yield model.forward
```

## 8. forward_split_prefill 详解

### 8.1 用途

`forward_split_prefill` 用于 **PDMux (Prefill-Decode Multiplexing)** 场景，将 Prefill 分解为**逐层执行**，与 Decode 交错进行：

```python
def forward_split_prefill(
    self,
    forward_batch: ForwardBatch,
    reinit_attn_backend: bool = False,
    forward_count: int = 1,  # 本次执行几层
) -> LogitsProcessorOutput:
    # 执行 [split_index, next_split_index) 层
    next_split_index = min(
        forward_batch.split_index + forward_count,
        self.model_config.num_hidden_layers,
    )
    ret = self.model.forward_split_prefill(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
        (forward_batch.split_index, next_split_index),
    )
    forward_batch.split_index = next_split_index
    return ret
```

### 8.2 与 Overlap Schedule 配合

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PDMux Split Prefill + Decode Overlap                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stream 0 (Decode):                                                          │
│    ─────decode batch A─────────decode batch A─────────decode batch A────    │
│                                                                              │
│  Stream 1 (Prefill):                                                         │
│    ─layer 0-4─────layer 5-9────layer 10-14───layer 15-19───...─lm_head─     │
│    │← chunk 1 →│← chunk 2 →│← chunk 3 →│← chunk 4 →│                        │
│                                                                              │
│  优势:                                                                       │
│  ├── Prefill 不会阻塞 Decode (低延迟)                                       │
│  ├── 多个 CUDA Stream 并行执行                                              │
│  └── 根据 SM 资源动态分配每次执行的层数                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 9. 三种 CUDA Graph 对比

### 9.1 概览

| 特性 | Decode CUDA Graph | Piecewise CUDA Graph | VIT CUDA Graph |
|------|-------------------|----------------------|----------------|
| **目标阶段** | Decode (is_decode=True) | Prefill/Extend | Vision Encoder |
| **捕获维度** | batch_size | num_tokens | 无独立实现 |
| **典型场景** | 逐 token 生成 | 长序列 Prefill 加速 | 通过 Piecewise 支持 |
| **文件** | cuda_graph_runner.py | piecewise_cuda_graph_runner.py | - |

### 9.2 Decode CUDA Graph

```python
# CudaGraphRunner
capture_forward_mode = ForwardMode.DECODE  # 或 TARGET_VERIFY

# 按 batch_size 捕获
self.capture_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
for bs in self.capture_bs:
    num_tokens = bs * 1  # Decode: 1 token per request
    self.graphs[bs] = capture_one_batch_size(bs, ...)
```

**开启条件**:
- `forward_mode.is_cuda_graph()` → True (DECODE, TARGET_VERIFY, DLLM_EXTEND)
- `batch_size <= max_bs` (在捕获范围内)
- 非 mixed batch (encoder-decoder 场景)
- capture_hidden_mode 匹配

**关闭条件**:
- `--disable-cuda-graph` 参数
- EAGLE Draft Worker 临时禁用 (初始化时)
- batch_size 超出捕获范围
- 返回 logprob 时如果 start_len 条件不满足

### 9.3 Piecewise CUDA Graph

```python
# PiecewiseCudaGraphRunner
capture_forward_mode = ForwardMode.EXTEND  # Prefill 阶段

# 按 num_tokens (不是 batch_size) 捕获
self.capture_num_tokens = [128, 256, 512, 1024, 2048, 4096, ...]
for num_tokens in self.capture_num_tokens:
    self.graphs[num_tokens] = capture_one_batch_size(num_tokens, ...)
```

**关键区别**:
1. **捕获的是 Language Model 部分** (不含 VIT)
2. **使用 `input_embeds`** 作为输入 (VIT 输出复制到固定 buffer)
3. **支持 torch.compile** (inductor/eager 编译器)

### 9.4 VIT CUDA Graph (Vision Encoder)

> **SGLang 有独立的 VIT CUDA Graph！** 通过 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 启用。

**核心实现**: `srt/multimodal/vit_cuda_graph_runner.py`

```python
class ViTCudaGraphRunner:
    """捕获 blocks + merger + deepstack merger 部分到 CUDA Graph"""
    
    def __init__(self, vit: nn.Module):
        self.block_graphs: Dict[int, torch.cuda.CUDAGraph] = {}  # seq_len -> graph
        self.block_output: Dict[int, torch.Tensor] = {}
        self.block_input: Dict[int, torch.Tensor] = {}
    
    def _get_graph_key(self, x_3d: torch.Tensor) -> int:
        return x_3d.shape[0]  # 使用 seq_len 作为 key (不是 batch_size!)
    
    def run(self, x, cu_seqlens, ...):
        graph_key = self._get_graph_key(x_3d)
        if graph_key not in self.block_graphs:
            self.create_graph(...)  # 懒加载创建
        return self.replay(graph_key, x_3d, ...)
```

**VIT CUDA Graph 特点**:
1. **按 seq_len 捕获** (不是 batch_size)，因为 image patch 数量决定 seq_len
2. **懒加载捕获** - 首次遇到新 shape 时创建图，非启动时预捕获
3. **支持 Qwen2.5-VL 窗口注意力** (`fullatt_block_indexes`, `cu_window_seqlens`)
4. **支持 Qwen3-VL deepstack** (`deepstack_visual_indexes`, `deepstack_merger_list`)

**Qwen3-VL 调用流程**:
```python
# qwen3_vl.py forward()
def forward(self, x, grid_thw):
    if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
        return self.forward_with_cuda_graph(x, grid_thw)
    # 正常执行...

def forward_with_cuda_graph(self, x, grid_thw):
    x = self.patch_embed(x)          # 不在 Graph 中
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)  # 不在 Graph 中
    x += pos_embeds
    # blocks + merger + deepstack 通过 CUDA Graph 执行
    return self.cuda_graph_runner.run(x=x, ...)
```

**开启条件**:
```bash
export SGLANG_VIT_ENABLE_CUDA_GRAPH=1
```

**注意**:
- VIT CUDA Graph **不使用 torch.compile**
- 仅支持 `triton_attn` 和 `fa3` attention backend
- `patch_embed` 和 `pos_embed` 在图外执行

### 9.5 开启/关闭条件总结

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CUDA Graph 开启/关闭条件                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Decode CUDA Graph:                                                          │
│  ├── 开启: forward_mode ∈ {DECODE, TARGET_VERIFY, DLLM_EXTEND}             │
│  │         AND batch_size <= max_bs                                         │
│  │         AND graph_runner.can_run() = True                                │
│  └── 关闭: --disable-cuda-graph                                             │
│            OR batch_size 超出范围                                            │
│            OR encoder_lens 有 0 值 (混合批次)                                │
│            OR capture_hidden_mode 不匹配                                     │
│                                                                              │
│  Piecewise CUDA Graph (LLM Prefill):                                         │
│  ├── 开启: --enable-piecewise-cuda-graph                                    │
│  │         AND num_tokens <= max_num_tokens                                 │
│  │         AND forward_mode = EXTEND                                        │
│  └── 关闭: 未启用该参数                                                      │
│            OR num_tokens > max_num_tokens                                   │
│            OR 返回 logprob 且 start_len < seq_len                           │
│            OR PP (Pipeline Parallel) 启用                                   │
│            OR torch_compile 同时启用 (冲突)                                  │
│                                                                              │
│  VIT CUDA Graph:                                                             │
│  ├── 开启: SGLANG_VIT_ENABLE_CUDA_GRAPH=1                                  │
│  │         AND mm_attention_backend ∈ {triton_attn, fa3}                   │
│  └── 关闭: 未设置环境变量                                                    │
│            OR 使用不支持的 attention backend                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.6 流量波动场景分析

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      流量波动对 CUDA Graph 的影响                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  低流量 (batch_size 小):                                                     │
│  ├── Decode CUDA Graph: ✓ 正常使用，padding 到最小捕获 bs                    │
│  └── Piecewise CUDA Graph: ✓ 正常使用                                       │
│                                                                              │
│  高流量 (batch_size 大):                                                     │
│  ├── Decode CUDA Graph:                                                     │
│  │   ├── bs <= max_bs: ✓ padding 到合适的 bs                                │
│  │   └── bs > max_bs: ✗ 降级到普通执行                                      │
│  └── Piecewise CUDA Graph: 取决于 num_tokens                                │
│                                                                              │
│  流量波动时:                                                                 │
│  ├── batch_size 变化: 选择不同的预捕获图 (bisect_left)                       │
│  ├── Padding 开销: 小 bs 需要更多 padding                                   │
│  └── 无需重新捕获图 (启动时已捕获所有 bs)                                    │
│                                                                              │
│  VIT 流量波动:                                                               │
│  ├── 不同 image size → 不同 seq_len → 不同 graph                           │
│  ├── 懒加载: 首次遇到新 size 时捕获 (有一次性开销)                           │
│  └── 图会累积: 处理过的 size 都会缓存                                        │
│                                                                              │
│  Decode 阶段是否会关闭 CUDA Graph?                                           │
│  └── 通常不会！只要 batch_size <= max_bs 就会使用                            │
│      max_bs 默认是 max_running_requests，一般足够大                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 10. torch.compile vs CUDA Graph 对比

### 10.1 技术原理对比

| 特性 | torch.compile | CUDA Graph |
|------|---------------|------------|
| **层级** | Python/PyTorch 图编译 | CUDA Runtime 级别 |
| **优化目标** | kernel fusion, 算子优化 | 消除 kernel launch 开销 |
| **动态性** | 支持动态 shape (有限制) | **固定 shape** |
| **JIT 编译** | 是 (首次运行慢) | 否 (捕获时一次性开销) |
| **适用阶段** | Compute-bound (Prefill) | Memory-bound (Decode) |

### 10.2 核心区别

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    torch.compile vs CUDA Graph 原理                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  torch.compile:                                                              │
│  ├── 分析 Python 代码，生成优化的 kernel                                    │
│  ├── Kernel Fusion: 多个小算子合并成一个大 kernel                           │
│  ├── Memory Planning: 优化中间变量分配                                      │
│  └── 适合: Prefill 阶段 (compute-bound, 计算密集)                           │
│                                                                              │
│  CUDA Graph:                                                                 │
│  ├── 录制一系列 CUDA kernel 调用                                            │
│  ├── 重放时: 单次 CPU 提交，GPU 连续执行所有 kernel                         │
│  ├── 消除: CPU→GPU 的 launch overhead                                       │
│  └── 适合: Decode 阶段 (memory-bound, launch overhead 占比大)               │
│                                                                              │
│  关键约束:                                                                   │
│  ├── CUDA Graph: 输入 shape 必须固定 (所以需要 padding)                     │
│  └── torch.compile: 可支持有限动态 shape (但需 recompile)                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3 组合使用场景

```python
# 1. Decode CUDA Graph + torch.compile (CudaGraphRunner)
class CudaGraphRunner:
    def __init__(self):
        if server_args.enable_torch_compile:
            # 在 CUDA Graph 捕获前先 torch.compile
            self.compile_bs = [bs for bs in self.capture_bs if bs <= torch_compile_max_bs]
    
    def capture_one_batch_size(self, bs, forward):
        with patch_model(model, bs in self.compile_bs):
            # 先 compile 优化 kernel，再捕获成 Graph
            graph = capture(forward(...))

# 2. Piecewise CUDA Graph + torch.compile (PiecewiseCudaGraphRunner)
class PiecewiseCudaGraphRunner:
    def __init__(self):
        # 必须配合使用
        self.compile_config = CompilationConfig(
            piecewise_cuda_graph_tokens,
            piecewise_cuda_graph_compiler,  # "eager" or "inductor"
        )
        with set_compiled(True), enable_piecewise_cuda_graph_compile():
            self.warmup_torch_compile(num_tokens)
            self.capture()

# 3. VIT CUDA Graph (不使用 torch.compile)
class ViTCudaGraphRunner:
    # 直接捕获，无 torch.compile
    pass
```

### 10.4 各模块组合使用情况

| 组件 | torch.compile | CUDA Graph | 说明 |
|------|---------------|------------|------|
| **Decode (LLM)** | ✓ 可选 | ✓ 默认开启 | torch.compile 优化 kernel，Graph 减少 launch |
| **Prefill (LLM)** | - | Piecewise 可选 | Piecewise 用于 MoE 模型，内置 torch.compile |
| **VIT Encoder** | ✗ | ✓ 可选 | 仅用 Graph，不用 compile |
| **EAGLE Draft** | ✓ 可选 | ✓ 可选 | 类似 Decode |

### 10.5 如何选择

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           组合使用建议                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  场景 1: 标准 LLM 推理                                                       │
│  └── 使用: Decode CUDA Graph (默认开启)                                     │
│      可选: --enable-torch-compile (小 bs 时 kernel 可进一步优化)            │
│                                                                              │
│  场景 2: MoE 模型 (DeepSeek, Qwen-MoE)                                       │
│  └── 使用: --enable-piecewise-cuda-graph                                    │
│      原因: MoE 的 EP 通信与 torch.compile 结合更好                           │
│                                                                              │
│  场景 3: 多模态 (Qwen2.5-VL, Qwen3-VL)                                       │
│  └── 使用: SGLANG_VIT_ENABLE_CUDA_GRAPH=1                                   │
│      原因: VIT 部分 shape 固定时可加速                                       │
│      注意: VIT 不使用 torch.compile                                         │
│                                                                              │
│  场景 4: 最大吞吐                                                            │
│  └── 禁用 compile (编译开销大)，仅用 CUDA Graph                              │
│                                                                              │
│  场景 5: 调试 / 开发                                                         │
│  └── --disable-cuda-graph (方便 debug，不影响 VIT Graph)                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.6 Piecewise CUDA Graph 代码示例

用于长序列 Prefill 的优化 (进一步代码细节可参考 §9.3):

```python
def init_piecewise_cuda_graphs(self):
    # 收集 attention 和 MoE 层
    for layer in self.model.model.layers:
        if hasattr(layer, "self_attn"):
            self.attention_layers.append(layer.self_attn.attn)
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            self.moe_layers.append(layer.mlp.experts)
    
    self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)
```

## 11. 初始化流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ModelRunner.initialize 流程                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. init_torch_distributed()                                                 │
│     ├── 初始化 TP/PP 进程组                                                  │
│     └── 设置 NCCL 通信                                                       │
│                                                                              │
│  2. load_model()                                                             │
│     ├── 加载模型权重                                                         │
│     └── 应用量化 (如果启用)                                                  │
│                                                                              │
│  3. init_attention_backend()                                                 │
│     └── 选择 FlashInfer/FlashAttention/Triton 后端                          │
│                                                                              │
│  4. init_lora_manager() (如果启用)                                           │
│                                                                              │
│  5. kernel_warmup()                                                          │
│     └── FlashInfer autotune                                                 │
│                                                                              │
│  6. init_device_graphs()                                                     │
│     ├── 创建 CudaGraphRunner                                                │
│     └── 捕获所有 batch size                                                  │
│                                                                              │
│  7. init_piecewise_cuda_graphs() (如果启用)                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 12. 内存管理

### 12.1 全局图内存池

```python
# cuda_graph_runner.py
global_graph_memory_pool = None

def get_global_graph_memory_pool():
    return global_graph_memory_pool

def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val

# 首次捕获时设置
if get_global_graph_memory_pool() is None:
    set_global_graph_memory_pool(torch.cuda.graph_pool_handle())
```

### 12.2 GraphInputBuffers

```python
@dataclass
class GraphInputBuffers:
    input_ids: torch.Tensor           # [max_num_token]
    req_pool_indices: torch.Tensor    # [max_bs]
    seq_lens: torch.Tensor            # [max_bs]
    seq_lens_cpu: torch.Tensor        # [max_bs] (host)
    out_cache_loc: torch.Tensor       # [max_num_token]
    positions: torch.Tensor           # [max_num_token]
    
    # 用于 CUDA Graph replay 时填充数据
    def populate_from_forward_batch(self, forward_batch, ...):
        self.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        self.seq_lens[:bs].copy_(forward_batch.seq_lens)
        ...
```

## 13. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `disable_cuda_graph` | False | 禁用 CUDA Graph |
| `cuda_graph_bs` | auto | 捕获的 batch size 列表 |
| `enable_torch_compile` | False | 启用 torch.compile |
| `torch_compile_max_bs` | 32 | torch.compile 最大 bs |
| `enable_piecewise_cuda_graph` | False | 启用分段 CUDA Graph |
| `disable_cuda_graph_padding` | False | 禁用 bs padding |

## 14. 调试技巧

### 14.1 查看捕获的 batch size

```python
# 启动日志中会显示
logger.info(f"Capture cuda graph bs {self.capture_bs}")
```

### 14.2 禁用 CUDA Graph 调试

```bash
python -m sglang.launch_server ... --disable-cuda-graph
```

### 14.3 Profile CUDA Graph

```python
# 启用 profiler
--enable-profile-cuda-graph
# 输出: cuda_graph_runner_memory_usage.pickle
```

## 15. 下一步

- **08**: Attention 后端 (FlashInfer, FlashAttention, Triton)
- **09**: Chunked Prefill 详解
