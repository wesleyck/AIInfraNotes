# 20. Qwen3-VL 模型解析

> **核心问题**: Qwen3-VL 作为 vLLM 中典型的多模态模型，其架构如何设计？
>
> **关键文件**:
> - `vllm/model_executor/models/qwen3_vl.py` - Qwen3-VL 完整实现
> - `vllm/model_executor/models/qwen2_5_vl.py` - 复用的 VisionAttention
> - `vllm/model_executor/models/vision.py` - Vision 通用工具

---

## 20.1 Qwen3-VL 整体架构

Qwen3-VL 是阿里巴巴推出的多模态大语言模型，其架构由三部分组成：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Qwen3-VL Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                 Visual Encoder (ViT)                           │ │
│  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐  │ │
│  │  │ Patch Embed  │→│ Transformer │→│ Merger (Deepstack)   │  │ │
│  │  │ Conv3D       │  │ Blocks      │  │ 多尺度特征融合       │  │ │
│  │  └──────────────┘  └─────────────┘  └──────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                 Vision-Language Fusion                         │ │
│  │  将视觉 embeddings 替换到文本 embeddings 中                     │ │
│  │  + MRoPE (3D 位置编码)                                         │ │
│  │  + Deepstack 多尺度特征注入                                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              Language Model (Qwen3ForCausalLM)                 │ │
│  │  标准 Transformer Decoder + 特殊的 Deepstack Attention         │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键特性

| 特性 | 说明 |
|------|------|
| **3D Patch Embedding** | 使用 Conv3D 同时处理时空维度 |
| **MRoPE** | 3D 旋转位置编码 (时间、高度、宽度) |
| **Deepstack** | 从 ViT 中间层提取多尺度特征 |
| **动态分辨率** | 支持任意分辨率图片/视频 |
| **视频理解** | 原生支持视频帧序列 |

---

## 20.2 Vision Transformer 详解

### 20.2.1 Qwen3_VisionPatchEmbed - 3D Patch Embedding

```python
# vllm/model_executor/models/qwen3_vl.py:142-168

class Qwen3_VisionPatchEmbed(nn.Module):
    """将像素值转换为 patch embeddings"""
    
    def __init__(
        self,
        patch_size: int = 14,         # 空间 patch 大小
        temporal_patch_size: int = 2, # 时间 patch 大小 (用于视频)
        in_channels: int = 3,         # RGB
        hidden_size: int = 1152,      # ViT hidden size
    ):
        super().__init__()
        # 3D 卷积: 同时处理时间和空间维度
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3dLayer(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,  # 非重叠 patch
            bias=True,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [L, C] 其中 L = T * H * W * patch_size^2 * temporal_patch_size
        L, C = x.shape
        # reshape 为 5D: [L, frames_per_patch, temp_patch, patch, patch]
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        # 3D 卷积
        x = self.proj(x).view(L, self.hidden_size)
        return x
```

**设计要点**：
- 使用 `Conv3D` 而非 `Conv2D`，天然支持视频的时间维度
- `temporal_patch_size=2` 意味着每 2 帧合并为一个 token
- 对于图片，可以理解为 `T=1` 的退化情况

### 20.2.2 Qwen3_VisionTransformer - 完整 ViT

```python
# vllm/model_executor/models/qwen3_vl.py:312-607

class Qwen3_VisionTransformer(nn.Module):
    def __init__(self, vision_config, ...):
        # 核心参数
        self.hidden_size = vision_config.hidden_size           # 1152
        self.num_heads = vision_config.num_heads               # 16
        self.patch_size = vision_config.patch_size             # 14
        self.spatial_merge_size = vision_config.spatial_merge_size  # 2
        self.temporal_patch_size = vision_config.temporal_patch_size  # 2
        
        # Deepstack: 从中间层提取多尺度特征
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        # 例如: [7, 15] 表示从第 7 和第 15 层提取特征
        
        # 模块
        self.patch_embed = Qwen3_VisionPatchEmbed(...)
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.rotary_pos_emb = get_rope(...)  # 2D RoPE for ViT
        
        self.blocks = nn.ModuleList([
            Qwen3_VisionBlock(...) for _ in range(depth)
        ])
        
        # 主 merger
        self.merger = Qwen3_VisionPatchMerger(...)
        
        # Deepstack mergers (每个中间层一个)
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3_VisionPatchMerger(...) 
            for _ in range(len(deepstack_visual_indexes))
        ])
```

### 20.2.3 Forward 流程

```python
# vllm/model_executor/models/qwen3_vl.py:532-580

def forward(self, x: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
    """
    Args:
        x: [total_patches, channels] 所有图片/视频帧的像素值
        grid_thw: [[t1,h1,w1], [t2,h2,w2], ...] 每个图片/视频的网格大小
    
    Returns:
        [total_tokens, out_hidden_size * (1 + num_deepstack_levels)]
    """
    # 1. Patch Embedding
    hidden_states = self.patch_embed(x)  # [L, hidden_size]
    
    # 2. 位置编码 (插值支持任意分辨率)
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds
    
    # 3. 计算 RoPE (2D)
    rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw)
    
    # 4. 计算 cu_seqlens (用于 FlashAttention varlen)
    cu_seqlens = compute_cu_seqlens(grid_thw)
    
    # 5. Transformer Blocks
    hidden_states = hidden_states.unsqueeze(1)  # [L, 1, hidden_size]
    deepstack_features = []
    
    for layer_idx, block in enumerate(self.blocks):
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
        )
        
        # Deepstack: 从指定层提取特征
        if layer_idx in self.deepstack_visual_indexes:
            idx = self.deepstack_visual_indexes.index(layer_idx)
            deepstack_feature = self.deepstack_merger_list[idx](hidden_states)
            deepstack_features.append(deepstack_feature)
    
    # 6. 主 merger: 空间降采样
    hidden_states = self.merger(hidden_states)  # [L/4, out_hidden_size]
    
    # 7. 拼接 deepstack 特征
    hidden_states = torch.cat(
        [hidden_states] + deepstack_features, 
        dim=1
    )  # [L/4, out_hidden_size * (1 + num_deepstack)]
    
    return hidden_states
```

---

## 20.3 Deepstack 多尺度特征融合

### 20.3.1 设计动机

传统 VLM 只使用 ViT 最后一层的输出，但研究表明：
- **浅层特征**：细节丰富，适合细粒度理解
- **深层特征**：语义丰富，适合高级理解

Qwen3-VL 的 Deepstack 机制同时利用多个层的特征：

```
ViT Block 0  →  ...
ViT Block 7  → Deepstack Merger 0 → feature_1
     ...
ViT Block 15 → Deepstack Merger 1 → feature_2
     ...
ViT Block 31 → Main Merger        → feature_main

Final Output = [feature_main, feature_1, feature_2]
             = [seq_len, hidden * 3]
```

### 20.3.2 Qwen3_VisionPatchMerger

```python
# vllm/model_executor/models/qwen3_vl.py:260-309

class Qwen3_VisionPatchMerger(nn.Module):
    """将 2x2 的 patch 合并为 1 个，同时投影到 LLM hidden size"""
    
    def __init__(
        self,
        d_model: int,           # LLM hidden size (如 8192)
        context_dim: int,       # ViT hidden size (如 1152)
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,  # Deepstack 使用
        ...
    ):
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        # 1152 * 4 = 4608
        
        self.norm = nn.LayerNorm(...)
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,  # 4608
            self.hidden_size,  # 4608
            ...
        )
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,  # 4608
            d_model,           # 8192
            ...
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq, 1, hidden_size]
        # 每 2x2 个 patch 合并
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x  # [seq/4, d_model]
```

**效果**：
- 输入: `[1024, 1, 1152]` (32x32 patches)
- 输出: `[256, 8192]` (16x16 tokens, LLM hidden size)
- 空间分辨率降低 4 倍 (2x2 → 1)

### 20.3.3 Deepstack 在 LLM 中的注入

```python
# vllm/model_executor/models/qwen3_vl.py:1918-1957

def _compute_deepstack_embeds(
    self,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: MultiModalEmbeddings,
    is_multimodal: torch.Tensor,
) -> tuple[torch.Tensor, MultiModalEmbeddings]:
    """分离主特征和多尺度特征"""
    
    # 拼接所有 multimodal embeddings
    mm_cat = torch.cat(multimodal_embeddings, dim=0)
    # [total_mm_tokens, hidden * (1 + num_deepstack)]
    
    # 分离主特征和多尺度特征
    mm_main, mm_multiscale = torch.split(
        mm_cat,
        [self.visual_dim, self.multiscale_dim],  # [hidden, hidden*2]
        dim=-1,
    )
    
    # 主特征用于替换 placeholder
    # 多尺度特征存储起来，在 LLM 的特定层注入
    deepstack_embeds = ...  # [num_levels, seq_len, hidden]
    
    return deepstack_embeds, mm_main
```

Deepstack 特征会在 LLM 的 forward 中通过 **Cross-Attention** 注入：
- 将多尺度视觉特征作为额外的 K/V 输入
- 在特定层（与 ViT 提取层对应）进行注入
- 增强模型对视觉细节的理解

---

## 20.4 MRoPE (Multi-head Rotary Position Embedding)

### 20.4.1 3D 位置编码

Qwen3-VL 使用 **3D RoPE** 而非传统的 1D RoPE：

```
传统 1D RoPE:
  position = [0, 1, 2, 3, 4, 5, ...]  # 一维序列

MRoPE (3D):
  position = [
    [t0, t0, t0, t1, t1, t1, ...],  # 时间维度
    [h0, h1, h2, h0, h1, h2, ...],  # 高度维度
    [w0, w1, w2, w0, w1, w2, ...],  # 宽度维度
  ]  # shape: [3, seq_len]
```

### 20.4.2 位置计算

```python
# vllm/model_executor/models/qwen3_vl.py:1821-1887

def compute_mrope_positions(
    self, input_tokens, mm_features
) -> tuple[torch.Tensor, int]:
    """计算 MRoPE 位置编码"""
    
    llm_pos_ids_list = []
    st = 0
    
    for offset, llm_grid_h, llm_grid_w in self.iter_mm_grid_hw(input_tokens, mm_features):
        # 处理文本段
        text_len = offset - st
        st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
        
        # 文本使用 1D 位置 (三个维度相同)
        text_positions = np.broadcast_to(
            np.arange(text_len), 
            (3, text_len)
        ) + st_idx
        llm_pos_ids_list.append(text_positions)
        st_idx += text_len
        
        # 处理视觉段 - 使用 3D 位置
        grid_indices = np.indices((1, llm_grid_h, llm_grid_w))
        # grid_indices.shape = [3, 1, H, W]
        frame_positions = grid_indices.reshape(3, -1) + st_idx
        llm_pos_ids_list.append(frame_positions)
        
        st = offset + actual_frame_tokens
    
    # 拼接所有位置
    llm_positions = np.concatenate(llm_pos_ids_list, axis=1)
    # shape: [3, total_seq_len]
    
    return torch.from_numpy(llm_positions), mrope_position_delta
```

### 20.4.3 示例

```
Input: "描述图片" + [16x16 image tokens] + "的内容"

文本 "描述图片":
  positions = [[0,1,2,3], [0,1,2,3], [0,1,2,3]]  # 3 个维度相同

图片 16x16:
  T 维度: [[4,4,4,...], ...]  # 时间都是 4
  H 维度: [[0,0,0,0,1,1,1,1,...], ...]  # 行号
  W 维度: [[0,1,2,3,0,1,2,3,...], ...]  # 列号

文本 "的内容":
  positions = [[260,261,262], [260,261,262], [260,261,262]]
```

---

## 20.5 embed_multimodal 与 embed_input_ids

### 20.5.1 embed_multimodal - 执行 ViT 编码

```python
# vllm/model_executor/models/qwen3_vl.py:1889-1916

def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
    """执行视觉编码器，返回视觉 embeddings"""
    
    # 1. 解析输入
    mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
    if not mm_input_by_modality:
        return None
    
    multimodal_embeddings: tuple[torch.Tensor, ...] = ()
    
    # 2. 按 modality 处理
    for modality in mm_input_by_modality:
        if modality == "image":
            image_input = mm_input_by_modality["image"]
            image_embeddings = self._process_image_input(image_input)
            # 可选: EVS (视频采样) 后处理
            if self.is_multimodal_pruning_enabled:
                image_embeddings = self._postprocess_image_embeds_evs(...)
            multimodal_embeddings += tuple(image_embeddings)
        
        if modality == "video":
            video_input = mm_input_by_modality["video"]
            video_embeddings = self._process_video_input(video_input)
            if self.is_multimodal_pruning_enabled:
                video_embeddings = self._postprocess_video_embeds_evs(...)
            multimodal_embeddings += tuple(video_embeddings)
    
    return multimodal_embeddings
```

### 20.5.2 embed_input_ids - 合并文本和视觉 embeddings

```python
# vllm/model_executor/models/qwen3_vl.py:1959-2000

def embed_input_ids(
    self,
    input_ids: torch.Tensor,
    multimodal_embeddings: MultiModalEmbeddings | None = None,
    *,
    is_multimodal: torch.Tensor | None = None,
    ...
) -> torch.Tensor:
    """将视觉 embeddings 合并到文本 embeddings 中"""
    
    # 1. 获取文本 embeddings
    inputs_embeds = self._embed_text_input_ids(
        input_ids,
        self.language_model.embed_input_ids,
        is_multimodal=is_multimodal,
        ...
    )
    
    if multimodal_embeddings is None:
        return inputs_embeds
    
    # 2. 处理 Deepstack
    if self.use_deepstack:
        deepstack_embeds, multimodal_embeddings = self._compute_deepstack_embeds(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
    else:
        deepstack_embeds = None
    
    # 3. 将视觉 embeddings 替换到对应位置
    inputs_embeds = _merge_multimodal_embeddings(
        inputs_embeds=inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )
    
    # 4. 保存 Deepstack embeddings (用于 LLM forward)
    if deepstack_embeds is not None:
        self._set_deepstack_input_embeds(deepstack_embeds)
    
    return inputs_embeds
```

---

## 20.6 与 vLLM 推理系统的集成

### 20.6.1 接口实现

Qwen3-VL 实现了多个 vLLM 接口：

```python
@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor, ...)
class Qwen3VLForConditionalGeneration(
    SupportsMultiModal,           # 支持多模态输入
    SupportsMRoPE,               # 支持 3D RoPE
    SupportsLoRA,                # 支持 LoRA
    SupportsPP,                  # 支持 Pipeline Parallelism
    SupportsMultiModalPruning,   # 支持 EVS (视频采样)
    SupportsEagle3,              # 支持 Eagle3 投机采样
):
    ...
```

### 20.6.2 关键方法映射

| vLLM 接口 | Qwen3-VL 实现 |
|-----------|---------------|
| `embed_multimodal()` | 执行 ViT，返回视觉 embeddings |
| `embed_input_ids()` | 合并文本和视觉 embeddings |
| `get_mrope_positions()` | 计算 3D 位置编码 |
| `forward()` | LLM forward (含 Deepstack 注入) |

### 20.6.3 调用流程

```
SchedulerOutput
    ↓
EncoderRunner.execute_mm_encoder()
    ↓
model.embed_multimodal(pixel_values=..., image_grid_thw=...)
    ↓ 返回
encoder_cache[mm_hash] = visual_embeddings  # 缓存
    ↓
EncoderRunner.get_inputs_embeds()
    ↓
model.embed_input_ids(input_ids, mm_embeds, is_mm_embed)
    ↓ 返回
mixed_embeddings  # 可以直接送入 LLM
    ↓
model.forward(inputs_embeds=mixed_embeddings, positions=mrope_positions, ...)
```

---

## 20.7 性能优化

### 20.7.1 Vision Attention Backend

```python
# vllm/model_executor/models/qwen3_vl.py:379-391

self.attn_backend = get_vit_attn_backend(
    head_size=head_dim,
    dtype=torch.get_default_dtype(),
)

if self.attn_backend not in {
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.TORCH_SDPA,
    AttentionBackendEnum.ROCM_AITER_FA,
}:
    raise RuntimeError(f"Qwen3-VL does not support {self.attn_backend}")
```

### 20.7.2 数据并行 ViT

```python
# vllm/model_executor/models/qwen3_vl.py:182-200

class Qwen3_VisionMLP(nn.Module):
    def __init__(self, ...):
        use_data_parallel = is_vit_use_data_parallel()
        
        self.linear_fc1 = ColumnParallelLinear(
            ...,
            disable_tp=use_data_parallel,  # DP 模式下禁用 TP
        )
```

通过 `mm_encoder_tp_mode="data"` 配置，可以在多 GPU 间使用数据并行而非张量并行处理 ViT。

### 20.7.3 位置编码缓存

```python
# vllm/model_executor/models/qwen3_vl.py:415-440

@staticmethod
@lru_cache(maxsize=1024)  # 缓存常用的位置编码
def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
    ...
```

---

## 20.8 小结

Qwen3-VL 的核心创新：

| 特性 | 设计 | 优势 |
|------|------|------|
| **3D Patch Embed** | Conv3D | 原生支持视频 |
| **MRoPE** | 3D 位置编码 | 保留空间结构 |
| **Deepstack** | 多层特征提取 | 多尺度理解 |
| **动态分辨率** | 位置编码插值 | 支持任意图片大小 |
| **FlashAttention** | 变长 attention | 高效处理多图片 |

```
架构图:

Input: text + [img1] + text + [img2] + ...

           pixel_values_1        pixel_values_2
                 ↓                     ↓
         ┌───────────────────────────────────────┐
         │           Qwen3_VisionTransformer      │
         │  ┌─────────┐  ┌─────────┐  ┌───────┐  │
         │  │ Block 0 │→│   ...   │→│Block31│  │
         │  └─────────┘  └─────────┘  └───────┘  │
         │       ↓            ↓           ↓      │
         │   (deepstack)  (deepstack)  (main)   │
         └───────────────────────────────────────┘
                         ↓
         [visual_emb_1, visual_emb_2]  (每个含 deepstack)
                         ↓
         ┌───────────────────────────────────────┐
         │          embed_input_ids()             │
         │  替换 placeholder → mixed_embeddings   │
         └───────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────────────┐
         │           Qwen3ForCausalLM            │
         │  forward(inputs_embeds, mrope_pos)    │
         │  + deepstack cross-attention           │
         └───────────────────────────────────────┘
                         ↓
                    logits
```

---

> **下一节**: [21-vit-implementation.md](./21-vit-implementation.md) - ViT 实现与优化
