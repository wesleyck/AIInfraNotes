# 21. ViT 实现与优化

> **核心问题**: 
> 1. 开始 chunk 时，ViT 的输入始终是整个图片还是部分？
> 2. ViT 有没有特殊的分片凑 batch 处理？
>
> **关键文件**:
> - `vllm/v1/worker/gpu/mm/encoder_runner.py` - Encoder 执行器
> - `vllm/model_executor/models/vision.py` - ViT 通用工具
> - `vllm/multimodal/utils.py` - 多模态工具函数

---

## 21.1 核心问题解答：ViT 输入是整张图片还是部分？

### 答案：ViT 始终处理整张图片，不会分片

**原因**：ViT 使用**双向 Self-Attention**，每个 patch 需要看到所有其他 patches，因此必须一次性输入完整图片。

```
                    整张图片 (224x224)
                           ↓
        ┌─────────────────────────────────────┐
        │           Patch Embedding           │
        │   224x224 → 16x16 = 256 patches     │
        └─────────────────────────────────────┘
                           ↓
        ┌─────────────────────────────────────┐
        │     ViT Transformer (双向 Attn)      │
        │  每个 patch attend 到所有 256 个     │
        │  必须一次性看到完整图片               │
        └─────────────────────────────────────┘
                           ↓
                  256 visual tokens
                           ↓
              encoder_cache[mm_hash] = tokens
```

### 代码证据

```python
# vllm/v1/worker/gpu/mm/encoder_runner.py:62-88

@torch.inference_mode()
def execute_mm_encoder(
    self,
    model: SupportsMultiModal,
    mm_hashes: list[str],
    mm_kwargs: list[MultiModalKwargsItem],
) -> list[torch.Tensor]:
    """执行多模态 encoder，一次性处理完整输入"""
    
    if not mm_hashes:
        return []
    
    encoder_outputs: list[torch.Tensor] = []
    
    # 按 modality 分组批量处理
    for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
        mm_kwargs,
        device=self.device,
        pin_memory=False,
    ):
        # 关键: 一次性执行 embed_multimodal
        # pixel_values 包含完整图片数据
        curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)
        sanity_check_mm_encoder_outputs(curr_group_outputs, expected_num_items=num_items)
        encoder_outputs.extend(curr_group_outputs)
    
    # 缓存完整的 encoder 输出
    for mm_hash, output in zip(mm_hashes, encoder_outputs):
        self.encoder_cache[mm_hash] = output
    
    return encoder_outputs
```

### "Chunked MM Input" 的真正含义

`disable_chunked_mm_input` 配置控制的是 **Decoder 端** 如何消费 ViT 输出，而非 ViT 本身：

```python
# vllm/v1/core/sched/scheduler.py:1133-1143

# 如果禁用 chunked mm input，不允许部分调度多模态 placeholder
if (
    self.scheduler_config.disable_chunked_mm_input
    and num_computed_tokens < start_pos
    and (num_computed_tokens + num_new_tokens) < (start_pos + num_encoder_tokens)
):
    # 回退到多模态项之前
    num_new_tokens = start_pos - num_computed_tokens
    break
```

**示例**：
```
Prompt: "描述这张图片 <placeholder...336个> 的内容"
          ↑ tokens 0-5    ↑ tokens 6-341    ↑ tokens 342-345

场景: 当前 budget 只能处理 100 个 tokens

disable_chunked_mm_input = False (默认):
  Step 1: tokens 0-100 (含部分 placeholder)
  Step 2: tokens 100-200 (placeholder 继续)
  Step 3: tokens 200-345 (完成)
  
  ViT 只在 Step 1 执行一次，输出缓存在 encoder_cache

disable_chunked_mm_input = True:
  Step 1: tokens 0-5 (文本，不进入 placeholder)
  Step 2: tokens 6-341 (完整 placeholder + ViT 执行)
  Step 3: tokens 342-345 (剩余文本)
  
  确保 placeholder tokens 不被拆分
```

---

## 21.2 核心问题解答：ViT 的分片凑 Batch 处理

### 答案：有三种优化策略

1. **跨请求 Batch**: 多个请求的图片合并成一个 batch
2. **Data Parallel 分片**: 在多 GPU 间按 batch 维度分片
3. **MRoPE 负载均衡**: 针对不同大小图片的智能分配

---

## 21.3 跨请求 Batch 处理

### 21.3.1 group_mm_kwargs_by_modality

```python
# vllm/multimodal/utils.py:469-496

def group_mm_kwargs_by_modality(
    mm_kwargs: list[MultiModalKwargsItem],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[str, int, BatchedTensorInputs], None, None]:
    """将多个请求的多模态输入按 modality 分组批量处理
    
    Args:
        mm_kwargs: 来自多个请求的 MultiModalKwargsItem 列表
    
    Yields:
        (modality, num_items, batched_kwargs) 三元组
    """
    from vllm.multimodal.inputs import MultiModalKwargsItems
    
    # 按 modality 分组 (images 一起, videos 一起)
    for modality, items in groupby(mm_kwargs, key=lambda item: item.modality):
        items_lst = list(items)
        mm_kwargs_items = MultiModalKwargsItems.from_seq(items_lst)
        
        # 合并成 batched tensors
        mm_kwargs_data = mm_kwargs_items.get_data(
            device=device,
            pin_memory=pin_memory,
        )
        
        yield modality, len(items_lst), mm_kwargs_data
```

### 21.3.2 Batch 组装示例

```
请求 A: image_1 (224x224)
请求 B: image_2 (384x384)  
请求 C: image_3 (224x224)
请求 D: video_1 (10 frames)

分组后:

Batch 1 (image modality):
  pixel_values: [img1_patches, img2_patches, img3_patches]
  image_grid_thw: [[1,16,16], [1,28,28], [1,16,16]]
  
Batch 2 (video modality):
  pixel_values_videos: [video1_patches]
  video_grid_thw: [[10,16,16]]

每个 batch 调用一次 model.embed_multimodal()
```

### 21.3.3 结果拆分

```python
# vllm/v1/worker/gpu/mm/encoder_runner.py:85-88

# 执行后按 mm_hash 拆分并缓存
for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
```

---

## 21.4 Data Parallel 分片

### 21.4.1 基本 DP 分片

当使用多 GPU 时，可以将 ViT 的 batch 分片到不同 GPU：

```python
# vllm/model_executor/models/vision.py:282-312

def run_dp_sharded_vision_model(
    image_input: torch.Tensor, 
    vision_model: torch.nn.Module
) -> torch.Tensor:
    """在多 GPU 间按 batch 维度分片处理 ViT
    
    Args:
        image_input: [num_images, ...] 所有图片的输入
        vision_model: ViT 模型
    
    Returns:
        [num_images, seq_len, hidden] 所有图片的输出
    """
    num_chunks = image_input.shape[0]
    mp_world_size = get_tensor_model_parallel_world_size()
    
    # 计算每个 GPU 处理的图片数
    num_chunks_per_rank = (num_chunks + mp_world_size - 1) // mp_world_size
    
    # 可能需要 padding 以确保每个 GPU 处理相同数量
    num_padded_chunks = num_chunks_per_rank * mp_world_size - num_chunks
    pad = (0,) * (2 * (image_input.dim() - 1)) + (0, num_padded_chunks)
    image_input_padded = torch.nn.functional.pad(image_input, pad)
    
    # 当前 GPU 只处理自己的分片
    rank = get_tensor_model_parallel_rank()
    image_input_per_rank = image_input_padded[
        rank * num_chunks_per_rank : (rank + 1) * num_chunks_per_rank, ...
    ]
    
    # 执行 ViT
    vision_embeddings = vision_model(image_input_per_rank)
    vision_embeddings = vision_embeddings.contiguous()
    
    # All-gather 收集所有 GPU 的结果
    vision_embeddings = tensor_model_parallel_all_gather(vision_embeddings, dim=0)
    
    # 移除 padding
    vision_embeddings = vision_embeddings[:num_chunks, ...]
    
    return vision_embeddings
```

**示意图**：
```
4 张图片, 2 个 GPU:

             image_input: [img1, img2, img3, img4]
                    ↓ 分片
    ┌─────────────────────┬─────────────────────┐
    │       GPU 0         │       GPU 1         │
    │   [img1, img2]      │   [img3, img4]      │
    │        ↓            │        ↓            │
    │   ViT forward       │   ViT forward       │
    │        ↓            │        ↓            │
    │   [emb1, emb2]      │   [emb3, emb4]      │
    └─────────────────────┴─────────────────────┘
                    ↓ all_gather
             [emb1, emb2, emb3, emb4]
```

### 21.4.2 配置方式

```python
# vLLM 启动参数
--mm-encoder-tp-mode data  # 使用数据并行处理 ViT
--mm-encoder-tp-mode tensor  # 使用张量并行处理 ViT (默认)
```

---

## 21.5 MRoPE 负载均衡分片

### 21.5.1 问题背景

对于支持动态分辨率的模型（如 Qwen2-VL, Qwen3-VL），不同图片的 token 数量差异很大：

```
img1: 1000x1000 → ~5000 tokens
img2: 100x100   → ~50 tokens
img3: 200x200   → ~200 tokens
img4: 50x50     → ~12 tokens

简单的 batch 分片:
  GPU 0: [img1, img2] → 5050 tokens
  GPU 1: [img3, img4] → 212 tokens
  
严重不均衡！GPU 0 会成为瓶颈
```

### 21.5.2 负载均衡算法

```python
# vllm/model_executor/models/vision.py:315-381

def get_load_balance_assignment(
    sizes: list[int],
    num_gpus: int = 2,
) -> tuple[list[int], list[int], list[int]]:
    """使用贪心算法进行负载均衡分配
    
    Args:
        sizes: 每张图片的 token 数量
        num_gpus: GPU 数量
    
    Returns:
        shuffle_indices: 重排索引
        gpu_sample_counts: 每个 GPU 处理的图片数
        grouped_sizes_per_gpu: 每个 GPU 的总负载
    """
    n_samples = len(sizes)
    
    # 贪心算法: 按 size 从大到小分配
    gpu_assignments = [list() for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus
    
    # 按 size 降序排序
    large_to_small_indices = sorted(
        range(n_samples), 
        key=lambda i: sizes[i], 
        reverse=True
    )
    
    for idx in large_to_small_indices:
        # 找到当前负载最小的 GPU
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]
    
    # 生成 shuffle 索引和统计信息
    shuffle_indices = []
    gpu_sample_counts = []
    for gpu_id in range(num_gpus):
        shuffle_indices.extend(gpu_assignments[gpu_id])
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))
    
    return (shuffle_indices, gpu_sample_counts, gpu_loads)
```

**示例**：
```
sizes = [5000, 50, 200, 12]
num_gpus = 2

贪心分配过程:
1. img1(5000) → GPU 0  (loads: [5000, 0])
2. img3(200)  → GPU 1  (loads: [5000, 200])
3. img2(50)   → GPU 1  (loads: [5000, 250])
4. img4(12)   → GPU 1  (loads: [5000, 262])

结果:
  GPU 0: [img1]           → 5000 tokens
  GPU 1: [img3, img2, img4] → 262 tokens
  
shuffle_indices = [0, 2, 1, 3]
gpu_sample_counts = [1, 3]
```

### 21.5.3 MRoPE 专用分片函数

```python
# vllm/model_executor/models/vision.py:384-571

def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw_list: list[list[int]],
    *,
    rope_type: Literal["rope_3d", "rope_2d"],
) -> tuple[torch.Tensor, ...]:
    """针对 MRoPE 模型的负载均衡 ViT 执行
    
    与普通 DP 分片的区别:
    1. 基于 token 数量而非图片数量进行负载均衡
    2. 需要重新计算每个分片的 cu_seqlens
    3. 需要处理不规则的输入形状
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    
    # 计算每张图片的 token 数
    sizes = [t * h * w for t, h, w in grid_thw_list]
    
    # 负载均衡分配
    shuffle_indices, gpu_sample_counts, gpu_loads = get_load_balance_assignment(
        sizes, tp_size
    )
    
    # 重排 pixel_values 和 grid_thw
    # ... 复杂的索引重排逻辑 ...
    
    # 每个 GPU 执行自己负责的图片
    local_outputs = vision_model(local_pixel_values, local_grid_thw)
    
    # All-gather + 恢复原始顺序
    # ...
    
    return outputs
```

---

## 21.6 预处理并行化

### 21.6.1 多线程媒体加载

```python
# vllm/multimodal/utils.py:47-50

# 全局线程池用于媒体加载
global_thread_pool = ThreadPoolExecutor(
    max_workers=envs.VLLM_MEDIA_LOADING_THREAD_COUNT
)

# 异步加载媒体
async def load_from_url_async(self, url: str, media_io: MediaIO, ...):
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        global_thread_pool, 
        media_io.load_bytes, 
        data
    )
    return await future
```

### 21.6.2 环境变量配置

```bash
# 控制媒体加载线程数
export VLLM_MEDIA_LOADING_THREAD_COUNT=8

# 控制图片预处理工作进程数 (部分模型)
export VLLM_IMAGE_PROCESSOR_WORKERS=4
```

---

## 21.7 ViT Attention Backend

### 21.7.1 支持的 Backend

```python
# vllm/model_executor/models/vision.py

def get_vit_attn_backend(head_size: int, dtype: torch.dtype) -> AttentionBackendEnum:
    """获取 ViT 使用的 attention backend"""
    
    # 优先使用 FlashAttention
    if is_flash_attn_available():
        return AttentionBackendEnum.FLASH_ATTN
    
    # 备选: PyTorch SDPA
    return AttentionBackendEnum.TORCH_SDPA
```

### 21.7.2 FlashAttention Varlen

ViT 使用 FlashAttention 的变长模式处理不同大小的图片：

```python
# vllm/model_executor/models/qwen2_5_vl.py (简化)

class Qwen2_5_VisionAttention:
    def forward(self, x, cu_seqlens, max_seqlen, ...):
        # 使用 varlen attention 处理多个不同大小的图片
        if self.attn_backend == AttentionBackendEnum.FLASH_ATTN:
            output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,  # ViT 使用双向 attention
            )
```

**cu_seqlens 示例**：
```
3 张图片: 256 + 576 + 144 = 976 tokens

cu_seqlens = [0, 256, 832, 976]

表示:
  图片 1: tokens 0-255
  图片 2: tokens 256-831
  图片 3: tokens 832-975

每张图片内部做 full attention，图片之间不交互
```

---

## 21.8 完整执行流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                 ViT Execution Pipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: Scheduler 决定调度                                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ for request in requests:                                       │ │
│  │     _try_schedule_encoder_inputs()                             │ │
│  │     检查 encoder_cache 命中                                     │ │
│  │     检查 compute budget                                        │ │
│  │ → scheduled_encoder_inputs = {req_id: [input_ids]}            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Step 2: 收集 MM Inputs                                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ encoder_runner.prepare_mm_inputs(scheduled_encoder_inputs)     │ │
│  │ → mm_hashes: [hash1, hash2, ...]                               │ │
│  │ → mm_kwargs: [item1, item2, ...]                               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Step 3: 按 Modality 分组                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ group_mm_kwargs_by_modality(mm_kwargs)                         │ │
│  │ → images_batch: {pixel_values: [...], grid_thw: [...]}        │ │
│  │ → videos_batch: {pixel_values_videos: [...], ...}             │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Step 4: 可选 - DP 分片 / 负载均衡                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ if mm_encoder_tp_mode == "data":                               │ │
│  │     run_dp_sharded_vision_model() 或                           │ │
│  │     run_dp_sharded_mrope_vision_model()                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Step 5: 执行 ViT                                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ model.embed_multimodal(**images_batch)                         │ │
│  │   → patch_embed(pixel_values)                                  │ │
│  │   → transformer blocks (FlashAttn varlen)                      │ │
│  │   → merger (空间降采样)                                         │ │
│  │ → [emb1, emb2, emb3, ...]  每张图片一个 tensor                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Step 6: 缓存结果                                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ for mm_hash, output in zip(mm_hashes, outputs):                │ │
│  │     encoder_cache[mm_hash] = output                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 21.9 小结

### 问题 1 答案：ViT 始终处理整张图片

| 方面 | 说明 |
|------|------|
| **原因** | ViT 使用双向 Self-Attention，需要看到完整输入 |
| **Chunked MM** | 控制 decoder 端如何消费 ViT 输出，非 ViT 本身 |
| **缓存** | ViT 输出整体缓存，一次计算多次使用 |

### 问题 4 答案：ViT 有多种 Batch 优化

| 优化策略 | 适用场景 | 代码位置 |
|----------|----------|----------|
| **跨请求 Batch** | 多请求同时到达 | `group_mm_kwargs_by_modality()` |
| **DP 分片** | 多 GPU 环境 | `run_dp_sharded_vision_model()` |
| **负载均衡** | 动态分辨率模型 | `get_load_balance_assignment()` |
| **预处理并行** | 媒体加载 | `global_thread_pool` |

```
性能优化全景图:

     多个请求的图片
           ↓
    ┌──────────────────┐
    │  跨请求 Batch     │  将多个请求的图片合并
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  按 Modality 分组 │  images/videos 分开处理
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  负载均衡分配     │  大图片和小图片混合分配
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  多 GPU DP 分片   │  每个 GPU 处理一部分
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  FlashAttn Varlen │  高效处理变长序列
    └──────────────────┘
           ↓
      ViT 输出
```

---

> **下一节**: [22-mm-processing.md](./22-mm-processing.md) - 多模态处理流程
