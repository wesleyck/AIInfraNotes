# 19. 多模态架构总览

> **核心问题**: vLLM 如何高效处理包含图片、视频、音频的多模态请求？
>
> **关键文件**:
> - `vllm/multimodal/inputs.py` - 核心数据结构定义
> - `vllm/v1/core/encoder_cache_manager.py` - Encoder 输出缓存管理
> - `vllm/v1/worker/gpu/mm/encoder_runner.py` - GPU 端多模态执行
> - `vllm/v1/core/sched/scheduler.py` - 调度器多模态处理

---

## 19.1 多模态处理的核心挑战

在纯文本 LLM 推理中，输入是 token IDs，直接通过 embedding table 转换为向量。但多模态模型面临更复杂的挑战：

```
文本推理流程:
  token_ids → Embedding Table → hidden_states → LLM Forward

多模态推理流程:
  token_ids + 图片/视频/音频
       ↓
  ┌─────────────────────────────────────────────────────┐
  │ 1. 多模态数据需要专门的 Encoder (如 ViT) 处理         │
  │ 2. Encoder 计算开销大，需要缓存复用                   │
  │ 3. 多模态 token 需要与文本 token 正确合并             │
  │ 4. 长序列中图片可能分布在不同位置                     │
  │ 5. 多个请求可能共享相同的多模态输入                   │
  └─────────────────────────────────────────────────────┘
       ↓
  mixed_embeddings → LLM Forward
```

vLLM 的多模态架构通过以下设计解决这些挑战：

1. **两层 Cache 架构**: Encoder Cache + KV Cache 分离
2. **按需调度**: 只在需要时执行 encoder 计算
3. **跨请求共享**: 相同多模态输入可复用 encoder 输出
4. **灵活的 Chunked 处理**: 支持多模态 token 分批进入 decoder

---

## 19.2 核心数据结构

### 19.2.1 PlaceholderRange - 占位符位置信息

```python
# vllm/multimodal/inputs.py:167-227

@dataclass
class PlaceholderRange:
    """表示多模态输入在 prompt 中的占位符位置"""
    
    offset: int
    """占位符在 prompt 中的起始位置"""
    
    length: int
    """占位符的 token 数量 (不是 embedding 数量!)"""
    
    is_embed: torch.Tensor | None = None
    """标记哪些位置是真正的 embedding (用于处理分隔符等)"""
    
    @cached_property
    def get_num_embeds(self) -> int:
        """返回实际 embedding 数量"""
        if self.embeds_cumsum is None:
            return self.length
        return int(self.embeds_cumsum[-1])
```

**示例**：
```
Prompt: "描述这张图片 <image_placeholder...> 的内容"
Token IDs: [1, 2, 3, 4, -1, -1, ..., -1, 5, 6, 7]
                       ↑ offset=4     ↑ length=336
                       
PlaceholderRange(offset=4, length=336)
  - 表示从位置4开始，有336个 placeholder tokens
  - 这些位置会被 ViT 输出的 embeddings 替换
```

### 19.2.2 MultiModalFeatureSpec - 多模态特征规格

```python
# vllm/multimodal/inputs.py:339-370

@dataclass
class MultiModalFeatureSpec:
    """每个多模态输入项的完整规格"""
    
    data: Optional[MultiModalKwargsItem]
    """多模态数据 (pixel_values, grid_thw 等)"""
    
    modality: str
    """模态类型: "image", "video", "audio" """
    
    identifier: str
    """唯一标识符 (mm_hash 或 uuid)，用于 Encoder Cache"""
    
    mm_position: PlaceholderRange
    """在 prompt 中的位置信息"""
    
    mm_hash: str | None = None
    """基础 hash 值 (用于 processor cache)"""
```

**设计要点**：
- `identifier` 是 Encoder Cache 的 key，相同图片内容会有相同的 hash
- `mm_position` 描述这个多模态输入对应 prompt 中的哪些 tokens
- 一个请求可以有多个 `MultiModalFeatureSpec` (多张图片)

### 19.2.3 MultiModalKwargsItem - Encoder 输入参数

```python
# vllm/multimodal/inputs.py (简化)

class MultiModalKwargsItem:
    """单个多模态项传给 encoder 的 kwargs"""
    
    modality: str       # "image" / "video" / "audio"
    
    # 对于图片，通常包含:
    # - pixel_values: torch.Tensor  # 预处理后的像素值
    # - image_grid_thw: list        # 网格信息 (Qwen-VL 特有)
    
    # 对于视频，通常包含:
    # - pixel_values_videos: torch.Tensor
    # - video_grid_thw: list
```

---

## 19.3 两层 Cache 架构

vLLM 多模态推理的核心创新是**两层分离的 Cache 架构**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Layer 1: Encoder Cache                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ EncoderCacheManager (Scheduler 端)                            │   │
│  │  cached: {mm_hash -> set[request_id]}  # 引用计数             │   │
│  │  freeable: OrderedDict[mm_hash, size]  # 可回收队列 (LRU)     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↕ 同步                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ encoder_cache: dict[mm_hash, torch.Tensor] (Worker 端)        │   │
│  │  实际存储 ViT/Audio Encoder 输出                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  特点:                                                               │
│  - 按多模态项整体缓存/驱逐 (不会部分驱逐)                            │
│  - 支持跨请求共享 (相同 mm_hash)                                    │
│  - LRU 驱逐策略                                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ embeddings 注入
┌─────────────────────────────────────────────────────────────────────┐
│                         Layer 2: KV Cache                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ BlockPool (与纯文本相同)                                       │   │
│  │  - 图片 tokens 和文本 tokens 存储方式完全相同                   │   │
│  │  - block_hash 计算会包含 mm_hash (支持 prefix caching)         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  特点:                                                               │
│  - 图片 token 进入 decoder 后与普通文本无区别                       │
│  - Prefix caching 通过 mm_hash 区分不同图片                         │
│  - 按 block 粒度管理，可部分驱逐                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 为什么需要两层 Cache？

| 需求 | Encoder Cache | KV Cache |
|------|---------------|----------|
| **存储内容** | ViT 输出的 embeddings | Attention 层的 K/V 张量 |
| **复用场景** | 相同图片 + 不同文本 | 相同前缀 (包括图片位置) |
| **计算代价** | ViT forward (昂贵) | Attention forward (更频繁) |
| **驱逐粒度** | 整个多模态项 | 单个 block |

**场景分析**：

```
请求 A: "描述图片1: <img1> 的内容"
请求 B: "分析图片1: <img1> 的风格"
请求 C: "描述图片2: <img2> 的内容"

Encoder Cache:
  - A 和 B 共享 img1 的 encoder 输出 ✓
  - C 需要重新计算 img2 的 encoder 输出

KV Cache (Prefix Caching):
  - A 和 B 文本不同，无法共享 prefix
  - 如果 A 和 C 前缀相同但图片不同，也无法共享 (mm_hash 不同)
```

---

## 19.4 处理流程概览

### 19.4.1 完整请求生命周期

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: 输入处理 (InputProcessor)                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  用户请求:                                                           │
│    prompt = "描述这张图片的内容"                                      │
│    images = [PIL.Image]                                             │
│                                                                      │
│  处理后:                                                             │
│    token_ids = [1, 2, ..., -1, -1, ..., -1, ..., 10]                │
│                          ↑ placeholder tokens                        │
│    mm_features = [                                                  │
│      MultiModalFeatureSpec(                                         │
│        identifier="hash_abc123",                                    │
│        mm_position=PlaceholderRange(offset=5, length=336),          │
│        data=MultiModalKwargsItem(pixel_values=..., grid_thw=...)    │
│      )                                                              │
│    ]                                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: 调度决策 (Scheduler)                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  _try_schedule_encoder_inputs():                                    │
│                                                                      │
│  for each mm_feature in request.mm_features:                        │
│      1. 检查是否与当前调度范围重叠                                    │
│      2. 检查 Encoder Cache 是否命中                                  │
│      3. 检查是否有足够的 compute budget                              │
│      4. 决定是否调度此 encoder input                                 │
│                                                                      │
│  输出:                                                               │
│    encoder_inputs_to_schedule = [0]  # 需要执行的 encoder input IDs  │
│    cached_encoder_inputs = []        # 已缓存的                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Encoder 执行 (EncoderRunner)                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  execute_mm_encoder():                                              │
│                                                                      │
│  1. 收集所有需要执行的 mm_kwargs                                     │
│  2. 按 modality 分组 (images/videos 分开处理)                        │
│  3. 批量执行 model.embed_multimodal(**kwargs)                       │
│  4. 缓存结果: encoder_cache[mm_hash] = output                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Embedding 合并                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  gather_mm_embeddings():                                            │
│    - 从 encoder_cache 获取需要的 embeddings                          │
│    - 确定每个 embedding 在当前 batch 中的位置                        │
│    - 返回 mm_embeds 和 is_mm_embed mask                              │
│                                                                      │
│  get_inputs_embeds():                                               │
│    - 调用 model.embed_input_ids(input_ids, mm_embeds, is_mm_embed)  │
│    - 将 mm_embeds 替换到 text_embeds 的对应位置                      │
│                                                                      │
│  text_embeds:  [E1, E2, E3, E4, E5, ...]                           │
│  is_mm_embed:  [F,  F,  T,  T,  T,  ...]                           │
│  mm_embeds:          [M1, M2, M3, ...]                              │
│                        ↓                                             │
│  mixed_embeds: [E1, E2, M1, M2, M3, ...]                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: LLM Forward                                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  model.forward(inputs_embeds=mixed_embeds, ...)                     │
│                                                                      │
│  - 此时多模态 tokens 与文本 tokens 已经融合                          │
│  - 后续处理与纯文本完全相同                                          │
│  - KV Cache 存储也完全相同                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 19.4.2 关键代码路径

```python
# 1. 调度器决定需要执行哪些 encoder inputs
# vllm/v1/core/sched/scheduler.py:1043-1199
def _try_schedule_encoder_inputs(self, request, num_computed_tokens, 
                                   num_new_tokens, encoder_compute_budget):
    encoder_inputs_to_schedule = []
    
    for i, mm_feature in enumerate(request.mm_features):
        # 检查是否需要在当前 step 处理
        start_pos = mm_feature.mm_position.offset
        if start_pos >= num_computed_tokens + num_new_tokens:
            break  # 还没到这个 encoder input
        
        # 检查 cache 命中
        if self.encoder_cache_manager.check_and_update_cache(request, i):
            continue  # 已缓存，跳过
        
        # 检查是否有 budget
        if not self.encoder_cache_manager.can_allocate(...):
            break  # 预算不足
        
        encoder_inputs_to_schedule.append(i)
    
    return encoder_inputs_to_schedule, ...

# 2. Worker 执行 encoder
# vllm/v1/worker/gpu/mm/encoder_runner.py:62-88
@torch.inference_mode()
def execute_mm_encoder(self, model, mm_hashes, mm_kwargs):
    encoder_outputs = []
    
    # 按 modality 分组批量处理
    for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(mm_kwargs):
        outputs = model.embed_multimodal(**mm_kwargs_group)
        encoder_outputs.extend(outputs)
    
    # 缓存结果
    for mm_hash, output in zip(mm_hashes, encoder_outputs):
        self.encoder_cache[mm_hash] = output
    
    return encoder_outputs

# 3. 合并 embeddings
# vllm/v1/worker/gpu/mm/encoder_runner.py:169-184
@torch.inference_mode()
def get_inputs_embeds(self, model, input_ids, mm_embeds, is_mm_embed):
    return model.embed_input_ids(
        input_ids,
        multimodal_embeddings=mm_embeds,
        is_multimodal=is_mm_embed,
    )
```

---

## 19.5 EncoderCacheManager 详解

### 19.5.1 核心状态

```python
# vllm/v1/core/encoder_cache_manager.py:18-79

class EncoderCacheManager:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size           # 总容量 (embedding 数量)
        self.num_free_slots = cache_size       # 当前空闲容量
        self.num_freeable_slots = cache_size   # 可回收容量 (含 freeable)
        
        # mm_hash -> 引用此 mm_data 的 request_ids 集合
        # 空集合表示已缓存但无人引用
        self.cached: dict[str, set[str]] = {}
        
        # 可回收的条目 (引用计数为 0)，LRU 顺序
        self.freeable: OrderedDict[str, int] = OrderedDict()
        
        # 已被驱逐的 mm_hash 列表 (通知 worker 清理)
        self.freed: list[str] = []
```

### 19.5.2 Cache 命中检查

```python
# vllm/v1/core/encoder_cache_manager.py:80-106

def check_and_update_cache(self, request: Request, input_id: int) -> bool:
    """检查 encoder output 是否已缓存，并更新引用"""
    
    mm_hash = request.mm_features[input_id].identifier
    
    # 完全没有缓存
    if mm_hash not in self.cached:
        return False
    
    # 缓存存在但无人引用 (在 freeable 队列中)
    if not self.cached[mm_hash]:
        # 从 freeable 中移除，因为现在又有人用了
        num_encoder_embeds = self.freeable.pop(mm_hash)
        self.num_freeable_slots -= num_encoder_embeds
    
    # 添加当前请求的引用
    self.cached[mm_hash].add(request.request_id)
    return True
```

### 19.5.3 分配与驱逐

```python
# vllm/v1/core/encoder_cache_manager.py:108-167

def can_allocate(self, request, input_id, encoder_compute_budget, 
                  num_embeds_to_schedule) -> bool:
    """检查是否有足够空间，必要时驱逐旧条目"""
    
    num_embeds = request.get_num_encoder_embeds(input_id)
    
    # 1. 检查 compute budget
    if num_embeds > encoder_compute_budget:
        return False
    
    num_embeds += num_embeds_to_schedule
    
    # 2. 足够空闲空间
    if num_embeds <= self.num_free_slots:
        return True
    
    # 3. 可回收空间也不够
    if num_embeds > self.num_freeable_slots:
        return False
    
    # 4. 需要驱逐 freeable 中的条目
    while num_embeds > self.num_free_slots:
        # LRU: 驱逐最老的
        mm_hash, num_free_embeds = self.freeable.popitem(last=False)
        del self.cached[mm_hash]
        self.freed.append(mm_hash)  # 通知 worker 清理
        self.num_free_slots += num_free_embeds
    
    return True
```

### 19.5.4 引用计数与释放

```python
# vllm/v1/core/encoder_cache_manager.py:210-242

def free_encoder_input(self, request: Request, input_id: int) -> None:
    """释放请求对某个 encoder input 的引用"""
    
    mm_hash = request.mm_features[input_id].identifier
    
    # 检查是否存在
    if not self.cached.get(mm_hash, None):
        return
    
    # 移除当前请求的引用
    self.cached[mm_hash].discard(request.request_id)
    
    # 如果引用变为空，加入 freeable 队列
    if not self.cached[mm_hash]:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.freeable[mm_hash] = num_encoder_embeds
        self.num_freeable_slots += num_encoder_embeds

def free(self, request: Request) -> None:
    """释放请求的所有 encoder input 引用 (请求结束时调用)"""
    input_ids = self.get_cached_input_ids(request)
    for input_id in input_ids:
        self.free_encoder_input(request, input_id)
```

---

## 19.6 Prefix Caching 与多模态

### 19.6.1 Block Hash 计算包含 mm_hash

```python
# vllm/v1/core/kv_cache_utils.py:387-448

def _gen_mm_extra_hash_keys(request, start_token_idx, end_token_idx, 
                             start_mm_idx) -> tuple[list[Any], int]:
    """为 block hash 生成多模态相关的额外 key"""
    
    extra_keys = []
    mm_features = request.mm_features
    
    for mm_feature in mm_features[start_mm_idx:]:
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        
        # 检查此 block 是否包含这个多模态输入
        if end_token_idx > offset and start_token_idx < offset + length:
            # 将 mm_hash 加入 extra_keys
            extra_keys.append(mm_feature.identifier)
    
    return extra_keys, next_mm_idx
```

这意味着：
- 相同文本 + 不同图片 → 不同 block_hash → 无法共享 prefix
- 相同文本 + 相同图片 → 相同 block_hash → 可以共享 prefix

### 19.6.2 测试验证

```python
# tests/v1/core/test_prefix_caching.py:1219-1235

# Block hash 包含 mm_hash
assert block_hashes[0] == sha256(
    (NONE_HASH, tuple(token_ids[:block_size]), ("aaa",))  # mm_hash="aaa"
)
assert block_hashes[1] == sha256(
    (block_hashes[0], tuple(token_ids[block_size:block_size*2]), ("aaa", "bbb"))
)

# 不同 mm_hash = 不同 block_hash
# 即使 token_ids 相同，图片不同也无法 prefix cache 命中
```

---

## 19.7 关键配置项

| 配置 | 位置 | 说明 |
|------|------|------|
| `encoder_cache_size` | `SchedulerConfig` | Encoder cache 容量 (embedding 数量) |
| `max_num_encoder_input_tokens` | `SchedulerConfig` | 单次可执行的 encoder tokens 预算 |
| `disable_chunked_mm_input` | `SchedulerConfig` | 禁止拆分多模态 placeholder |
| `mm_encoder_tp_mode` | `MultiModalConfig` | ViT 的 TP 模式 ("tensor" / "data") |
| `limit_mm_per_prompt` | 启动参数 | 每个请求最多多少个多模态输入 |

---

## 19.8 小结

vLLM 的多模态架构通过精心设计的两层 Cache 和按需调度机制，实现了：

1. **高效复用**: Encoder Cache 支持跨请求共享，避免重复计算
2. **灵活调度**: Encoder 执行与 decoder 解耦，按需触发
3. **统一管理**: 多模态 tokens 进入 decoder 后与文本统一处理
4. **Prefix Caching**: 通过 mm_hash 区分不同多模态内容

```
核心架构图:

User Request
     ↓
┌─────────────────────────────────────────┐
│         InputProcessor                   │
│   解析多模态数据，生成 mm_features       │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│          Scheduler                       │
│   _try_schedule_encoder_inputs()        │
│   检查 cache，决定执行哪些 encoders      │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│        EncoderRunner                     │
│   execute_mm_encoder()                   │
│   批量执行 ViT，缓存结果                 │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│      Embedding Merge                     │
│   gather_mm_embeddings()                │
│   get_inputs_embeds()                   │
│   将 mm_embeds 注入 text_embeds         │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│          LLM Forward                     │
│   mixed_embeds → hidden_states          │
│   KV Cache 存储与纯文本相同              │
└─────────────────────────────────────────┘
```

---

> **下一节**: [20-qwen3vl-model.md](./20-qwen3vl-model.md) - Qwen3-VL 模型解析
