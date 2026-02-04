# 22. 多模态处理流程

> **核心问题**: 
> 1. 会不会存在 req1: text1+image1, req2: text1+image2 图片部分 cache 命中？
> 2. 整张图片的 KV cache 保存是有特殊逻辑还是跟正常 LLM 没有区别？如何保证完整性？
>
> **关键文件**:
> - `vllm/v1/core/encoder_cache_manager.py` - Encoder Cache 管理
> - `vllm/v1/core/kv_cache_utils.py` - KV Cache 与多模态
> - `vllm/v1/core/sched/scheduler.py` - 调度器多模态逻辑

---

## 22.1 核心问题解答：不同图片间的 Cache 命中

### 答案：不同图片之间不会 Cache 命中，相同图片可以跨请求共享

```
场景分析:

req1: "描述图片" + image1 (hash: "aaa")
req2: "描述图片" + image2 (hash: "bbb")
req3: "分析图片" + image1 (hash: "aaa")

Encoder Cache:
  "aaa" → image1 的 ViT 输出
  "bbb" → image2 的 ViT 输出
  
  req1 和 req3 共享 "aaa" ✓ (相同图片)
  req1 和 req2 无法共享 ✗ (不同图片，hash 不同)

KV Cache (Prefix Caching):
  block_hash = hash(parent_hash, token_ids, mm_hash)
  
  req1 和 req2: token_ids 相同，但 mm_hash 不同 → 不同 block_hash → 无法共享
  req1 和 req3: token_ids 和 mm_hash 都相同 → 相同 block_hash → 可以共享
```

---

## 22.2 Cache 机制详解

### 22.2.1 Encoder Cache - 基于 mm_hash 的共享

```python
# vllm/v1/core/encoder_cache_manager.py:80-106

def check_and_update_cache(self, request: Request, input_id: int) -> bool:
    """检查 encoder output 是否已缓存"""
    
    # 获取多模态输入的唯一标识
    mm_hash = request.mm_features[input_id].identifier
    
    # 情况 1: 完全没有缓存
    if mm_hash not in self.cached:
        return False  # 需要执行 encoder
    
    # 情况 2: 有缓存但无人引用 (在 freeable 队列中)
    if not self.cached[mm_hash]:
        # 从 freeable 移除，重新激活
        num_encoder_embeds = self.freeable.pop(mm_hash)
        self.num_freeable_slots -= num_encoder_embeds
    
    # 添加当前请求的引用
    self.cached[mm_hash].add(request.request_id)
    return True  # 命中缓存，无需执行 encoder
```

**mm_hash 的计算**:

```python
# 简化版 - 实际实现在 multimodal/processing 中
def compute_mm_hash(image_data):
    # 基于图片内容计算 hash
    return hashlib.sha256(image_data.tobytes()).hexdigest()

# 相同图片 → 相同 mm_hash → 可以共享
# 不同图片 → 不同 mm_hash → 无法共享
```

### 22.2.2 KV Cache - Block Hash 包含 mm_hash

```python
# vllm/v1/core/kv_cache_utils.py:387-448

def _gen_mm_extra_hash_keys(
    request: Request, 
    start_token_idx: int, 
    end_token_idx: int, 
    start_mm_idx: int
) -> tuple[list[Any], int]:
    """为 block hash 生成多模态相关的额外 key"""
    
    extra_keys: list[Any] = []
    mm_features = request.mm_features
    
    if not mm_features:
        return extra_keys, start_mm_idx
    
    curr_mm_idx = start_mm_idx
    while curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        
        # 检查当前 block 是否包含这个多模态输入
        if end_token_idx > offset:
            if start_token_idx > offset + length:
                # 这个 block 已经过了当前 mm 输入
                curr_mm_idx += 1
                continue
            
            # 这个 block 包含当前 mm 输入
            # 将 mm_hash 加入 extra_keys!
            extra_keys.append(mm_feature.identifier)  # ← 关键
            
            if end_token_idx >= offset + length:
                curr_mm_idx += 1
            else:
                break
        else:
            break
    
    return extra_keys, curr_mm_idx
```

**Block Hash 计算**:

```python
# vllm/v1/core/kv_cache_utils.py:525-552

def hash_block_tokens(
    hash_function,
    parent_block_hash,
    curr_block_token_ids,
    extra_keys,  # 包含 mm_hash!
):
    """计算 block hash"""
    
    return hash_function((
        parent_block_hash,
        tuple(curr_block_token_ids),
        extra_keys,  # (mm_hash1, mm_hash2, ...)
    ))
```

### 22.2.3 测试验证

```python
# tests/v1/core/test_prefix_caching.py:1219-1235

def test_mm_prefix_caching():
    """验证多模态 prefix caching"""
    
    # 构造包含图片的 token 序列
    # T=文本, P=placeholder
    common_token_ids = list(range(10)) + [-1] * 6  # [T...T, P...P]
    mm_positions = [PlaceholderRange(offset=11, length=10)]
    mm_hashes = ["aaa"]
    
    # 创建请求
    req0 = make_request("0", common_token_ids, mm_hashes=mm_hashes)
    
    # 验证 block hash 包含 mm_hash
    assert block_hashes[0] == sha256((
        NONE_HASH,                    # parent hash
        tuple(token_ids[:block_size]), # token ids
        ("aaa",)                       # mm_hash!
    ))
    
    # 相同 token_ids + 不同 mm_hash = 不同 block_hash
    req1 = make_request("1", common_token_ids, mm_hashes=["bbb"])
    # req1 的 block_hash 与 req0 不同，无法共享
```

---

## 22.3 核心问题解答：KV Cache 与 Encoder Cache 的区别

### 答案：KV Cache 与普通文本完全相同，无特殊存储逻辑

```
┌─────────────────────────────────────────────────────────────────────┐
│                      两层 Cache 对比                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Encoder Cache                                ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ 存储内容:  ViT 输出的 embeddings                                 ││
│  │ 存储位置:  encoder_cache: dict[mm_hash, torch.Tensor]           ││
│  │ 粒度:      按多模态项 (整张图片)                                 ││
│  │ 驱逐:      LRU，整体驱逐，不会部分驱逐                           ││
│  │ 共享:      基于 mm_hash，相同图片可跨请求共享                    ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    KV Cache                                      ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ 存储内容:  Attention 层的 K/V 张量                               ││
│  │ 存储位置:  BlockPool (paged blocks)                              ││
│  │ 粒度:      按 block (block_size tokens)                          ││
│  │ 驱逐:      LRU，按 block 驱逐，可部分驱逐                        ││
│  │ 共享:      基于 block_hash (包含 mm_hash)                        ││
│  │ 特殊性:    图片 tokens 与文本 tokens 存储方式完全相同！          ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 22.3.1 图片 Token 在 Decoder 中的处理

```python
# vllm/v1/worker/gpu/mm/encoder_runner.py:169-184

def get_inputs_embeds(self, model, input_ids, mm_embeds, is_mm_embed):
    """将视觉 embeddings 合并到文本 embeddings"""
    
    return model.embed_input_ids(
        input_ids,
        multimodal_embeddings=mm_embeds,  # 图片 embeddings
        is_multimodal=is_mm_embed,        # 标记哪些位置是图片
    )

# 在 model.embed_input_ids 内部:
def embed_input_ids(self, input_ids, multimodal_embeddings, is_multimodal):
    # 1. 获取文本 embeddings
    text_embeds = self.embed_tokens(input_ids)
    
    # 2. 将图片 embeddings 替换到对应位置
    inputs_embeds = _merge_multimodal_embeddings(
        inputs_embeds=text_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )
    
    return inputs_embeds  # 混合的 embeddings
```

**合并后的 embeddings**:
```
text_embeds:   [E1, E2, E3, E4, E5, E6, E7, E8, ...]
is_multimodal: [F,  F,  T,  T,  T,  T,  F,  F,  ...]
mm_embeds:           [M1, M2, M3, M4]

合并结果:      [E1, E2, M1, M2, M3, M4, E7, E8, ...]
                      ↑ 图片 tokens 被替换

进入 Decoder 后，所有 tokens 统一处理，KV Cache 存储无区别
```

### 22.3.2 KV Cache 存储示意

```
Block 0: [K0, K1, K2, ... K15]  # 可能包含 text+image tokens 混合
Block 1: [K16, K17, ... K31]   # 可能全是 image tokens
Block 2: [K32, K33, ... K47]   # 可能是 image+text 混合

每个 Block:
  - 存储的是 Attention 层计算出的 K/V
  - 无论原始 token 是 text 还是 image，存储格式完全相同
  - block_hash 通过包含 mm_hash 来区分不同图片
```

---

## 22.4 图片 Cache 完整性保证

### 22.4.1 Encoder Cache 完整性

Encoder Cache 按**多模态项整体**缓存和驱逐：

```python
# vllm/v1/core/encoder_cache_manager.py:162-167

# 驱逐时整体驱逐
while num_embeds > self.num_free_slots:
    # 整个 mm_hash 对应的输出一起驱逐
    mm_hash, num_free_embeds = self.freeable.popitem(last=False)
    del self.cached[mm_hash]  # 删除整个条目
    self.freed.append(mm_hash)
    self.num_free_slots += num_free_embeds

# 不会出现:
# - 一张图片的部分 embeddings 被驱逐
# - 另一部分保留
```

### 22.4.2 disable_chunked_mm_input 配置

```python
# vllm/v1/core/sched/scheduler.py:1133-1143

# 禁止将多模态 placeholder 拆分调度
if (
    self.scheduler_config.disable_chunked_mm_input
    and num_computed_tokens < start_pos
    and (num_computed_tokens + num_new_tokens) < (start_pos + num_encoder_tokens)
):
    # 回退: 只调度到 placeholder 之前
    num_new_tokens = start_pos - num_computed_tokens
    break
```

**效果**:
```
Prompt: [T, T, T, T, P, P, P, P, P, P, T, T]  # T=text, P=placeholder
               ↑ 4      ↑ 6 个 placeholder

disable_chunked_mm_input = True:
  Step 1: 调度 [T, T, T, T]  (不进入 placeholder)
  Step 2: 调度 [P, P, P, P, P, P]  (完整 placeholder)
  Step 3: 调度 [T, T]  (剩余文本)
  
  保证 placeholder 不被拆分
```

### 22.4.3 引用计数机制

```python
# vllm/v1/core/encoder_cache_manager.py

class EncoderCacheManager:
    def __init__(self, cache_size):
        # 引用计数: mm_hash -> set[request_id]
        self.cached: dict[str, set[str]] = {}
        
        # 可回收队列 (引用计数为 0)
        self.freeable: OrderedDict[str, int] = OrderedDict()
```

**生命周期**:
```
1. 请求 A 需要 image1:
   - check_and_update_cache() 未命中
   - 执行 ViT
   - allocate(): cached["aaa"] = {"req_A"}

2. 请求 B 也需要 image1:
   - check_and_update_cache() 命中!
   - cached["aaa"] = {"req_A", "req_B"}

3. 请求 A 完成:
   - free_encoder_input(): cached["aaa"] = {"req_B"}

4. 请求 B 完成:
   - free_encoder_input(): cached["aaa"] = {}
   - 引用为 0，加入 freeable 队列

5. 内存不足时驱逐:
   - can_allocate() 中触发
   - 从 freeable 中 LRU 驱逐
```

---

## 22.5 跨请求共享详解

### 22.5.1 共享场景分析

| 场景 | Encoder Cache 共享 | KV Cache 共享 |
|------|-------------------|---------------|
| 相同文本 + 相同图片 | ✓ (mm_hash 相同) | ✓ (block_hash 相同) |
| 相同文本 + 不同图片 | ✗ (mm_hash 不同) | ✗ (block_hash 不同) |
| 不同文本 + 相同图片 | ✓ (mm_hash 相同) | ✗ (token_ids 不同) |
| 不同文本 + 不同图片 | ✗ | ✗ |

### 22.5.2 Encoder Cache 共享示例

```python
# 场景: 多个用户上传相同图片

# 用户 1
req1 = Request(
    prompt="描述这张图片",
    image=cat.jpg,  # mm_hash = "abc123"
)

# 用户 2 (10 秒后)
req2 = Request(
    prompt="分析图片中的内容",  # 不同的 prompt
    image=cat.jpg,  # 相同的图片, mm_hash = "abc123"
)

# 调度器处理 req2:
def _try_schedule_encoder_inputs(req2, ...):
    mm_hash = req2.mm_features[0].identifier  # "abc123"
    
    # 检查缓存
    if encoder_cache_manager.check_and_update_cache(req2, 0):
        # 命中! 跳过 ViT 执行
        return [], ...  # 不需要调度 encoder
    
# 结果: req2 直接复用 req1 的 ViT 输出，节省计算
```

### 22.5.3 KV Cache 共享示例

```python
# 场景: 完全相同的请求 (系统 prompt + 相同图片)

system_prompt = "你是一个图像分析助手"
common_image = dashboard.png  # mm_hash = "xyz789"

# 请求 1
req1 = Request(
    prompt=system_prompt + " [IMAGE] 分析这个仪表盘",
    image=common_image,
)

# 请求 2 (类似的请求)
req2 = Request(
    prompt=system_prompt + " [IMAGE] 分析这个仪表盘的趋势",
    image=common_image,
)

# Block Hash 计算:
# Block 0: hash(None, tokens[0:16], ("xyz789",))  # 包含 system_prompt + 部分 image
# Block 1: hash(block0_hash, tokens[16:32], ("xyz789",))  # 更多 image tokens

# req1 和 req2:
# - 前缀 token_ids 相同
# - mm_hash 相同
# → block_hash 相同 → Prefix Cache 命中!
```

---

## 22.6 端到端处理流程

### 22.6.1 完整请求生命周期

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Request Lifecycle                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: 输入处理                                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ InputProcessor.process()                                       │ │
│  │   1. tokenize(prompt) → token_ids                              │ │
│  │   2. preprocess(image) → pixel_values                          │ │
│  │   3. compute_mm_hash(image) → mm_hash                          │ │
│  │   4. create_mm_feature(pixel_values, mm_hash, position)        │ │
│  │                                                                 │ │
│  │ 输出: Request with token_ids, mm_features                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Phase 2: 调度                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Scheduler.schedule()                                           │ │
│  │   1. 检查 KV Cache 空间                                         │ │
│  │   2. _try_schedule_encoder_inputs()                            │ │
│  │      - check_and_update_cache() → 检查 Encoder Cache           │ │
│  │      - can_allocate() → 检查空间，必要时驱逐                   │ │
│  │   3. 决定 encoder_inputs_to_schedule                           │ │
│  │                                                                 │ │
│  │ 输出: SchedulerOutput with scheduled_encoder_inputs            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Phase 3: Encoder 执行 (如果需要)                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ EncoderRunner.execute_mm_encoder()                             │ │
│  │   1. prepare_mm_inputs() → 收集需要执行的 mm_kwargs            │ │
│  │   2. group_mm_kwargs_by_modality() → 按模态分组                │ │
│  │   3. model.embed_multimodal() → 执行 ViT                       │ │
│  │   4. encoder_cache[mm_hash] = output → 缓存结果                │ │
│  │                                                                 │ │
│  │ 输出: encoder_cache 中有可用的 embeddings                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Phase 4: Embedding 合并                                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ EncoderRunner.gather_mm_embeddings()                           │ │
│  │   1. 从 encoder_cache 获取需要的 embeddings                     │ │
│  │   2. 确定每个 embedding 在 batch 中的位置                      │ │
│  │                                                                 │ │
│  │ EncoderRunner.get_inputs_embeds()                              │ │
│  │   1. model.embed_input_ids(input_ids, mm_embeds, is_mm_embed)  │ │
│  │   2. 图片 embeddings 替换到 text embeddings 中                 │ │
│  │                                                                 │ │
│  │ 输出: mixed_embeddings                                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Phase 5: LLM Forward                                                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ model.forward(inputs_embeds=mixed_embeddings, ...)             │ │
│  │   - 所有 tokens 统一处理                                        │ │
│  │   - KV Cache 按 block 存储 (与纯文本相同)                       │ │
│  │                                                                 │ │
│  │ 输出: logits                                                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Phase 6: 请求完成                                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ encoder_cache_manager.free(request)                            │ │
│  │   - 释放 Encoder Cache 引用                                     │ │
│  │   - 引用计数为 0 时加入 freeable 队列                          │ │
│  │   - 后续需要空间时 LRU 驱逐                                     │ │
│  │                                                                 │ │
│  │ kv_cache_manager.free(request)                                 │ │
│  │   - 释放 KV Cache blocks                                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 22.7 配置与调优

### 22.7.1 关键配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `encoder_cache_size` | 自动计算 | Encoder Cache 容量 (embeddings 数量) |
| `max_num_encoder_input_tokens` | 自动计算 | 单步最大 encoder tokens |
| `disable_chunked_mm_input` | False | 禁止拆分多模态 placeholder |
| `mm_encoder_tp_mode` | "tensor" | ViT 并行模式 ("tensor"/"data") |
| `limit_mm_per_prompt` | 无限制 | 每个请求最多多模态输入数 |

### 22.7.2 性能调优建议

```python
# 场景 1: 大量相同图片的请求
# 建议: 增大 encoder_cache_size
vllm serve model --encoder-cache-size 1000

# 场景 2: 长视频/大图片
# 建议: 使用 DP 模式并行处理 ViT
vllm serve model --mm-encoder-tp-mode data

# 场景 3: 需要完整图片上下文
# 建议: 禁用 chunked mm input
vllm serve model --disable-chunked-mm-input
```

---

## 22.8 小结

### 问题 2 答案总结

| 问题 | 答案 |
|------|------|
| **image1 和 image2 会 cache 命中吗？** | ❌ 不会，mm_hash 不同 |
| **驱逐时会部分驱逐吗？** | ❌ 不会，Encoder Cache 整体驱逐 |
| **相同图片可以跨请求共享吗？** | ✅ 可以，通过 mm_hash 匹配 |

### 问题 3 答案总结

| 问题 | 答案 |
|------|------|
| **KV Cache 有特殊存储逻辑吗？** | ❌ 没有，与文本完全相同 |
| **如何保证图片完整性？** | Encoder Cache 整体缓存 + `disable_chunked_mm_input` |
| **Encoder Cache 和 KV Cache 区别？** | Encoder Cache 存 ViT 输出，KV Cache 存 Attention K/V |

```
最终架构总结:

              图片数据
                 ↓
         ┌──────────────┐
         │ 计算 mm_hash │
         └──────────────┘
                 ↓
         ┌──────────────┐
         │ Encoder Cache│ ← mm_hash 作为 key
         │   检查       │   相同图片可跨请求共享
         └──────────────┘
                 ↓
        (命中?)─────→ 直接使用缓存
           ↓ 未命中
         ┌──────────────┐
         │   执行 ViT   │ ← 整张图片一次性处理
         └──────────────┘
                 ↓
         ┌──────────────┐
         │ 缓存到       │
         │ encoder_cache│
         └──────────────┘
                 ↓
         ┌──────────────┐
         │ 合并到       │
         │ text embeds  │
         └──────────────┘
                 ↓
         ┌──────────────┐
         │ LLM Forward  │ ← 图片 tokens 与文本统一处理
         │              │   KV Cache 存储无区别
         └──────────────┘
                 ↓
         ┌──────────────┐
         │  KV Cache    │ ← block_hash 包含 mm_hash
         │  (BlockPool) │   相同前缀+相同图片可共享
         └──────────────┘
```

---

> **多模态章节完成**
>
> 回顾:
> - [19-multimodal-overview.md](./19-multimodal-overview.md) - 架构总览
> - [20-qwen3vl-model.md](./20-qwen3vl-model.md) - Qwen3-VL 模型
> - [21-vit-implementation.md](./21-vit-implementation.md) - ViT 实现 (问题1,4)
> - [22-mm-processing.md](./22-mm-processing.md) - 处理流程 (问题2,3)
>
> **下一节**: [23-hybrid-model-arch.md](./23-hybrid-model-arch.md) - 混合模型架构设计
