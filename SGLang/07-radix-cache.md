# SGLang RadixCache 前缀缓存详解

> **默认场景**: Qwen3.5 混合架构模型（Full Attention + Linear Attention/GatedDeltaNet + MoE + MTP）
>
> **启用特性**: PD 分离 + Chunked Prefill + ViT DP + Overlap Schedule + 多模态缓存 + EPLB + MTP + 线性注意力

## 本章定位
- 主题范围: RadixCache 命中、锁与驱逐。

## 设计 Why（为什么这么设计）
- 前缀树缓存提升复用率，但必须与锁和驱逐策略协同。
- 核心取舍: 吞吐 vs 时延、显存 vs 计算、通用性 vs 特化。

## 阅读建议（进阶）
1. 先抓目标函数和边界条件，再读具体实现。
2. 先看调用链和状态变化，再看局部优化细节。
3. 源码锚点以“路径 + 类/函数”为主，避免依赖易漂移行号。

## 1. RadixCache 概览

**核心文件**:
- `python/sglang/srt/mem_cache/radix_cache.py` - 基础 RadixCache
- `python/sglang/srt/mem_cache/base_prefix_cache.py` - 抽象基类
- `python/sglang/srt/mem_cache/evict_policy.py` - 逐出策略
- `python/sglang/srt/mem_cache/swa_radix_cache.py` - SWA 变体
- `python/sglang/srt/mem_cache/mamba_radix_cache.py` - Mamba 变体
- `python/sglang/srt/mem_cache/hiradix_cache.py` - 层级缓存变体

### 1.1 什么是 RadixAttention?

RadixAttention 是 SGLang 的核心创新，通过 **Radix Tree (基数树)** 数据结构高效管理 KV Cache 的前缀共享。

```mermaid
flowchart TD
    subgraph Traditional["传统方案: 每个请求独立存储完整 KV Cache"]
        ReqA["请求 A: System Prompt + User A Query -> 独立 KV Cache"]
        ReqB["请求 B: System Prompt + User B Query -> 独立 KV Cache, 重复存储!"]
    end

    subgraph RadixWay["RadixAttention: 共享公共前缀"]
        SP["System Prompt - 共享"]
        UA["User A Query"]
        UB["User B Query"]
        SP --> UA
        SP --> UB
    end

    Traditional -.->|"节省内存 + 无需重复计算"| RadixWay
```

### 1.2 类继承关系

```mermaid
flowchart TD
    Base["BasePrefixCache - 抽象基类"]
    Base --> RC["RadixCache - 标准 Radix Tree 缓存"]
    Base --> SWA["SWARadixCache - Sliding Window Attention 变体"]
    Base --> Mamba["MambaRadixCache - Mamba-Hybrid 模型变体"]
    RC --> Hi["HiRadixCache - 继承自 RadixCache"]
    Base --> Chunk["ChunkCache - 简化版, 禁用 radix_cache 时使用"]
    Base --> Cpp["RadixCacheCpp - C++ 优化实现, 实验性"]
```

## 2. 核心数据结构

### 2.1 RadixKey

```python
class RadixKey:
    def __init__(self, token_ids: List[int], extra_key: Optional[str] = None,
                 is_bigram: bool = False):
        self.token_ids = token_ids  # Token ID 序列
        self.extra_key = extra_key  # 额外键 (LoRA ID, cache_salt 等)
        self.is_bigram = is_bigram  # 是否为 EAGLE bigram key
```

**用途**:
- `token_ids`: 实际的 token 序列，用于前缀匹配
- `extra_key`: 隔离不同 LoRA 或数据集的缓存命名空间

### 2.2 TreeNode

```python
class TreeNode:
    def __init__(self, id=None, priority=0):
        self.children = defaultdict(TreeNode)  # 子节点 (按首 token 索引)
        self.parent: TreeNode = None           # 父节点
        self.key: RadixKey = None              # 本节点存储的 token 序列
        self.value: torch.Tensor = None        # KV Cache 索引

        # 锁和访问控制
        self.lock_ref = 0                      # 引用计数 (防止逐出)
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()
        self.hit_count = 0                     # 访问次数 (LFU 用)
        self.priority = priority               # 优先级 (Priority eviction)

        # Hierarchical Cache 相关
        self.host_value: torch.Tensor = None   # Host 端 KV 索引
        self.host_ref_counter = 0              # Host 引用计数
        self.hash_value: List[str] = None      # SHA256 块哈希

```

### 2.3 TreeNode.value 的含义与使用

- **类型**: `Optional[torch.Tensor]` (dtype=int64)
- **内容**: 存储的是 KV Cache 的**物理 slot 索引**（与 Allocator 分配的索引一致）

**数据流向**：TreeNode.value 的写入和读取分两个阶段：

```
写入（请求完成时）:
  req_to_token_pool[req_id, 0..seq_len]  →  TreeNode.value
  ↑                                          ↑
  模型前向过程中通过 set_kv_buffer()       cache_finished_req() / cache_unfinished_req()
  写入的 KV slot 索引                      将这些索引拷贝到 Radix Tree 节点

读取（新请求匹配时）:
  match_prefix() 沿 Tree 路径拼接所有节点的 value
  → prefix_indices = [node1.value, node2.value, ...]
  → 写入新请求的 req_to_token_pool[new_req_id, 0..prefix_len]
  → 新请求的 Attention 层直接用这些 slot 索引读取已缓存的 KV（零拷贝复用）
```

- `key` 存储 token IDs（用于匹配），`value` 存储对应的 KV 物理位置（用于复用）。两者一一对应：`key[i]` 这个 token 的 KV 数据存在 `value[i]` 指向的物理 buffer 位置。

### 2.4 Radix Tree 结构示例

```mermaid
graph TD
    Root["[root]"] --> Node12["[1, 2]<br/>KV: [idx0, idx1]"]
    Root --> Node8["[8,9,10,11,12]<br/>KV: [idx4..idx8]"]

    Node12 --> Node3["[3]<br/>KV: [idx2]"]
    Node12 --> Node4567["[4,5,6,7]<br/>KV: [idx3..idx6]"]

    Node3 --> Node45["[4, 5]<br/>KV: [idx7, idx8]"]

    style Root fill:#f9f,stroke:#333
    style Node12 fill:#bbf,stroke:#333
    style Node3 fill:#bfb,stroke:#333
    style Node4567 fill:#bfb,stroke:#333
    style Node45 fill:#fbf,stroke:#333
    style Node8 fill:#bbf,stroke:#333
```

**插入序列说明**:

```mermaid
graph TD
    subgraph Sequences["插入序列"]
        direction LR
        S1["[1, 2, 3]"]
        S2["[1, 2, 3, 4, 5]"]
        S3["[1, 2, 4, 5, 6, 7]"]
        S4["[8, 9, 10, 11, 12]"]
    end

    subgraph Tree["形成的 Radix Tree"]
        R["root"]
        N12["[1, 2]"]
        N89["[8,9,10,11,12]"]
        N3["[3]"]
        N4567["[4,5,6,7]"]
        N45["[4, 5]"]

        R --> N12
        R --> N89
        N12 --> N3
        N12 --> N4567
        N3 --> N45
    end

    Sequences -.->|"构建"| Tree
```

每个节点的 value 存储对应 token 的 KV 索引

## 3. 核心操作

### 3.1 match_prefix

**最长前缀匹配**

```python
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    """
    查找 key 在 Radix Tree 中的最长公共前缀

    参数:
        params.key: RadixKey - token_ids + extra_key
        params.cow_mamba: bool - Mamba Copy-On-Write (MambaRadixCache 专用)
        params.req: Req - 请求对象 (MambaRadixCache 专用)

    匹配逻辑:
    - 内部先调用 maybe_bigram_convert() 处理 EAGLE bigram 转换
    - 底层使用 **Token ID 逐个比较** (`_key_match_page_size1`) 或按页比较 (`_key_match_paged`)
    - `extra_key` (如 LoRA ID) 通过 `get_child_key_fn` 影响子节点查找路径，
      实现逻辑上的命名空间隔离，不同 `extra_key` 的请求不会匹配到彼此的节点

    返回:
        MatchResult(
            device_indices: 匹配到的 KV 索引
            last_device_node: 匹配终止的节点
            last_host_node: (HiCache) Host 端匹配节点
            host_hit_length: Host 端命中长度 (HiCache)
            mamba_branching_seqlen: Mamba 分支点 (MambaRadixCache)
        )
    """
```

**匹配流程**:

```mermaid
flowchart LR
    subgraph Input
        Key["key = [1, 2, 3, 4, 5, 6]"]
    end

    Root["root"] --> |"children[1]"| N12["[1,2]<br/>匹配 2"]
    N12 --> |"children[3]"| N3["[3]<br/>匹配 1"]
    N3 --> |"children[4]"| N45["[4,5]<br/>匹配 2"]
    N45 --> |"children[6]"| Miss["不存在"]

    Miss --> Result["返回: 5 个 token 的 KV 索引"]

    style Result fill:#90EE90
    style Miss fill:#FFB6C1
```

**详细步骤**:

```mermaid
flowchart TD
    Start["输入: key = [1, 2, 3, 4, 5, 6]"] --> Step1
    Step1["1: 从 root 开始, 查找子节点 children[1]"] --> Step2
    Step2["2: 找到节点 [1, 2], key_match_fn 比较:<br/>key[0:2] == node.key[0:2], 匹配长度 = 2"] --> Step3
    Step3["3: 继续查找子节点 children[3]"] --> Step4
    Step4["4: 找到节点 [3], 匹配长度 = 1"] --> Step5
    Step5["5: 继续查找子节点 children[4], 找到 [4, 5]"] --> Step6
    Step6["6: 匹配 [4, 5] vs key[3:5], 完全匹配"] --> Step7
    Step7["7: 查找 children[6], 不存在, 结束"] --> Result
    Result["返回: indices = concat of [1,2], [3], [4,5] 的 KV 索引"]

    style Result fill:#90EE90
```

### 3.2 insert

**插入新序列**

```python
def insert(self, params: InsertParams) -> InsertResult:
    """
    插入 token 序列及其 KV 索引到树中

    参数:
        params.key: RadixKey - token_ids + extra_key
        params.value: torch.Tensor - KV 索引
        params.priority: int - 优先级
        params.chunked: bool - 是否为 chunked prefill 中间插入

    返回: InsertResult(prefix_len=已存在的前缀长度, mamba_exist=...)
    """
```

**关键细节**:
- 如果前缀已存在，只插入新增部分
- 返回 `total_prefix_length` 告知调用方哪些 KV 已存在（可释放）

### 3.3 _split_node

**节点分裂**

当匹配在节点中间结束时，需要分裂节点：

```mermaid
flowchart TD
    subgraph Before["分裂前"]
        P1["parent"] --> N1["[1,2,3,4,5]"]
    end

    subgraph After["匹配 [1,2,3,6,7] 后分裂"]
        P2["parent"] --> N2["[1,2,3] - 新节点"]
        N2 --> N3["[4,5] - 原节点"]
        N2 --> N4["[6,7] - 新插入"]
    end

    Before -.->|"split"| After
```

```python
def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
    new_node = TreeNode(priority=child.priority)
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len]
    new_node.children = {get_child_key(child.key[split_len:]): child}
    child.key = child.key[split_len:]
    child.value = child.value[split_len:]
    # ... 更新父子关系
```

## 4. 锁机制 (lock_ref)

### 4.1 为什么需要锁?

```mermaid
flowchart TD
    subgraph Problem["问题场景"]
        A1["请求 A 正在使用节点 [1,2,3] 的 KV Cache 进行推理"]
        A2["此时内存不足, evict 尝试释放该节点"]
        A3["导致请求 A 访问已释放内存, 崩溃!"]
        A1 --> A2 --> A3
    end

    subgraph Solution["解决方案: 引用计数锁"]
        B1["inc_lock_ref: 请求开始使用时, 锁定从 last_node 到 root 的路径"]
        B2["dec_lock_ref: 请求完成时, 解锁"]
        B3["evict 只能逐出 lock_ref == 0 的节点"]
        B1 --> B2 --> B3
    end

    Problem -.->|"解决"| Solution
```

### 4.2 实现细节

```python
def inc_lock_ref(self, node: TreeNode):
    """锁定从 node 到 root 的整个路径"""
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:
            # 从 evictable 转为 protected
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
            delta -= len(node.key)
        node.lock_ref += 1
        self._update_leaf_status(node)  # 维护 evictable_leaves set
        node = node.parent
    return delta

def dec_lock_ref(self, node: TreeNode):
    """解锁路径"""
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 1:
            # 从 protected 转为 evictable
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)
            delta += len(node.key)
        node.lock_ref -= 1
        self._update_leaf_status(node)  # 维护 evictable_leaves set
        node = node.parent
    return delta
```

> **`_update_leaf_status(node)`**: 每次 lock_ref 变化后都调用，判断节点是否应加入/移出 `evictable_leaves` set。判断逻辑：节点未被驱逐 (`not evicted`) 且 `lock_ref == 0` 且所有子节点都已被驱逐时，才是 evictable leaf。

## 5. 逐出策略

**文件**: `python/sglang/srt/mem_cache/evict_policy.py`

| 策略 | 优先级计算 | 适用场景 |
|------|-----------|---------|
| **LRU** | `last_access_time` | 热点数据保留 (默认) |
| **LFU** | `(hit_count, last_access_time)` | 高频访问优先 |
| **FIFO** | `creation_time` | 按创建顺序 |
| **MRU** | `-last_access_time` | 最近使用优先逐出 |
| **FILO** | `-creation_time` | 后进先出 |
| **Priority** | `(priority, last_access_time)` | 请求优先级感知 |

### 5.1 evict() 流程

```python
def evict(self, params: EvictParams) -> EvictResult:
    num_tokens = params.num_tokens
    leaves = list(self.evictable_leaves)  # 使用维护好的 evictable_leaves set
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free(x.value)  # 释放 KV
        num_evicted += len(x.value)
        self._delete_leaf(x)                            # 删除节点

        # 如果父节点变成叶子且未被锁定，可能成为下一个逐出候选
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))

    return EvictResult(num_tokens_evicted=num_evicted)
```

> **关键区别**: 实际实现使用 `self.evictable_leaves` (一个 `set`) 而非“每次逐出前全树收集叶子节点”的遍历策略。`evictable_leaves` 由 `_update_leaf_status()` 实时维护，避免每次 evict 都遍历整棵树。

## 6. 请求生命周期中的缓存操作

### 6.1 cache_finished_req

**作用**: 请求生命周期结束时调用，完成两件事：
1. **固化 KV Cache**：将该请求的 token 序列和对应的 KV slot 索引插入 Radix Tree，使**下一轮具有相同前缀的请求**可以直接命中并复用这些 KV 数据。
2. **释放资源**：释放与树中已有节点重复的 KV 索引、释放 page 对齐截断的 unaligned tail、解锁节点（`dec_lock_ref` → `lock_ref` 降为 0 → 节点变为**可驱逐**状态）。被固化到 Tree 的 KV slot **不会立即 free**，而是留在 Tree 中等待后续请求复用或 LRU 驱逐时才释放。`req_to_token_pool.free(req_pool_idx)` 由 Scheduler 在外部调用，不在此方法内。

**关键动作**:

```mermaid
flowchart TD
    S1["1: pop_committed_kv_cache<br/>获取已提交的 KV 长度"]
    S2["2: 构建 RadixKey<br/>token_ids, extra_key, page_align"]
    S3["3: insert(InsertParams(key, value, priority))<br/>返回: InsertResult(prefix_len=已存在前缀长度)"]
    S4["4: free kv_indices[cache_protected_len : new_prefix_len]<br/>释放与树中已有节点重复的 KV"]
    S5["5: free kv_indices[len(keys):]<br/>释放 page_align 截断的 unaligned tail"]
    S6["6: dec_lock_ref(req.last_node)<br/>解锁之前持有的节点"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6
```

> **注意**: `req_to_token_pool.free(req_pool_idx)` 不在 `cache_finished_req` 内部执行，而是由 Scheduler 在外部调用。Step 5 释放的是 page 对齐后多余的 KV 索引（unaligned tail）。

### 6.2 cache_unfinished_req

用于 **Chunked Prefill**，请求未完成但需要中间缓存：

```python
def cache_unfinished_req(self, req: Req, chunked=False):
    # 插入当前已处理的 token
    new_prefix_len = self.insert(radix_key, kv_indices)

    # 释放重复部分
    self.token_to_kv_pool_allocator.free(kv_indices[cache_protected_len:new_prefix_len])

    # 重新匹配以获取最新的节点引用
    match_result = self.match_prefix(radix_key)
    new_last_node = match_result.last_device_node

    # 更新请求的 prefix_indices 供下一个 chunk 使用
    req.prefix_indices = new_indices
    req.cache_protected_len = len(new_indices)

    # 更新锁
    self.dec_lock_ref(req.last_node)
    self.inc_lock_ref(new_last_node)
    req.last_node = new_last_node
```

## 7. Page Size 与对齐

当 `page_size > 1` 时，匹配和插入以 page 为单位：

```python
if self.page_size != 1:
    page_aligned_len = len(key) // self.page_size * self.page_size
    key = key[:page_aligned_len]
```

**影响**:
- 匹配粒度变粗 (以 page_size 个 token 为单位)
- 部分页 (partial page) 需要特殊处理，不能进入 RadixCache

## 8. extra_key 命名空间隔离

```python
# 不同 LoRA 的请求共享相同 system prompt 也不会混用
key1 = RadixKey([1,2,3], extra_key="lora_adapter_A")
key2 = RadixKey([1,2,3], extra_key="lora_adapter_B")
# key1 和 key2 被视为完全不同的缓存条目
```

命名空间隔离确保:
- 不同 LoRA adapter 的 KV Cache 不会混用
- 不同 `cache_salt` 的请求隔离

## 9. EAGLE Bigram 转换

对于 EAGLE 投机解码，使用 bigram key：

```python
def convert_to_bigram_key(token_ids: List[int]) -> List[int]:
    """
    [1, 2, 3, 4] → [(1,2), (2,3), (3,4)]
    """
    return [combine(token_ids[i], token_ids[i+1]) for i in range(len(token_ids)-1)]
```

这允许 EAGLE 的 draft model 共享 target model 的部分 KV Cache。

## 10. Scheduler 集成

```python
- 源码锚点: `python/sglang/srt/managers/scheduler.py`
if disable_radix_cache and chunked_prefill:
    # 禁用 radix cache 时使用简化版
    if self.is_hybrid_swa:
        self.tree_cache = SWAChunkCache(params)
    else:
        self.tree_cache = ChunkCache(params)
else:
    if SGLANG_EXPERIMENTAL_CPP_RADIX_TREE:
        self.tree_cache = RadixCacheCpp(params, server_args)  # C++ 实验性实现
    elif self.enable_hierarchical_cache:
        self.tree_cache = HiRadixCache(params, server_args)   # 层级缓存 (继承 RadixCache)
    elif self.is_hybrid_swa:
        self.tree_cache = SWARadixCache(params)               # SWA 混合架构
    elif self.is_hybrid_ssm:
        self.tree_cache = MambaRadixCache(params)             # Mamba 混合架构
    elif server_args.enable_lmcache:
        self.tree_cache = LMCRadixCache(params, ...)          # LMCache 集成
    else:
        self.tree_cache = RadixCache(params)                  # 默认
```

## 11. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `disable_radix_cache` | False | 禁用前缀缓存 |
| `radix_eviction_policy` | "lru" | 逐出策略 |
| `page_size` | 16 | 页大小 |
| `enable_hierarchical_cache` | False | 启用层级缓存 |

## 12. 多模态 Chunked Prefill 与缓存命中案例

### 12.1 场景设定

```
假设:
  chunk_size = 512 tokens
  image1 token 数 = 1024 (> chunk_size，需要 2 个 chunk)
  image2 token 数 = 768  (> chunk_size，需要 2 个 chunk)
  image3 token 数 = 896  (> chunk_size，需要 2 个 chunk)
  text1 token 数 = 100

请求:
  req1: text1 + image1 + image2
  req2: text1 + image1 + image3
```

### 12.2 req1 处理流程 (首次请求)

```mermaid
flowchart TD
    Input["原始输入: text1:100 + image1:1024 + image2:768<br/>总 token: 1892"]

    Input --> C1
    subgraph C1["Chunk 1 - 512 tokens"]
        C1A["text1:100 + image1 前 412 tokens"]
        C1B["match_prefix -> cache miss, 首次请求"]
        C1C["计算 KV Cache"]
        C1D["cache_unfinished_req -> 插入 RadixCache"]
        C1A --> C1B --> C1C --> C1D
    end

    C1 --> C2
    subgraph C2["Chunk 2 - 512 tokens"]
        C2A["image1 剩余 612 tokens, 只取前 512"]
        C2B["match_prefix -> hit 512 tokens, chunk1 已缓存"]
        C2C["计算 512 tokens 的 KV"]
        C2D["cache_unfinished_req -> 更新 RadixCache"]
        C2A --> C2B --> C2C --> C2D
    end

    C2 --> C3
    subgraph C3["Chunk 3 - 512 tokens"]
        C3A["image1 最后 100 + image2 前 412 tokens"]
        C3B["match_prefix -> hit 1024 tokens"]
        C3C["继续计算和缓存"]
        C3A --> C3B --> C3C
    end

    C3 --> C4
    subgraph C4["Chunk 4 - 剩余 356 tokens"]
        C4A["image2 剩余 356 tokens"]
        C4B["cache_finished_req -> 完成, 插入最终状态"]
        C4A --> C4B
    end
```

### 12.3 req2 部分命中场景

```mermaid
flowchart TD
    Input["req2 输入: text1:100 + image1:1024 + image3:896"]

    subgraph CacheState["RadixCache 当前状态 - req1 完成后"]
        CR["root"]
        CN["text1 + image1 + image2<br/>1892 tokens 的 KV 索引"]
        CR --> CN
    end

    Input --> R1
    subgraph R1["req2 Chunk 1 - 512 tokens"]
        R1A["key = text1:100 + image1 前 412"]
        R1B["match_prefix -> 完全命中 512 tokens"]
        R1C["无需计算, 直接复用 KV"]
        R1A --> R1B --> R1C
    end

    R1 --> R2
    subgraph R2["req2 Chunk 2 - 512 tokens"]
        R2A["key = text1:100 + ... + image1 前 924"]
        R2B["match_prefix -> 完全命中 1024 tokens"]
        R2C["无需计算"]
        R2A --> R2B --> R2C
    end

    R2 --> R3
    subgraph R3["req2 Chunk 3 - 512 tokens - 关键分歧点!"]
        R3A["key = text1:100 + image1:1024 + image3 前 388"]
        R3B["match_prefix 匹配过程:<br/>树中: text1+image1+image2...<br/>请求: text1+image1+image3...<br/>匹配到位置 1124 时发现分歧"]
        R3C["触发 _split_node 分裂节点"]
        R3D["返回: hit_len = 1124 tokens, page_aligned"]
        R3E["需要计算: image3 前 388 tokens 的 KV"]
        R3F["计算后 cache_unfinished_req"]
        R3A --> R3B --> R3C --> R3D --> R3E --> R3F
    end

    R3 --> SplitResult
    subgraph SplitResult["分裂后的 RadixCache"]
        SR["root"]
        SN1["text1 + image1<br/>1124 tokens - 新分裂节点"]
        SN2["image2...<br/>768 tokens"]
        SN3["image3...<br/>待计算"]
        SR --> SN1
        SN1 --> SN2
        SN1 --> SN3
    end

    SplitResult --> R4
    subgraph R4["req2 Chunk 4 - 剩余 508 tokens"]
        R4A["image3 剩余 508 tokens"]
        R4B["cache_finished_req -> 完成"]
        R4A --> R4B
    end
```

### 12.4 req3 并发场景：req1 未完成时 req3 到达（与 req2 输入相同）

> **核心问题**: 缓存复用是否必须等请求生命周期结束（`cache_finished_req`）？还是在请求执行过程中就可以被后续请求命中？

**结论**: **不需要等待**。`cache_unfinished_req()` 使得每完成一个 chunk，中间结果就立即固化到 Radix Tree，后续请求在下一个调度轮次即可命中。

#### 场景设定

```
req1: text1:100 + image1:1024 + image2:768  (总 1892 tokens)
req3: text1:100 + image1:1024 + image3:896  (总 2020 tokens, 与 req2 输入相同)
chunk_size = 512

关键时序: req3 在 req1 完成 Chunk 2 后、Chunk 3 前到达
```

#### Scheduler 单线程事件循环

SGLang Scheduler 是**单线程事件循环**，每个 step 只处理一个 prefill batch。Chunked 请求在每个 chunk 完成后会被移出 running batch，在下一轮重新调度。这保证了 `cache_unfinished_req` → `match_prefix` 的顺序一致性：

```
scheduler.py:
    self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
    → insert 当前已完成 chunk 的 KV 到 Tree
    → req_to_token_pool.free(chunked_req.req_pool_idx)  # 释放旧槽位
    → 下一轮 get_new_batch_prefill() 重新分配 req_pool_idx
```

#### 完整时序

```
Step 1: req1 Chunk 1 (tokens 0-511)
─────────────────────────────────────────────────────────────────────
  Scheduler:
    → get_new_batch_prefill(): 从 waiting_queue 取出 req1
    → match_prefix(req1) → cache miss (首次)
    → PrefillAdder: extend_len=1892 > chunk_size=512, 标记为 chunked
    → 计算 tokens[0..511] 的 KV
  完成后:
    → cache_unfinished_req(req1): insert [0..511] 到 Tree
    → Tree: [root] → [0..511] (lock_ref=1, req1 持有)
    → req1 移出 running batch, 等待下一轮重新调度

Step 2: req1 Chunk 2 (tokens 512-1023)
─────────────────────────────────────────────────────────────────────
  Scheduler:
    → get_new_batch_prefill(): req1 作为 self.chunked_req 优先重新加入
    → 接续前缀 [0..511], 计算 tokens[512..1023]
  完成后:
    → cache_unfinished_req(req1): insert [0..1023] 到 Tree
    → Tree: [root] → [0..1023] (lock_ref=1)

  同时: req3 到达, 加入 waiting_queue (此时 req1 未完成!)

Step 3: req1 Chunk 3 + req3 被调度 (可能同一 batch)
─────────────────────────────────────────────────────────────────────
  Scheduler:
    → get_new_batch_prefill():
      1) req1 (chunked_req) 优先加入, 继续处理 tokens[1024..1535]
      2) req3 从 waiting_queue 取出
         → match_prefix(req3): 对比 Tree 中 [0..1023]
         → req3 tokens[0..1023] = text1 + image1 前 924  ← 与 req1 完全相同!
         → 命中 1024 tokens! inc_lock_ref → lock_ref=2
         → extend_len = 2020-1024 = 996 > chunk_size=512, req3 也变成 chunked
         → req3 被标记为 chunked, 只计算 tokens[1024..1535]

  注意: 一个 batch 中只允许一个 chunked_req, 所以 req1 和 req3 不会同时
  作为 chunked_req。实际调度是:
    → req1 Chunk 3 先执行 (chunked_req 优先)
    → req1 cache_unfinished_req → Tree 扩展到 [0..1535]
    → 下一轮 req3 才被调度

Step 4: req3 首次实际计算
─────────────────────────────────────────────────────────────────────
  此时 Tree 状态: [root] → [0..1535] (req1 已推进到 1536 tokens)

  Scheduler:
    → match_prefix(req3): 比对 req3 tokens[0..2019] 与 Tree [0..1535]
    → tokens[0..1123] (text1 + 完整 image1) 完全匹配!
    → token 1124: req3=image3_pad vs Tree=image2_pad → 分歧!
    → 命中 1124 tokens (page_aligned 后可能是 1120)

  Tree 变化 (req3 的 cache_unfinished_req 触发 _split_node):
    [root] → [text1+image1: 1124 tokens]
                  ├── [image2...: 412 tokens]  ← req1 的分支
                  └── [image3...: 512 tokens]  ← req3 的新分支

  req3 实际计算: 2020 - 1124 = 896 tokens 中的前 512 tokens

Step 5-6: req1 Chunk 4 (完成) + req3 Chunk 2 (完成)
─────────────────────────────────────────────────────────────────────
  req1: cache_finished_req → dec_lock_ref → image2 分支 lock_ref 降为 0
  req3: cache_finished_req → dec_lock_ref → 所有节点可驱逐

  最终 Tree:
    [root] → [text1+image1: 1124 tokens] (lock_ref=0)
                  ├── [image2...: 768 tokens]  (lock_ref=0)
                  └── [image3...: 896 tokens]  (lock_ref=0)
```

#### 关键结论

```mermaid
flowchart TD
    Q1["Q: 缓存复用需要等请求完成吗?"] --> A1["A: 不需要!"]
    A1 --> M1["cache_unfinished_req 在每个 chunk 完成后<br/>立即将中间结果固化到 Radix Tree"]
    A1 --> M2["后续请求在下一个调度轮次<br/>match_prefix 即可命中"]
    A1 --> M3["这就是 SGLang 的 online prefix caching<br/>不仅缓存已完成请求, 还实时共享运行中请求的进度"]

    Q2["Q: req3 实际节省了多少?"] --> A2["本例中 req3 命中 1124/2020 tokens ≈ 55.6%<br/>节省了 text1 + 完整 image1 的重复计算"]
    Q2 --> A3["如果 req3 晚到几个 step, req1 已推进更远<br/>命中可能更多 (但本例 image3≠image2 所以上限就是 1124)"]

    Q3["Q: lock_ref 如何保证安全?"] --> A4["req1 持有 lock_ref=1, req3 到达后变 2<br/>两者共享的节点不会被 evict<br/>req1 完成后降为 1, req3 完成后降为 0"]
```

> **代码路径**: `python/sglang/srt/managers/scheduler.py` → `radix_cache.py cache_unfinished_req` → `insert` + `match_prefix` → `inc_lock_ref`

### 12.5 Page Alignment 对部分命中的影响

```python
# page_size = 16 时的对齐
actual_match = 1124
page_aligned_match = (1124 // 16) * 16  # = 1120

# 结果: 虽然精确匹配到 1124，但只能复用 1120 tokens
# 后 4 tokens 需要重新计算 (页对齐开销)
```

### 12.6 多模态缓存隔离机制

> **注意**: `extra_key` 用于 **LoRA ID 和 cache_salt**，不是多模态图片隔离！

```python
- 源码锚点: `python/sglang/srt/managers/schedule_batch.py`
# extra key for classifying the request (e.g. cache_salt)
if lora_id is not None:
    extra_key = (extra_key or "") + lora_id  # LoRA ID 拼接到 extra_key
```

**多模态图片如何隔离?**

图片通过 **pad_value (图片 hash)** 嵌入到 token_ids 中：

```python
# MultimodalDataItem.set_pad_value()
self.hash = hash_feature(self.feature)  # 图片内容 hash
self.pad_value = self.hash % (1 << 30)  # 作为 token 占位符

# 最终 token_ids 示例:
# [text_tokens..., pad_value_image1, pad_value_image1, ..., pad_value_image2, ...]
```

因此，不同图片组合的请求：
- **不同 pad_value** -> 不同 token_ids -> 自然不会匹配
- **不需要 extra_key** 来隔离多模态

```mermaid
flowchart LR
    subgraph req1_tokens["req1 token_ids"]
        T1R1["100, 101, 102"]
        I1R1["999999, 999999<br/>image1"]
        I2R1["888888, 888888<br/>image2"]
        T1R1 --> I1R1 --> I2R1
    end

    subgraph req2_tokens["req2 token_ids"]
        T1R2["100, 101, 102"]
        I1R2["999999, 999999<br/>image1"]
        I3R2["777777, 777777<br/>image3"]
        T1R2 --> I1R2 --> I3R2
    end

    I1R1 -.->|"image1 相同, 可命中"| I1R2
    I2R1 -.->|"image2 vs image3 不同, 分歧点"| I3R2
```

### 12.7 并发请求间的缓存共享

**关键点**: 请求**无需等待生命周期结束**即可共享缓存。`cache_unfinished_req()` 使得请求**每完成一个 chunk 就立即将进度固化到 Radix Tree**，后续请求可以实时复用。

#### 时序示例：req1 未完成时 req3 进入

假设 req1 和 req3 的输入完全相同，启用 chunked prefill（chunk_size=512）：

```
时间轴:
─────────────────────────────────────────────────────────
t0: req1 到达 (2048 tokens)
    → Chunk 1 (0-511) 计算完成
    → cache_unfinished_req(): 将 [0..511] 的 KV 插入 Tree, inc_lock_ref()
    → Tree: [root] → [0..511] (lock_ref=1, req1 持有)

t1: req1 Chunk 2 (512-1023) 计算完成
    → cache_unfinished_req(): 将 [0..1023] 的 KV 更新到 Tree
    → Tree: [root] → [0..1023] (lock_ref=1, req1 持有)

t2: req3 到达 (相同 2048 tokens), 此时 req1 仍在运行!
    → match_prefix(): 匹配到 [0..1023] 的节点 (req1 已缓存的部分)
    → 命中 1024 tokens! 直接复用 KV, 无需重算
    → inc_lock_ref() → lock_ref=2 (req1 + req3 共同持有)
    → req3 只需从 token 1024 开始计算

t3: req1 Chunk 3 (1024-1535) 计算完成
    → cache_unfinished_req(): Tree 更新为 [0..1535]
    → req3 的下一个 chunk 可能再次命中更多

t4: req1 完成
    → cache_finished_req(): dec_lock_ref() → lock_ref=1 (只剩 req3)
    → 节点仍不可驱逐 (lock_ref > 0)

t5: req3 完成
    → cache_finished_req(): dec_lock_ref() → lock_ref=0
    → 节点变为可驱逐 (等待 LRU)
─────────────────────────────────────────────────────────
```

**核心机制**：
- `cache_unfinished_req()` 是 chunked prefill 的关键 — 每个 chunk 完成后都将中间结果固化到 Tree
- 后续请求**不需要等前一个请求完成**，只要前一个请求已提交的 chunk 就能立刻复用
- `lock_ref` 引用计数确保多个请求共同持有的节点不会被提前驱逐
- 这就是 SGLang 的 **online prefix caching** — 不仅缓存历史请求，还实时共享正在运行的请求

---

## 13. MambaRadixCache 详解

### 13.1 Mamba-Hybrid 模型的特殊挑战

```mermaid
flowchart TD
    subgraph Transformer["Transformer - KV Cache"]
        TA["无状态: 每个 token 的 KV 独立计算"]
        TB["可分割: token[0:100] 的 KV 可以与 token[100:200] 分开存储"]
        TC["前缀共享: A->B->C 和 A->B->D 可共享 A->B 的 KV"]
    end

    subgraph MambaSSM["Mamba - SSM State"]
        MA["有状态: 每个 token 依赖之前所有 token 累积的状态"]
        MB["不可分割: 状态必须从起始位置完整计算"]
        MC["分支问题: A->B->C 和 A->B->D 在 B 之后的状态完全不同!"]
    end

    subgraph BranchExample["Mamba 分支示例"]
        AB["A -> B"]
        StateAB["state_AB = f state_A, B"]
        C["C"]
        D["D"]
        StateABC["state_ABC = f state_AB, C"]
        StateABD["state_ABD = f state_AB, D"]
        Diff["两者完全不同, 不能共享!"]

        AB --> StateAB
        StateAB --> C --> StateABC
        StateAB --> D --> StateABD
        StateABC -.-> Diff
        StateABD -.-> Diff
    end
```

### 13.2 MambaRadixCache 双锁机制

```python
# MambaRadixCache 的 TreeNode
class TreeNode:
    # KV Cache 相关 (Attention 层)
    self.value: torch.Tensor = None           # KV 索引
    self.full_lock_ref = 0                    # KV 锁

    # Mamba State 相关 (SSM 层)
    self.mamba_value: torch.Tensor = None     # Mamba 状态索引
    self.mamba_lock_ref = 0                   # Mamba 锁

    # LRU 链表 (分开管理)
    self.prev, self.next = None, None         # KV LRU
    self.mamba_prev, self.mamba_next = None, None  # Mamba LRU
```

**不变量 (Invariant)**:

```mermaid
flowchart LR
    Inv["full_lock_ref >= mamba_lock_ref - 总是成立"]
    R1["如果需要 Mamba 状态, 必然也需要 KV Cache"]
    R2["但使用 KV Cache 不一定需要 Mamba 状态"]
    Inv --> R1
    Inv --> R2
```

### 13.3 Mamba Tombstone (墓碑节点)

当节点被分裂时，Mamba 状态无法分割：

```mermaid
flowchart TD
    subgraph Before["分裂前"]
        BN["node: key=[A,B,C,D]<br/>value=[kv0,kv1,kv2,kv3]<br/>mamba_value=state_ABCD"]
    end

    subgraph After["分裂后 - 匹配 [A,B,E] 时"]
        AN["new_node: key=[A,B]<br/>value=[kv0,kv1]<br/>mamba_value=None - Tombstone!"]
        AC["child: key=[C,D]<br/>value=[kv2,kv3]<br/>mamba_value=state_ABCD"]
        ANC["new_child: key=[E]<br/>value=[kv4]<br/>mamba_value=state_ABE"]
        AN --> AC
        AN --> ANC
    end

    Before -.->|"split"| After

    subgraph Tombstone["Tombstone 节点特征"]
        T1["KV Cache 可用 - value 有效"]
        T2["Mamba 状态不可用 - mamba_value=None"]
        T3["后续请求可复用 KV, 但必须从 tombstone 点重新计算 Mamba 状态"]
    end

    style AN fill:#ff9999
```

**Tombstone 存在的根本原因**:

Mamba 状态 (SSM state) 是**不可分割**的，只在 Chunk 边界 (`mamba_cache_chunk_size`，可配置) 有意义。当节点在中间位置被分裂 (Split) 时，新的中间节点无法拥有有意义的 Mamba 状态，因此置为 `None`（即 tombstone）。但 KV Cache 是 per-token 的，分裂后依然有效。

**KV Cache 与 Mamba State 的独立性**:

> **是的，KV cache 和 Mamba state 是完全独立的。** 它们由不同的物理存储管理（KVCache pool vs MambaPool），互不依赖。一个 token 的 KV cache 存在并不意味着它的 Mamba state 也存在，反之亦然。
>
> **Tombstone 的实际含义**就是这种独立性的体现：节点的 KV cache 有效（`value` 非空），但 Mamba state 无效（`mamba_value=None`）。当新请求匹配到 tombstone 节点时：
> - **Attention 层**：直接复用该节点的 KV cache（零拷贝）
> - **Mamba 层**：需要**回退到最近一个有 mamba_value 的祖先节点**，从那个 checkpoint 开始重新跑 Mamba 前向，更新 state
>
> 这就是 `_match_prefix_helper` 中 `best_value_len` / `best_last_node` 追踪最深有效 mamba 节点的逻辑 — 沿着 Tree 走到最远的 KV 匹配，但返回的 mamba 复用位置退回到最深的非 tombstone 节点。

### 13.4 cow_mamba (Copy-On-Write) 机制

cow_mamba 逻辑位于 `_match_post_processor()` 方法中（不在 `match_prefix` 本体），通过 `MatchPrefixParams` 传入参数：

```python
def _match_post_processor(self, params: MatchPrefixParams,
                          value, last_node, best_value_len) -> MatchResult:
    cow_mamba = params.cow_mamba
    req = params.req

    # ... LRU 更新、mamba_branching_seqlen 计算 ...

    # Copy mamba state to req local space if cow is true
    if cow_mamba and last_node.mamba_value is not None:
        if req.mamba_pool_idx is None:
            dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
            # 分配失败时: 锁定 last_node 防止被驱逐, 触发 evict, 再重试
            if dst_index is None:
                self.inc_lock_ref(last_node)
                self.evict(EvictParams(num_tokens=0, mamba_num=1))
                dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                self.dec_lock_ref(last_node)
            src_index = last_node.mamba_value
            self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)
            req.mamba_pool_idx = dst_index[0]
        else:
            # 已有 mamba_pool_idx, 直接覆盖
            src_index = last_node.mamba_value
            dst_index = req.mamba_pool_idx.unsqueeze(0)
            self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)
```

**为什么需要 Copy-On-Write?**

```mermaid
flowchart TD
    subgraph Problem["问题"]
        P1["req1 和 req2 都命中 [A,B] 节点"]
        P2["req1 继续生成 [C,D,E], 更新 Mamba 状态"]
        P3["req2 继续生成 [F,G], 也需要更新 Mamba 状态"]
        P4["如果共享状态, req1 的更新会破坏 req2!"]
        P1 --> P2 --> P3 --> P4
    end

    subgraph Solution["解决方案"]
        S1["match_prefix 时复制状态到请求私有空间"]
        S2["后续更新只影响请求自己的状态副本"]
        S1 --> S2
    end

    Problem -.->|"Copy-On-Write"| Solution
```

### 13.5 mamba_branching_seqlen

```python
# _match_prefix_helper 返回值 (注意: 返回完整 value list, 不做截断)
return value, best_last_node, best_value_len
# value: List[torch.Tensor] - 所有匹配节点的 KV 索引列表 (未截断)
# best_last_node: TreeNode - 最深的有 mamba_value 的节点
# best_value_len: int - best_last_node 对应的 value 列表长度

# mamba_branching_seqlen 在 _match_post_processor 中计算:
if len(value) > best_value_len:
    mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size  # 可配置
    total_matched = sum(len(v) for v in value)
    mamba_cache_chunk_aligned_seqlen = (
        total_matched // mamba_cache_chunk_size
    ) * mamba_cache_chunk_size
    mamba_branching_seqlen = (
        mamba_cache_chunk_aligned_seqlen if mamba_cache_chunk_aligned_seqlen > 0 else None
    )
```

**作用**: 告诉调度器从哪个位置开始需要重新计算 Mamba 状态

```mermaid
flowchart LR
    subgraph MatchResult["匹配结果"]
        MR1["KV hit = 1000 tokens"]
        MR2["mamba_value 只在 token 800 处有效<br/>800 之后是 tombstone"]
        MR3["mamba_branching_seqlen = 800<br/>mamba_cache_chunk_size 对齐 (可配置)"]
    end

    subgraph SchedulerAction["调度器处理"]
        SA1["token[0:800]: 直接复用 KV + Mamba 状态"]
        SA2["token[800:1000]: 复用 KV, 但需要重新运行 Mamba 层计算状态"]
    end

    MatchResult --> SchedulerAction
```

### 13.6 Chunked Prefill 与 Mamba 的特殊关联

```mermaid
flowchart TD
    subgraph Constraint1["约束 1: Mamba 状态必须在 mamba_cache_chunk_size 边界保存"]
        C1["原因: Mamba 内部使用 chunk-based 算法<br/>FLA = Flash Linear Attention<br/>状态只在 chunk 边界是完整的<br/>chunk 大小通过 mamba_cache_chunk_size 配置"]
    end

    subgraph Constraint2["约束 2: cache_unfinished_req 只缓存到 mamba_last_track_seqlen"]
        C2A["if enable_mamba_extra_buffer:"]
        C2B["cache_len = req.mamba_last_track_seqlen"]
        C2C["else: cache_len = len token_ids"]
        C2A --> C2B
        C2A --> C2C
    end

    subgraph Constraint3["约束 3: Mamba 状态需要 ping-pong buffer 交换"]
        C3A["req.mamba_ping_pong_track_buffer[0]<br/>当前 chunk 输入状态"]
        C3B["req.mamba_ping_pong_track_buffer[1]<br/>当前 chunk 输出状态"]
        C3C["完成后交换, 下一 chunk 继续"]
        C3A --> C3B --> C3C
    end
```

### 13.7 Mamba Eviction 策略

```python
# 两个独立的 LRU 链表
self.full_lru_list = LRUList(mamba=False)   # 管理 KV Cache
self.mamba_lru_list = LRUList(mamba=True)   # 管理 Mamba 状态

def evict_mamba(self, mamba_num: int) -> int:
    """只逐出 Mamba 状态，保留 KV Cache"""
    x = self.mamba_lru_list.get_lru_no_lock()
    mamba_num_evicted = 0
    while mamba_num_evicted < mamba_num and self.mamba_lru_list.in_list(x):
        if len(x.children) > 0:
            # 情况 1: 内部节点 → tombstone (保留节点结构)
            self.req_to_token_pool.mamba_pool.free(x.mamba_value)
            mamba_num_evicted += len(x.mamba_value)
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
            self.mamba_lru_list.remove_node(x)
            self._tombstone_internal_node(x)  # mamba_value=None, 节点保留
        else:
            # 情况 2: 叶子节点 → 完全删除 (释放 KV + Mamba)
            _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True)
            mamba_num_evicted += mamba_evicted_delta
        x = x_next

def evict(self, num_tokens: int):
    """逐出 KV Cache (同时会释放 Mamba 状态)"""
    # 先尝试只逐出 Mamba 状态释放内存
    # 如果不够，再逐出 KV Cache
```

**逐出优先级**:
1. 先逐出 Mamba 状态：内部节点变为 tombstone（保留 KV），叶子节点完全删除
2. 如果 KV 内存不足，再通过 `evict_full` 逐出完整节点

---

## 14. SWA RadixCache

**文件**: `python/sglang/srt/mem_cache/swa_radix_cache.py` (1188行)

管理 Full Attention 和 SWA 两种 attention 层的缓存。Llama4、Step3p5 等 SWA 混合架构模型需要同时维护两套缓存树。

> **注意**: Qwen3.5 的混合架构是 Full Attention + Linear Attention (GatedDeltaNet)，不使用 SWA。SWA 混合架构适用于 Llama4、Step3p5、GptOss、MiMoV2 等模型。

### 核心类

| 类 | 行号 | 说明 |
|----|------|------|
| `TreeNode` | L58 | 缓存树节点 |
| `LRUList` | L118 | LRU 逐出列表 |
| `SWARadixCache(BasePrefixCache)` | L339 | SWA Radix 缓存主类 |

## 15. BasePrefixCache 接口重构

**文件**: `python/sglang/srt/mem_cache/base_prefix_cache.py`

SGLang 引入了结构化参数类，统一了各 RadixCache 子类的接口：

| 参数类 | 说明 |
|--------|------|
| `MatchPrefixParams`  | 前缀匹配参数：key、cow_mamba、req |
| `InsertParams`  | 插入参数：key、value、mamba_value、prev_prefix_len、swa_evicted_seqlen、chunked、priority |
| `InsertResult`  | 插入结果：prefix_len、mamba_exist |
| `EvictParams`  | 逐出参数：num_tokens、swa_num_tokens、mamba_num |
| `EvictResult`  | 逐出结果：num_tokens_evicted、swa_num_tokens_evicted、mamba_num_evicted |
| `MatchResult`  | 匹配结果：device/host 索引、命中长度、mamba 分支信息 |

## 16. 存储后端更新

**文件**: `srt/mem_cache/storage/`

SGLang 支持多个外部存储后端：

| 后端 | 目录 | 说明 |
|------|------|------|
| NIXL | `nixl/` | NVIDIA NIXL 高性能传输 |
| aibrix_kvcache | `aibrix_kvcache/` | AIBrix KV Cache 存储 |
| EIC | `eic/` | EIC 存储后端 |
| HF3FS | `hf3fs/` | HuggingFace 3FS 文件系统 |
| LMCache | `lmcache/` | LMCache 集成 |
| Mooncake | `mooncake_store/` | Mooncake 分布式存储 |

## 17. Mamba RadixCache

**文件**: `python/sglang/srt/mem_cache/mamba_radix_cache.py` (1232行)

为 Qwen3.5 等包含线性注意力层的模型提供 Mamba 状态的 Radix 缓存支持。

| 类 | 行号 | 说明 |
|----|------|------|
| `TreeNode` | L63 | 缓存树节点 |
| `LRUList` | L117 | LRU 逐出列表 |
| `MambaRadixCache(BasePrefixCache)` | L371 | Mamba Radix 缓存主类 |

## 18. 下一步

- **08**: ModelRunner 与 CUDA Graph
- **09**: Attention 后端 (FlashInfer, FlashAttention)

## 与其他章节关系
- 为 `03/04/05` 提供前缀复用基础。


## 最小可验证实验
- 固定模型和负载，仅切换本章机制开关。
- 记录 TTFT、TPOT、吞吐、显存峰值与回退率。
- 总结收益场景、退化场景、推荐默认值。


## 常见误解
- 命中前缀就能完全复用。
