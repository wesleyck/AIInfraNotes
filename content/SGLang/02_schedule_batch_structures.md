# SGLang 调度数据结构学习笔记

> schedule_batch.py 精简学习指南

---

## 1. 文件结构概览

| 行数范围 | 内容 | 重要性 |
|----------|------|--------|
| 104-173 | FinishReason 类（5 个） | ⭐ 可跳过 |
| 176-452 | 多模态相关类 | ⭐⭐ 已学过 |
| 455-481 | `RequestStage` 枚举 | ⭐⭐⭐ 必看 |
| **484-1152** | **`Req` 类** | ⭐⭐⭐⭐ **核心** |
| **1155-2185** | **`ScheduleBatch` 类** | ⭐⭐⭐⭐ **核心** |
| 2188-2278 | `ModelWorkerBatch` 类 | ⭐⭐⭐ 理解数据即可 |

---

## 2. 三个类的关系

```
Req (请求)
 │
 └── 多个 Req 组成 ──► ScheduleBatch (调度批次)
                            │
                            └── 转换为 ──► ModelWorkerBatch (GPU批次)
                                                │
                                                └── 最终转换 ──► ForwardBatch
```

---

## 3. Req 类核心属性（共 6 组）

```python
class Req:
    # ===== 组1: 身份标识 =====
    self.rid                    # 请求 ID
    self.origin_input_ids       # 原始 token ids
    self.output_ids             # 生成的 token ids
    self.fill_ids               # origin_input_ids + output_ids
    
    # ===== 组2: KV Cache 管理 =====
    self.req_pool_idx           # 在 ReqToTokenPool 中的索引
    self.prefix_indices         # 共享 prefix 的 KV cache 索引
    self.extend_input_len       # 需要 prefill 的 token 数
    self.cached_tokens          # 已缓存的 token 数
    
    # ===== 组3: 状态控制 =====
    self.finished_reason        # 完成原因
    self.is_chunked             # 是否在 chunked prefill 中
    self.is_retracted           # 是否被回退
    self.stream                 # 是否流式输出
    
    # ===== 组4: 多模态 =====
    self.multimodal_inputs      # MultimodalInputs
    
    # ===== 组5: 采样参数 =====
    self.sampling_params
    
    # ===== 组6: Logprob 相关 =====
    self.return_logprob
    self.logprob_start_len
```

### Req 核心方法

| 方法 | 作用 | 何时调用 |
|------|------|----------|
| `init_next_round_input()` | 初始化下一轮输入，匹配 prefix cache | Prefill/Decode 开始前 |
| `check_finished()` | 检查是否完成 | 每次 decode 后 |
| `reset_for_retract()` | 回退请求状态 | 内存不足时 |
| `seqlen` (property) | 当前序列长度 | 随时 |
| `finished()` | 是否完成 | 状态检查 |

---

## 4. ScheduleBatch 类核心属性（共 4 组）

```python
class ScheduleBatch:
    # ===== 组1: 请求列表 =====
    self.reqs: List[Req]              # 批次中的所有请求
    
    # ===== 组2: 内存池引用 =====
    self.req_to_token_pool           # 请求 -> token 索引池
    self.token_to_kv_pool_allocator  # token -> KV cache 分配器
    self.tree_cache                   # Radix Tree Cache
    
    # ===== 组3: 批次状态 =====
    self.forward_mode: ForwardMode   # PREFILL/DECODE/EXTEND
    self.input_ids                    # 当前批次的 input_ids
    self.seq_lens                     # 每个请求的序列长度
    
    # ===== 组4: 用于 GPU 的张量 =====
    self.req_pool_indices            # 请求在池中的索引
    self.out_cache_loc               # 输出 KV cache 的位置
```

### ScheduleBatch 核心方法

| 方法 | 作用 | 调用时机 |
|------|------|----------|
| `init_new()` | 创建新批次 | 调度器选择请求后 |
| `prepare_for_extend()` | 准备 prefill/extend 批次 | Prefill 前 |
| `prepare_for_decode()` | 准备 decode 批次 | Decode 前 |
| `filter_batch()` | 过滤完成的请求 | 每轮结束 |
| `retract_decode()` | 回退请求（内存不足） | 内存压力时 |
| `get_model_worker_batch()` | 转换为 GPU 批次 | 发给 Worker 前 |

---

## 5. ModelWorkerBatch（@dataclass）

```python
@dataclass
class ModelWorkerBatch:
    forward_mode: ForwardMode     # 前向模式
    input_ids: torch.Tensor       # 输入 token ids (GPU)
    req_pool_indices: torch.Tensor # 请求索引 (GPU) 
    seq_lens: torch.Tensor        # 序列长度 (GPU)
    out_cache_loc: torch.Tensor   # KV cache 写入位置
    # ... 其他 GPU 张量
```

---

## 6. ForwardMode 枚举

```python
class ForwardMode(IntEnum):
    PREFILL = 0   # 预填充（新请求）
    DECODE = 1    # 解码（生成下一个 token）
    EXTEND = 2    # 扩展（续写/追加 context）
    IDLE = 3      # 空闲
```

---

## 7. 推荐学习路径

```
第1步 (10分钟): ForwardMode 枚举
       ↓
第2步 (20分钟): Req 核心属性（6 组）
       ↓
第3步 (20分钟): ScheduleBatch 核心属性 + prepare_for_extend()
       ↓
第4步 (10分钟): ModelWorkerBatch 字段
       ↓
第5步: 按需深入其他方法
```

---

## 8. 关键流程：请求生命周期

```
新请求入队
    ↓
Scheduler.get_new_batch_for_prefill()
    ↓ 创建
ScheduleBatch.init_new(reqs=[...])
    ↓
ScheduleBatch.prepare_for_extend()  ← 分配 KV cache，准备 input_ids
    ↓
ScheduleBatch.get_model_worker_batch() → ModelWorkerBatch
    ↓ 发给 GPU Worker
Model Forward
    ↓ 返回
ScheduleBatch.filter_batch()  ← 移除完成的请求
    ↓
ScheduleBatch.prepare_for_decode()  ← 准备下一轮 decode
    ↓
循环直到所有请求完成
```
