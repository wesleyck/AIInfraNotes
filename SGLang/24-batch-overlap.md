# SGLang Batch Overlap 批处理重叠

> **默认场景**: Qwen3.5 混合架构模型（Full Attention + Linear Attention/GatedDeltaNet + MoE + MTP）
>
> **启用特性**: PD 分离 + Chunked Prefill + ViT DP + Overlap Schedule + 多模态缓存 + EPLB + MTP + 线性注意力

## 1. 概述

Batch Overlap 是 v0.5.9 新增的模块，提供 ModelRunner 级别的计算-通信重叠优化。与 Scheduler 级别的 `event_loop_overlap`（CPU/GPU 重叠）不同，Batch Overlap 关注的是 **GPU 内部**的计算与通信重叠，特别适用于 MoE 模型（如 Qwen3.5）的 all-to-all 通信场景。

**文件**: `srt/batch_overlap/`

| 文件 | 行数 | 说明 |
|------|------|------|
| `single_batch_overlap.py` | 145 | SBO：单批重叠 |
| `two_batch_overlap.py` | 1074 | TBO：双批重叠 |
| `operations.py` | 214 | 重叠操作定义 |
| `operations_strategy.py` | 296 | 操作策略选择 |

## 2. 两种重叠模式

### 2.1 SBO (Single Batch Overlap)

**文件**: `srt/batch_overlap/single_batch_overlap.py` (145行)

单批重叠：在一个 batch 的 forward 内部，将 MoE 层的 all-to-all 通信与 attention 计算重叠。

```
时间轴 ──────────────────────────────────────────→

GPU Compute:  ┌─ Attention ─┐    ┌─ Attention ─┐
              │  Layer i     │    │  Layer i+1   │
              └──────────────┘    └──────────────┘
GPU Comm:          ┌─ MoE All-to-All (Layer i) ─┐
                   └─────────────────────────────┘
                         ↕ 重叠执行
```

SBO 的核心思想：MoE 层的 expert dispatch (all-to-all) 通信可以与下一层的 attention 计算并行执行。

### 2.2 TBO (Two Batch Overlap)

**文件**: `srt/batch_overlap/two_batch_overlap.py` (1074行)

双批重叠：两个 batch 的 forward 交替执行，一个做计算时另一个做通信。这是更激进的重叠策略。

```
时间轴 ──────────────────────────────────────────────────→

Batch A:  ┌─ Compute ─┐          ┌─ Compute ─┐
          │ (forward)  │          │ (forward)  │
          └────────────┘          └────────────┘
Batch B:       ┌─ Comm ─┐  ┌─ Compute ─┐
               │(all2all)│  │ (forward)  │
               └─────────┘  └────────────┘
                    ↕ A 的计算与 B 的通信重叠
```

TBO 需要专用的 Attention 后端（`tbo_backend.py`）配合，因为两个 batch 需要共享 KV Cache 和 attention 状态。

## 3. Operations 与 Strategy

### 3.1 Operations

**文件**: `srt/batch_overlap/operations.py` (214行)

定义了重叠操作的基本单元，包括：
- 计算操作（attention forward、MoE forward）
- 通信操作（all-to-all dispatch、all-to-all combine）
- 同步操作（stream 同步、barrier）

### 3.2 Operations Strategy

**文件**: `srt/batch_overlap/operations_strategy.py` (296行)

根据模型架构和硬件配置选择最优的操作策略：
- 决定哪些操作可以重叠
- 确定操作的执行顺序
- 处理依赖关系

## 4. 与其他组件的协作

### 4.1 与 Scheduler 的关系

```
Scheduler.event_loop_overlap()     ← CPU/GPU 重叠（调度 vs 前向）
    └─ run_batch()
        └─ ModelRunner.forward()
            └─ batch_overlap        ← GPU 内部重叠（计算 vs 通信）
```

两层重叠互不冲突，可以同时启用。

### 4.2 与 ModelRunner 的集成

ModelRunner 在 `forward()` 方法中检测是否启用 batch overlap，如果启用则使用 SBO 或 TBO 的 forward 路径替代标准 forward。

### 4.3 与 TBO Attention Backend 的协作

**文件**: `srt/layers/attention/tbo_backend.py` (约300行)

TBO 模式需要专用的 attention 后端，因为：
- 两个 batch 需要交替访问 KV Cache
- attention metadata 需要支持双 batch 的索引
- 需要额外的 stream 管理来协调两个 batch 的执行

## 5. 适用场景

Batch Overlap 主要适用于：
- **MoE 模型**（如 Qwen3.5）：MoE 层的 all-to-all 通信开销大，重叠可以显著提升吞吐
- **EP (Expert Parallel)** 场景：跨节点的 expert dispatch 通信延迟高
- **大规模部署**：多节点部署时通信开销占比更高，重叠收益更大

不适用于：
- 非 MoE 模型（无 all-to-all 通信）
- 单卡部署（无跨卡通信）

## 6. 下一步

- **08**: ModelRunner 中 batch overlap 的集成细节
- **09**: TBO Attention Backend 的实现
- **13**: EP 并行策略与 EPLB 负载均衡
