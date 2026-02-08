## Review (01-04 调度部分)

### 01_architecture.md
### 02_core_data_structures.md

1. 6.3部分，对于model workbatch的用途写的 跨进程传输  这个对吗？是多卡传输的意思吗？
2. 第七部分的数据流转示例中，画图表示流程

### 03_scheduler.md

1. 

### 05_memory_pool.md

1. 新增 §10 (显存占用分析与 OOM 机制)，原 §10 重编号为 §11

---

## 结构重组记录

**2025-02-08**: 03_scheduler.md + 05_memory_pool.md 结构重组

**03_scheduler.md**:
- 移除 L4/L5 大标题分隔 (`# L4: 精确细节补充` / `# L5: 高级主题`)
- §9 (Retraction 机制) 扩展:
  - §9.3: new_token_ratio 动态调整 (核心作用、初始化、衰减、OOM 回调)
  - §9.4: 在 PrefillAdder 中的使用 (rem_total_token_offset 计算)
  - §9.5: 完整生命周期图 (mermaid 流程图)
- 删除原 §23 (new_token_ratio 动态调整) - 已整合到 §9
- 重新编号: §24-31 → §23-30

**05_memory_pool.md**:
- 新增 §10 (显存占用分析与 OOM 机制):
  - §10.1: 显存占用分类 (静态 vs 动态)
  - §10.2: 静态显存 (模型权重、KV Cache)
  - §10.3: 动态显存 (模型激活、ViT 激活、图片像素)
  - §10.4: OOM 类型对比 (SGLang 逻辑 OOM vs CUDA OOM)
  - §10.5: OOM 预防机制 (自动估算、GPU默认配置、VLM调整、用户可调参数)
  - §10.6: 碎片问题 (KV Cache 无碎片、多模态可能有)
  - §10.7: 显存监控
- 原 §10 重编号为 §11

**2025-02-08**: 03_scheduler.md FutureMap 整合重组

**03_scheduler.md**:
- 新增 §2.8 (FutureMap 详解) 作为 Overlap 机制核心组件:
  - §2.8.1: 核心问题 (GPU 异步执行下的数据依赖)
  - §2.8.2: Future Token 机制 (负索引占位符)
  - §2.8.3: 设计优势 (省略 GPU→CPU→GPU 往返)
  - §2.8.4: 数据结构 (循环 Buffer)
  - §2.8.5: 生命周期 (alloc → resolve → store)
  - §2.8.6: 投机解码的扩展
  - §2.8.7: EOS 导致的额外 Forward
- 删除原 §11 (run_batch 详细流程) - FutureMap 概览已整合到 §2.8
- 删除原 §20 (FutureMap / resolve_future 机制) - 已整合到 §2.8
- 重新编号: §12-30 → §11-28 (最终 28 个章节)
