# v0.5.7 → v0.5.9 笔记更新清单

基座模型从 Qwen3-VL-235B-A22B-Thinking 更换为 Qwen3.5（混合架构）。

## CLAUDE.md

- [ ] 默认学习场景从 `Qwen/Qwen3-VL-235B-A22B-Thinking` 改为 `Qwen3.5`（混合架构：Full Attention + Linear Attention + MoE + MTP）
- [ ] 启用特性补充：EPLB、MTP、线性注意力、Mamba 缓存

## 00-index.md

- [ ] 版本号 v0.5.7 → v0.5.9
- [ ] 基座模型说明更新为 Qwen3.5
- [ ] 笔记目录表新增 `24-batch-overlap.md`（Phase 3 执行层，批处理重叠）
- [ ] 核心文件速查表行号全面更新
- [ ] 新增文件条目：`batch_overlap/`, `eplb/`, `elastic_ep/`, `dllm/`, `mem_cache/swa_memory_pool.py`, `mem_cache/swa_radix_cache.py`, `disaggregation/kv_events.py`

## 01-architecture.md [B级]

- [ ] 基座模型上下文从 Qwen3-VL-235B 替换为 Qwen3.5（混合架构）
- [ ] Scheduler mixin 继承链更新：新增 `SchedulerRuntimeCheckerMixin`、`SchedulerRecvSkipper`；删除 `scheduler_enhancer.py` 相关描述
- [ ] 新增小节「Batch Overlap 调度模式」：SBO 和 TBO 概念，与 `event_loop_overlap` 的关系
- [ ] 新增 Anthropic API 入口：`entrypoints/anthropic/`
- [ ] 新增 PrefillDelayer 组件简介（DP Attention 场景下的 prefill 延迟协商）
- [ ] 目录结构图更新：新增 `batch_overlap/`, `dllm/`, `elastic_ep/`, `eplb/` 等目录

## 02-core-data-structures.md [B级]

- [ ] Qwen3.5 模型上下文替换
- [ ] `ScheduleBatch` 字段更新：检查 SWA 相关字段、batch_overlap 相关字段
- [ ] `ForwardBatch` 字段更新
- [ ] 新增 `ForwardBatchDeepSeekMHAMixin`
- [ ] 新增 `input_buffers.py` 说明
- [ ] `ForwardMode` 枚举检查：是否有新增值
- [ ] 行号全面校准

## 03-scheduler.md [S级]

- [ ] Qwen3.5 模型上下文替换
- [ ] Scheduler 类继承结构重写：新增 `SchedulerRuntimeCheckerMixin`，删除 `scheduler_enhancer.py` 相关内容
- [ ] 新增章节「PrefillDelayer」
- [ ] 新增章节「SchedulerRecvSkipper」
- [ ] 新增章节「SchedulerRuntimeCheckerMixin」
- [ ] 新增章节「SchedulerInputBlocker」
- [ ] 事件循环与 `batch_overlap/` 的集成点更新
- [ ] 行号全面校准

## 04-schedule-policy.md [B级]

- [ ] Qwen3.5 上下文替换
- [ ] PrefillAdder 变化检查
- [ ] PrefillDelayer 与调度策略的协作关系
- [ ] SWA 模型的调度策略差异

## 05-chunked-prefill.md [B级]

- [ ] Qwen3.5 上下文替换
- [ ] PrefillDelayer 与 Chunked Prefill 的交互
- [ ] SWA 模型的分块策略特殊处理
- [ ] 行号更新

## 06-memory-pool.md [A级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增章节「SWA 内存池」
- [ ] 新增章节「Sparsity 支持」
- [ ] `memory_pool.py` 更新
- [ ] `memory_pool_host.py` 更新
- [ ] Mamba/Linear Attention 状态管理
- [ ] KV Cache 卸载

## 07-radix-cache.md [B级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增章节「SWA RadixCache」
- [ ] `BasePrefixCache` 接口重构
- [ ] RadixCache 重构
- [ ] 存储后端更新
- [ ] Mamba RadixCache 更新

## 08-model-runner.md [A级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增章节「Batch Overlap 集成」
- [ ] 新增章节「model_runner_kv_cache_mixin.py」
- [ ] `model_runner.py` 更新
- [ ] `input_buffers.py` 说明
- [ ] `hook_manager.py` 说明
- [ ] `cpu_graph_runner.py` 说明
- [ ] `piecewise_cuda_graph_runner.py` 说明
- [ ] 行号全面更新

## 09-attention-backends.md [S级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增章节「线性注意力后端」
- [ ] 新增章节「FLA (Flash Linear Attention)」
- [ ] 新增章节「Wave 后端」
- [ ] 新增章节「TRT-LLM MHA 后端」
- [ ] 新增章节「TRT-LLM MLA 后端」
- [ ] 新增章节「CUTLASS MLA 后端」
- [ ] 新增章节「TBO 后端」
- [ ] 新增章节「混合线性注意力后端」
- [ ] 新增章节「Torch Flex 后端」
- [ ] NSA 后端扩展
- [ ] `attention_registry.py` 更新
- [ ] 概览图重绘
- [ ] `RadixLinearAttention` 说明

## 10-model-loading.md [C级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增 `models/deepseek_common/` 说明
- [ ] 更新支持模型列表
- [ ] `checkpoint_engine/` 目录说明

## 11-multimodal.md [C级]

- [ ] Qwen3.5 上下文替换
- [ ] `multimodal_processor.py` 新增处理器
- [ ] `async_mm_data_processor.py` 更新
- [ ] `mm_utils.py` 更新

## 12-speculative-decoding.md [B级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增章节「Standalone Worker V2」
- [ ] Multi-Layer EAGLE 更新
- [ ] `eagle_info_v2.py` 说明
- [ ] `draft_utils.py` 说明

## 13-parallel-strategies.md [A级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增章节「EPLB」
- [ ] 新增章节「Elastic EP」
- [ ] 新增章节「DLLM」
- [ ] EP 部分扩展
- [ ] Context Parallel 重构变化

## 14-pd-disaggregation.md [S级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增章节「KV 事件管理」
- [ ] 新增章节「KV Cache 卸载」
- [ ] 新增章节「多硬件后端」
- [ ] 新增章节「Encode Server/Receiver」
- [ ] `decode.py` 大幅扩展
- [ ] `prefill.py` 更新
- [ ] 核心文件表全面更新

## 15-sgl-kernel-overview.md [B级]

- [ ] 新增章节「多平台支持」
- [ ] 删除 AOT Marlin kernels 说明
- [ ] 新增 `expert_specialization/` 目录
- [ ] 目录结构图更新

## 16-attention-kernels.md [B级]

- [ ] CUTLASS MLA kernel
- [ ] Wave Attention kernel
- [ ] NSA kernel 扩展
- [ ] CPU FlashAttention

## 17-moe-kernels.md [A级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增章节「Token Dispatcher 多后端」
- [ ] 新增章节「MoE Runner」
- [ ] `cutlass_moe.py` 更新
- [ ] 新增 `cutlass_w4a8_moe.py`
- [ ] 新增 `flashinfer_cutedsl_moe.py`
- [ ] 新增 `kt_ep_wrapper.py`
- [ ] 新增 `routed_experts_capturer.py`

## 18-quantization.md [A级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增「Quark 量化」
- [ ] 新增「Petit 量化」
- [ ] 新增「MXFP4 量化」
- [ ] 新增「FP4 量化」
- [ ] 新增「W4AFP8 量化」
- [ ] Compressed Tensors MoE 重构
- [ ] ModelSlim MoE 重构
- [ ] 量化格式总览表更新

## 19-sampling-and-generation.md [C级]

- [ ] Qwen3.5 上下文替换
- [ ] 确定性采样哈希函数改进
- [ ] `custom_logit_processor.py` 更新

## 20-constrained-generation.md [B级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增「Reasoner Grammar Backend」
- [ ] Grammar 后端列表更新

## 21-reasoning-and-function-call.md [A级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增 25+ 函数调用检测器
- [ ] `function_call_parser.py` 更新
- [ ] `core_types.py` 更新

## 22-embedding-and-rerank.md [C级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增 `sparse_pooler.py` 说明
- [ ] 检查 Embedding/Rerank 端点变化

## 23-lora.md [A级]

- [ ] Qwen3.5 上下文替换
- [ ] 新增「LoRA Overlap Loader」
- [ ] 新增后端扩展
- [ ] `lora_manager.py` 更新
- [ ] `lora_registry.py` 更新

## 24-batch-overlap.md [新增]

- [ ] 创建全新笔记
- [ ] SBO (Single Batch Overlap)
- [ ] TBO (Two Batch Overlap)
- [ ] Operations 与 Operations Strategy
- [ ] 与 Scheduler、ModelRunner、TBO Backend 的协作
- [ ] 适用场景：MoE 模型的计算-通信重叠优化
