# vLLM 推理框架深度学习笔记

> **学习目标**: 以 Qwen3-VL 235B 多模态模型 + PD分离 + ChunkPrefill 和 Qwen3-Next (混合DeltaNet) 为例，深入探究 vLLM 的核心设计
>
> **代码版本**: vLLM v1 架构 (基于 `vllm/v1/` 目录)
>
> **学习方式**: 源码阅读为主，深入到 CUDA Kernel 级别

---

## 目录结构

```
notes/
├── 00-index.md                    # 本文件 - 总览和学习路线图
│
├── Part 1: 基础架构
│   ├── 01-architecture-overview.md    # vLLM v1 整体架构
│   ├── 02-core-data-structures.md     # 核心数据结构详解
│   ├── 03-scheduler-design.md         # 调度器核心设计
│   └── 04-request-lifecycle.md        # 请求生命周期
│
├── Part 2: 内存管理
│   ├── 05-kv-cache.md                 # KV Cache 管理机制
│   ├── 06-block-manager.md            # Block 分配与回收
│   └── 07-paged-attention.md          # PagedAttention 实现
│
├── Part 3: 执行优化
│   ├── 08-model-runner.md             # ModelRunner 核心架构
│   ├── 09-cuda-graph.md               # CUDA Graph 优化机制
│   ├── 10-piecewise-cudagraph.md      # Piecewise CUDA Graph 详解
│   ├── 11-torch-compile.md            # torch.compile 集成与优化
│   ├── 12-chunked-prefill.md          # 分块预填充
│   ├── 13-pd-disaggregation.md        # Prefill-Decode 分离
│   └── 14-kv-transfer.md              # KV 传输机制 (NIXL/P2P)
│
├── Part 4: Attention 机制
│   ├── 15-attention-backend.md        # Attention Backend 架构
│   ├── 16-flash-attention.md          # FlashAttention 实现
│   ├── 17-mla-attention.md            # MLA (DeepSeek) 实现
│   └── 18-gdn-attention.md            # GDN/DeltaNet 实现
│
├── Part 5: 多模态
│   ├── 19-multimodal-overview.md      # 多模态架构总览
│   ├── 20-qwen3vl-model.md            # Qwen3-VL 模型解析
│   ├── 21-vit-implementation.md       # ViT 实现与优化
│   └── 22-mm-processing.md            # 多模态处理流程
│
├── Part 6: 混合模型
│   ├── 23-hybrid-model-arch.md        # 混合模型架构设计
│   ├── 24-qwen3next-model.md          # Qwen3-Next/DeltaNet 解析
│   ├── 25-mamba-integration.md        # Mamba 状态管理
│   └── 26-fla-ops.md                  # Flash Linear Attention Ops
│
├── Part 7: 投机采样
│   ├── 27-spec-decode-overview.md     # 投机采样总览
│   ├── 28-eagle3.md                   # Eagle3 实现
│   └── 29-mtp.md                      # Multi-Token Prediction
│
├── Part 8: 量化
│   ├── 30-quantization-overview.md    # 量化框架设计
│   ├── 31-fp8-quantization.md         # FP8 量化实现
│   ├── 32-awq-gptq.md                 # AWQ/GPTQ 实现
│   └── 33-marlin-kernel.md            # Marlin Kernel 优化
│
├── Part 9: CUDA Kernels
│   ├── 34-kernel-overview.md          # Kernel 架构总览
│   ├── 35-paged-attn-kernel.md        # PagedAttention Kernel
│   ├── 36-activation-kernels.md       # 激活函数 Kernel
│   ├── 37-moe-kernels.md              # MoE Kernels
│   └── 38-rope-kernels.md             # RoPE 位置编码 Kernel
│
├── Part 10: 特殊模型支持
│   ├── 39-embedding-model.md          # Embedding 模型支持
│   ├── 40-reranker-model.md           # Reranker/Cross-Encoder 支持
│   └── 41-reward-model.md             # Reward Model 支持
│
└── Part 11: 输出解析与工具调用
    ├── 42-structured-output.md        # 结构化输出 (JSON Schema)
    ├── 43-tool-call-parsing.md        # Tool Call 解析机制
    └── 44-reasoning-parsing.md        # Reasoning/思维链解析
```

---

## 学习路线图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Phase 1: 基础理解 (Week 1-2)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  架构总览 ──→ 核心数据结构 ──→ 调度器设计 ──→ KV Cache 机制                    │
│     ↓                                                                       │
│  理解 vLLM 的核心创新：分页 KV Cache + 连续批处理                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Phase 2: 执行优化 (Week 3-4)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ModelRunner ──→ CUDA Graph ──→ Piecewise ──→ torch.compile                 │
│       ↓                                                                     │
│  Chunked Prefill ──→ PD 分离 ──→ Attention Backend                          │
│       ↓                                                                     │
│  结合 Qwen3-VL 235B 理解长序列处理 + 分布式推理                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Phase 3: 多模态深入 (Week 5-6)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Qwen3-VL 模型 ──→ ViT 实现 ──→ Deepstack 融合 ──→ MRoPE                     │
│       ↓                                                                     │
│  理解视觉 Token 处理、多尺度特征提取、3D 位置编码                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Phase 4: 混合模型 (Week 7-8)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Qwen3-Next 架构 ──→ DeltaNet 原理 ──→ GDN Backend ──→ FLA Ops              │
│       ↓                                                                     │
│  理解 Linear Attention 与 Full Attention 的混合、状态管理                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Phase 5: 高级优化 (Week 9-10)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Eagle3 投机采样 ──→ MTP ──→ FP8 量化 ──→ Marlin Kernel                      │
│       ↓                                                                     │
│  理解推理加速技术、量化部署                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Phase 6: Kernel 深入 (Week 11-12)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  PagedAttention Kernel ──→ Activation Kernels ──→ MoE Kernels               │
│       ↓                                                                     │
│  C++/CUDA 实现细节、性能优化技巧                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块与关键文件映射

### Part 1-2: 基础架构与内存管理

| 模块 | 关键文件 | 重点内容 |
|------|---------|---------|
| **架构入口** | `vllm/v1/engine/core.py` | EngineCore 主循环 |
| **调度器** | `vllm/v1/core/sched/scheduler.py` | Scheduler 类, `schedule()` 方法 |
| **请求** | `vllm/v1/request.py` | Request, RequestStatus 枚举 |
| **KV Cache** | `vllm/v1/core/kv_cache_manager.py` | KVCacheManager, 块分配 |
| **Block Pool** | `vllm/v1/core/block_pool.py` | BlockPool, prefix caching |
| **调度输出** | `vllm/v1/core/sched/output.py` | SchedulerOutput, NewRequestData |

### Part 3: 执行优化

| 模块 | 关键文件 | 重点内容 |
|------|---------|---------|
| **ModelRunner** | `vllm/v1/worker/gpu_model_runner.py` | GPUModelRunner, `execute_model()` |
| **CUDA Graph** | `vllm/compilation/cuda_graph.py` | CUDAGraphWrapper, CUDAGraphEntry |
| **Dispatcher** | `vllm/v1/cudagraph_dispatcher.py` | CudagraphDispatcher, `dispatch()` |
| **torch.compile** | `vllm/compilation/backends.py` | VllmBackend, `split_graph()` |
| **分片后端** | `vllm/compilation/piecewise_backend.py` | PiecewiseBackend |
| **编译配置** | `vllm/config/compilation.py` | CompilationConfig, CUDAGraphMode |
| **PD 分离** | `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` | NixlConnector |

### Part 4: Attention 机制

| 模块 | 关键文件 | 重点内容 |
|------|---------|---------|
| **Backend 基类** | `vllm/v1/attention/backend.py` | AttentionBackend 抽象类 |
| **FlashAttention** | `vllm/v1/attention/backends/flash_attn.py` | FlashAttentionImpl |
| **MLA** | `vllm/v1/attention/backends/mla/flashmla.py` | FlashMLABackend |
| **GDN** | `vllm/v1/attention/backends/gdn_attn.py` | GDNAttentionBackend |
| **注册表** | `vllm/v1/attention/backends/registry.py` | AttentionBackendEnum |

### Part 5: 多模态

| 模块 | 关键文件 | 重点内容 |
|------|---------|---------|
| **多模态核心** | `vllm/multimodal/` | 图像/视频/音频处理 |
| **Qwen3-VL** | `vllm/model_executor/models/qwen3_vl.py` | Qwen3_VisionTransformer |
| **ViT Attention** | 同上 | Qwen2_5_VisionAttention |
| **Deepstack** | 同上 | Qwen3_VisionPatchMerger |
| **处理器** | 同上 | Qwen3VLMultiModalProcessor |

### Part 6: 混合模型

| 模块 | 关键文件 | 重点内容 |
|------|---------|---------|
| **Qwen3-Next** | `vllm/model_executor/models/qwen3_next.py` | Qwen3NextGatedDeltaNet |
| **MTP** | `vllm/model_executor/models/qwen3_next_mtp.py` | Qwen3NextMTP |
| **混合接口** | `vllm/model_executor/models/interfaces.py` | IsHybrid Protocol |
| **FLA Ops** | `vllm/model_executor/layers/fla/ops/` | chunk_gated_delta_rule |
| **Mamba 工具** | `vllm/model_executor/layers/mamba/mamba_utils.py` | 状态管理 |

### Part 7: 投机采样

| 模块 | 关键文件 | 重点内容 |
|------|---------|---------|
| **投机核心** | `vllm/v1/spec_decode/eagle.py` | EAGLE 投机解码 |
| **Eagle3** | `vllm/model_executor/models/llama_eagle3.py` | Eagle3 模型 |
| **MTP** | `vllm/v1/spec_decode/` | Multi-Token Prediction |
| **拒绝采样** | `vllm/v1/worker/gpu/spec_decode/rejection_sample.py` | 验证采样 |

### Part 8: 量化

| 模块 | 关键文件 | 重点内容 |
|------|---------|---------|
| **量化注册** | `vllm/model_executor/layers/quantization/__init__.py` | 方法注册 |
| **FP8** | `vllm/model_executor/layers/quantization/fp8.py` | Fp8Config |
| **AWQ** | `vllm/model_executor/layers/quantization/awq.py` | AWQConfig |
| **GPTQ** | `vllm/model_executor/layers/quantization/gptq_marlin.py` | Marlin 优化 |
| **Marlin** | `vllm/model_executor/layers/quantization/utils/marlin_utils.py` | 工具函数 |

### Part 9: CUDA Kernels

| 模块 | 关键文件 | 重点内容 |
|------|---------|---------|
| **Attention** | `csrc/attention/attention_kernels.cuh` | paged_attention_kernel |
| **MoE** | `csrc/moe/topk_softmax_kernels.cu` | TopK + Softmax |
| **激活** | `csrc/activation_kernels.cu` | SiLU, GELU |
| **LayerNorm** | `csrc/layernorm_kernels.cu` | RMSNorm |
| **RoPE** | `csrc/pos_encoding_kernels.cu` | 位置编码 |
| **量化** | `csrc/quantization/marlin/marlin.cu` | Marlin kernel |

### Part 10-11: 特殊模型与工具调用

| 模块 | 关键文件 | 重点内容 |
|------|---------|---------|
| **Embedding** | `vllm/model_executor/models/` | 嵌入模型支持 |
| **结构化输出** | `vllm/v1/engine/` | JSON Schema 解析 |
| **Tool Call** | `vllm/entrypoints/` | 工具调用处理 |

---

## 代码库目录结构概览

```
vllm/
├── v1/                           # V1 架构 (新版本) ★ 重点学习
│   ├── core/
│   │   ├── sched/               # 调度器
│   │   ├── kv_cache_manager.py  # KV Cache 管理
│   │   └── block_pool.py        # Block 池
│   ├── attention/
│   │   ├── backends/            # Attention 后端实现
│   │   └── ops/                 # Attention 操作
│   ├── worker/
│   │   ├── gpu_model_runner.py  # GPU ModelRunner
│   │   └── gpu/                 # GPU Worker 组件
│   ├── spec_decode/             # 投机解码
│   └── engine/                  # 引擎核心
│
├── model_executor/
│   ├── models/                  # 200+ 模型实现 ★
│   │   ├── qwen3_vl.py         # Qwen3-VL
│   │   ├── qwen3_next.py       # Qwen3-Next
│   │   └── ...
│   └── layers/
│       ├── quantization/        # 量化层
│       ├── fused_moe/          # MoE 实现
│       └── fla/                # FLA Ops
│
├── compilation/                  # 编译优化 ★
│   ├── cuda_graph.py           # CUDA Graph
│   ├── backends.py             # 编译后端
│   └── decorators.py           # @support_torch_compile
│
├── distributed/
│   └── kv_transfer/            # PD 分离 KV 传输
│
├── multimodal/                  # 多模态处理
│
├── attention/                   # Attention 层抽象
│
└── config/                      # 配置类定义

csrc/                            # C++/CUDA Kernels ★
├── attention/                   # Attention kernels
├── moe/                        # MoE kernels
├── quantization/               # 量化 kernels
│   └── marlin/                # Marlin 实现
├── activation_kernels.cu
├── layernorm_kernels.cu
├── pos_encoding_kernels.cu
└── sampler.cu
```

---

## 关键概念速查

### 调度相关

| 概念 | 说明 |
|------|------|
| **Continuous Batching** | 连续批处理，动态添加/移除请求 |
| **Chunked Prefill** | 将长 prefill 分块处理，避免阻塞 decode |
| **Prefix Caching** | 缓存公共前缀的 KV，减少重复计算 |
| **Preemption** | 内存不足时抢占低优先级请求 |

### 内存管理

| 概念 | 说明 |
|------|------|
| **PagedAttention** | 分页 KV Cache，类似 OS 虚拟内存 |
| **Block** | KV Cache 的基本分配单位 |
| **Block Table** | 逻辑块到物理块的映射 |
| **Copy-on-Write** | 分支序列共享 KV，写时复制 |

### 执行优化

| 概念 | 说明 |
|------|------|
| **CUDA Graph** | 捕获 kernel 调用序列，减少 launch 开销 |
| **Piecewise CUDA Graph** | 分片 cudagraph，attention 在 graph 外 |
| **torch.compile** | PyTorch 2.0 编译优化 |
| **Inductor** | torch.compile 的默认后端 |

### Attention

| 概念 | 说明 |
|------|------|
| **FlashAttention** | 内存高效的精确 attention |
| **MLA** | Multi-head Latent Attention (DeepSeek) |
| **GDN** | Gated DeltaNet (Qwen3-Next) |
| **MRoPE** | Multi-head Rotary Position Embedding |

### 投机采样

| 概念 | 说明 |
|------|------|
| **Draft Model** | 小模型快速生成候选 token |
| **Target Model** | 大模型验证候选 token |
| **EAGLE** | 无需 draft model 的投机方法 |
| **MTP** | Multi-Token Prediction |

---

## 学习建议

1. **先整体后细节**: 先理解 V1 架构的整体流程，再深入各模块
2. **以模型为线索**: 以 Qwen3-VL 和 Qwen3-Next 为主线贯穿学习
3. **代码与文档结合**: 结合 `docs/` 目录的设计文档理解代码
4. **动手调试**: 使用 debugger 跟踪请求的完整生命周期
5. **画图辅助**: 对复杂流程用 Mermaid 图梳理

---

## 参考资源

- **vLLM 官方文档**: https://docs.vllm.ai/
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **PagedAttention 论文**: https://arxiv.org/abs/2309.06180
- **FlashAttention 论文**: https://arxiv.org/abs/2205.14135
- **EAGLE 论文**: https://arxiv.org/abs/2401.15077

---

> **下一步**: [01-architecture-overview.md](./01-architecture-overview.md) - vLLM V1 整体架构
