## Review

## 10_multimodal (已修正 2026-02-18)

以下问题已在本次修正中解决：

1. [x] **P2-7**: 区分两个同名 `mm_utils.py` 文件 — 在 §1 添加提示框，全文引用加路径前缀
2. [x] **P0-1**: 新增 §4A「跨进程多模态数据传输」— 四阶段传输表、broadcast_pyobj 实现、prepare_for_extend 双路径、Mermaid 时序图、三种传输优化
3. [x] **P1-2**: 修正 §4.3 DP Scheduling vs ViT DP 概念混淆 — 添加区分框、修正 Mermaid 图去掉 DataParallelController
4. [x] **P1-3**: 修正 §4.3.3 "输入分发"为"本地切片" — 说明不存在 NCCL scatter
5. [x] **P1-4**: 补充 §9.3.1a ViT DP 权重完整复制说明 — tp_size=1 机制、显存权衡分析
6. [x] **P1-5**: 修正 §6.1 融合流程 — 替换为 SGLang 实际调用链 (general_mm_embed_routine → embed_mm_inputs)
7. [x] **P1-6**: 修正 §9.3.2 负载均衡算法描述 — image_to_tp_rank 改为 shuffle_indices，添加切片示例
8. [x] **P2-8**: 补充 §6.3 Deepstack 机制 — ViT 中间层捕获、separate_deepstack_embeds、decoder 注入

## 09_model_loading

1. 概述里面的图画的逻辑有问题把，是不是runner.init后不是分支的逻辑，而是顺序逻辑
2. 核心类层次这部分写的不太好
3. 调用链里面的启动路径跟1 概述部分是不是重合了部分，这部分启动路径这写的也不太好
