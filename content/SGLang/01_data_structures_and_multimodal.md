# SGLang 数据结构与多模态处理笔记

> 基于 Qwen3-VL MoE 场景的代码分析

---

## 1. 请求数据结构变化流程

```
用户请求 (HTTP/API)
       ↓
┌─────────────────────────────────────────────────────────────────┐
│  GenerateReqInput (io_struct.py:165)                            │
│  • 面向 API 的原始输入格式                                        │
│  • 支持 text/input_ids/input_embeds + image/video/audio_data    │
│  • 支持 batch 和 parallel sampling                               │
└─────────────────────────────────────────────────────────────────┘
       ↓  TokenizerManager 处理
┌─────────────────────────────────────────────────────────────────┐
│  TokenizedGenerateReqInput (io_struct.py:678)                   │
│  • 单条请求，已分词                                               │
│  • input_ids: List[int]                                         │
│  • mm_inputs: dict (多模态数据已处理)                             │
│  • sampling_params: SamplingParams (已解析)                      │
└─────────────────────────────────────────────────────────────────┘
       ↓  Scheduler 接收
┌─────────────────────────────────────────────────────────────────┐
│  Req (schedule_batch.py:484)                                    │
│  • 调度器内部的请求表示                                           │
│  • 包含运行时状态：KV cache、prefix indices、output_ids           │
│  • multimodal_inputs: MultimodalInputs                          │
└─────────────────────────────────────────────────────────────────┘
       ↓  组成批次
┌─────────────────────────────────────────────────────────────────┐
│  ScheduleBatch / ModelWorkerBatch / ForwardBatch                │
│  • GPU 端的张量表示                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么不用一个类一路到底？

| 原因 | 说明 |
|------|------|
| **职责边界清晰** | API 层不需要知道 GPU tensor，调度层不需要知道 HTTP stream |
| **跨进程通信效率** | 只传输必要字段，减少序列化开销 |
| **渐进式处理** | 每层都可以添加处理后的新信息 |
| **可测试性** | 可以独立测试每一层的逻辑 |

---

## 2. Mixin 设计模式

### 核心思想

```python
# 传统继承：Dog is-a Animal
# Mixin 组合：Dog has-ability-of Swimming, Running

class Scheduler(
    SchedulerOutputProcessorMixin,    # 输出处理能力
    SchedulerMetricsMixin,            # 指标收集能力
    SchedulerPPMixin,                 # Pipeline Parallel 能力
    ...
):
```

### 好处

1. **代码模块化** - 避免「上帝类」，每个能力独立一个文件
2. **可选组合** - 按需启用功能
3. **单一职责** - 每个 Mixin 只做一件事
4. **易于测试** - 可以单独测试每个 Mixin

---

## 3. Qwen3-VL 多模态处理流程

### 完整调用链

```
TokenizerManager._tokenize_one_request()  (tokenizer_manager.py:637)
       │
       ├─► self.mm_data_processor.process()  (AsyncMMDataProcessor)
       │       │
       │       └─► QwenVLImageProcessor.process_mm_data_async()
       │
       └─► self._create_tokenized_object(mm_inputs=...)
              └─► TokenizedGenerateReqInput
```

### 详细流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ① load_mm_data() (base_processor.py:561)                               │
│     ├─► submit_data_loading_tasks() - 提交并行下载任务                    │
│     │      └─► io_executor.submit(_load_single_item, ...)               │
│     │              ↓                                                     │
│     │     ┌─────────────────────────────────────────────────────────┐   │
│     │     │ _load_single_item() (base_processor.py:381)             │   │
│     │     │   └─► load_image()  (utils/common.py:843)               │   │
│     │     │         • HTTP GET 下载图片 (同步 requests.get)          │   │
│     │     │         • base64/本地文件读取                            │   │
│     │     │         • 返回 PIL.Image                                │   │
│     │     └─────────────────────────────────────────────────────────┘   │
│     └─► 返回 BaseMultiModalProcessorOutput(images=[PIL.Image, ...])     │
└─────────────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  ② process_and_combine_mm_data() (base_processor.py:765)                │
│     └─► process_mm_data() → HuggingFace Processor 调用                   │
│           • 如果有 BaseImageProcessorFast，可在 GPU 上运行               │
│           • 返回 pixel_values tensor, image_grid_thw 等                 │
│     └─► collect_mm_items_from_processor_output()                         │
│           → List[MultimodalDataItem]                                     │
└─────────────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  ③ Qwen 特有：计算 mrope_positions (qwen_vl.py:374)                     │
│     └─► MRotaryEmbedding.get_rope_index(                                │
│            spatial_merge_size, image_token_id, input_ids, image_grid_thw│
│         )                                                               │
│     → 返回 mrope_positions, mrope_position_delta                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 返回的 mm_inputs 结构

```python
{
    "input_ids": [token_ids...],
    "mm_items": [MultimodalDataItem(
                   modality=Modality.IMAGE,
                   feature=pixel_values,  # ← 图片 tensor 在这里！
                   offsets=[(start, end),...],
                 )],
    "mrope_positions": tensor,       # Qwen 专属
    "mrope_position_delta": tensor,  # Qwen 专属
    "im_token_id": 151655,
    ...
}
```

---

## 4. Q&A 汇总

| 问题 | 答案 |
|------|------|
| **图片下载在哪里？** | `load_image()` in `utils/common.py:843` |
| **下载是同步还是异步？** | 单张同步，多张通过 **ThreadPoolExecutor 并行** |
| **数据如何转 tensor？** | HuggingFace `_processor()` 调用 |
| **HF processor 是 CPU 还是 GPU？** | 默认 CPU，有 `BaseImageProcessorFast` 可用 CUDA |
| **mrope 何时生成？** | `QwenVLImageProcessor.process_mm_data_async()` 末尾 |
| **图片数据最终在哪里？** | `TokenizedGenerateReqInput.mm_inputs["mm_items"][0].feature` |
| **基类提供下载吗？** | 是，`BaseMultimodalProcessor._load_single_item()` |
| **所有多模态类都要定义接口吗？** | 必须实现 `process_mm_data_async()` 或 `process_mm_data()` |

---

## 5. 类继承关系

```
BaseMultimodalProcessor (base_processor.py)
├── load_mm_data()                  # 下载图片
├── _load_single_item()             # 单张加载 (静态方法)
├── process_mm_data()               # 调用 HF processor
└── process_and_combine_mm_data()   # 组合处理

     ↓ 继承
QwenVLImageProcessor (qwen_vl.py)
├── models = [Qwen2VL, Qwen2.5VL, Qwen3VL, Qwen3VLMoe, ...]
├── __init__(): 设置 mm_tokens, vision_start_token_id 等
└── process_mm_data_async(): 
    ├── 调用父类方法
    └── 调用 MRotaryEmbedding.get_rope_index()  ← Qwen 专属
```

---

## 6. 性能关键点

| 环节 | 机制 |
|------|------|
| **图片下载** | ThreadPoolExecutor 并行（默认 4 线程，`SGLANG_IO_WORKERS`） |
| **CPU 密集处理** | ProcessPoolExecutor 可用（`SGLANG_CPU_WORKERS`） |
| **HF Processor** | 可选 CUDA 加速（需要 `BaseImageProcessorFast`） |
| **Tensor 存储** | 默认移到 CPU，除非设置 `keep_mm_feature_on_device` |
