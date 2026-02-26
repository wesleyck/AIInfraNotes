# 11. 多模态处理 (Multimodal Processing)

## 本章定位
- 主题范围: 多模态预处理与执行交互。

## 设计 Why（为什么这么设计）
- 多模态把异构输入映射为统一执行表示，复杂度集中在入口。
- 核心取舍: 吞吐 vs 时延、显存 vs 计算、通用性 vs 特化。

## 阅读建议（进阶）
1. 先抓目标函数和边界条件，再读具体实现。
2. 先看调用链和状态变化，再看局部优化细节。

## 1. 概述

SGLang 的多模态系统处理图像、视频、音频等非文本输入，支持 Qwen3.5、LLaVA、InternVL 等模型。

> **⚠ 同名文件区分**：本文涉及两个 `mm_utils.py`，功能完全不同：
>
> | 路径 | 职责 | 核心函数 |
> |------|------|---------|
> | `python/sglang/srt/multimodal/mm_utils.py` | ViT DP 并行编码 | `get_dp_encoder_lb_assignment()`, `run_dp_sharded_mrope_vision_model()` |
> | `python/sglang/srt/managers/mm_utils.py` | 嵌入融合、缓存 | `embed_mm_inputs()`, `get_embedding_and_mask()`, `general_mm_embed_routine()` |
>
> 后文引用时均使用 **完整路径前缀** 以避免歧义。

```mermaid
flowchart TB
    subgraph Lifecycle["多模态完整生命周期"]
        direction TB
        S1["1: 请求接收<br/>HTTP/API 解析多模态输入"]
        S2["2: 预处理 + Tokenize<br/>图像加载 → resize → HF processor → tokenize"]
        S3["3: Scheduler 调度<br/>构建 ScheduleBatch, H2D 搬运"]
        S4["4: VIT 编码<br/>像素值 → Vision Encoder → image_embeddings"]
        S5["5: 嵌入融合<br/>text_embeddings 中替换为 image_embeddings"]
        S6["6: LLM 推理<br/>Prefill / Decode"]

        S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end
```

**各阶段详细说明**：

| 阶段 | 进程 | 核心操作 | 关键函数 |
|------|------|---------|---------|
| 1. 请求接收 | FastAPI | 解析 `images`（base64/URL）和 `text`（含 `<image>` placeholder） | `openai_v1_chat_completions()` |
| 2. 预处理 + Tokenize | TokenizerManager | `AsyncMMDataProcessor.process()` → 加载图像/视频/音频 → HF processor（resize + normalize + to_tensor）→ 展开 placeholder tokens → 计算 MRoPE 位置编码 → 生成 `MultimodalDataItem` + hash | `process_mm_data_async()` |
| 3. Scheduler 调度 | Scheduler | 创建 `ScheduleBatch`，`prepare_for_extend()` 将 pixel_values 从 CPU 搬到 GPU | `get_next_batch_to_run()` |
| 4. VIT 编码 | ModelRunner | 像素值 → patch_embedding + position_embedding → Transformer Blocks（可用 CUDA Graph）→ 输出 image_embeddings | `vision_model.forward()` |
| 5. 嵌入融合 | ModelRunner | `embed_tokens(input_ids)` 生成 text_embeddings，在 image_offsets 位置替换为 image_embeddings | `embed_mm_inputs()` |
| 6. LLM 推理 | ModelRunner | 使用融合后的 embeddings 进行 Prefill / Decode | `language_model.forward()` |

## 2. 核心类层次

### 2.1 BaseMultimodalProcessor

`BaseMultimodalProcessor` 是所有多模态 processor 的抽象基类，定义了从原始输入到 `MultimodalDataItem` 的标准处理流程。它在 `TokenizerManager` 进程中运行，是多模态数据进入推理引擎的第一个入口。

**设计要点**：
- **双 executor 并行**：I/O 密集任务（图片下载/URL 加载）使用 `ThreadPoolExecutor`，CPU 密集任务（图片 resize/预处理）使用 `ProcessPoolExecutor`。分离的原因是 I/O 任务受网络延迟限制，线程即可；而 CPU 预处理（如大图 resize）会阻塞 GIL，必须用多进程
- **`ATTR_NAME_TO_MODALITY` 路由**：HuggingFace processor 输出的属性名（如 `pixel_values`、`audio_features`）是扁平的字典。此映射表将属性名路由到对应的 `Modality` 枚举，使 `process_and_combine_mm_data()` 能将 processor 输出拆分到正确的 `MultimodalDataItem` 中
- **`FEATURE_NAMES`**：标识哪些属性是"主特征"（需要 GPU 计算的大张量），与 `model_specific_data` 中的元数据属性区分

```python
# python/sglang/srt/multimodal/processors/base_processor.py
class BaseMultimodalProcessor(ABC):
    """Base class for all multimodal processors."""

    models = []  # 支持的模型类列表

    def __init__(self, hf_config, server_args, _processor, transport_mode, *args, **kwargs):
        self.hf_config = hf_config
        self._processor = _processor  # HuggingFace AutoProcessor
        self.server_args = server_args
        self.transport_mode = transport_mode

        # 每帧估算 token 数 (粗略值，实际因模型和图像而异)
        self.NUM_TOKEN_PER_FRAME = 330

        # 双 executor 并行处理
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_IO_WORKERS", 4))
        )  # ThreadPool: I/O 密集任务 (图片加载/URL 下载)
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("fork"),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )  # ProcessPool: CPU 密集任务 (图片预处理/resize)

        # 属性名 → 模态类型映射
        self.ATTR_NAME_TO_MODALITY = {
            "pixel_values": Modality.IMAGE,
            "image_grid_thw": Modality.IMAGE,
            "audio_features": Modality.AUDIO,
            "pixel_values_videos": Modality.VIDEO,
            ...  # 共 20+ 个属性映射
        }

        # 特征字段名列表
        self.FEATURE_NAMES = [
            "pixel_values",           # 图像像素值
            "pixel_values_videos",    # 视频像素值
            "audio_features",         # 音频特征
            "input_features",         # 通用输入特征
        ]

        # 条件初始化 CUDA IPC 内存池
        if SGL_USE_CUDA_IPC and not skip_mm_pool:
            self.cudaipc_mmfeature_pool = MmItemMemoryPool(...)
```

> `get_mm_data()` 不在基类中定义，而是由各子类实现（如 `QwenVLImageProcessor.get_mm_data()`）。

### 2.2 MultimodalSpecialTokens

`MultimodalSpecialTokens` 管理多模态输入中的特殊 token（如 `<image>`、`<video>`），负责将输入文本按模态 token 分割，以便后续将文本片段和多模态 placeholder 正确对齐。

**设计要点**：
- **为什么需要 regex 匹配**：不同模型的多模态 token 格式差异很大。Qwen 使用 `<|vision_start|>(<|image_pad|>)+<|vision_end|>` 这样的包裹序列，简单的字符串匹配无法处理展开后的变长 token 序列。regex 提供了统一的匹配能力，同时支持精确匹配（单 token）和模式匹配（变长序列）
- **`build()` 初始化链**：`build()` 按顺序执行三步初始化：`convert_to_strs()`（token id → 字符串）→ `parse_regex()`（构建单模态 pattern）→ `get_combined_regex()`（合并为分割用 combined pattern）。这个链式设计确保了 tokenizer 依赖在初始化时一次性解析完毕，运行时只需 regex 匹配

```python
# python/sglang/srt/multimodal/processors/base_processor.py
@dataclasses.dataclass
class MultimodalSpecialTokens:
    """Manages special tokens for multimodal inputs."""

    # 基础 token 字段
    image_token: Optional[Union[str, List[str]]] = None   # "<image>"
    video_token: Optional[Union[str, List[str]]] = None   # "<video>"
    audio_token: Optional[Union[str, List[str]]] = None   # "<audio>"
    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None

    # 正则匹配字段 (用于处理非标准 token 格式)
    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None
    combined_regex: Optional[re.Pattern] = None            # 合并正则，用于分割输入文本

    def build(self, processor) -> "MultimodalSpecialTokens":
        """初始化入口：convert_to_strs → parse_regex → get_combined_regex"""
        self.convert_to_strs(processor)    # token id → 字符串 (通过 tokenizer)
        self.parse_regex()                  # 构建单模态匹配 pattern
        self.get_combined_regex()           # 合并为分割用 combined pattern
        return self

    def get_modality_of_token(self, token: str) -> Optional[Modality]:
        """返回 token 对应的模态类型（先精确匹配，再 regex 匹配）"""
        ...

    def get_combined_regex(self) -> re.Pattern:
        """构建合并正则，用于将输入字符串按多模态 token 分割"""
        ...
```

### 2.3 QwenVLImageProcessor (Qwen3.5 专用)

`QwenVLImageProcessor` 是 Qwen 系列 VL 模型的专用 processor，覆盖 Qwen2-VL 到 Qwen3.5 全系列。它在 `BaseMultimodalProcessor` 基础上增加了 Qwen 特有的处理逻辑。

**设计要点**：
- **MRoPE 位置编码计算**：Qwen-VL 使用 3D RoPE（temporal × height × width），在预处理阶段就需要根据 `grid_thw` 计算每个 patch 的 3D 位置索引。这些位置信息通过 `mrope_positions` 传递给 Scheduler，最终在 attention 计算中使用
- **`image_token_regex`**：匹配展开后的完整 token 序列（`<|vision_start|>(<|image_pad|>)+<|vision_end|>`），而非单个 token。`build()` 方法会自动为未设置 regex 的模态生成默认 regex

```python
# python/sglang/srt/multimodal/processors/qwen_vl.py
class QwenVLImageProcessor(BaseMultimodalProcessor):
    """Compatible with Qwen-VL & Qwen-Omni Series."""

    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3OmniMoeForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(...)

        # Qwen 特有配置
        self.image_factor = IMAGE_FACTOR  # 28
        self.min_pixels = MIN_PIXELS      # 4 * 28 * 28
        self.max_pixels = MAX_PIXELS      # 16384 * 28 * 28

        # 特殊 token
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=hf_config.video_token_id,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    async def process_mm_data_async(self, image_data, input_text, request_obj, *args, **kwargs):
        # Step 1: 加载原始数据 (base class 方法)
        base_output = self.load_mm_data(
            prompt=input_text, image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        # Step 2: 视频预处理 (Qwen 特有的帧采样 + resize)
        if base_output.videos:
            videos_processed = [
                await preprocess_video(video, video_config=self.video_config)
                for video in base_output.videos
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        # Step 3: 调用 HF processor 并组合为 MultimodalDataItem 列表
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens, video_metadata=video_metadata, ...
        )

        # Step 4: 计算 MRoPE 位置编码 (3D: temporal × height × width)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(...)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "mrope_positions": mrope_positions,
            ...
        }
```

### 2.4 AsyncMMDataProcessor

`AsyncMMDataProcessor` 是 `TokenizerManager` 调用多模态处理的直接入口，作为异步包装器将底层 processor 的同步/异步接口统一为 async API。

**调用链**: `TokenizerManager._tokenize_one_request()` → `AsyncMMDataProcessor.process()` → `QwenVLImageProcessor.process_mm_data_async()`

```python
# python/sglang/srt/managers/async_mm_data_processor.py
class AsyncMMDataProcessor:
    """TokenizerManager 调用多模态处理的直接入口，异步包装器。"""

    def __init__(self, mm_processor, *, max_concurrent_calls=None, timeout_s=None):
        self.mm_processor = mm_processor
        self.semaphore = asyncio.Semaphore(max_concurrent_calls) if max_concurrent_calls else None
        self.timeout_s = timeout_s

        # 自动检测 async/sync 处理器
        self._proc_async = getattr(mm_processor, "process_mm_data_async", None)
        self.is_async = asyncio.iscoroutinefunction(self._proc_async)
        # sync 回退: 使用 ThreadPoolExecutor 避免阻塞事件循环
        self.fallback_exec = ThreadPoolExecutor(...) if not self.is_async else None

    async def process(self, *, image_data, audio_data, input_text_or_ids, request_obj, **kwargs):
        async def _invoke():
            if self.is_async:
                return await self._proc_async(...)  # 原生 async 路径
            return await loop.run_in_executor(self.fallback_exec, sync_fn)  # sync 回退

        # 可选: Semaphore 并发限制 + timeout
        if self.semaphore:
            async with self.semaphore:
                return await asyncio.wait_for(_invoke(), timeout=self.timeout_s)
        return await _invoke()
```

> **设计要点**:
> - Semaphore 控制并发数，防止大量多模态请求同时占用 CPU/GPU
> - timeout 机制防止单个请求的图像加载/预处理卡死整个 pipeline
> - 自动检测底层 processor 是否为 async，sync processor 自动包装到线程池

### 2.5 MultimodalDataItem

`MultimodalDataItem` 是单个模态输入的统一数据容器。例如一个请求包含 3 张图 + 1 段音频，会生成 2 个 `MultimodalDataItem`（一个 IMAGE，一个 AUDIO）。它贯穿从预处理到嵌入融合的整个生命周期。

**设计要点**：
- **`pad_value` 机制**：每个多模态 item 通过 `set_pad_value()` 基于特征内容的 hash 生成一个唯一整数值（>= 1,000,000），用于替换 `input_ids` 中对应的 placeholder tokens。在嵌入融合阶段，系统通过匹配这些 pad_value 来定位需要替换为视觉/音频嵌入的位置。`MM_PAD_SHIFT_VALUE = 1_000_000` 远大于任何模型的 vocab_size（通常 < 200,000），确保不与正常 token ID 冲突
- **`__getattr__` 透明访问**：模型特有的属性（如 Qwen 的 `image_grid_thw`、InternVL 的 `images_spatial_crop`）存储在 `model_specific_data` 字典中，但可以直接通过 `item.image_grid_thw` 访问，简化调用方代码

```python
# python/sglang/srt/managers/schedule_batch.py
@dataclasses.dataclass
class MultimodalDataItem:
    """单个模态的所有输入数据。"""

    modality: Modality
    hash: int = None          # 特征内容的 hash，用于缓存匹配
    pad_value: int = None     # 替换 input_ids 中 placeholder 的唯一值
    offsets: Optional[list] = None  # 每个 item 在 input_ids 中的 (start, end) 位置

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL

    feature: Union[torch.Tensor, np.ndarray] = None           # 原始特征 (pixel_values 等)
    precomputed_embeddings: Optional[torch.Tensor] = None      # 预计算嵌入 (二选一)

    model_specific_data: dict[str, Any] = field(default_factory=dict)  # 模型特有元数据

    def __getattr__(self, name):
        # 透明访问 model_specific_data，如 item.image_grid_thw
        if name in self.model_specific_data:
            return self.model_specific_data[name]
        raise AttributeError(...)

    def set_pad_value(self):
        """计算 hash → 生成唯一 pad_value，确保不与 vocab token 冲突。"""
        self.hash = hash_feature(self.feature or self.precomputed_embeddings)
        self.pad_value = MM_PAD_SHIFT_VALUE + (self.hash % (1 << 30))
```

## 3. 图片下载 + 预处理

多模态输入在进入推理引擎前需要经过三个阶段的处理，每个阶段使用不同的并行策略：

```
图片下载(URL→PIL) → 预处理(smart_resize/视频帧采样) → HF Processor(std/norm/resize/to_tensor)
```

| 阶段 | 执行器 | 类型 | 说明 |
|------|--------|------|------|
| 图片下载 | `io_executor` (ThreadPoolExecutor) | I/O 密集 | URL 下载、base64 解码、文件读取 |
| 预处理 | `cpu_executor` (ProcessPoolExecutor) | CPU 密集 | smart_resize、视频帧采样、尺寸调整 |
| HF Processor | 主要 CPU，部分 GPU | 混合 | 标准化(std/norm)、resize、to_tensor。Qwen 的 `process_vision_info` 中部分操作可能使用 GPU |

三个阶段的分工使得 I/O 等待、CPU 计算和 GPU 预处理可以流水线并行，避免单一瓶颈。

### 3.1 smart_resize (Qwen-VL)

```python
# python/sglang/srt/multimodal/processors/qwen_vl.py
def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,      # 28
    min_pixels: int = MIN_PIXELS,    # 4 * 28 * 28
    max_pixels: int = MAX_PIXELS,    # 16384 * 28 * 28
) -> Tuple[int, int]:
    """
    Rescales the image ensuring:
    1. Both dimensions are divisible by 'factor'
    2. Total pixels within [min_pixels, max_pixels]
    3. Aspect ratio maintained
    """

    # 调整到 factor 的倍数
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    # 检查像素限制
    if h_bar * w_bar > max_pixels:
        # 缩小
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        # 放大
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar
```

### 3.2 grid_thw 计算

Qwen-VL 使用 3D grid (temporal, height, width) 表示视觉 patches：

```python
def _compute_grid_thw(images):
    """计算每张图像的 grid 尺寸"""
    grid_thw = []
    for img in images:
        h, w = img.size
        # patches = (h // 28) * (w // 28)
        t = 1  # 图像: temporal=1, 视频: temporal=num_frames
        grid_h = h // IMAGE_FACTOR
        grid_w = w // IMAGE_FACTOR
        grid_thw.append((t, grid_h, grid_w))
    return grid_thw
```

### 3.3 视频处理

```python
# python/sglang/srt/multimodal/processors/qwen_vl.py
def preprocess_video(vr, image_factor=28, video_config={}):
    """Process video for Qwen-VL."""

    total_frames = len(vr)
    video_fps = vr.get_avg_fps()

    # 1. 计算采样帧数
    nframes = smart_nframes(video_config, total_frames, video_fps)

    # 2. 均匀采样
    frame_indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()

    # 3. 调整尺寸 (考虑总像素限制)
    h, w = frames.shape[1:3]
    new_h, new_w = smart_resize(
        h, w,
        factor=image_factor,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=VIDEO_MAX_PIXELS // nframes,  # 分摊到每帧
    )

    return frames, (nframes, new_h // image_factor, new_w // image_factor)
```

### 3.4 process_and_combine_mm_data() — 多模态数据处理核心入口

`BaseMultimodalProcessor.process_and_combine_mm_data()` 是预处理阶段的核心函数（`python/sglang/srt/multimodal/processors/base_processor.py` L974-1114），负责将原始多模态输入转换为统一的 `MultimodalDataItem` 列表。

```python
def process_and_combine_mm_data(
    self,
    base_output: BaseMultiModalProcessorOutput,
    mm_tokens: MultimodalSpecialTokens,
    **kwargs,
) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
```

处理流程分为 4 个阶段：

1. **按模态分类**：将输入分为 `raw_images`、`raw_videos`、`raw_audios` 和 `dict_items`（预计算 embedding 或 processor_output 格式）

2. **原始数据处理**：调用 `_process_and_collect_mm_items()` 执行 HuggingFace AutoProcessor，生成 `pixel_values`、`input_ids` 等

3. **字典格式处理**：处理两种特殊输入格式
   - `processor_output`：已经过 HF processor 的数据，直接收集
   - `precomputed_embedding`：预计算的 embedding，跳过 VIT 编码

4. **CUDA IPC 包装**：当 `SGL_USE_CUDA_IPC=1` 时，将 GPU tensor 包装为 `CudaIpcTensorTransportProxy`，实现跨进程零拷贝传输

```python
# python/sglang/srt/multimodal/processors/base_processor.py
if SGL_USE_CUDA_IPC:
    for item in all_collected_items:
        if isinstance(item.feature, torch.Tensor) and item.feature.is_cuda:
            sync_flag, available_slice = (
                self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(
                    item.feature
                )
            )
            if isinstance(available_slice, torch.Tensor):
                available_slice.copy_(item.feature.view(torch.int8).view(-1),
                                     non_blocking=True)
                item.feature = CudaIpcTensorTransportProxy(
                    data=available_slice,
                    info_data=item.feature,
                    sync_buffer_meta=sync_flag,
                )
```

> **设计要点**：CUDA IPC 使用预分配的内存池（`cudaipc_mmfeature_pool`）而非直接共享 tensor handle，避免了跨进程 GPU 内存泄漏问题。

## 4. 多模态缓存 (MultimodalCache)

### 4.1 缓存机制

```python
# python/sglang/srt/mem_cache/multimodal_cache.py
class MultimodalCache(ABC):
    """Abstract base for multimodal embedding cache."""

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        """Combine multiple item hashes into one."""
        if not mm_hashes:
            return None
        return hash(tuple(mm_hashes))

    @abstractmethod
    def get(self, mm_hashes: List[int], combined_hash=None) -> Optional[torch.Tensor]:
        """Get cached embedding by hash."""
        raise NotImplementedError

    @abstractmethod
    def set(self, mm_hash: int, embedding: torch.Tensor, allocator) -> bool:
        """Store embedding with hash."""
        raise NotImplementedError

class MultiModalStaticCache(MultimodalCache):
    """LRU cache for multimodal embeddings."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.mm_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.current_size = 0

    def set(self, mm_hash, embedding, loc=None):
        data_size = embedding.element_size() * embedding.numel()

        # LRU 逐出
        while self.current_size + data_size > self.max_size:
            if not self.mm_cache:
                return False
            lru_hash, lru_embedding = self.mm_cache.popitem(last=False)
            self.current_size -= _get_tensor_size(lru_embedding)

        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True
```

### 4.2 缓存工作流

缓存的 get/set 发生在 **模型 forward 阶段**（而非 Scheduler 调度阶段）。入口是 `get_embedding_and_mask()`，它依次尝试两种路径：

```python
# python/sglang/srt/managers/mm_utils.py
# 1. 优先尝试预计算 embedding（来自 precomputed_embedding 格式的输入）
embedding = _get_precomputed_embedding(
    embedding_items, prefix_length, extend_length, items_offset_list
)
# 2. 未命中则走 chunked prefill 路径（内部使用 embedding_cache）
if embedding is None:
    embedding = _get_chunked_prefill_embedding(
        data_embedding_func, embedding_items, items_size,
        prefix_length, extend_length, items_offset_list,
    )
```

当前实际调用的缓存路径是 `_get_chunked_prefill_embedding()`：

```python
# python/sglang/srt/managers/mm_utils.py
item_hashes = [item.hash for item in embedding_items_per_req]
embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)

embedding_per_req = embedding_cache.get(item_hashes)        # cache GET
if embedding_per_req is None:
    embedding_per_req = data_embedding_func(embedding_items_per_req)  # VIT 编码
    if not embedding_cache.set(embedding_items_hash, embedding_per_req):  # cache SET
        print_warning_once("Multimodal embedding cache is full...")
```

> **注意**：此函数标记为 `# TODO: To be obsoleted`，未来将被 `_get_chunked_prefill_embedding_for_chunked_items()` 替代（见 §4.2.1）。

#### 4.2.1 函数过渡状态

| 函数 | 行号 | 状态 | Cache 设备 |
|------|------|------|-----------|
| `_get_chunked_prefill_embedding()` | mm_utils.py | 当前默认，标记 `TODO: To be obsoleted` | GPU |
| `_get_chunked_prefill_embedding_for_chunked_items()` | mm_utils.py | 未来替代，当前未被调用 | CPU（`.detach().cpu()`） |

切换后的主要变化：
- Cache 从 GPU 移到 CPU，释放 GPU 显存
- 读取时需 `.to(target_device)` 搬回 GPU，增加少量延迟
- 支持按 chunk 粒度缓存（而非按整个请求），提升 chunked prefill 场景的缓存命中率

### 4.3 GPU Feature Buffer (预分配 GPU 缓冲区)

GPU Feature Buffer 是一块预分配的 GPU 内存，用于在 `MultimodalInputs.from_dict()` 构建多模态输入时避免频繁的小张量 GPU 分配。

**调用时机**：Scheduler 进程收到 tokenized 数据后，调用 `MultimodalInputs.from_dict()` 构建 `MultimodalInputs` 对象时使用。此时 feature tensor 需要临时放到 GPU 上进行快速 hash 计算，计算完成后移回 CPU。

**设计动机**：多模态请求中每个 item 的 feature 大小不一，频繁的小张量 GPU 分配/释放会导致 CUDA 内存碎片化。预分配一块连续 buffer，通过偏移量管理，可以显著减少分配开销并加速 hash 计算。

```python
# python/sglang/srt/managers/mm_utils.py
_GPU_FEATURE_BUFFER: Optional[torch.Tensor] = None
_BUFFER_OFFSET = 0

def init_feature_buffer(device):
    """预分配 GPU buffer，用于快速 hash 计算和特征暂存。"""
    size_mb = envs.SGLANG_MM_BUFFER_SIZE_MB.get()
    num_elements = int(size_mb * 1024 * 1024 / 4)
    _GPU_FEATURE_BUFFER = torch.empty(num_elements, dtype=torch.float32, device=device)

def try_add_to_buffer(tensor: torch.Tensor) -> torch.Tensor:
    """尝试将 tensor 复制到预分配 buffer 中，返回 buffer view。"""
    if _BUFFER_OFFSET + tensor.numel() <= _GPU_FEATURE_BUFFER.numel():
        buffer_view = _GPU_FEATURE_BUFFER[offset : offset + tensor.numel()]
        buffer_view.copy_(tensor.flatten(), non_blocking=True)
        return buffer_view.view(tensor.shape)
    return tensor  # buffer 不足时返回原 tensor
```

> 通过 `SGLANG_MM_BUFFER_SIZE_MB` 环境变量控制大小（默认值由 envs 模块定义）。每个 batch 开始时调用 `reset_buffer_offset()` 重置偏移量。

### 4.4 跨进程多模态数据传输

本节解答核心问题：**多模态场景下，pixel_values 在多卡 TP + ViT DP 配置中，是通过 CPU pickle 序列化跨进程传输，还是 GPU 间直接拷贝？**

**结论**：默认路径是 **CPU pickle 序列化**。ViT DP 的输出聚合是唯一使用 NCCL GPU-to-GPU 的环节。

**序列化瓶颈分析**：阶段 1-2 是纯 CPU pickle 操作，大图像场景确实存在序列化开销。优化手段包括 CUDA IPC（§4.4.5）、`--keep-mm-feature-on-device`、`--enable-broadcast-mm-inputs-process`。为什么不用共享内存或共享 GPU memory？跨进程 GPU 内存管理复杂度高，CUDA IPC 已是最接近零拷贝的方案，且仅限同 GPU 进程间通信。

#### 4.4.1 传输路径四阶段总结

| 阶段 | 路径 | 机制 | 数据格式 | 源码 |
|------|------|------|---------|------|
| 1. TokenizerManager → Scheduler Rank 0 | ZMQ `send_pyobj` | pickle 序列化 CPU tensor | bytes over TCP/IPC | `tokenizer_manager.py` |
| 2. Scheduler Rank 0 → 其他 TP Rank | `broadcast_pyobj()` | pickle.dumps → ByteTensor → dist.broadcast (CPU group) | bytes via CPU process group | `common.py` |
| 3. CPU → GPU (每个 Rank) | `prepare_for_extend()` | `.to(device, non_blocking=True)` 或 CudaIpc | async H2D copy | `schedule_batch.py` |
| 4. ViT DP 输出聚合 | `all_gather` | NCCL all_gather | GPU tensor (GPU-to-GPU) | `multimodal/mm_utils.py` |

只有**阶段 4** 是 GPU 直连通信，前三个阶段全部经过 CPU。

#### 4.4.2 broadcast_pyobj 实现

Rank 0 收到请求后需要广播给其他 TP Rank。这里使用的是 **pickle + CPU dist.broadcast**，而非 NCCL：

```python
# python/sglang/srt/utils/common.py
def broadcast_pyobj(data, rank, dist_group, src=0, force_cpu_device=True):
    device = torch.device("cpu")  # force_cpu_device=True (默认)

    if rank == src:
        # Rank 0: pickle 序列化 → numpy → ByteTensor → broadcast
        serialized_data = pickle.dumps(data)
        size = len(serialized_data)
        tensor_data = torch.ByteTensor(
            np.frombuffer(serialized_data, dtype=np.uint8)
        ).to(device)
        tensor_size = torch.tensor([size], dtype=torch.long, device=device)

        dist.broadcast(tensor_size, src=src, group=dist_group)  # 先广播大小
        dist.broadcast(tensor_data, src=src, group=dist_group)  # 再广播数据
    else:
        # 其他 Rank: 接收 → numpy → pickle.loads
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
```

> **关键点**：`force_cpu_device=True` 意味着使用 CPU process group（Gloo 后端），不走 NCCL。pixel_values 作为 CPU tensor 被 pickle 序列化传输。

#### 4.4.3 prepare_for_extend 中的双路径

每个 Rank 收到请求后，在 `prepare_for_extend()` 中将 pixel_values 从 CPU 搬到 GPU：

```python
# python/sglang/srt/managers/schedule_batch.py
for mm_input in multimodal_inputs:
    if mm_input is None:
        continue
    for mm_item in mm_input.mm_items:
        pixel_values = getattr(mm_item, "feature", None)
        if isinstance(pixel_values, torch.Tensor):
            # 默认路径: CPU → GPU async copy
            mm_item.feature = pixel_values.to(self.device, non_blocking=True)
        elif isinstance(pixel_values, CudaIpcTensorTransportProxy):
            # CUDA IPC 路径: GPU → GPU 零拷贝重建
            mm_item.feature = pixel_values.reconstruct_on_target_device(
                torch.cuda.current_device()
            )
```

#### 4.4.4 完整数据流时序图

```mermaid
sequenceDiagram
    participant Client as HTTP Client
    participant TM as TokenizerManager<br/>(主进程)
    participant S0 as Scheduler Rank 0
    participant S1 as Scheduler Rank 1-N
    participant GPU as GPU (每个 Rank)
    participant VIT as ViT DP Forward

    Client->>TM: POST /v1/chat (images)
    Note over TM: QwenVLImageProcessor<br/>process_mm_data_async()<br/>→ pixel_values (CPU tensor)

    TM->>S0: ZMQ send_pyobj<br/>pickle(mm_inputs dict)

    Note over S0: pickle.loads → mm_inputs dict

    rect rgb(255, 243, 205)
        Note over S0,S1: broadcast_pyobj (CPU Gloo)
        S0->>S1: dist.broadcast(pickle.dumps(recv_reqs))
        Note over S1: pickle.loads → 每个 Rank 拥有<br/>完整 mm_inputs dict
    end

    Note over S0: MultimodalInputs.from_dict()
    Note over S1: MultimodalInputs.from_dict()

    rect rgb(204, 229, 255)
        Note over S0,GPU: prepare_for_extend (H2D)
        S0->>GPU: pixel_values.to(device, non_blocking=True)
        S1->>GPU: pixel_values.to(device, non_blocking=True)
    end

    rect rgb(212, 237, 218)
        Note over VIT: ViT DP 并行编码
        Note over GPU: 每个 Rank 本地切片 pixel_values<br/>执行 VIT forward
        GPU->>GPU: NCCL all_gather<br/>(唯一 GPU-to-GPU 传输)
    end
```

#### 4.4.5 三种传输优化

默认路径（CPU pickle）在大图像或多 TP Rank 场景下可能成为瓶颈。SGLang 提供三种优化手段：

**(1) CUDA IPC Transport (`SGLANG_USE_CUDA_IPC_TRANSPORT=1`)**

在 TokenizerManager 中通过 CUDA IPC handle 共享 GPU 内存，Scheduler 端通过 `reconstruct_on_target_device()` 零拷贝重建 tensor。

- **限制**：仅限同一 GPU 上的进程间通信（intra-GPU）。TP 模式下不同 Rank 使用不同 GPU，CUDA IPC 不可用，会自动 fallback 到默认 pickle 路径
- **适用场景**：TokenizerManager 和 Scheduler 在同一 GPU 上运行时

**(2) Keep Feature on Device (`--keep-mm-feature-on-device`)**

```python
# python/sglang/srt/multimodal/processors/base_processor.py
if not self.server_args.keep_mm_feature_on_device:
    # 默认: 预处理后移到 CPU
    for feature_name in self.FEATURE_NAMES:
        if feature_name in result and isinstance(result[feature_name], torch.Tensor):
            result[feature_name] = result[feature_name].to("cpu")
# 启用后: 预处理结果保留在 GPU，跳过 CPU roundtrip
```

- **适用场景**：HF Processor 在 GPU 上预处理的情况（如 Qwen 的部分操作），或 TokenizerManager 和 Scheduler 在同一 GPU 的场景
- **注意**：如果预处理本身就是纯 CPU 操作（feature 在 CPU 上），此选项无效

**(3) Broadcast MM Inputs Process (`--enable-broadcast-mm-inputs-process`)**

启用后，TP Rank 0 执行一次 `from_dict()` 物化完整对象，然后广播给其他 TP Rank：

```python
# python/sglang/srt/managers/scheduler.py
def _process_and_broadcast_mm_inputs(self, recv_reqs):
    if self.is_entry_rank:
        image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
        if group_world_size > 1:
            torch.distributed.broadcast_object_list(
                [image_inputs], src=self.entry_rank,
                group=self.dp_tp_cpu_group
            )
    else:
        obj_list = [None]
        torch.distributed.broadcast_object_list(
            obj_list, src=self.entry_rank,
            group=self.dp_tp_cpu_group
        )
        image_inputs = obj_list[0]
```

> **收益**：减少 CPU 占用，避免单线程 Scheduler 被大量 `from_dict` 阻塞，在高并发多模态请求场景下尤为明显。

## 5. ViT DP 并行编码

### 5.1 概念区分：ViT DP vs DP Scheduling

> **⚠ 不要混淆**：SGLang 中有两种 "DP"：
>
> | 概念 | 启用方式 | 含义 |
> |------|---------|------|
> | **DP Scheduling** | `--dp-size N` | 启动 **N 个独立 Scheduler 进程**，由 DataParallelController 分发请求 |
> | **ViT DP** | `--mm-enable-dp-encoder` | **单个 Scheduler** 内，TP 组的多个 Rank 分担 ViT 编码，复用 TP 通信组做 all-gather |
>
> 本章分析的是 **ViT DP**（`--mm-enable-dp-encoder`），不涉及 DataParallelController。

**关键架构说明**：
- ViT DP 下只有 **一个 Scheduler**（逻辑上一个因为每个调度的逻辑一致得到一样的结果，所以相当于一个），TP 组的多个 Rank 分担 ViT 编码工作
- `tp_size` 即 ViT DP 的并行度（例如 `--tp 4` 则 4 个 Rank 分担编码）
- Rank 0 是 entry rank，接收 ZMQ 请求并通过 `broadcast_pyobj` 广播给其他 Rank
- 所有 Rank 独立计算负载均衡分配（确定性算法，输入相同则结果一致），无需额外通信
- 每个 Rank 是独立进程，cache 通过 Python 全局变量隔离

### 5.2 为什么需要 ViT DP

默认情况下，VIT 编码在每个 TP rank 上执行完全相同的计算（冗余），浪费计算资源：

```mermaid
flowchart TB
    subgraph NoDP["默认模式: 每个 GPU 处理所有图像 (冗余)"]
        ND0["GPU_0: VIT(img0,img1) -> emb0,emb1"]
        ND1["GPU_1: VIT(img0,img1) -> emb0,emb1 冗余"]
    end

    subgraph DPMode["DP 模式: 每个 GPU 处理部分图像, 然后 All-Gather"]
        D0["GPU_0: VIT(img0) -> emb0"]
        D1["GPU_1: VIT(img1) -> emb1"]
        AG["All-Gather -> 全部 embeddings"]
        D0 --> AG
        D1 --> AG
    end

    NoDP -->|"效率提升"| DPMode

    style NoDP fill:#ffcccc
    style DPMode fill:#ccffcc
```

### 5.3 权重复制机制

启用 `--mm-enable-dp-encoder` 后，ViT 的所有层使用 `tp_size=1, tp_rank=0` 初始化，**每个 TP Rank 持有完整的 ViT 权重副本**（而非 TP 分片）：

```python
# python/sglang/srt/models/qwen3_vl.py
class VisionMLP(nn.Module):
    def __init__(self, ..., use_data_parallel: bool = False):
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
        self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        # ColumnParallelLinear / RowParallelLinear 使用 tp_size=1
        # → 加载完整权重，不做分片
        # → RowParallelLinear 不执行 all-reduce
```

**影响**：

| 方面 | 默认 TP 模式 | ViT DP 模式 |
|------|-------------|-------------|
| 每 Rank ViT 权重量 | `ViT_size / tp_size` | `ViT_size`（完整） |
| 额外显存开销 | 0 | `(1 - 1/tp_size) * ViT_size` |
| ViT 通信开销 | 每层 all-reduce | 仅最终 all-gather |
| ViT 计算量 | 全量图像 | 分片图像（`/tp_size`） |

以 Qwen3.5-397B-A17B 的 ViT（约 1.1B 参数，FP16 约 2.2GB）为例：
- `tp=8` 默认模式：每 Rank 约 275MB ViT 权重，但每 Rank 编码所有图像
- `tp=8` ViT DP 模式：每 Rank 约 2.2GB ViT 权重（多占 ~1.9GB），但每 Rank 仅编码 1/8 图像

权衡取舍是：**用权重复制的额外显存换取 ViT 计算的线性加速和消除逐层 all-reduce**。

### 5.4 负载均衡算法

不同图像的 patch 数量可能差异很大，简单轮询分配会导致负载不均衡。以下使用 2 个 Rank 的示例更好地体现负载均衡效果：

```python
# python/sglang/srt/multimodal/mm_utils.py
def get_dp_encoder_lb_assignment(
    sizes: list[int],      # 每张图像的 patch 数量
    num_gpus: int = 2,     # GPU 数量 (= tp_size)
) -> tuple[list[int], list[int], list[int]]:
    """
    贪心算法：按图像大小负载均衡分配到 GPU。

    Returns:
        shuffle_indices:  按 GPU 分组排列的图像索引序列
        gpu_sample_counts: 每 GPU 分配的图像数
        gpu_loads:         每 GPU 的总 patch 数

    Example (2 Rank):
        sizes = [1000, 100, 200, 50], num_gpus = 2
        # 贪心分配: GPU_0=[img0(1000), img3(50)], GPU_1=[img2(200), img1(100)]
        # 拼接: shuffle_indices = [0, 3, 2, 1]
        #        ^--GPU_0--^  ^--GPU_1--^
        # gpu_sample_counts = [2, 2]
        # gpu_loads = [1050, 300]

    使用方式 (调用者 run_dp_sharded_mrope_vision_model):
        cum = [0, *accumulate(gpu_sample_counts)]  # [0, 2, 4]
        image_idxs_local = shuffle_indices[cum[rank]:cum[rank+1]]
        # Rank 0 → shuffle_indices[0:2] = [0, 3] → 编码 img0, img3
        # Rank 1 → shuffle_indices[2:4] = [2, 1] → 编码 img2, img1
    """

    # 1. 按大小降序排列
    large_to_small_indices = sorted(
        range(len(sizes)), key=lambda i: sizes[i], reverse=True
    )

    # 2. 贪心分配：每次分配给当前负载最小的 GPU
    gpu_loads = [0] * num_gpus
    gpu_assignments = [[] for _ in range(num_gpus)]

    for idx in large_to_small_indices:
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    # 3. 按 GPU 顺序拼接
    shuffle_indices = []
    gpu_sample_counts = []
    for gpu_id in range(num_gpus):
        shuffle_indices.extend(gpu_assignments[gpu_id])
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return shuffle_indices, gpu_sample_counts, gpu_loads
```

> **为什么每个 Rank 独立计算而非分发结果？** 负载均衡算法是确定性的（相同输入必然产生相同输出），且 CPU 开销极低（仅排序+贪心分配），远小于一次额外通信的开销。

```mermaid
flowchart LR
    subgraph Input["输入: patches = [1000, 100, 200, 50]"]
        direction TB
        P0["img0: 1000"]
        P1["img1: 100"]
        P2["img2: 200"]
        P3["img3: 50"]
    end

    subgraph Sort["Step 1: 降序排序"]
        S["[1000, 200, 100, 50]<br/>索引: [0, 2, 1, 3]"]
    end

    subgraph Greedy["Step 2: 贪心分配 (2 GPU)"]
        direction TB
        G1["Round 1: img0(1000) → GPU_0 (负载:1000)"]
        G2["Round 2: img2(200) → GPU_1 (负载:200)"]
        G3["Round 3: img1(100) → GPU_1 (负载:300)"]
        G4["Round 4: img3(50) → GPU_1 (负载:350)"]
    end

    subgraph Result["分配结果"]
        direction TB
        R0["GPU_0: [img0, img3]<br/>负载=1050"]
        R1["GPU_1: [img2, img1]<br/>负载=300"]
    end

    Input --> Sort --> Greedy --> Result

    style G1 fill:#d4edda
    style G2 fill:#d4edda
    style G3 fill:#d4edda
    style G4 fill:#d4edda
```

### 5.5 数据流转: pixel_value → VIT embedding

完整的数据流转分为 3 个阶段：

> **注意**：阶段 1 不存在 NCCL scatter 或跨 Rank 传输。所有 TP Rank 已通过 `broadcast_pyobj`（见 §4.4.2）拥有完整 `pixel_values`，每个 Rank 根据 `tp_rank` 做本地 `torch.cat` 切片提取自己负责的子集。

```mermaid
flowchart TB
    subgraph Phase1["阶段 1: 本地切片 (非跨 Rank 传输)"]
        direction TB
        P1_IN["所有 Rank 已持有完整 pixel_values<br/>(来自 broadcast_pyobj + prepare_for_extend H2D)"]
        P1_LB["负载均衡分配 (纯计算, 每 Rank 独立执行)"]
        P1_R0["Rank 0: pixel_values_local = torch.cat(本 Rank 分配的图像)"]
        P1_R1["Rank 1: pixel_values_local = torch.cat(本 Rank 分配的图像)"]

        P1_IN --> P1_LB
        P1_LB --> P1_R0
        P1_LB --> P1_R1
    end

    subgraph Phase2["阶段 2: 并行 VIT 编码"]
        direction TB
        P2_R0["Rank 0: VIT(pixel_local) → embed_local [N0, D]"]
        P2_R1["Rank 1: VIT(pixel_local) → embed_local [N1, D]"]
    end

    subgraph Phase3["阶段 3: 聚合与缓存"]
        direction TB
        P3_PAD["Padding 到 max_len"]
        P3_AG["NCCL All-Gather<br/>GPU → GPU"]
        P3_REORDER["去 Padding + 恢复原始顺序"]
        P3_CACHE["存入 Cache<br/>(当前: GPU tensor)"]
        P3_FUSE["融合到 input_embeds 送入 LLM"]

        P3_PAD --> P3_AG --> P3_REORDER
        P3_REORDER --> P3_CACHE
        P3_REORDER --> P3_FUSE
    end

    P1_R0 --> P2_R0
    P1_R1 --> P2_R1

    P2_R0 --> P3_PAD
    P2_R1 --> P3_PAD

    style Phase2 fill:#d4edda,stroke:#28a745
    style P3_AG fill:#cce5ff,stroke:#007bff,stroke-width:2px
    style P3_CACHE fill:#fff3cd,stroke:#ffc107,stroke-width:2px
```

**代码实现**：

```python
# python/sglang/srt/multimodal/mm_utils.py
def run_dp_sharded_mrope_vision_model(vision_model, pixel_values, grid_thw_list, rope_type):
    tp_size = get_attention_tp_size()
    tp_rank = get_attention_tp_rank()

    # 1. 计算每张图像的 patch 数
    patches_per_image = [t * h * w for t, h, w in grid_thw_list]

    # 2. 负载均衡分配
    (shuffle_indices, gpu_sample_counts, grouped_pixel_values_len) = \
        get_dp_encoder_lb_assignment(patches_per_image, tp_size)

    # 3. 提取本 Rank 负责的 pixel_values (本地切片，非跨 Rank 传输)
    pixel_values_local = torch.cat([
        pixel_values[cum_patches[i]:cum_patches[i + 1]]
        for i in image_idxs_local
    ])

    # 4. 本地 VIT 编码
    image_embeds_local = vision_model(pixel_values_local, local_grid_thw)

    # 5. Padding + All-Gather (NCCL)
    padded = pad_to_length(image_embeds_local, max_len)
    gathered_embeds = get_attention_tp_group().all_gather(padded, dim=0)

    # 6. 去 Padding，按原始顺序重组
    return reorder_embeddings(gathered_embeds, original_indices)
```

### 5.6 Cache 架构

**结论**：**每个 TP Rank 进程有独立的 Cache 副本**（非共享内存）。

每个 TP Rank 运行在独立的 Python 进程中（通过 `multiprocessing` 启动），Python 全局变量 `embedding_cache` 在每个进程中独立实例化。

```python
# python/sglang/srt/managers/mm_utils.py
# 全局变量: 每个进程独立
embedding_cache: Optional[MultiModalStaticCache] = None

def init_mm_embedding_cache(max_size: int = 0):
    global embedding_cache
    embedding_cache = MultiModalStaticCache(max_size)  # 进程内单例

# python/sglang/srt/managers/scheduler.py
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)
```

```mermaid
flowchart TB
    subgraph CacheArch["Cache 架构 (ViT DP, 单 Scheduler 2 TP Rank)"]
        direction TB

        subgraph Rank0["TP Rank 0 进程"]
            S0["Scheduler Rank 0<br/>(entry rank, 接收 ZMQ)"]
            C0["embedding_cache<br/>(全局单例)<br/>**独立 GPU 内存**"]
            G0["GPU 0"]
        end

        subgraph Rank1["TP Rank 1 进程"]
            S1["Scheduler Rank 1<br/>(接收 broadcast_pyobj)"]
            C1["embedding_cache<br/>(全局单例)<br/>**独立 GPU 内存**"]
            G1["GPU 1"]
        end

        S0 -->|"broadcast_pyobj"| S1
        S0 --> C0
        S1 --> C1
        C0 <--> G0
        C1 <--> G1

        G0 <-->|"NCCL All-Gather"| G1
    end

    style C0 fill:#d4edda,stroke:#28a745
    style C1 fill:#d4edda,stroke:#28a745
```

> **注意**：当前默认路径下 cache 存储的是 **GPU tensor**（直接存 VIT 输出），因此图中标注为"独立 GPU 内存"。未来切换到 CPU cache 后将变为 CPU 内存。

### 5.7 传输方式总结

> 详细的端到端传输路径分析见 **§4.4**。本表按实际调用时序排列：

| 顺序 | 传输类型 | 方式 | 说明 |
|------|---------|------|------|
| 1 | TokenizerManager → Scheduler | ZMQ `send_pyobj` | pickle 序列化 CPU tensor |
| 2 | Rank 0 → 其他 TP Rank | `broadcast_pyobj` (CPU Gloo) | pickle + dist.broadcast |
| 3 | CPU → GPU | `prepare_for_extend` H2D | `.to(device, non_blocking=True)` |
| 4 | GPU ↔ GPU | NCCL All-Gather | ViT embeddings 聚合 |

### 5.8 完整请求生命周期

```mermaid
flowchart TB
    subgraph Phase1["阶段 1: HTTP 请求接收"]
        A1["FastAPI Router"] --> A2["GenerateReqInput"]
        A2 --> A3["text + images[]"]
    end

    subgraph Phase2["阶段 2: TokenizerManager 预处理"]
        B1["QwenVLImageProcessor"]
        B2["加载图像 (PIL)"]
        B3["smart_resize()"]
        B4["计算 grid_thw"]
        B5["pixel_values + input_ids"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph Phase3["阶段 3: Scheduler 调度"]
        C1["Scheduler get_next_batch_to_run()"]
        C2["pad_input_ids 替换 placeholder"]
        C3["prepare_for_extend H2D 搬运"]
        C1 --> C2 --> C3
    end

    subgraph Phase4["阶段 4: ModelRunner.forward()"]
        D1["ForwardBatch 创建"]
        D2["Qwen3VLForConditionalGeneration.forward()"]
        D1 --> D2
    end

    subgraph Phase5["阶段 5: VIT 编码 + Cache"]
        direction TB
        E0["检查 MultimodalCache"]
        E1["cache miss: get_dp_encoder_lb_assignment()"]
        E2["run_dp_sharded_mrope_vision_model()"]
        E3["各 GPU 并行 VIT 编码"]
        E4["all_gather()"]
        E5["恢复原始顺序 + cache set"]
        E0 --> E1 --> E2 --> E3 --> E4 --> E5
    end

    subgraph Phase6["阶段 6: 嵌入融合 + LLM 推理"]
        F1["embed_mm_inputs()"]
        F2["text_embeds + image_embeds 融合"]
        F3["Qwen3LLMModel.forward()"]
        F4["LogitsProcessor → 采样"]
        F1 --> F2 --> F3 --> F4
    end

    subgraph Phase7["阶段 7: Decode + 返回"]
        G1["Decode 循环"] --> G2["返回结果"]
    end

    Phase1 --> Phase2 --> Phase3 --> Phase4 --> Phase5 --> Phase6 --> Phase7

    style Phase5 fill:#e6f3ff,stroke:#0066cc,stroke-width:2px
```

**涉及的核心类和函数**：

| 阶段 | 类/文件 | 核心函数 |
|------|---------|----------|
| 1. 请求接收 | `http_server.py` | `openai_v1_chat_completions()` |
| 2. 预处理 | `QwenVLImageProcessor` | `process_mm_data_async()`, `smart_resize()` |
| 3. 调度 | `Scheduler` / `schedule_batch.py` | `get_next_batch_to_run()`, `prepare_for_extend()` |
| 4. 模型前向 | `Qwen3VLForConditionalGeneration` | `forward()` |
| 4. 跨进程传输 | `common.py` | `broadcast_pyobj()` |
| 5. VIT 编码 | `managers/mm_utils.py` | `_get_chunked_prefill_embedding()` |
| 5. VIT DP | `multimodal/mm_utils.py` | `get_dp_encoder_lb_assignment()`, `run_dp_sharded_mrope_vision_model()` |
| 6. 融合 | `managers/mm_utils.py` | `embed_mm_inputs()`, `general_mm_embed_routine()` |
| 7. 采样 | `LogitsProcessor` | `forward()` |

## 6. VIT CUDA Graph

### 6.1 设计动机与 Trade-off

**为什么用 CUDA Graph**：ViT 的 Transformer blocks 通常有 20-40 层，每层包含多个 kernel（attention、FFN、LayerNorm 等）。逐层 kernel launch 的 CPU 开销在小 batch 场景下占比显著。CUDA Graph 将整个 `blocks + merger + deepstack_merger` 捕获为单次 replay，消除所有中间 launch 开销。

**Graph Key 含义**：`x_3d.shape[0]` = patch 总数（序列长度），不同序列长度需要不同 graph。

**Padding 策略**：不做 padding。按 seq_len 分桶，每个唯一 seq_len 捕获独立 graph。首次遇到新 seq_len 时 warmup + capture，后续直接 replay。

**额外计算开销**：无 padding 浪费，但 graph 数量可能较多（不同图像尺寸产生不同 seq_len）。

**主要收益**：消除 kernel launch 开销。ViT 通常有 20-40 层 Transformer block，每层多个 kernel，launch 开销累积可观。

**与 torch.compile 的关系**：两者互补——compile 优化 kernel 实现（算子融合、内存优化），CUDA Graph 消除 launch 开销。当前 ViT 代码未使用 compile。

**使用限制**：仅支持 `triton_attn` 和 `fa3` backend，不支持 `flashinfer`。

### 6.2 ViTCudaGraphRunner

```python
# python/sglang/srt/multimodal/vit_cuda_graph_runner.py
class ViTCudaGraphRunner:
    """
    Generic ViT CUDA Graph Runner.
    Captures "blocks + merger + deepstack merger" into CUDA graph.
    Lazily captures graphs for each unique sequence length.
    """

    def __init__(self, vit: nn.Module):
        self.vit = vit
        # graph_key -> buffers / graphs
        self.block_input: Dict[Hashable, torch.Tensor] = {}
        self.block_ws: Dict[Hashable, torch.Tensor] = {}
        self.block_graphs: Dict[Hashable, torch.cuda.CUDAGraph] = {}
        self.block_output: Dict[Hashable, torch.Tensor] = {}
        # captured seqlens buffers (地址必须稳定以支持 graph replay)
        self.cu_full_len: Dict[Hashable, torch.Tensor] = {}
        self.cu_window_len: Dict[Hashable, torch.Tensor] = {}

    def _get_graph_key(self, x_3d: torch.Tensor) -> int:
        """Graph key = sequence length (patch 总数)."""
        return x_3d.shape[0]

    def run(self, x_3d, cu_seqlens, rotary_pos_emb, ...):
        """Run VIT with CUDA Graph if available."""
        graph_key = self._get_graph_key(x_3d)
        if graph_key not in self.block_graphs:
            return self._capture_and_run(graph_key, x_3d, cu_seqlens, ...)
        else:
            return self._replay(graph_key, x_3d, rotary_pos_emb, ...)
```

### 6.3 Graph 捕获

```python
def _capture_and_run(self, graph_key, x_3d, cu_seqlens, ...):
    """Capture VIT forward as CUDA Graph."""

    # 1. 分配输入/输出 buffer
    self.block_input[graph_key] = torch.empty_like(x_3d)
    self.block_ws[graph_key] = torch.empty(...)  # workspace

    # 2. warmup (确保 CUDA 内核已编译)
    with torch.no_grad():
        _ = self._run_blocks(self.block_input[graph_key], ...)

    # 3. 捕获
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = self._run_blocks(self.block_input[graph_key], ...)

    self.block_graphs[graph_key] = graph
    self.block_output[graph_key] = output
```

### 6.4 启用方式

```bash
# 启用 VIT CUDA Graph
export SGLANG_VIT_ENABLE_CUDA_GRAPH=1

# 仅支持部分 attention backend
# 支持: triton_attn, fa3
# 不支持: flashinfer (目前)
```

> **启用判断位置**：环境变量定义在 `python/sglang/srt/environ.py`（`EnvBool(False)`），各模型文件在 ViT forward 入口处检查：
>
> ```python
> # models/qwen3_vl.py
> if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
>     return self.forward_with_cuda_graph(x, grid_thw)
>
> # models/qwen2_5_vl.py
> self.enable_cg = _is_cuda and envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()
>
> # layers/attention/vision.py
> if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
>     ...  # 使用 graph-compatible 的 attention 实现
> ```
>
> ViTCudaGraphRunner 本身不做启用判断，由调用方决定是否走 CUDA Graph 路径。

## 7. 嵌入融合

### 7.1 融合流程 (Qwen3.5)

> **重要区分**：SGLang 的嵌入融合**不在**模型的 `forward()` 中直接完成（与 HuggingFace 原始实现不同）。融合逻辑位于 `python/sglang/srt/managers/mm_utils.py` 的 `embed_mm_inputs()` 中，由 `general_mm_embed_routine()` 统一调用。

**实际调用链**：

```
Qwen3VLForConditionalGeneration.forward()           # models/qwen3_vl.py
  └─ general_mm_embed_routine()                      # managers/mm_utils.py
       ├─ embed_mm_inputs()                          # managers/mm_utils.py
       │    ├─ get_embedding_and_mask()              # managers/mm_utils.py
       │    │    └─ data_embedding_func()            # = self.get_image_feature()
       │    │         └─ run_dp_sharded_mrope_vision_model()  # (ViT DP 时)
       │    │              或 self.visual()           # (非 DP 时)
       │    ├─ input_ids.clamp_(0, vocab_size-1)     # 清除 hash pad_value
       │    ├─ input_embeds = embed_tokens(input_ids) # 文本嵌入
       │    ├─ input_embeds[indices] = embedding      # 融合: scatter 替换
       │    └─ separate_deepstack_embeds()            # (Deepstack 时)
       └─ language_model(input_embeds=input_embeds)   # 送入 LLM
```

```python
# python/sglang/srt/models/qwen3_vl.py
def forward(self, input_ids, positions, forward_batch, ...):
    if self.is_mrope_enabled:
        positions = forward_batch.mrope_positions

    # 调用通用融合例程，不直接处理 pixel_values
    hidden_states = general_mm_embed_routine(
        input_ids=input_ids,
        forward_batch=forward_batch,
        language_model=self.model,
        multimodal_model=self,        # 提供 get_image_feature()
        positions=positions,
        use_deepstack=self.use_deepstack,
    )

# python/sglang/srt/managers/mm_utils.py
# scatter embeddings into input embedding
for i, modality, embedding, mask in zip(...):
    if embedding is None or mask is None:
        continue
    indices = torch.where(mask.squeeze(dim=-1))[0]
    input_embeds[indices] = embedding.to(input_embeds.device, input_embeds.dtype)

# pad_value 来自内容 hash，可能是很大的整数，会导致 embedding lookup 崩溃
vocab_size = input_embedding.num_embeddings
input_ids.clamp_(min=0, max=vocab_size - 1)
input_embeds = input_embedding(input_ids)
```

### 7.2 mrope (Multi-Resolution RoPE)

Qwen-VL 使用 3D RoPE 处理不同分辨率：

```python
def get_rope_index(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """Calculate 3D position indices for mrope."""

    # 文本 token: [0, 1, 2, 3, ...]
    # 图像 token: [t, h, w] 坐标

    position_ids = torch.zeros((batch_size, seq_len, 3), ...)

    for i, (t, h, w) in enumerate(image_grid_thw):
        # 图像 patches 的 3D 位置
        temporal = torch.arange(t).repeat_interleave(h * w)
        height = torch.arange(h).repeat(t * w)
        width = torch.arange(w).repeat(t * h)

        position_ids[i, image_start:image_end, 0] = temporal
        position_ids[i, image_start:image_end, 1] = height
        position_ids[i, image_start:image_end, 2] = width

    return position_ids
```

### 7.3 Deepstack 机制 (Qwen3.5)

Qwen3.5 引入 **Deepstack** 机制：在 ViT 的中间层捕获特征，注入到 LLM decoder 的浅层，使浅层 decoder 也获得多分辨率视觉特征。

**工作原理**：
- ViT 在指定中间层（`deepstack_visual_indexes`，如 `{8: 0, 16: 1, 24: 2}`）捕获特征，每个中间层通过独立的 `deepstack_merger` 降维
- ViT 输出维度 = `hidden_size * (1 + num_deepstack)`，比正常 ViT 输出多出 deepstack 特征部分
- LLM decoder 在 `deepstack_embed_to_decoder_layer` 映射的指定层，通过 `post_residual_addition` 将 deepstack 特征加到 hidden_states

#### 7.3.1 ViT 端：中间层特征捕获

```python
# python/sglang/srt/models/qwen3_vl.py
# ViT forward 中，在指定层捕获 deepstack 特征
for layer_num, blk in enumerate(self.blocks):
    x = blk(x, cu_seqlens=cu_seqlens, ...)

    if layer_num in self.deepstack_visual_indexes:
        # 通过独立的 merger 降维
        deepstack_feature = self.deepstack_merger_list[num_deepstack_captured](x)
        deepstack_feature_lists.append(deepstack_feature)
        num_deepstack_captured += 1

# 最终层通过主 merger
x = self.merger(x)

# 拼接: [seq_len, hidden_size * (1 + num_deepstack)]
hidden_states = torch.cat([x] + deepstack_feature_lists, dim=1)
```

#### 7.3.2 融合端：特征分离与注入

```python
# python/sglang/srt/models/qwen3_vl.py
def separate_deepstack_embeds(self, embedding):
    separate_index = self.config.hidden_size
    input_embeds = embedding[:, :separate_index]          # 主视觉特征
    input_deepstack_embeds = embedding[:, separate_index:] # deepstack 特征
    return input_embeds, input_deepstack_embeds
```

在 `embed_mm_inputs()` 中：
- 主视觉特征 scatter 到 `input_embeds` 中替换 pad token
- deepstack 特征存入 `input_deepstack_embeds`（零初始化同长张量），在 pad token 位置写入
- `input_deepstack_embeds` 通过 `kwargs["input_deepstack_embeds"]` 传入 decoder
- decoder 在 `deepstack_embed_to_decoder_layer` 映射的指定层将 deepstack 特征融合到 hidden_states

### 7.4 Chunked Prefill 的 CPU Offload 机制

```python
# python/sglang/srt/managers/mm_utils.py
# 融合完成后，将 GPU features offload 到 CPU
if mm_inputs_list:
    for mm_input_obj in mm_inputs_list:
        if mm_input_obj and hasattr(mm_input_obj, "mm_items"):
            for mm_item in mm_input_obj.mm_items:
                feature = getattr(mm_item, "feature", None)
                if isinstance(feature, torch.Tensor) and feature.is_cuda:
                    mm_item.feature = feature.to("cpu", non_blocking=True)
```

> **设计动机**：此逻辑与 cache 存储位置无关——即使 cache 移到 CPU，offload 仍然需要。释放的是原始 `pixel_values` / `feature`，不是 cached embedding。
>
> 在 chunked prefill 场景下，同一请求跨多个 batch 处理。MultimodalCache 是 best-effort 的，可能被 LRU 驱逐。CPU offload 确保即使 cache miss，原始数据仍可访问（作为 fallback）。
>
> 两个目标：(1) 释放宝贵的 GPU 显存给后续 batch 的动态激活使用；(2) 保留数据可访问性作为 cache miss 的可靠回退。

### 7.5 请求完成后的 MM 内存清理

```python
# python/sglang/srt/managers/scheduler.py
def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
    for req in batch.reqs:
        if not req.finished() or not (mm_inputs := req.multimodal_inputs):
            continue
        # session 请求保留 mm_inputs，供后续轮次使用
        if req.session_id:
            continue
        # 非 session 请求: 清理 features 并释放引用
        for item in mm_inputs.mm_items:
            item.feature = None
        req.multimodal_inputs = None
```

> 每个 batch 处理完成后，Scheduler 调用此方法清理已完成请求的多模态数据。关键区分:
> - 普通请求: 立即清理 `feature` 和 `multimodal_inputs`，释放 GPU/CPU 内存
> - Session 请求: 保留 `mm_inputs`，因为同一 session 的后续请求可能需要引用之前的多模态上下文
>
> cache 大小是预估的，不会因 cache 本身溢出。但不及时清理 feature 引用会导致 GPU 显存碎片化，在高并发多模态场景下，动态激活（VIT/LLM forward 中间张量）可能因剩余显存不足而 OOM。

## 8. 支持的模态和模型

### 8.1 模态类型

```python
# python/sglang/srt/managers/schedule_batch.py
class Modality(Enum):
    IMAGE = auto()
    MULTI_IMAGES = auto()  # 多图场景，与 IMAGE 共享 image_token_id
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        return Modality[modality_str.upper()]

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]
        # 注意: MULTI_IMAGES 不在 all() 中，因为它是 IMAGE 的变体
```

> `MULTI_IMAGES` 用于区分单图和多图场景。`is_image()` 方法同时匹配 `IMAGE` 和 `MULTI_IMAGES`，而 `Modality.all()` 只返回三种基础模态。在 `embed_mm_inputs()` 遍历 `Modality.all()` 时，MULTI_IMAGES 的 item 会被 `is_modality(Modality.IMAGE)` 匹配到 IMAGE 分支统一处理。

### 8.2 支持的模型

| 模型 | 支持模态 | Processor 类 / 模型文件 |
|------|----------|------------------------|
| **Qwen3.5** | 图像, 视频 | `QwenVLImageProcessor` |
| **Qwen2.5-VL** / **Qwen2-VL** | 图像, 视频 | `QwenVLImageProcessor` |
| **Qwen3-Omni** | 图像, 视频, 音频 | `QwenVLImageProcessor` |
| **Qwen2-Audio** | 音频 | `models/qwen2_audio.py` |
| **Gemma3** / **Gemma3n** | 图像 | `models/gemma3_mm.py` / `processors/gemma3n.py` |
| **LLaVA** / **LLaVA-OneVision** | 图像, 视频 | `LLaVAImageProcessor` |
| **InternVL** / **InternVL2.5** | 图像 | `InternVLImageProcessor` |
| **GLM-4V** | 图像 | `GLM4VImageProcessor` |
| **MiniCPM-V** | 图像, 视频 | `MiniCPMImageProcessor` |
| **Pixtral** | 图像 | `models/pixtral.py` |
| **Phi-3.5-Vision** / **Phi-4-MM** | 图像 | `Phi4MMImageProcessor` |
| **MLlama** | 图像 | `models/mllama.py` |
| **MLlama4** | 图像 | `models/mllama4.py` |
| **KimiVL** | 图像 | `models/kimi_vl.py` |
| **DeepSeek-VL2** | 图像 | `processors/deepseek_vl_v2.py` |
| **Janus-Pro** | 图像 | `processors/janus_pro.py` |
| **NVILA** / **JetVLM** | 图像 | `models/nvila.py` |
| **NanoNemotronVL** | 图像 | `models/nano_nemotron_vl.py` |
| **Sarashina2Vision** | 图像 | `models/sarashina2_vision.py` |
| **CLIP** | 图像 (Embedding) | `models/clip.py` |
| **DotSVLM** | 图像 | `models/dots_vlm.py` |
| **PaddleOCR-VL** | 图像 | `processors/paddleocr_vlm.py` |

## 9. 优化策略

### 9.1 图像缓存

```bash
# 启用多模态缓存
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-7B \
    --mm-cache-size 2048  # MB
```

### 9.2 VIT CUDA Graph

```bash
# 对于 Triton/FA3 backend
export SGLANG_VIT_ENABLE_CUDA_GRAPH=1
```

### 9.3 EVS (Efficient Video Sampling)

EVS（[arXiv:2510.14624](https://arxiv.org/abs/2510.14624)）是一种视频 token 裁剪优化，在 ViT 编码完成后，通过计算相邻帧之间的相似度，裁剪冗余视频帧的 token。与传统的帧采样（在 ViT 之前丢弃帧）不同，EVS 在 ViT 编码之后操作，保留了视觉信息的同时减少了 LLM 需要处理的 token 数量。

```python
# python/sglang/srt/multimodal/evs/evs_module.py
@dataclass(kw_only=True)
class EVSEmbeddingResult(EmbeddingResult):
    """ViT 编码后的裁剪结果，包含每帧保留的 token 数。"""
    num_tokens_per_frame: list[int]
    # 例如 [256, 180, 195, 256]: frame 0 保留全部 256 tokens，
    # frame 1-2 因与前帧相似被裁剪
```

关键组件：
- `compute_retention_mask`（`evs_core.py`）：计算每帧的保留掩码
- `replace_offsets_with_tokens_per_frame`（`evs_core.py`）：根据裁剪结果重新分配 placeholder
- `EVS` 基类（`evs_module.py`）：模型继承此类并实现 `create_evs_config()`，当 `video_pruning_rate > 0` 时自动替换 `get_video_feature()` 为 EVS 裁剪版本

## 10. 调试技巧

### 10.1 查看图像处理

```python
# 打印 grid_thw
processor = QwenVLImageProcessor(...)
result = processor.process_mm_data(text, images=[img])
print(f"grid_thw: {result['image_grid_thw']}")
print(f"pixel_values shape: {result['pixel_values'].shape}")
```

### 10.2 检查缓存命中

```python
# 设置日志级别
export SGLANG_LOG_LEVEL=debug

# 查看日志中的缓存命中信息
# [DEBUG] MultimodalCache hit for hash=12345678
```

### 10.3 禁用缓存 (调试)

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-7B \
    --mm-cache-size 0  # 禁用缓存
```

---

## 11. 多模态更新

Qwen3.5 继承 Qwen3VL 的 Vision 架构，但 text backbone 是混合架构（Full Attention + Linear Attention + MoE）。多模态处理流程与 Qwen3-VL 基本一致，主要变化在 text backbone 的推理路径。

## 与其他章节关系
- 扩展 `01/03/08` 到多模态。

## 最小可验证实验
- 固定模型和负载，仅切换本章机制开关。
- 记录 TTFT、TPOT、吞吐、显存峰值与回退率。
- 总结收益场景、退化场景、推荐默认值。

## 常见误解
- 多模态仅是输入前处理问题。

