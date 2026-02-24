# 09. 模型加载 (Model Loading)

## 1. 概述：模型加载要解决什么问题

模型加载系统解决三个核心任务：

1. **发现 (Discovery)**：从 HF Config 的 `architectures` 字段找到对应的 SGLang 模型类，定位权重文件
2. **构造 (Construction)**：在 GPU 上创建空的 `nn.Module` 结构（参数已分配但未初始化值）
3. **物化 (Materialization)**：从磁盘流式加载权重，逐张量写入 GPU 参数

整体数据流：

```
                         ┌──────────────────────────────┐
                         │  config.json (HF)            │
                         │  architectures → 模型类       │
                         └──────────┬───────────────────┘
                                    │ get_model_architecture()
                                    ▼
                         ┌──────────────────────────────┐
                         │  model_class(**kwargs)        │
                         │  with target_device:          │
                         │    → GPU 上空参数              │
                         └──────────┬───────────────────┘
                                    │
    ┌───────────────────┐           │ model.load_weights(weights)
    │  safetensors 文件  │           │
    │  (磁盘)           ├──yield──▶ │  for name, tensor in weights:
    │                   │  (CPU)    │    param.data.copy_(tensor)
    └───────────────────┘           │    → CPU→GPU DMA
                                    ▼
                         ┌──────────────────────────────┐
                         │  GPU 上完整模型               │
                         │  + 量化后处理                 │
                         └──────────────────────────────┘
```

关键设计选择：
- **不经过** `from_pretrained()`，完全绕过 Transformers 的加载路径
- **流式逐张量**加载，不需要先全量加载到 CPU 再拷贝
- 峰值 CPU 内存 ≈ 一个 shard 文件大小（mmap 模式下更低）

---

## 2. 调用链全景 (Q1: 从服务启动到权重就位，经过哪些类/函数？)

### 2.1 完整调用链

从调度器到加载器的实际调用链（核心节点）：

| 层级 | 调用 | 位置 | 职责 |
|------|------|------|------|
| 1 | `Scheduler.init_model_worker()` | `scheduler.py:468` | 创建 Worker 进程/线程 |
| 2 | `TpModelWorker.__init__()` | `tp_worker.py:208` | 构造 `ModelConfig`，持有一个或多个 `ModelRunner`（MTP/EAGLE 会有列表） |
| 3 | `ModelRunner.__init__()` | `model_runner.py:261` | 保存配置，不做重活 |
| 4 | `ModelRunner.init_torch_distributed()` | `model_runner.py:693` | 初始化 TP/PP/EP 分布式通信组 |
| 5 | `initialize_model_parallel(...)` | `parallel_state.py:1573` | 建立 TP/PP/EP 组——**必须在 load_model 之前**，因为并行层构造时需要 rank/size |
| 6 | `ModelRunner.load_model()` | `model_runner.py:806` | 5 阶段加载（见 §2.2） |
| 7 | `get_model_loader(load_config)` | `loader.py:2657` | 按 `LoadFormat` 分发到具体加载器 |
| 8 | `loader.load_model(...)` | 如 `loader.py:616` | 构造模型 + 加载权重 + 量化后处理 |

### 2.2 `ModelRunner.load_model()` 的 5 个阶段

`model_runner.py:806`，约 185 行，远不止"调 loader"那么简单：

**Stage A: 前置检查 (L806-825)**
- 记录加载前可用 GPU 内存
- `torch.set_num_threads(1)` — 限制 PyTorch CPU 计算线程数（BLAS/OpenMP），避免与 I/O 线程竞争
- 检查 CUDA compute capability，低于 sm80 时自动降级 dtype 到 `float16`

> **线程数配置说明**：SGLang 中有两个不同的线程数设置：
> - `torch.set_num_threads(1)` (`model_runner.py:813-814`)：控制 PyTorch CPU 计算线程（BLAS/OpenMP），设为 1 防止多线程 BLAS 运算与加载 I/O 线程争抢 CPU
> - `DEFAULT_NUM_THREADS = 8` (`weight_utils.py:364-365`)：safetensors I/O 读取的 `ThreadPoolExecutor` 线程池大小，由 writer 进程 fork 出的 Python ThreadPool 负责并行读取多个 shard 文件

**Stage B: 构建 LoadConfig (L828-855)**
- 组装 `LoadConfig` dataclass，注入 `load_format`、`download_dir`、`tp_rank`、`draft_model_idx` 等
- CPU 设备场景下调整 unaligned CPU TP 配置

**Stage C: 获取加载器并执行 (L876-898)**
- `monkey_patch_vllm_parallel_state()` 兼容 vLLM 并行状态
- `memory_saver_adapter.region()` 包裹加载过程，可选 CPU 权重备份
- `get_model_loader(load_config)` → `loader.load_model(...)` 执行核心加载

**Stage D: 核心加载（在 loader 内部）**
- 见 §3 模型发现与构造、§4 权重物化

**Stage E: 后处理 (L900-991)**
- FP8 KV cache scale 加载
- 滑动窗口大小检测
- 记录权重 GPU 内存占用
- **RoPE 缓存预扩展**（见下方详解）
- **屏障同步**：`dist.monitored_barrier(...)` 确保所有 TP rank 都完成加载（Mooncake 后端使用 `dist.barrier` 替代）

**Stage E 详解: RoPE 缓存预扩展** (`model_runner.py:967-973`)

模型加载完成后、CUDA Graph capture 之前，调用 `reserve_rope_cache_for_long_sequences()` (`utils/common.py:3679`) 预扩展所有 RoPE 层的 cos/sin cache：

```python
# utils/common.py:3679-3716
def reserve_rope_cache_for_long_sequences(model, server_args, model_config):
    SAFETY_FACTOR = envs.SGLANG_SPEC_EXPANSION_SAFETY_FACTOR.get()
    MARGIN = envs.SGLANG_ROPE_CACHE_SAFETY_MARGIN.get()
    ALIGN = envs.SGLANG_ROPE_CACHE_ALIGN.get()

    # 1) 估算基础上下文上界
    base_ctx = server_args.context_length or model_config.context_len or 2048

    # 2) 投机解码扩展
    steps = server_args.speculative_num_steps or 0
    draft = server_args.speculative_num_draft_tokens or 0
    reserve = base_ctx + steps * draft * SAFETY_FACTOR + MARGIN

    # 3) 对齐以减少重分配频率
    reserve = (reserve + ALIGN - 1) // ALIGN * ALIGN

    # 4) 递归扩展所有 RoPE 层
    def reserve_rope_cache_recursive(module):
        for child in module.children():
            if hasattr(child, "_ensure_cos_sin_cache_length"):
                child._ensure_cos_sin_cache_length(reserve - 1)
            else:
                reserve_rope_cache_recursive(child)
    reserve_rope_cache_recursive(model)
```

**为什么需要预扩展？** CUDA Graph capture 会固化所有 tensor 形状，运行时不能动态扩展。如果 cos_sin_cache 不够长，推理时遇到长序列会因形状不匹配而崩溃。

**`_ensure_cos_sin_cache_length` 的增量计算机制**：不重新计算全量 cache，而是只计算 `[cur_len, new_len)` 范围的 cos/sin 值，通过 `torch.cat` 追加到现有 `cos_sin_cache` 末尾。

### 2.3 关键配置类

**`LoadConfig`** (`load_config.py:37`)：加载配置 dataclass

核心字段：
- `load_format: LoadFormat` — 加载格式枚举
- `download_dir` — 权重缓存目录
- `model_loader_extra_config` — 多线程加载等附加参数
- `tp_rank` — 当前 TP rank（用于分片加载）
- `draft_model_idx` — MTP 草稿层索引

**`BaseModelLoader`** (`loader.py:280`)：加载器抽象基类

```python
class BaseModelLoader(ABC):
    def __init__(self, load_config: LoadConfig): ...
    def download_model(self, model_config: ModelConfig) -> None: ...
    def load_model(self, *, model_config, device_config) -> nn.Module: ...
```

---

## 3. 模型发现与构造 (Q2: SGLang 自定义了什么，直接用了 HF 什么？)

### 3.1 模型注册表 (Registry)

`python/sglang/srt/models/registry.py`

`ModelRegistry` 是单例 `@dataclass`，在导入时扫描 `sglang/srt/models/` 下所有 `.py` 文件。每个模型文件通过 `EntryClass` 变量注册：

```python
# 单个类 -- qwen3.py:587
EntryClass = Qwen3ForCausalLM

# 多个架构共享同一实现 -- llama.py:758
EntryClass = [LlamaForCausalLM, Phi3ForCausalLM, InternLM3ForCausalLM]
```

**关键约束**：类的 `__name__` 必须与 HF `config.json` 中 `"architectures"` 字段完全匹配，这是 Registry 的查找 key。

### 3.2 从 HF `architectures` 到 SGLang 类

`python/sglang/srt/model_loader/utils.py:84`

`get_model_architecture()` 的完整决策路径：

```
hf_config.architectures (如 ["Qwen3VLForConditionalGeneration"])
    │
    ├─ Registry 有原生实现 且 ModelImpl.AUTO → 直接使用原生类
    │
    ├─ Registry 无原生实现 或 ModelImpl.TRANSFORMERS
    │   └─ resolve_transformers_arch(): 检查 HF 模型的 is_backend_compatible()
    │       ├─ 兼容 → "TransformersForCausalLM" (桥接器)
    │       └─ 不兼容 → 报错
    │
    └─ ModelImpl.MINDSPORE → "MindSporeForCausalLM"
```

最终 `ModelRegistry.resolve_model_cls(architectures)` 查表返回 `(model_class, arch_name)`。如果所有架构都不匹配，会追加 `"TransformersForCausalLM"` 作为兜底。

### 3.3 SGLang vs HF 组件对比

| 组件 | SGLang 做法 | HF 做法 | 为什么 |
|------|------------|---------|--------|
| Config | 直接用 `AutoConfig` | `AutoConfig.from_pretrained()` | 配置解析与推理无关 |
| 模型类 | 原生实现 140+ 模型 | `nn.Module` 基类 | 推理热路径必须重写 |
| Attention | `RadixAttention` + Backend | HF attention impl | KV Cache 管理方式完全不同 |
| Linear | `QKV/Column/RowParallelLinear` | `nn.Linear` | TP 分片是内建的 |
| Tokenizer | 直接用 `AutoTokenizer` | `AutoTokenizer` | CPU 操作非瓶颈 |
| 视觉编码器 | 视情况而定 | `CLIPVisionModel` 等 | 需 TP 时重写，否则黑盒 |
| Processor | 直接用 `AutoProcessor` | `AutoProcessor` | 预处理非瓶颈 |
| 激活函数 | 复用 `ACT2FN` | `transformers.activations` | 简单注册表 |

**设计原则**：推理热路径（Attention, Linear, Embedding）必须重写；非热路径（Config, Tokenizer, Processor）直接复用 HF。

### 3.4 `_CONFIG_REGISTRY`: 自定义 HF 配置注册

`python/sglang/srt/utils/hf_transformers_utils.py:72-97`

对于 HuggingFace 尚未合入的新模型，SGLang 维护内部配置注册表 `_CONFIG_REGISTRY`（约 20 种模型类型），通过 `AutoConfig.register()` 注册到 HF 自动检测系统，使 `AutoConfig.from_pretrained()` 能正确识别它们。

### 3.5 `_initialize_model()` 详解

`python/sglang/srt/model_loader/loader.py:228`

这是一个关键的独立函数（非 Loader 方法），负责从配置到可加载权重的 `nn.Module` 的全部构造：

```python
def _initialize_model(model_config, load_config) -> nn.Module:
    model_class, _ = get_model_architecture(model_config)    # 1. 解析模型类
    packed_modules_mapping = getattr(model_class, "packed_modules_mapping", {})  # 2. fused 模块映射
    remap_prefix = getattr(model_class, "remap_prefix", None)  # 3. 前缀重映射
    quant_config = _get_quantization_config(                 # 4. 量化配置
        model_config, load_config, packed_modules_mapping, remap_prefix)
    hf_to_sglang_mapper = getattr(model_class, "hf_to_sglang_mapper", None)
    if hf_to_sglang_mapper and quant_config:                 # 5. 量化 key 映射
        quant_config.apply_sglang_mapper(hf_to_sglang_mapper)
    kwargs = {"config": model_config.hf_config, "quant_config": quant_config}
    return model_class(**kwargs)                              # 6. 构造模型实例
```

**为什么参数直接在 GPU 上？** 调用方使用 `with target_device:` 上下文：

```python
# loader.py:631-636
with set_default_torch_dtype(model_config.dtype):
    with target_device:                           # ← PyTorch device context
        model = _initialize_model(model_config, self.load_config)
```

在 `with target_device:` 内，所有 `nn.Parameter()` / `torch.empty()` 默认分配在 GPU 上（未初始化值）。这样设计的目的是**避免 CPU→GPU 全量拷贝**——参数壳子直接建在 GPU，后续 `load_weights()` 逐步 `param.data.copy_(loaded_weight)` 覆盖写入。

> **`DummyModelLoader` 特殊处理**：`DummyModelLoader` 用于测试/profiling 场景，在构造模型后额外调用 `initialize_dummy_weights(model)` 用 `[-1e-3, 1e-3]` 范围的随机值填充所有参数，跳过真正的权重加载。

---

## 4. 权重物化：从磁盘文件到 GPU 张量 (Q3: 权重如何从 safetensors 变成 GPU 张量？)

这是模型加载的核心环节，回答"权重到底是怎么从磁盘到 GPU 的"这个关键问题。

### 4.1 整体流程概览

```
safetensors 文件 (磁盘)
    │
    │ safetensors.safe_open(device="cpu") + mmap
    ▼
单个 tensor (CPU 内存)
    │
    │ yield name, tensor          ← Python generator，逐张量产出
    ▼
model.load_weights() 循环消费
    │
    │ param.data.copy_(tensor)    ← CPU→GPU DMA 传输
    ▼
GPU 参数 (已初始化)
```

**关键结论**（先说答案）：
- **不需要**先全量加载到 CPU 再拷贝，是**流式逐张量**加载
- 峰值 CPU 内存 = 一个 shard 文件大小（mmap 模式下按需加载，更低）
- **不经过** Transformers 的 `from_pretrained`，完全绕过
- 每个张量的生命周期：磁盘 → CPU（短暂） → GPU → CPU 张量被 GC 回收

### 4.2 权重迭代器详解

SGLang 提供 4 种权重迭代器，都是 Python generator，yield `(name, tensor)` 对：

| 迭代器 | yield 设备 | 并行方式 | 适用场景 |
|--------|-----------|---------|---------|
| `safetensors_weights_iterator` | CPU | 串行逐文件 | 默认路径 |
| `multi_thread_safetensors_weights_iterator` | CPU | 多线程(默认8) | 大模型加速 |
| `fastsafetensors_weights_iterator` | **GPU** | GPU Direct Storage | 有 GDS 硬件时 |
| `pt_weights_iterator` | CPU | 串行 | `.pt/.bin` 格式 |

迭代器选择逻辑在 `_get_weights_iterator()` (`loader.py:459`)：

```
safetensors 文件？
    ├─ LoadFormat.FASTSAFETENSORS → fastsafetensors_weights_iterator (GPU Direct)
    ├─ enable_multithread_load → multi_thread_safetensors_weights_iterator
    └─ 默认 → safetensors_weights_iterator
.pt/.bin 文件？
    ├─ enable_multithread_load → multi_thread_pt_weights_iterator
    └─ 默认 → pt_weights_iterator
```

**`safetensors_weights_iterator` 的 yield 机制**（`weight_utils.py:678`）：

```python
def safetensors_weights_iterator(hf_weights_files, ..., disable_mmap=False):
    for st_file in tqdm(hf_weights_files, ...):
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())  # 全量读入内存
                for name, param in result.items():
                    yield name, param
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)  # mmap: 按需读取
```

流式加载的本质：
- `safe_open(device="cpu")` 使用 **mmap** 打开文件，不立即占用物理内存
- `f.get_tensor(name)` 触发实际磁盘 I/O，返回**一个** CPU 张量
- Python generator 的 `yield` 暂停执行，消费端取一个才读一个
- 消费端处理完后，CPU 张量失去引用，被 GC 回收——内存占用始终是"一个张量"级别

**`weight_loader_disable_mmap` 机制**：
- **mmap 开启（默认）**：通过 `safetensors.safe_open(device="cpu")` 使用 OS 内存映射，未访问部分不占物理内存，多 rank 可共享页面缓存
- **mmap 关闭**：通过 `disable_mmap=True`，使用 `open(st_file, "rb")` + `f.read()` 全量读入内存
- mmap 优势：未访问的 shard 部分不占物理 RAM；同一节点上多个 rank 进程读同一文件时，OS 自动共享页面缓存，避免重复 I/O

### 4.3 `model.load_weights()` 消费端

以 Qwen3 为例（`qwen3.py:485`），`load_weights` 使用 Python 独特的 **for-else** 模式：

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(self.named_parameters())

    for name, loaded_weight in weights:         # ← 消费 generator
        # ... PP 层过滤、skip 逻辑 ...

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            param.weight_loader(param, loaded_weight, shard_id)   # 融合参数路径
            break
        else:
            # 没匹配到 stacked 规则，走普通路径
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)                    # 默认路径
```

语义：
- **匹配到 stacked 规则**（如 `q_proj` → `qkv_proj`）：走并行层的 `weight_loader`，按 `shard_id` 写入融合参数的对应切片
- **没匹配到**（for 循环正常结束，执行 else）：走 `default_weight_loader` 或参数自带的 loader

### 4.4 `default_weight_loader` — CPU→GPU 的最终一步

`weight_utils.py:937`

```python
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            param.data.fill_(loaded_weight.item())   # 标量广播
        else:
            assert param.size() == loaded_weight.size()
            param.data.copy_(loaded_weight)           # ← 关键一行
    except Exception:
        raise
```

`param.data.copy_(loaded_weight)` 这一行完成了 CPU→GPU 传输：
- `param` 在 GPU 上（§3.5 的 `with target_device:` 分配）
- `loaded_weight` 在 CPU 上（§4.2 的迭代器 yield）
- PyTorch 的 `copy_()` 检测到设备不同，自动触发 **CPU→GPU DMA 传输**
- 这是 `cudaMemcpy(H2D)` 的 Python 封装，是同步操作

### 4.5 `WeightsMapper` 名称映射

`python/sglang/srt/models/utils.py:40`

在迭代器和 `load_weights` 之间，部分模型（主要是 Qwen VL 系列）通过 `WeightsMapper` 做 HF → SGLang 命名转换：

```python
hf_to_sglang_mapper = WeightsMapper(
    orig_to_new_substr={"attn.qkv": "attn.qkv_proj"},
    orig_to_new_prefix={
        "model.visual.": "visual.",
        "model.language_model.": "language_model.model.",
    },
)
```

`WeightsMapper` 支持三类映射：`orig_to_new_substr`、`orig_to_new_prefix`、`orig_to_new_suffix`，每类按 key 长度降序排列，优先最长匹配。映射值为 `None` 表示丢弃该权重。

映射应用时机：在 `DefaultModelLoader.load_model()` 中，通过 `_get_all_weights()` 返回的迭代器已经应用了 `source.prefix`，而 `WeightsMapper` 则在模型的 `load_weights()` 内部消费权重时应用。

### 4.6 `_get_all_weights` — primary + secondary

`loader.py:529`

```python
def _get_all_weights(self, model_config, model):
    primary_weights = DefaultModelLoader.Source.init_new(model_config, model)
    yield from self._get_weights_iterator(primary_weights)

    secondary_weights = getattr(model, "secondary_weights", ())
    for source in secondary_weights:
        yield from self._get_weights_iterator(source)
```

`secondary_weights` 机制用于需要从**额外来源**加载权重的场景（如附属模块、额外编码器等），但绝大多数模型只使用 primary 路径。

> **Python Generator 语义说明**：
> - **`yield name, tensor`**：将函数变为 generator，惰性执行——调用者 `next()` 一次才执行到下一个 `yield`，实现"消费一个读一个"的流式加载
> - **`yield from iterable`**：委托给子 generator，等价于 `for item in iterable: yield item`，但更高效。`_get_all_weights` 使用 `yield from` 将 `_get_weights_iterator` 的输出透传给消费者
> - **流式设计意义**：整个权重加载链（`_get_all_weights` → `_get_weights_iterator` → `safetensors_weights_iterator`）都是 generator 组成的管道。每个 tensor 从磁盘读取后立刻被消费端（`load_weights`）处理并 `copy_` 到 GPU，CPU 端只驻留一个 tensor 的内存

### 4.7 `stacked_params_mapping` vs `packed_modules_mapping`

这两个容易混淆的映射服务于不同目的：

| 映射 | 定义位置 | 消费者 | 用途 |
|------|---------|--------|------|
| `stacked_params_mapping` | `load_weights()` 方法内 | `load_weights` 本身 | 决定 checkpoint 权重如何路由到融合参数 |
| `packed_modules_mapping` | 模型类属性 | 量化系统 | 告诉量化配置哪些权重是融合的，做 scale 对齐 |

例如 `stacked_params_mapping` 中 `("qkv_proj", "q_proj", "q")` 告诉 `load_weights`：当遇到名称含 `q_proj` 的权重时，重命名为 `qkv_proj`，以 shard_id `"q"` 调用 `QKVParallelLinear.weight_loader`。

---

## 5. 量化加载 (Q4: 预量化权重 vs 在线量化，加载路径有何不同？)

### 5.1 两种场景对比

| 场景 | 权重文件内容 | 加载时 | 后处理时 |
|------|------------|-------|---------|
| 预量化 (GPTQ/AWQ/W8A8) | 已打包的 int/fp8 权重 + scale | 直接加载打包格式 | Marlin repacking、转置等 |
| 在线量化 (FP8 dynamic) | 正常 float 权重 | 正常加载 float | `per_channel_quant_fp8()` 等 |

### 5.2 量化配置检测

`_get_quantization_config()` (`loader.py:190`):

1. 从 `model_config.quantization` 读取量化方法名
2. `get_quant_config()` (`weight_utils.py:168`) 从 `hf_config.quantization_config` 读取详细配置
3. 检查设备 compute capability 是否满足量化方法的最低要求
4. 检查 dtype 兼容性

视觉模型会 fallback 到 `text_config.quantization_config`，确保在多模态模型中正确检测量化配置。

### 5.3 `process_weights_after_loading` 后处理

触发点在 `DefaultModelLoader.load_weights_and_postprocess()` (`loader.py:644`):

```python
model.load_weights(weights)                           # 先加载权重
for _, module in model.named_modules():
    quant_method = getattr(module, "quant_method", None)
    if quant_method is not None:
        with device_loading_context(module, target_device):  # CPU offload 时临时移到 GPU
            quant_method.process_weights_after_loading(module)
```

典型后处理动作：

| 方法 | `process_weights_after_loading()` 典型动作 |
|------|------------------------------------------|
| W8A8 FP8 | checkpoint 已 FP8 → 转置权重并设 scale；否则在线量化到 FP8 |
| GPTQ | 规范化 qweight/qzeros/scales/g_idx；可选 `gptq_shuffle` |
| GPTQ Marlin | `gptq_marlin_repack` + `marlin_permute_scales` + workspace |
| AWQ | 基础路径冻结参数；Marlin 路径做 `awq_marlin_repack` |
| BitsAndBytes | 4/8bit 状态解析；非预量化时在线 `quantize_4bit(quant_type="nf4")` |
| Unquant | 大多数场景 no-op；CPU AMX 做权重打包 |

`device_loading_context` 的作用：在 CPU offload 场景下，临时将参数移到 GPU 做量化后处理（如 Marlin repacking 需要 GPU），完成后移回 CPU。

### 5.4 BitsAndBytes 特殊路径

BitsAndBytes 有专门的 `BitsAndBytesModelLoader` (`loader.py:1562`)，两个关键分支：
- **预量化分支**：checkpoint 中已有 quant state（含 nf4/fp4），直接解析
- **非预量化分支**：先按 TP 切片，再调用 `bitsandbytes` 4bit 量化

---

## 6. 多卡并行与权重分片 (Q5: 每个 rank 加载全量还是分片？)

### 6.1 核心问题：每个 rank 加载全量文件还是分片文件？

答案：**每个 rank 读取全量文件，但只取自己的切片**。

```
Rank 0: safetensors 全量文件 → 迭代器 yield → weight_loader 取 [0:shard_size] → GPU 0
Rank 1: safetensors 全量文件 → 迭代器 yield → weight_loader 取 [shard_size:2*shard_size] → GPU 1
...
```

为什么不预分片？
- safetensors **mmap** 使得未访问的部分不占物理内存
- 分片逻辑在**并行层的 `weight_loader`** 中，由每个层自己决定切片方式
- 这样简化了加载器设计——不需要预计算哪些 rank 需要哪些文件

### 6.2 各并行层的分片方式

| 层 | 分片维度 | 关键逻辑 |
|----|---------|---------|
| `ColumnParallelLinear` | 输出维 | `loaded_weight.narrow(output_dim, tp_rank * shard_size, shard_size)` |
| `RowParallelLinear` | 输入维 | `loaded_weight.narrow(input_dim, tp_rank * shard_size, shard_size)` |
| `QKVParallelLinear` | Q:按 head; KV:可复制 | `num_kv_head_replicas` 处理 GQA |
| `MergedColumnParallel` | 每 shard 输出维 | gate_up_proj: 先按 shard_id 偏移再按 tp_rank |
| `VocabParallelEmbedding` | 词表维 | mask + all_reduce |
| `FusedMoE` | EP+MoE-TP | 双层分片 |

**`ColumnParallelLinear.weight_loader`** (`linear.py:363`)：

```python
def weight_loader(self, param, loaded_weight):
    output_dim = getattr(param, "output_dim", None)
    if output_dim is not None:
        shard_size = param.data.shape[output_dim]
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
    # 最终走 default_weight_loader 的 copy_
```

**`QKVParallelLinear` 与 GQA 复制**：

当 `tp_size > total_num_kv_heads` 时（如 tp=16, kv_heads=8）：
- `num_kv_head_replicas = tp_size / total_num_kv_heads = 2`
- Q 分片用 `shard_id = tp_rank`
- K/V 分片用 `shard_id = tp_rank // num_kv_head_replicas`
- 每 2 个 rank 共享一份 K/V 分片

**`MergedColumnParallelLinear`** (以 `gate_up_proj` 为例, `linear.py:523`):

`output_sizes=[intermediate, intermediate]` 时：
- `loaded_shard_id=0`（gate）：`shard_offset = 0`, `shard_size = intermediate / tp_size`
- `loaded_shard_id=1`（up）：`shard_offset = intermediate / tp_size`, `shard_size = intermediate / tp_size`
- 再加上 `tp_rank` 偏移取本卡分片

**MoE: EP + MoE-TP 双层分片**

MoE 并行将 TP 组分解为两级：`tp_size = moe_ep_size * moe_tp_size`。

- EP 过滤：`_map_global_expert_id_to_local_expert_id(...)` 跳过不属于本 EP rank 的 expert。共享专家不参与 EP 分片，所有 rank 完整持有。
- MoE-TP 分片：`w13`(gate/up) 按 `shard_size * moe_tp_rank` 切片，`w2`(down) 按输入维切片。

### 6.3 Pipeline Parallel

**层分配**：`make_layers()` (`utils/common.py:573`) 计算 `start_layer`/`end_layer`，非本 rank 的层用 `PPMissingLayer` 占位。

**加载时过滤**（`qwen3.py:499`）：
```python
layer_id = get_layer_id(name)
if layer_id < self.model.start_layer or layer_id >= self.model.end_layer:
    continue  # 跳过非本 PP rank 的层
```

**`tie_word_embeddings` 跨 PP 处理**（`qwen3.py:516-525`）：
- 首 rank：从 weights 中找到 `embed_tokens.weight`
- 末 rank：用 `embed_tokens.weight` 填充 `lm_head.weight`

### 6.4 `monkey_patch_vllm_parallel_state`

`model_runner.py:877`

临时替换 vLLM 的并行状态函数为 SGLang 的实现，确保 vLLM 的权重加载代码（SGLang 的 linear.py 等适配自 vLLM）尊重 SGLang 的 TP 配置。加载完成后 `reverse=True` 恢复。

---

## 7. MTP 与投机采样 (Q6: MTP 权重过滤是怎么回事？)

### 7.1 多 ModelRunner 架构

Multi-Token Prediction (MTP) 使用多个草稿层，每个层创建独立的 `ModelRunner`：

```python
# tp_worker.py:271-293
for i in range(server_args.speculative_num_draft_tokens):
    runner = ModelRunner(draft_model_idx=i, ...)
    self.model_runner_list.append(runner)
```

### 7.2 权重过滤机制

`loader.py:509-524`

checkpoint 中 MTP 权重的命名格式：`model.mtp.layers.<idx>.xxx`

当 `load_config.draft_model_idx` 非空时，加载器过滤逻辑：

```python
pattern = r"model.mtp.layers.(\d+)."
for name, tensor in weights_iterator:
    group = re.match(pattern, name)
    if group is not None:
        idx = int(group.group(1))
        if idx != self.load_config.draft_model_idx:
            continue                                    # 跳过其他层
        new_name = name.replace(group.group(), "model.mtp.layers.0.")  # 重命名为 layers.0
    else:
        new_name = name
    filtered_weights.append((new_name, tensor))
```

### 7.3 为什么这么设计？

- 每个 draft 层**结构相同但权重不同**（包含 eh_proj + MLA attention + transformer layer + norms）
- 独立 `ModelRunner` 允许独立的 CUDA Graph capture
- 共享 KV Cache pool（通过 `req_to_token_pool` 参数传递）
- 所有层重命名为 `layers.0.` 是因为每个 ModelRunner 内部只需要一层结构

---

## 8. 多模态场景 (Q7: 视觉编码器如何加载？)

### 8.1 视觉编码器加载路径

视觉编码器是模型的子模块，在**同一个 `load_weights()` 中一起加载**，不走单独的加载路径。

```
checkpoint 文件:
  model.layers.0.self_attn.q_proj.weight    → 语言模型层
  model.layers.0.mlp.gate_proj.weight       → 语言模型层
  visual.blocks.0.attn.qkv_proj.weight      → 视觉编码器   ← 同一个迭代器
  visual.blocks.0.mlp.fc1.weight            → 视觉编码器
```

### 8.2 `hf_to_sglang_mapper` 名称映射

Qwen VL 系列（Qwen2-VL、Qwen2.5-VL、Qwen3-VL）需要做 HF → SGLang 命名转换，因为 HF 和 SGLang 的模块层次不同：

```python
# qwen2_vl.py:430
hf_to_sglang_mapper = WeightsMapper(
    orig_to_new_substr={"attn.qkv": "attn.qkv_proj"},
    orig_to_new_prefix={
        "model.language_model.": "language_model.model.",
        "model.visual.": "visual.",
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    },
)
```

`quant_config.apply_sglang_mapper(hf_to_sglang_mapper)` 确保量化 scale 的 key 也被同步映射。

### 8.3 视觉编码器两套实现

SGLang 同时维护了两套 CLIP/SigLIP 实现：

| 版本 | 来源 | 特点 | 使用场景 |
|------|------|------|---------|
| HF 原版 | `from transformers import CLIPVisionModel` | 不支持 TP | llava、nvila 等（单卡/不需视觉 TP） |
| SGLang 重写版 | `sglang/srt/models/clip.py` | `ColumnParallelLinear`/`RowParallelLinear` 替换 `nn.Linear` | gemma3_mm 等需要视觉 TP 时 |

选择逻辑：需要视觉编码器 TP 分片时用 SGLang 版本，否则用 HF 黑盒版本。

### 8.4 Processor 直接复用 HF

`python/sglang/srt/utils/hf_transformers_utils.py:514`

多模态模型的 Processor 通过 `AutoProcessor.from_pretrained(...)` 直接复用 HF，少数模型有特殊处理（如 Qwen2-VL 注入默认 `size`）。

---

## 9. 特殊加载器速查

`get_model_loader(...)` 入口：`loader.py:2657`

| 加载器 | LoadFormat | 核心差异 | 适用场景 |
|-------|-----------|---------|---------|
| `DefaultModelLoader` | AUTO/SAFETENSORS/PT/FASTSAFETENSORS | 标准流式加载 | 绝大多数模型 |
| `LayeredModelLoader` | LAYERED | meta 设备→逐层物化→逐层量化，降峰值内存 | 大模型+在线量化 |
| `BitsAndBytesModelLoader` | BITSANDBYTES | 4/8bit 专用路径 | BNB 量化 |
| `GGUFModelLoader` | GGUF | GGUF 格式解析 | GGUF 权重 |
| `ShardedStateLoader` | SHARDED_STATE | 按 rank 读预分片文件 | 预分片 checkpoint |
| `QuantizedRLModelLoader` | FLASH_RL | 首轮加载+量化，后续快速重绑重载 | RL 训练场景 |
| `RemoteModelLoader` | REMOTE | 远端存储加载 | 远端权重 |
| `RemoteInstanceModelLoader` | REMOTE_INSTANCE | NCCL/传输引擎加载 | 远端实例 |
| `DummyModelLoader` | DUMMY | 随机权重 | 测试和 profiling |

### `LayeredModelLoader` 三步机制

`loader.py:662`

与 `DefaultModelLoader` 的区别在于**逐层物化**，降低峰值内存：

```python
def load_model(self, *, model_config, device_config):
    # Step 1: 在 meta 设备上构造模型（无内存开销）
    with torch.device("meta"):
        model = _initialize_model(model_config, self.load_config)

    # Step 2: 递归逐层物化
    def fill_module(module, fqn, weights):
        for name, submod in module.named_children():
            fill_module(submod, fqn + [name], weights)
        module.to_empty(device=target_device, recurse=False)  # meta → GPU
        model.load_weights_to_module(fqn_path, weights)        # 填权重
        # 可选: 逐层量化

    fill_module(model, [], weights)
```

这样在任一时刻只有一层的参数在 GPU 上未量化，量化后的参数占用更少，峰值内存更低。

---

## 10. 与 Transformers 的关系总结

### 10.1 对比表

| 维度 | HuggingFace | SGLang 原生 |
|------|------------|------------|
| 基类 | `PreTrainedModel` | `nn.Module` |
| Forward 签名 | `forward(input_ids, attention_mask, position_ids, past_key_values, ...)` | `forward(input_ids, positions, forward_batch)` |
| KV Cache | 显式 `past_key_values` 传入传出 | 隐式管理: `ForwardBatch.attn_backend` + memory pool |
| 输入形状 | `[B, S]` (batch, seq_len) | 扁平 token 流 `(total_tokens,)` |
| TP 支持 | 依赖外部并行工具 | 原生并行层与并行组 |
| 输出 | `CausalLMOutputWithPast` | `LogitsProcessorOutput` |
| 线性层返回 | 单 tensor | `(output, output_bias)` tuple |

### 10.2 `TransformersForCausalLM` 桥接器

`python/sglang/srt/models/transformers.py:141`

让任何支持 `is_backend_compatible()` 的 HF 模型在 SGLang 中运行：

1. `AutoModel.from_config(config, attn_implementation="sglang")`（注意是 `from_config` 而非 `from_pretrained`）
2. 注册自定义注意力：`ALL_ATTENTION_FUNCTIONS["sglang"] = sglang_flash_attention_forward`，桥接 HF 4D 格式与 SGLang 2D 格式
3. `tensor_parallel(tp_size)` 读取 HF 模型的 `base_model_tp_plan`，替换 `nn.Linear` 为 SGLang 并行层
4. 输入 embedding 替换为 `VocabParallelEmbedding`
5. 输出头创建 `ParallelLMHead` + `LogitsProcessor`

**关键约束**：若 HF 模型未定义 `base_model_tp_plan` 且 `tp_size > 1`，直接报错。必须通过 `is_backend_compatible()` 检查。

**HF 桥接模型可享受的 SGLang 优化能力**：

| 优化特性 | 可用性 | 说明 |
|---------|--------|------|
| RadixAttention | 可用 | `attn_implementation="sglang"` 替换 HF 原生 attention，桥接为 `RadixAttention` |
| KV Cache (Radix Cache) | 可用 | 共享 SGLang 的前缀缓存机制 |
| TP 并行 | 部分可用 | 需要 HF 模型定义 `base_model_tp_plan`，否则 `tp_size > 1` 直接报错 |
| 调度 (Scheduler) | 可用 | 连续批处理、overlap schedule 等全部可用 |
| Fused MLP Kernels | 不可用 | 不使用 SGLang 的 fused kernels（gate_up_proj 等） |
| 量化 | 受限 | 依赖 HF 自身的量化支持，不使用 SGLang 的量化方法 |

### 10.3 设计原则

| 原则 | 说明 |
|------|------|
| **推理热路径必须重写** | Attention、Linear、Embedding — KV Cache 管理和 TP 分片是内建的 |
| **非热路径直接复用 HF** | Config、Tokenizer、Processor、简单投影层、激活函数 |
| **视觉编码器是中间地带** | 单卡/小模型用 HF 黑盒；需 TP 时用 SGLang 重写版 |
| **TransformersForCausalLM 桥解决长尾** | 140+ 原生实现覆盖主流模型，桥让新模型开箱即用 |
