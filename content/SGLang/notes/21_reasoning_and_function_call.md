# SGLang 推理解析与函数调用详解

> **默认场景**: Qwen/Qwen3-VL-235B-A22B-Thinking 多模态模型
>
> **核心组件**: ReasoningParser, FunctionCallParser, BaseFormatDetector, StructuralTag

## 1. 概览

SGLang 提供两个独立但可协作的解析系统，处理思维链 (Chain-of-Thought) 推理和函数/工具调用：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          双解析系统                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  模型输出流                                                                   │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────┐                                             │
│  │    ReasoningParser          │                                             │
│  │    (推理内容分离)            │                                             │
│  │                             │                                             │
│  │  输入: "<think>思考</think>回答"                                          │
│  │  输出: reasoning_text + normal_text                                       │
│  └──────────────┬──────────────┘                                             │
│                 │ normal_text                                                 │
│                 ▼                                                             │
│  ┌─────────────────────────────┐                                             │
│  │    FunctionCallParser       │                                             │
│  │    (工具调用解析)            │                                             │
│  │                             │                                             │
│  │  输入: "<tool_call>{"name":"get_weather",...}</tool_call>"                │
│  │  输出: normal_text + List[ToolCallItem]                                  │
│  └─────────────────────────────┘                                             │
│                                                                              │
│  两者独立工作: 推理解析先提取 thinking 部分，                                  │
│  剩余 normal_text 再交给函数调用解析                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**核心文件**:
- `srt/parser/reasoning_parser.py` — 推理内容解析器
- `srt/function_call/function_call_parser.py` — 函数调用解析器
- `srt/function_call/base_format_detector.py` — 函数调用检测器基类
- `srt/function_call/core_types.py` — 核心数据类型
- `srt/entrypoints/openai/serving_chat.py` — serving 层集成
- `srt/constrained/reasoner_grammar_backend.py` — Reasoner Grammar 包装器

## 2. ReasoningParser 架构

**文件**: `srt/parser/reasoning_parser.py`

### 2.1 类体系

```
ReasoningParser (入口类)
    │
    └── DetectorMap → BaseReasoningFormatDetector (基类)
                          │
                          ├── DeepSeekR1Detector     # <think>...</think>, force_reasoning=True
                          ├── Qwen3Detector          # <think>...</think>
                          ├── KimiDetector           # ◁think▷...◁/think▷
                          ├── GptOssDetector          # <|channel|>analysis<|message|>...<|end|>
                          ├── MiniMaxAppendThinkDetector  # 在输出前追加 <think>
                          └── NanoV3Detector          # <think>...</think>
```

### 2.2 模型注册表

```python
ReasoningParser.DetectorMap = {
    "deepseek-r1":          DeepSeekR1Detector,    # force_reasoning=True (始终假定在思考)
    "deepseek-v3":          Qwen3Detector,
    "glm45":                Qwen3Detector,
    "gpt-oss":              GptOssDetector,        # 使用 HarmonyParser
    "kimi":                 KimiDetector,          # Unicode 标记符
    "kimi_k2":              DeepSeekR1Detector,
    "qwen3":                Qwen3Detector,
    "qwen3-thinking":       Qwen3Detector,         # force_reasoning=True (强制覆盖)
    "minimax":              Qwen3Detector,          # force_reasoning=True (强制覆盖)
    "minimax-append-think": MiniMaxAppendThinkDetector,
    "step3":                DeepSeekR1Detector,
    "nano_v3":              NanoV3Detector,
    "interns1":             Qwen3Detector,
}
```

### 2.3 BaseReasoningFormatDetector 基类

```python
class BaseReasoningFormatDetector:
    def __init__(self, think_start_token, think_end_token, force_reasoning, stream_reasoning):
        self.think_start_token = think_start_token    # 如 "<think>"
        self.think_end_token = think_end_token        # 如 "</think>"
        self._in_reasoning = force_reasoning          # 初始状态
        self.stream_reasoning = stream_reasoning      # 是否流式输出推理内容
        self._buffer = ""                             # 流式缓冲区

    def detect_and_parse(self, text) -> StreamingParseResult:
        """一次性解析: 完整文本 → reasoning_text + normal_text"""

    def parse_streaming_increment(self, new_text) -> StreamingParseResult:
        """流式增量解析: 逐块输入 → 逐块输出"""
```

### 2.4 流式增量解析状态机

`parse_streaming_increment()` 实现了一个简洁的状态机：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   parse_streaming_increment 状态机                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  new_text 到达                                                               │
│      │                                                                       │
│      ▼                                                                       │
│  _buffer += new_text                                                        │
│      │                                                                       │
│      ├── buffer 是 start/end token 前缀? ──→ 继续缓冲 (返回空)             │
│      │                                                                       │
│      ├── 包含 think_start_token?                                            │
│      │   └── 剥离标记, _in_reasoning = True                                │
│      │                                                                       │
│      ├── _in_reasoning 且包含 think_end_token?                              │
│      │   ├── 切分: reasoning_text | normal_text                            │
│      │   ├── _in_reasoning = False                                         │
│      │   └── 清空 _buffer                                                  │
│      │                                                                       │
│      ├── _in_reasoning 且 stream_reasoning?                                │
│      │   └── 立即输出 reasoning_text, 清空 _buffer                         │
│      │                                                                       │
│      ├── _in_reasoning 且 !stream_reasoning?                               │
│      │   └── 继续缓冲 (返回空)                                             │
│      │                                                                       │
│      └── !_in_reasoning?                                                    │
│          └── 输出 normal_text, 清空 _buffer                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 StreamingParseResult

```python
class StreamingParseResult:
    normal_text: str = ""       # 非推理内容 (会传给用户/函数调用解析)
    reasoning_text: str = ""    # 推理内容 (thinking 部分)
```

### 2.6 特殊检测器

**DeepSeekR1Detector**: `force_reasoning=True`，模型一开始就被视为在思考，直到遇到 `</think>` 才切换。

**GptOssDetector**: 使用独立的 `HarmonyParser` 解析器处理 T4 格式 (`<|channel|>analysis<|message|>...<|end|>`)，支持多事件类型 (reasoning, normal, tool_call)。

**KimiDetector**: 使用 Unicode 字符作为标记 (`◁think▷` / `◁/think▷`)。

## 3. FunctionCallParser 架构

**文件**: `srt/function_call/function_call_parser.py`

### 3.1 类体系

```
FunctionCallParser (入口类)
    │
    └── ToolCallParserEnum → BaseFormatDetector (基类)
                                 │
                                 ├── DeepSeekV3Detector     # deepseekv3
                                 ├── DeepSeekV31Detector    # deepseekv31
                                 ├── DeepSeekV32Detector    # deepseekv32
                                 ├── Glm4MoeDetector        # glm, glm45
                                 ├── Glm47MoeDetector       # glm47
                                 ├── GptOssDetector         # gpt-oss
                                 ├── KimiK2Detector         # kimi_k2
                                 ├── Llama32Detector        # llama3
                                 ├── MiMoDetector           # mimo
                                 ├── MistralDetector        # mistral
                                 ├── PythonicDetector       # pythonic
                                 ├── Qwen25Detector         # qwen, qwen25
                                 ├── Qwen3CoderDetector     # qwen3_coder
                                 ├── Step3Detector          # step3
                                 ├── MinimaxM2Detector      # minimax-m2
                                 └── InternlmDetector       # interns1
```

### 3.2 核心数据类型

**文件**: `srt/function_call/core_types.py`

```python
class ToolCallItem(BaseModel):
    tool_index: int                 # 在 tools 列表中的索引
    name: Optional[str] = None     # 函数名
    parameters: str                # JSON 字符串形式的参数

class StreamingParseResult(BaseModel):
    normal_text: str = ""          # 非工具调用文本
    calls: List[ToolCallItem] = [] # 解析出的工具调用

@dataclass
class StructureInfo:
    begin: str       # 工具调用开始标记 (如 '<tool_call>\n{"name": "get_weather"')
    end: str         # 工具调用结束标记 (如 '</tool_call>')
    trigger: str     # 触发约束生成的 token
```

### 3.3 BaseFormatDetector 流式状态管理

**文件**: `srt/function_call/base_format_detector.py`

```python
class BaseFormatDetector(ABC):
    def __init__(self):
        self._buffer = ""                      # 流式缓冲区
        self.prev_tool_call_arr = []           # 上一轮解析的工具调用 (用于增量比较)
        self.current_tool_id = -1              # 当前正在解析的工具 ID
        self.current_tool_name_sent = False    # 当前工具名是否已发送
        self.streamed_args_for_tool = []       # 每个工具已流式发送的参数 JSON

        # 子类覆盖
        self.bot_token = ""                    # 工具调用开始标记
        self.eot_token = ""                    # 工具调用结束标记
        self.tool_call_separator = ", "        # 多工具调用分隔符

    @abstractmethod
    def has_tool_call(self, text: str) -> bool: ...
    @abstractmethod
    def detect_and_parse(self, text, tools) -> StreamingParseResult: ...
    @abstractmethod
    def structure_info(self) -> _GetInfoFunc: ...
```

### 3.4 流式解析状态机

基类 `parse_streaming_increment()` 实现了通用的流式工具调用解析：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               parse_streaming_increment 流程                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  new_text 到达 → _buffer += new_text                                        │
│      │                                                                       │
│      ├── 无工具调用标记 且 非 partial bot_token?                             │
│      │   └── 输出 normal_text, 清空 _buffer                                │
│      │                                                                       │
│      ├── 可能是 partial bot_token?                                          │
│      │   └── 继续缓冲 (返回空)                                             │
│      │                                                                       │
│      └── 检测到工具调用 (has_tool_call 或 separator)                        │
│          │                                                                   │
│          ▼                                                                   │
│      partial_json_loads(buffer[start_idx:])                                 │
│          │                                                                   │
│          ├── Case 1: 工具名尚未发送                                         │
│          │   ├── 验证函数名在 tools 中存在                                  │
│          │   ├── 发送 ToolCallItem(name=name, parameters="")               │
│          │   └── current_tool_name_sent = True                             │
│          │                                                                   │
│          └── Case 2: 工具名已发送, 流式参数                                │
│              ├── 对比 prev_tool_call_arr 计算增量 diff                     │
│              ├── 发送 ToolCallItem(parameters=argument_diff)               │
│              ├── streamed_args_for_tool[id] += diff                        │
│              │                                                               │
│              └── JSON 完整? → current_tool_id++, 重置状态                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

关键设计：
- 使用 `partial_json_loads` 解析不完整的 JSON，容忍流式传输中的截断
- 通过 `_find_common_prefix` 计算参数增量，避免重复发送
- `tool_index` 从 0 递增，每个工具完成后 `current_tool_id++`

### 3.5 代表性格式示例

**DeepSeek V3 格式**:
```
<｜tool▁call▁begin｜>function_name
{"param": "value"}<｜tool▁call▁end｜>
```

**Qwen 2.5 格式**:
```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Beijing"}}
</tool_call>
```

**Llama 3.2 格式**:
```
<|python_tag|>function_name.call(param="value")
```

**Mistral 格式**:
```
[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}]
```

## 4. StructuralTag 约束生成

FunctionCallParser 能为函数调用生成结构化标签约束，与 Grammar Backend（见 20 章）协作：

```python
# FunctionCallParser.get_structure_tag()
def get_structure_tag(self):
    for tool in self.tools:
        info = self.detector.structure_info()(tool.function.name)
        # info.begin = '<tool_call>\n{"name": "get_weather", "arguments": '
        # info.end   = '}\n</tool_call>'
        # info.trigger = '<tool_call>'
        structures.append(StructuresResponseFormat(
            begin=info.begin,
            schema=function.parameters,  # JSON Schema 约束参数格式
            end=info.end,
        ))
    return LegacyStructuralTagResponseFormat(
        type="structural_tag",
        structures=structures,
        triggers=triggers,
    )
```

约束选择逻辑：
```python
# get_structure_constraint()
if tool_choice == "auto" and (strict tools or TOOL_STRICT_LEVEL >= FUNCTION):
    → structural_tag 约束 (模型自由决定是否调用，但调用时参数受约束)

elif tool_choice == "required" or isinstance(tool_choice, ToolChoice):
    → json_schema 约束 (强制生成符合 schema 的 JSON)
```

## 5. Serving 层集成

**文件**: `srt/entrypoints/openai/serving_chat.py`

### 5.1 流式处理管线

```
模型输出 (逐 token)
    │
    ▼
separate_reasoning 启用?
    ├── Yes → ReasoningParser.parse_stream_chunk(delta)
    │         ├── reasoning_text → SSE: reasoning_content 字段
    │         └── normal_text → 继续
    └── No  → delta 原样传递
    │
    ▼
tools 已配置?
    ├── Yes → _process_tool_call_stream(delta)
    │         ├── FunctionCallParser.parse_stream_chunk(delta)
    │         ├── normal_text → SSE: content 字段
    │         └── calls → SSE: tool_calls 字段 (name/arguments)
    └── No  → SSE: content 字段
```

### 5.2 _process_tool_call_stream

```python
async def _process_tool_call_stream(self, index, delta, parser_dict, content, request, has_tool_calls):
    # 1. 延迟初始化 parser (每个 choice index 一个)
    if index not in parser_dict:
        if request.tool_choice == "required":
            parser_dict[index] = JsonArrayParser()         # 直接 JSON 解析
        else:
            parser_dict[index] = FunctionCallParser(tools, tool_call_parser)

    # 2. 解析增量
    normal_text, calls = parser.parse_stream_chunk(delta)

    # 3. 输出 normal_text → SSE content chunk
    if normal_text:
        yield SSE(delta=DeltaMessage(content=normal_text))

    # 4. 输出 tool calls → SSE tool_calls chunk
    for call_item in calls:
        yield SSE(delta=DeltaMessage(tool_calls=[...]))
```

### 5.3 非流式处理

```python
# 非流式场景
if reasoning_parser and request.separate_reasoning:
    reasoning_text, normal_text = ReasoningParser.parse_non_stream(full_text)

if function_call_parser:
    normal_text, tool_calls = FunctionCallParser.parse_non_stream(normal_text)
```

## 6. Scheduler 集成

### 6.1 think_end_id 注入

**文件**: `srt/managers/scheduler.py:447`

Scheduler 初始化时，如果启用了 `reasoning_parser`，会将 think_end_token 编码为 token ID 注入 tokenizer：

```python
# scheduler.py:447
if self.server_args.reasoning_parser and self.tokenizer:
    reasoning_parser = ReasoningParser(
        model_type=self.server_args.reasoning_parser, stream_reasoning=False
    )
    self.tokenizer.think_end_id = self.tokenizer.encode(
        reasoning_parser.detector.think_end_token, add_special_tokens=False
    )[0]
```

此 `think_end_id` 被用于：
- ReasonerGrammarBackend 判断 thinking 阶段结束
- 控制约束生成在 thinking 阶段暂停

### 6.2 ReasonerGrammarBackend

**文件**: `srt/constrained/reasoner_grammar_backend.py`

包装普通 Grammar Backend，在 thinking 阶段跳过约束：

```python
class ReasonerGrammarObject:
    def __init__(self, wrapped_grammar, think_end_id):
        self.wrapped_grammar = wrapped_grammar
        self.think_end_id = think_end_id
        self.tokens_after_think_end = -1  # -1 表示仍在 thinking

    def accept_token(self, token):
        if token == self.think_end_id:
            self.tokens_after_think_end = 0   # thinking 结束, 激活约束
        elif self.tokens_after_think_end >= 0:
            self.wrapped_grammar.accept_token(token)  # 委托给内层 grammar

    def fill_vocab_mask(self, mask, idx):
        if self.tokens_after_think_end < 0:
            pass  # thinking 阶段: 不限制词表 (全部允许)
        else:
            self.wrapped_grammar.fill_vocab_mask(mask, idx)  # 约束阶段
```

时序：
```
模型输出: <think>思考过程</think>{"name":"value"}
          ├── thinking 阶段 ──┤├── 约束阶段 ──────┤
          │  vocab_mask 全开   ││  grammar 约束     │
          │  accept_token 忽略 ││  accept_token 委托│
```

## 7. 启用方式

### 7.1 命令行参数

```bash
# 启用推理解析
sglang serve --model ... --reasoning-parser qwen3

# 启用函数调用
sglang serve --model ... --tool-call-parser qwen25

# 两者同时启用
sglang serve --model ... \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen25 \
    --grammar-backend xgrammar
```

### 7.2 请求参数

```python
# 推理解析: 通过 separate_reasoning=True 启用推理内容分离
response = client.chat.completions.create(
    model="qwen3",
    messages=[...],
    extra_body={"separate_reasoning": True},  # 分离 reasoning_content
)
# response.choices[0].message.reasoning_content = "思考过程..."
# response.choices[0].message.content = "最终回答"

# 函数调用: 通过 tools 参数
response = client.chat.completions.create(
    model="qwen25",
    messages=[...],
    tools=[{"type": "function", "function": {"name": "get_weather", ...}}],
    tool_choice="auto",  # auto / required / {"type":"function","function":{"name":"..."}}
)
```

## 8. 下一步

- **22 (待定)**: LoRA 适配器支持 (S-LoRA, Punica)
