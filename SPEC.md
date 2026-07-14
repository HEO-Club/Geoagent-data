# 图片地理定位 Agent 训练数据集生成流水线 — 项目规格

版本：2.2 | 生成时间：2026-07-14  
修订说明：基于 SPEC_REVIEW.md（v2.0）阻塞/重要/一般问题与覆盖决策修订；v2.2 修复 map_query 输出字段与 params.latlng 重名（F1）：Observation 改为 `resolved_latlng`，并补充 query/latlng 交叉约束与 status 条件 Observation 规则。

## 0. 给 Cursor 的元指令（先读这一段）

- 这是一个数据工程流水线项目，不是 Web 应用，不是 agent 应用。
- 本项目的产物是用于 SFT 训练的 agent 轨迹数据集（JSONL），供三个 LoRA 分别训练。
- 请严格按照本文档的模块划分与 data schema 实现，不要自行更改 schema 字段名和结构，它们是训练格式的契约。
- 逐模块实现，不要一次性生成全部代码。每个模块实现后应可独立测试。
- 所有 LLM 调用通过 API/SDK 完成（不涉及任何网页交互）。
- 所有结构化输出必须用 Pydantic + Instructor 强约束，禁止解析自由文本。
- 代码需带类型注解、docstring，关键逻辑加中文注释。
- 优先保证可测试、可调试、可断点续跑，其次才是性能。
- 模型名只能从配置读取；SDK 调用必须封装在 adapter 中；不得把具体模型名散落在业务代码中。

## 1. 项目目标与背景

### 1.1 最终目标

训练一个"通过图片进行地理定位的多 agent 系统"。系统由三个协作 agent 组成，共享同一个视觉语言模型基座，各自挂载独立 LoRA 权重：

- **Agent1（粗定位 LoRA / COARSE）**：演绎推理，从宏观视觉特征缩小到国家/地区级别
- **Agent2（精定位 LoRA / FINE）**：假设验证，从粗定位结论精确锁定到坐标级别
- **Agent3（验证 LoRA / VERIFIER）**：交叉验证，验证坐标与图像特征是否自洽，不通过则打回

每个 agent 独立执行 ReAct 循环：

观察输入 → 思考(Thought) → 调用工具(Action) → 得到结果(Observation) → 再思考 → ... → 输出结论

### 1.2 本项目要做什么

从各大自媒体平台爬取的「用图片做地理定位」的讲解视频中，结合预先提取好的带时间戳文字稿，把人类博主的定位推理过程和操作行为蒸馏成三套独立的 ReAct 轨迹，分别供三个 LoRA 训练。

### 1.3 可用输入资源（每条视频）

- 原始视频文件（mp4 等）
- 带时间戳文字稿（预先提取好，格式见第4节 TranscriptSegment schema）
  - 这是博主语音的精确转录，直接作为 narration 来源，无需 VLM 重复转录
  - 同时用于识别 agent 阶段切换点、精确定位答案泄漏边界
- 真实定位答案（从视频元数据或结尾提取的 groundtruth 坐标）
  - **groundtruth 只能进入 stage6**（验证、泄漏检测、过滤），不得进入 stage5 轨迹重构 LLM prompt

### 1.4 三个 Agent 的职责边界

**Agent1 粗定位（COARSE）：**

- 输入：原始图片
- 推理模式：演绎（这看起来像哪里）
- 主要 tool：zoom_inspect / ocr / sun_position_calc / web_search（仅 `purpose=broad_discovery`）
- 输出：`LocationHypothesis`（写入 `Trajectory.coarse_output`）
- handoff：`coarse_handoff=None`，`fine_handoff=None`

**Agent2 精定位（FINE）：**

- 输入：原始图片 + Agent1 的 `LocationHypothesis`（`coarse_handoff`）
- 推理模式：假设验证（去确认具体地点）
- 主要 tool：web_search（`broad_discovery` 或 `precise_lookup`）/ reverse_image_search / map_query / zoom_inspect / ocr
- 输出：`SubmitAnswerResult`（最后一步必须调用 terminal tool `submit_answer`，写入 `Trajectory.fine_output`）
- handoff：`coarse_handoff` 必填；`fine_handoff=None`
- `map_query` 用法示例：可用 `params.query`（地名）和/或 `params.latlng`（待查坐标）查询；读取 Observation 时使用 `resolved_latlng`（解析后标准坐标），**不得**把输入参数名 `latlng` 当作输出字段。

**Agent3 验证（VERIFIER）：**

- 输入：原始图片 + Agent2 的 `SubmitAnswerResult`（`fine_handoff`）；`coarse_handoff` 可选
- 推理模式：交叉验证（将该坐标与图像特征对照，**把 Agent2 结果当作候选答案验证**，不得看见 groundtruth）
- 主要 tool：map_query / web_search（仅 `purpose=verification`）
- 输出：`VerificationResult`（写入 `Trajectory.verifier_output`）
- handoff：`fine_handoff` 必填；`coarse_handoff` 可选
- `map_query` 用法示例：可用候选坐标作为 `params.latlng`（可同时带 `query` 消歧）；核对 Observation 的 `resolved_latlng` / `formatted_address` / `place_type` 是否与图像线索自洽。

### 1.5 四条数据质量铁律

- **防答案泄漏**：利用文字稿时间戳精确定位博主口头宣布答案的时间点；泄漏检查在 **stage6（模型外）** 完成。阶段时间规则见 1.6。
- **Observation 三条件**：真实可得（production tool 真实执行，draft tool VLM 合成但符合真实 API 风格）、格式一致（套统一 schema）、逻辑连贯（能支撑紧随其后的 Thought）。
- **宁缺毋滥**：验证不通过（推不到真值、跳步、幻觉）的样本进入 rejected，不写入最终训练 JSONL。
- **返工样本是高价值数据**：区分视频内真实纠错（`video_observed`）与系统打回（`system_feedback`），优先收集，不过滤。

### 1.6 答案泄漏与时间规则

**泄漏内容规则（stage6 程序化检查 + LLM-as-judge 辅助）：**

- Agent1 不得出现最终城市、精确地点或坐标。
- Agent2 只有在 Observation 支持后才能使用具体地点；最后一步允许 `submit_answer`。
- Agent3 可以看到 Agent2 的候选地点和坐标，但不得看到 groundtruth；必须把 Agent2 结果作为候选答案进行验证。
- groundtruth、真实地名和由 groundtruth 反向解析的地址 **不得进入 stage5 的任何 LLM prompt**。

**时间规则：**

- COARSE 和 FINE 默认只使用 `answer_timestamp` **之前**的证据。
- VERIFIER 可以使用答案宣布 **之后**的验证片段。
- 博主直接宣布答案的语句 **不能**作为验证证据。

## 2. 系统总体架构

```
【前置地基】(先实现)
├─ Tool Registry（tool_registry.json，单一事实来源）
├─ Tool Schema 规则体系（命名 / 语义 / Draft 创建 / Observation 合成）
├─ Tool Executor（production：本地可导入函数；draft：VLM 合成）
├─ 动态校验（pipeline/tools/validation.py）
└─ 轨迹 / 数据集 Schema（chat messages + role，服务 loss masking）

【主流水线】(每条视频依次经过；顺序固定)
stage0  解析带时间戳文字稿，定位答案时间戳，划分三个 Agent 与返工时间区间
        → PreprocessResult
stage1  按 Agent 时间区间抽帧（不按 Move），生成 TimedScreenAction
stage2  按时间重叠与语义边界生成 Move（TranscriptSegment ⟂ TimedScreenAction）
stage3  Move → NormalizedStep（匹配 / 组合 / 创建 Draft / fallback / thought_only）
stage4  执行 Action → ObservationExecutionResult
stage5  生成三 Agent 主轨迹与 revision 轨迹（本阶段禁止访问 groundtruth）
stage6  使用 groundtruth 做验证、泄漏检查、质量评分 → TrajectoryVerificationReport
stage7  生成三个 LoRA 的 DatasetEntry 与 JSONL 分片，再由单 writer 合并

【编排层】
Orchestrator（自写 Python + asyncio + tenacity），串联所有阶段，
支持并发、重试、断点续跑。批量入口 batch_run.py。

【Tool 生命周期管理】
tool_registry.json + manage_tools.py（升档/查询）
升档触发：明确 CLI 命令或 CI/CD 事件（系统不得自动猜测代码是否补全）
```

## 3. 目录结构（请按此创建）

```
geo-agent-dataset/
├── pipeline/
│   ├── __init__.py
│   ├── schemas.py              # 所有 Pydantic schema（契约核心）
│   ├── config.py               # 配置与 API key 加载（pydantic-settings）
│   ├── llm.py                  # LLM adapter 封装（Instructor + 结构化输出）
│   ├── stage0_preprocess.py
│   ├── stage1_parse.py
│   ├── stage2_moves.py
│   ├── stage3_normalize.py
│   ├── stage4_observe.py
│   ├── stage5_reconstruct.py
│   ├── stage6_verify.py
│   ├── stage7_format.py
│   └── tools/
│       ├── __init__.py
│       ├── base.py             # execute_action 分发器
│       ├── registry.py         # tool_registry 读写（含文件锁）
│       ├── validation.py       # 动态 params/observation 校验
│       ├── web_search.py
│       ├── reverse_image_search.py
│       ├── map_query.py
│       ├── ocr.py
│       ├── zoom_inspect.py
│       └── sun_position.py
├── run_one_video.py
├── batch_run.py
├── manage_tools.py
├── tool_registry.json
├── tests/
├── data/
│   ├── raw_videos/
│   ├── transcripts/
│   ├── intermediate/
│   │   └── {video_id}/
│   │       ├── manifest.json
│   │       ├── stage0_preprocess.json
│   │       ├── stage1_screen_actions.json
│   │       ├── stage2_moves.json
│   │       ├── stage3_normalized_steps.json
│   │       ├── stage4_observations.json
│   │       ├── stage5_trajectories.json
│   │       ├── stage6_verification.json
│   │       └── stage7_entries.json
│   └── output/
│       ├── shards/             # 每视频分片，禁止多协程直接写最终 JSONL
│       ├── agent1_coarse.jsonl
│       ├── agent2_fine.jsonl
│       └── agent3_verifier.jsonl
├── requirements.txt            # 兼容版本范围；生产可用 lock 文件
├── .env.example
├── .gitignore
└── SPEC.md
```

## 4. 数据 Schema 定义（pipeline/schemas.py）

这是全项目的契约核心。请用 Pydantic v2 实现，字段名不得更改。

### 4.1 输入资源 Schema

```python
class TranscriptSegment(BaseModel):
    start: float  # 秒
    end: float
    text: str  # 博主原话

class VideoInput(BaseModel):
    video_path: str
    transcript: list[TranscriptSegment]
    groundtruth: tuple[float, float]  # (lat, lng)；仅 stage6 消费
    source_platform: str
```

### 4.2 Agent 角色定义

```python
class AgentRole(str, Enum):
    COARSE = "coarse_locator"
    FINE = "fine_locator"
    VERIFIER = "verifier"
```

### 4.3 Tool 体系 Schema

```python
class ToolTier(str, Enum):
    DRAFT = "draft"            # 已有 Schema，无真实实现；observation 由 VLM 合成
    PRODUCTION = "production"  # 已有可调用并通过测试的真实实现

# example / default 支持的值类型（实现时可用 Union / TypeAlias）
ParamValue = str | int | float | bool | list[float] | list[str] | tuple[float, float]
# bbox: [x, y, w, h]（float×4）→ list[float] len=4
# latlng: (lat, lng) → tuple[float, float]
# lat_range: (min_lat, max_lat) → tuple[float, float]
# string_list: list[str]

class ParamField(BaseModel):
    name: str
    type: Literal[
        "string", "float", "int", "bool",
        "bbox", "latlng", "lat_range", "string_list"
    ]
    required: bool
    description: str  # 10~60 字符；格式："含义描述，如[示例值]"
    example: ParamValue  # 始终必填，且必须符合声明类型
    default: ParamValue | None = None
    enum_values: list[str] | None = None  # 仅 type=string 时可用；其他类型必须为 None

    # validator 约束：
    # - required=True → default 必须为 None
    # - required=False → 允许设置 default（也可为 None，表示运行时可不传且无默认）
    # - enum_values 非空时，example/default（若有）必须落在枚举内

class ObservationField(BaseModel):
    name: str
    type: Literal[
        "string", "float", "int", "bool", "string_list",
        "result_list", "latlng", "lat_range", "bbox"
    ]
    nullable: bool
    description: str  # 10~80 字符；nullable=true 时须说明何时为 null
    item_fields: list["ObservationField"] | None = None
    # type=result_list → item_fields 必填，长度 2~5，禁止再嵌套 result_list
    # 其他 type → item_fields 必须为 None

class ToolDefinition(BaseModel):
    name: str
    description: str
    tier: ToolTier
    params: list[ParamField]
    observation_fields: list[ObservationField]  # 原 observation 字段更名为 observation_fields
    allowed_agents: list[AgentRole]
    is_terminal: bool = False
    executor_ref: Optional[str] = None
    # draft：executor_ref 必须为 None
    # production：executor_ref 必须非空，且为可导入的 Python 路径，例如：
    #   "pipeline.tools.web_search.execute"
    #   "pipeline.tools.ocr.execute"
    # 不要把每个 Tool 设计成独立 HTTP 服务

    # draft 元数据
    created_at: str  # ISO8601，系统自动注入
    source_video_timestamp: Optional[float] = None
    source_narration: Optional[str] = None
    derived_from_existing_tools: list[str] = []
    # 注意：derived_from_existing_tools 是否存在于 Registry
    # 不在 Pydantic model_validator 中检查，改在 register_tool 时用 Registry snapshot 检查

    # validator 摘要：
    # - is_terminal=False → observation_fields 不得为空；必须包含 name=status（非空枚举）
    #   与 name=error_message（nullable=True）
    # - is_terminal=True → observation_fields 必须为空列表；executor_ref 可为 None
    #   （submit_answer 为本地终结动作，不产生 observation）
    # - params 数量 ≤ 5；observation_fields 数量 ≤ 8（terminal 的空列表除外）
    # - params 字段名与 observation_fields 字段名不得有交集
    # - params 禁止名为 image / image_path / frame / video（图像由 execute_action 统一传入）
```

**普通（非 terminal）Tool 的 Observation 固定字段：**

每个非 terminal Tool 的 `observation_fields` **必须**包含：

| 字段 | 类型 | nullable | 说明 |
|------|------|----------|------|
| status | string（enum: success / empty / error） | False | 执行状态 |
| error_message | string | True | status=error 时填写，否则 null |

业务字段（如 results、formatted_address）可继续 nullable。  
**删除**旧版「必须有且仅有至少一个 nullable=False 核心业务字段」的表述。  
新规则：普通 Tool 必须有非空 `status`；terminal Tool 不产生 Observation。

### 4.4 Tool 名称规则（A规则，validator 强制检查）

```python
# 硬规则（注册与 Draft 创建时强制）：
# A1: 小写 snake_case；仅 [a-z0-9_]；长度 3~64
# A2: 不得以下划线开头或结尾；不得包含连续下划线
# A3: 禁止无意义名称：tool / helper / utility / new_tool / custom_tool 等
# A4: 自动生成的 Draft Tool（非种子 Tool）名称至少包含两个语义 token（至少一段下划线分隔）
# A5: 注册时执行名称精确去重 + 语义去重（与现有 tool 高度相似则拒绝）

# 种子 Tool 名称白名单（不得重命名，且豁免 A4 的“至少两 token”若已满足则无需额外处理）：
SEED_TOOL_NAMES = {
    "web_search",
    "reverse_image_search",
    "map_query",
    "ocr",
    "zoom_inspect",
    "sun_position_calc",
    "submit_answer",
}

# LLM 命名提示（非 validator 注册条件）：
ALLOWED_VERBS_HINT = {
    "get", "search", "query", "detect",
    "extract", "calculate", "lookup", "compare", "estimate"
}
# 仅作为生成 Draft Tool 名称时的 prompt 提示，不得作为硬性注册校验条件。

# params 字段名：全小写下划线；禁止 data/input/output/value/result/info 等无意义词
# observation_fields 字段名不得与 params 字段名有交集
```

### 4.5 Tool 语义合法性规则（F规则）

```python
# F1: params 字段名集合 与 observation_fields 字段名集合 不得有交集
#     （含种子 Tool，无豁免。例：map_query 输出坐标必须为 resolved_latlng，
#      不得与 params.latlng 同名）
# F2: 普通 Tool 必须包含非空 status 字段（enum success|empty|error）；
#     terminal Tool 的 observation_fields 必须为空（不产生 Observation）
# F3: params 中有 bbox 类型时，observation_fields 中不得再有 bbox 类型
# F4: description 应清晰描述工具用途（中文或中英混合均可）；不再强制动词-描述关键词绑定
# F5: params 禁止 image/image_path/frame/video
# F6: derived_from_existing_tools 的存在性 → register_tool(registry_snapshot) 检查，不在 model_validator
# F7: params ≤ 5；observation_fields ≤ 8（terminal 空列表除外）
# F8: type=result_list 时 item_fields 必填且禁止嵌套 result_list；每项必须严格符合 item_fields
# F9: draft → executor_ref is None；production → executor_ref 非空且可导入
# F10: Tool 级交叉字段约束（如 map_query 的 query|latlng 至少其一；
#      以及 status 条件 Observation 规则）由 validate_action_params /
#      validate_observation 实现，不能仅靠单字段 required / nullable 表达
```

### 4.6 创建 Draft Tool 的条件（G规则，match_or_register_tool 里检查）

样本出现次数可以统计，但 **不得**作为注册或升档门槛。

只有 **同时满足** 以下条件才创建 Draft Tool：

1. 无法匹配现有 Tool；
2. 无法用现有 Tool 组合表达；
3. 输入输出语义明确；
4. 在地理定位中具有复用可能；
5. 未来可能实现真实 executor；
6. 不是滚动页面、移动鼠标、切换标签等纯 UI 操作；
7. 不是只对当前视频成立的一次性操作；
8. 与现有 Tool 不重复或高度相似（语义相似度阈值建议 0.85；名称编辑距离 ≤ 3 直接拒绝）。

不满足时：强制匹配现有 Tool、返回组合方案、或 `normalization_mode=fallback` / `thought_only`，**禁止无标记硬套**。

### 4.7 Draft Tool Observation 合成规则（H规则，VLM prompt 约束）

```python
# H1: 必须按 observation_fields 每个字段逐一填写，不得增减
# H2: nullable=true：视频中未出现对应信息时填 null，不得猜测
# H3: result_list：条目数 2~5，且每项符合 item_fields
# H4: string 禁止 "未知"/"不确定"/"N/A"；信息不可得时应为 nullable 字段填 null
# H5: 必须通过 validate_observation；Instructor 最多重试 DRAFT_TOOL_MAX_RETRY 次，
#     仍失败 → 该样本进入 rejected
# H6: 不得从 groundtruth 获得任何信息
```

### 4.8 初始 Tool 注册表（7 个种子 Tool）

种子 Tool **保留现有名称**。  
`tier` 规则：只有真实实现完成并通过 smoke test 的才能标为 `production`；尚未实现的标为 `draft`，`executor_ref=None`。

下列清单给出目标 schema；实现落地时按上述规则设置 `tier` / `executor_ref`。

**web_search**

- allowed_agents: [COARSE, FINE, VERIFIER]
- is_terminal: false
- params:
  - query(string, required)
  - top_k(int, required=False, default=3)
  - purpose(string, required, enum_values=["broad_discovery","precise_lookup","verification"])
- observation_fields:
  - status (string, nullable=False, enum success|empty|error)
  - error_message (string, nullable=True)
  - results (result_list, nullable=True；empty/error 可为 null)
    - item_fields: title(string), snippet(string), url(string)
- purpose 角色硬约束（stage3 + executor 双处检查）：
  - COARSE → 只能 broad_discovery
  - FINE → broad_discovery 或 precise_lookup
  - VERIFIER → 只能 verification
- 说明：purpose 可结构化约束用途；query 是否真正符合用途仍需规则或 judge 检查

**reverse_image_search**

- allowed_agents: [FINE]
- params: bbox(bbox, optional)
- observation_fields: status, error_message, matches(result_list → title/snippet/url)

**map_query**

- allowed_agents: [FINE, VERIFIER]
- is_terminal: false
- params:
  - query(string, optional) — 地名或地址查询文本
  - latlng(latlng, optional) — 用户提供的待查询坐标
- observation_fields:
  - status (string, nullable=False；enum success|empty|error)
  - error_message (string, nullable=True)
  - formatted_address (string, nullable)
  - resolved_latlng (latlng, nullable) — 地图服务解析/匹配/纠偏后的标准坐标
  - place_type (string, nullable)
- 字段语义（F1 强制区分输入/输出名）：
  - `params.latlng`：调用方提供的待查询坐标
  - `observation_fields.resolved_latlng`：服务返回的标准坐标（**不得**再命名为 `latlng`）
- params 交叉约束（不能仅靠单字段 `required` 表达；由 Tool 级交叉字段 validator 或 `validate_action_params` 实现）：
  1. `query` 与 `latlng` **至少提供一个**
  2. 允许只提供 `query`
  3. 允许只提供 `latlng`
  4. **允许同时提供**二者（用于约束搜索范围或消歧）
  5. 二者都缺失 → Action 参数校验失败
- Observation 条件规则（由 `validate_observation` 强制）：
  1. `status=success` → `resolved_latlng` 必须非空；`formatted_address` / `place_type` 可为 null
  2. `status=empty` → `resolved_latlng` / `formatted_address` / `place_type` 可为 null；`error_message` 必须为 null
  3. `status=error` → `error_message` 必须非空；其他业务字段可为 null
- Observation 示例（success）：

```json
{
  "status": "success",
  "error_message": null,
  "formatted_address": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
  "resolved_latlng": [48.8584, 2.2945],
  "place_type": "tourist_attraction"
}
```

**ocr**

- allowed_agents: [COARSE, FINE]
- params: bbox(bbox, optional)
- observation_fields: status, error_message, texts(string_list)

**zoom_inspect**

- allowed_agents: [COARSE, FINE]
- params: bbox(bbox, required)
- observation_fields: status, error_message, description(string)

**sun_position_calc**

- allowed_agents: [COARSE]
- params: shadow_direction_deg(float, optional), estimated_local_time(string, optional)
- observation_fields:
  - status, error_message
  - possible_latitude_range (lat_range, nullable)  # (min_lat, max_lat)，不是 latlng
  - note (string, nullable)

**submit_answer**（terminal）

- allowed_agents: [FINE]
- is_terminal: true
- observation_fields: []
- params:
  - latitude(float, required)
  - longitude(float, required)
  - location_name(string, required)
  - confidence(float, required)
  - reasoning(string, required)
- Agent2 最后一步必须调用本 tool；其 params 必须能解析为 `SubmitAnswerResult`

### 4.9 Agent 交接物与输出 Schema

```python
class LocationHypothesis(BaseModel):
    possible_countries: list[str]
    possible_regions: list[str]
    reasoning_summary: str
    confidence: float = Field(ge=0, le=1)
    key_clues_remaining: list[str]

class SubmitAnswerResult(BaseModel):
    # 与 submit_answer params 对齐；Agent2 → Agent3 的结构化交接物
    latitude: float
    longitude: float
    location_name: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str

class VerificationResult(BaseModel):
    verdict: Literal["pass", "fail"]
    failed_checks: list[str]
    suggested_recheck: str
    return_to_agent: Optional[Literal[1, 2]] = None
    # pass → None
    # fail + 1 → 打回 COARSE
    # fail + 2 → 打回 FINE
```

### 4.10 阶段中间产物 Schema

```python
class AgentTimeSegment(BaseModel):
    agent_role: AgentRole
    start_time: float
    end_time: float
    # AgentRole 不得作为持久化 JSON 对象的 key；使用对象列表

class PreprocessResult(BaseModel):
    answer_timestamp: float
    agent_segments: list[AgentTimeSegment]
    revision_segments: list[tuple[float, float]]

class TimedScreenAction(BaseModel):
    start_time: float
    end_time: float
    description: str
    visible_clues: list[str] = []

class Move(BaseModel):
    start_time: float
    end_time: float
    narration: str  # 来自文字稿，非 VLM 生成
    screen_action: Optional[str] = None
    visible_clues: list[str] = []
    agent_role: AgentRole

class Action(BaseModel):
    tool: str  # 必须存在于 tool_registry
    params: dict

class NormalizationMode(str, Enum):
    MATCHED = "matched"
    COMPOSED = "composed"
    DRAFT_CREATED = "draft_created"
    FALLBACK = "fallback"
    THOUGHT_ONLY = "thought_only"

class NormalizedStep(BaseModel):
    move: Move
    thought_draft: str
    actions: list[Action]  # 一个 Move 可对应多个 Action；thought_only 时为空列表
    normalization_mode: NormalizationMode
    matched_tool_confidence: Optional[float] = None
    fallback_reason: Optional[str] = None
    # screen_action 为空 → 允许 thought_only，不得伪造 Tool Action

class ObservationSource(str, Enum):
    REAL_EXECUTION = "real_execution"
    VLM_SYNTHESIZED = "vlm_synthesized"

class ObservationExecutionResult(BaseModel):
    action: Action
    observation: Optional[dict] = None
    source: Optional[ObservationSource] = None
    status: Literal["success", "empty", "error", "skipped"]
    error_message: Optional[str] = None
    cache_hit: bool = False
    # submit_answer / terminal：
    #   observation=None, source=None, status="skipped"
```

### 4.11 轨迹 Schema

```python
class TrajectoryStep(BaseModel):
    thought: str
    action: Action
    observation: Optional[dict] = None  # terminal 步为 None
    observation_source: Optional[ObservationSource] = None  # terminal 步为 None

class RevisionSource(str, Enum):
    VIDEO_OBSERVED = "video_observed"      # 视频中博主真实纠错
    SYSTEM_FEEDBACK = "system_feedback"  # Agent3 打回

class RevisionContext(BaseModel):
    source: RevisionSource
    parent_trajectory_id: str
    target_agent: AgentRole  # return_to_agent=1→COARSE；=2→FINE
    revision_round: int
    verification_result: Optional[VerificationResult] = None  # system_feedback 时必填
    video_segment: Optional[tuple[float, float]] = None       # video_observed 时必填

class Trajectory(BaseModel):
    id: str
    agent_role: AgentRole
    system_prompt: str
    user_query: str
    image_path: str
    steps: list[TrajectoryStep]

    # 交接
    coarse_handoff: Optional[LocationHypothesis] = None
    fine_handoff: Optional[SubmitAnswerResult] = None
    # COARSE：两者皆空
    # FINE：coarse_handoff 必填
    # VERIFIER：fine_handoff 必填；coarse_handoff 可选

    # 结构化输出（每个角色只填自己的）
    coarse_output: Optional[LocationHypothesis] = None
    fine_output: Optional[SubmitAnswerResult] = None
    verifier_output: Optional[VerificationResult] = None

    # 返工
    is_revision: bool = False
    parent_trajectory_id: Optional[str] = None
    revision_round: int = 0
    revision_source: Optional[RevisionSource] = None
    revision_input: Optional[VerificationResult] = None  # system_feedback 时保留
```

### 4.12 验证报告与数据集条目

```python
class TrajectoryVerificationReport(BaseModel):
    passed: bool
    quality_score: float  # 0~1
    distance_error_km: Optional[float] = None  # 仅 FINE 等有坐标时有值
    hard_fail_reasons: list[str] = []
    soft_warnings: list[str] = []
    leakage_detected: bool = False

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class DatasetEntry(BaseModel):
    id: str
    source_video: str
    agent_role: AgentRole
    groundtruth: tuple[float, float]  # 元数据；训练时不进入 assistant 可见推理目标外滥用
    messages: list[ChatMessage]
    coarse_handoff: Optional[LocationHypothesis] = None
    fine_handoff: Optional[SubmitAnswerResult] = None
    is_revision: bool = False
    parent_trajectory_id: Optional[str] = None
    revision_round: int = 0
    revision_source: Optional[RevisionSource] = None
    revision_input: Optional[VerificationResult] = None
    contains_draft_tools: bool = False
    draft_tool_names: list[str] = []
    quality_score: float
    verified: bool
    distance_error_km: Optional[float] = None
    # Loss mask：
    # assistant → 算 loss（Thought + Action）
    # system / user / tool → 全部 mask
```

### 4.13 Checkpoint / Manifest

```python
class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"  # 升档或上游变更后失效，需重跑

class StageManifestEntry(BaseModel):
    stage: str
    status: StageStatus
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None

class VideoManifest(BaseModel):
    video_id: str
    stages: list[StageManifestEntry]
```

统一目录：`data/intermediate/{video_id}/`，文件名见第3节。

## 5. 各模块职责与函数签名

### 5.1 config.py

用 pydantic-settings 从 .env 读取所有配置。禁止硬编码 key。模型名只能从配置读取。

配置项包括：

- APP_ENV, ALLOW_REAL_API
- GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
- TAVILY_API_KEY, SERPAPI_KEY, GOOGLE_MAPS_KEY
- GEMINI_MODEL（默认值仅写在配置/ENV，不写死在业务代码）
- MAX_CONCURRENT_VIDEOS
- ANSWER_LEAK_CHECK_ENABLED
- DRAFT_TOOL_MAX_RETRY（默认 3）
- MAX_REVISION_ROUNDS（默认 2）
- TOOL_REGISTRY_PATH, INTERMEDIATE_DIR, OUTPUT_DIR, CACHE_DIR
- DISTANCE_ERROR_THRESHOLD_KM（默认 25）

### 5.2 llm.py

```python
def call_structured(
    prompt: str,
    response_model: type[BaseModel],
    images: Optional[list[str]] = None,
    video: Optional[str] = None,
    model: Optional[str] = None,  # None → 从配置读取
) -> BaseModel:
    """调用 LLM 并强制返回 response_model；不合法自动重试。经 adapter 封装 SDK。"""

def call_text(prompt: str, model: Optional[str] = None) -> str:
    """纯文本调用（如 LLM-as-judge）。禁止传入 groundtruth。"""
```

### 5.3 tools/registry.py

```python
def load_registry() -> dict[str, ToolDefinition]:
    """从 tool_registry.json 读取全部 tool 定义。"""

def register_tool(tool: ToolDefinition) -> None:
    """
    写入新 tool。必须：
    1. 获取跨进程文件锁
    2. 重新读取最新 Registry
    3. 再次执行名称/语义去重
    4. 用 Registry snapshot 检查 derived_from_existing_tools
    5. 写临时文件 → fsync → 原子替换
    自动注入 created_at。
    """

def promote_tool(tool_name: str, executor_ref: str) -> dict:
    """
    升档流程（由 CLI 或 CI/CD 显式触发；无样本数量门槛）：
    1. 备份 registry 与受影响历史数据
    2. 检查 executor_ref 可导入
    3. smoke test 通过
    4. 用真实 executor 对样例 Action 校验 params → 真实 Observation 通过 schema
    5. 原子更新 registry：tier=production，executor_ref 写入
    6. 查找 draft_tool_names 含该 tool 的历史样本
    7. 对这些样本重跑 stage4 → stage5 → stage6 → stage7
       （不能只替换 Observation 后只跑 stage6；Thought 可能依赖旧合成 Observation）
    8. 失败则回滚 registry 与数据备份；成功则输出升档报告
    """

def get_tools_for_agent(role: AgentRole) -> list[ToolDefinition]:
    """按 allowed_agents 过滤。"""
```

### 5.4 tools/validation.py

```python
def apply_param_defaults(tool: ToolDefinition, params: dict) -> dict:
    """为缺失的可选参数补齐 default；不修改已传入值。"""

def validate_action_params(tool: ToolDefinition, params: dict) -> dict:
    """
    先 apply_param_defaults，再校验：
    - 拒绝额外参数
    - 检查必填、类型、范围、枚举
    - 检查 bbox / latlng / lat_range / string_list
    - 检查 web_search.purpose 与调用方 AgentRole 的硬约束（若上下文提供 role）
    - map_query：query 与 latlng 至少提供一个；允许只传其一或同时传；二者皆缺则失败
      （Tool 级交叉字段约束，不能仅靠单字段 required）
    返回规范化后的 params；失败抛 ValidationError。
    """

def validate_observation(tool: ToolDefinition, observation: dict | None) -> dict | None:
    """
    terminal → observation 必须为 None。
    非 terminal：
    - 拒绝额外字段
    - 按 observation_fields 检查必填/可空/类型/枚举
    - result_list 每项严格符合 item_fields
    - map_query status 条件规则：
        success → resolved_latlng 非空；formatted_address/place_type 可空
        empty → resolved_latlng/formatted_address/place_type 可空；error_message 必须为 null
        error → error_message 非空；其他业务字段可空
      （输出坐标字段名必须为 resolved_latlng，不得使用 latlng）
    production 返回非法 Observation → 不得用 VLM 伪装成真实结果；标记 error / 样本 rejected
    draft 校验失败 → 最多重试 DRAFT_TOOL_MAX_RETRY，仍失败 → rejected
    """
```

### 5.5 stage0_preprocess.py

```python
def locate_answer_timestamp(transcript: list[TranscriptSegment]) -> float: ...

def segment_by_agent_role(
    transcript: list[TranscriptSegment],
    answer_timestamp: float,
) -> list[AgentTimeSegment]:
    """
    识别三个阶段时间区间。
    COARSE/FINE 默认落在 answer_timestamp 之前；
    VERIFIER 可落在宣布答案之后的验证片段。
    """

def detect_revision_segments(
    transcript: list[TranscriptSegment],
) -> list[tuple[float, float]]: ...

def preprocess(video_input: VideoInput) -> PreprocessResult:
    """串联上述函数，返回 PreprocessResult（禁止裸 dict）。"""
```

### 5.6 stage1_parse.py

```python
def extract_keyframes(
    video_path: str,
    time_range: tuple[float, float],
    fps: float = 1.0,
) -> list[str]:
    """按 Agent 时间区间抽帧（不按 Move；Move 在 stage2 才生成）。"""

def detect_screen_actions(
    keyframes: list[str],
    narration_context: str,
    time_range: tuple[float, float],
) -> list[TimedScreenAction]:
    """VLM 识别屏幕操作，产出带时间戳的 TimedScreenAction 列表。"""
```

### 5.7 stage2_moves.py

```python
def build_moves(
    transcript_segment: list[TranscriptSegment],
    screen_actions: list[TimedScreenAction],
    agent_role: AgentRole,
    time_range: tuple[float, float],
) -> list[Move]:
    """
    按时间重叠对齐 TranscriptSegment 与 TimedScreenAction，
    再按语气/转折等语义边界切分 Move。
    不得按列表下标一一配对。
    """

def build_all_agent_moves(
    video_input: VideoInput,
    preprocess_result: PreprocessResult,
    screen_actions_by_role: dict[AgentRole, list[TimedScreenAction]],
) -> dict[AgentRole, list[Move]]: ...
```

### 5.8 stage3_normalize.py

```python
def match_or_register_tool(
    screen_action: str,
    narration: str,
    agent_role: AgentRole,
    existing_tools: list[ToolDefinition],
    all_moves: list[Move],
) -> tuple[list[Action], NormalizationMode, Optional[str]]:
    """
    按 G 规则决定：匹配 / 组合 / 建 Draft / fallback。
    同时检查 allowed_agents 与 web_search.purpose 角色约束。
    返回 actions（组合可为多个）、mode、fallback_reason。
    """

def normalize_to_steps(
    moves: list[Move],
    agent_role: AgentRole,
) -> list[NormalizedStep]:
    """
    每个 Move → NormalizedStep。
    screen_action 为空 → thought_only，actions=[]。
    一个 Move 可对应多个 Action（composed）。
    """
```

### 5.9 tools/base.py

```python
def execute_action(
    action: Action,
    image_path: str,
    agent_role: AgentRole,
) -> ObservationExecutionResult:
    """
    分发器：
    1. 查 registry；检查 allowed_agents 与 purpose 约束
    2. validate_action_params
    3. terminal → status=skipped，observation/source=None
    4. production → import executor_ref，真实执行，validate_observation；
       非法结果不得用 VLM 伪装
    5. draft → VLM 合成（H 规则），validate_observation，失败重试
    6. diskcache：key 至少包含
       tool_name, tool_schema_hash, executor_version,
       normalized_params_hash, image_content_hash,
       model_name（VLM 时）, prompt_version（VLM 时）
    """
```

### 5.10 stage4_observe.py

```python
def generate_observations(
    normalized_steps: list[NormalizedStep],
    image_path: str,
    agent_role: AgentRole,
) -> list[ObservationExecutionResult]:
    """
    展开 normalized_steps 中的全部 Action，逐个 execute_action。
    返回 ObservationExecutionResult 列表。
    thought_only 步不产生 execution result。
    """
```

### 5.11 stage5_reconstruct.py

**本阶段所有函数签名不得包含 groundtruth。**

```python
def reconstruct_single_trajectory(
    steps: list[NormalizedStep],
    observations: list[ObservationExecutionResult],
    agent_role: AgentRole,
    answer_timestamp: float,
    image_path: str,
    coarse_handoff: Optional[LocationHypothesis] = None,
    fine_handoff: Optional[SubmitAnswerResult] = None,
    is_revision: bool = False,
    revision_context: Optional[RevisionContext] = None,
) -> Trajectory:
    """
    组装 T→A→O，用 LLM 改写为前向推理链。
    禁止将 groundtruth / 真值地名 / 反向地理编码地址写入 prompt。
    Agent1 → 填写 coarse_output=LocationHypothesis
    Agent2 → 最后一步 submit_answer，填写 fine_output=SubmitAnswerResult
    Agent3 → 填写 verifier_output=VerificationResult；把 fine_handoff 当候选验证
    terminal 步 observation 与 observation_source 均为 None
    """

def reconstruct_all_trajectories(
    all_steps: dict[AgentRole, list[NormalizedStep]],
    all_observations: dict[AgentRole, list[ObservationExecutionResult]],
    answer_timestamp: float,
    image_path: str,
) -> dict[AgentRole, Trajectory]:
    """
    为三 Agent 重构主轨迹并传递交接物：
    Agent1.coarse_output → Agent2.coarse_handoff
    Agent2.fine_output → Agent3.fine_handoff
    """

def reconstruct_revision_trajectories(
    parent_trajectories: dict[AgentRole, Trajectory],
    verification: VerificationResult,
    all_steps: dict[AgentRole, list[NormalizedStep]],
    all_observations: dict[AgentRole, list[ObservationExecutionResult]],
    answer_timestamp: float,
    image_path: str,
    revision_round: int,
    max_revision_rounds: int,  # 来自 MAX_REVISION_ROUNDS，默认 2
    video_revision_segments: Optional[list[tuple[float, float]]] = None,
) -> list[Trajectory]:
    """
    闭合返工路径：
    - system_feedback：return_to_agent=1→COARSE；=2→FINE；构造 RevisionContext
    - video_observed：使用 video_revision_segments 生成高价值返工轨迹
    - revision_round >= max_revision_rounds 仍不通过 → 进入 rejected，禁止无限循环
    """
```

### 5.12 stage6_verify.py

```python
def verify_and_score(
    traj: Trajectory,
    groundtruth: tuple[float, float],
) -> TrajectoryVerificationReport:
    """
    groundtruth 仅在本阶段使用。
    Agent1：LocationHypothesis 是否覆盖真值国家/地区（模型外地理知识库/编码）
    Agent2：geopy 距离误差；超过 DISTANCE_ERROR_THRESHOLD_KM → hard fail
    Agent3：verdict 是否与 Agent2 误差一致性匹配
    全员：泄漏检测（见 1.6）+ LLM-as-judge（合理性；prompt 仍不得含 raw groundtruth 坐标作为“应输出答案”）
    返回 TrajectoryVerificationReport，禁止含义不清的三元组。
    """
```

### 5.13 stage7_format.py

```python
def to_dataset_entry(traj: Trajectory, meta: dict) -> DatasetEntry: ...

def format_all_and_save(
    trajectories: list[Trajectory],
    meta: dict,
    output_dir: str,
    video_id: str,
) -> None:
    """
    按 agent_role 写入 data/output/shards/{video_id}_agent{1|2|3}.jsonl；
    最终由单 writer 合并为 agent1_coarse.jsonl / agent2_fine.jsonl / agent3_verifier.jsonl。
    禁止多个协程直接追加同一个最终 JSONL。
    """
```

### 5.14 run_one_video.py / batch_run.py

- `run_one_video.py`：串联 stage0–7；每阶段落盘到 `data/intermediate/{video_id}/`；更新 `manifest.json`。
- `batch_run.py`：asyncio 并发 + tenacity；跳过 completed；分片写 JSONL，结束时单 writer 合并。

### 5.15 manage_tools.py

```bash
python manage_tools.py list
python manage_tools.py list --tier draft
python manage_tools.py promote <tool_name> --executor-ref pipeline.tools.<name>.execute
python manage_tools.py stats
```

升档无样本数量门槛；必须通过 import + smoke test + schema 校验；失败回滚。

## 6. Tool 升档流程

触发条件（唯一允许的触发源）：

- 开发者执行 `manage_tools.py promote ...`；或
- CI/CD 显式升档事件

系统 **不得** 自动猜测“代码是否已补全”而升档。  
升档不设样本数量门槛。

升档前必须检查：

1. `executor_ref` 可以导入；
2. smoke test 通过；
3. Action 参数通过 `validate_action_params`；
4. 真实 Observation 通过 `validate_observation`。

升档影响历史数据时，必须重跑：

**stage4 → stage5 → stage6 → stage7**

不能只替换 Observation 后重跑 stage6。

过程必须支持：备份 → 原子更新 → 失败回滚。  
升档后因真实 Observation 与旧 Thought 不一致导致验证失败而丢弃样本，是预期行为。

**Schema 字段重命名与历史 Observation（以 map_query 为例）：**

- 若升档或 registry 修订将 Observation 字段从旧名 `latlng` 改为 `resolved_latlng`，**禁止**仅在落盘 JSON 里做键名替换后继续沿用旧 Thought。
- 必须：备份 → 更新 registry schema → 重跑 **stage4 → stage5 → stage6 → stage7**，使 Observation、Thought 与下游校验一致使用 `resolved_latlng`。
- 历史样本中若仍出现 `map_query` Observation 键 `latlng`（无 `resolved_latlng`），`validate_observation` 必须拒绝；不得静默兼容旧键名。

## 7. 依赖与工具清单（requirements.txt）

使用兼容版本范围；生产环境建议另提供 lock 文件。

```
# LLM + 结构化输出
google-genai
anthropic
openai
instructor
pydantic>=2
pydantic-settings

# 视频处理
yt-dlp
faster-whisper
scenedetect
# 系统依赖：ffmpeg

# Tools
tavily-python
google-search-results
googlemaps
geopy
paddleocr
pillow
opencv-python
astral
pysolar

# 工程
tenacity
diskcache
duckdb
langfuse
python-dotenv
editdistance
filelock
pytest
pytest-asyncio
ruff
mypy
```

训练阶段（本仓库之外）：LLaMA-Factory 或 ms-swift；本仓库只生产 JSONL。

## 8. 开发顺序（请严格按此顺序）

### 里程碑一：地基

1. schemas.py — 全部 Pydantic schema + A/F/G/H 相关 validator
2. tools/validation.py
3. tool_registry.json — 种子 tool（按实现情况标 draft/production）
4. tools/registry.py — 文件锁 + 原子写
5. config.py + .env.example
6. llm.py adapters
7. tools/ 真实实现 + execute_action（先做确定性 tool）

### 里程碑二：单视频 Agent1 闭环

8. stage0 → stage7（仅 Agent1）
9. run_one_video.py 人工质检

### 里程碑三：三 Agent + Draft + 升档 + 返工

10. Agent2/3、Draft 创建、promote、revision 路径

### 里程碑四：规模化

11. 其余 production API、batch_run、cache、可观测性

## 9. 测试要求

- 每个 tool 单元测试，mock 外部 API；test 环境禁止真实付费调用。
- schema validator：违反 A/F/G 的样本必须被拒绝；种子名保留且可通过。
- `map_query`：F1 通过（params 含 `latlng`、observation 含 `resolved_latlng`，无重名）；若 observation 仍用 `latlng` 必须被拒绝。
- `map_query` params 交叉约束：仅 query、仅 latlng、二者同时 → 通过；二者皆缺 → 失败。
- `map_query` Observation：success / empty / error 三条 status 条件规则的正反例。
- Draft 创建流程测试。
- stage3→stage7 小样本（mock）测试 role / terminal / handoff；Agent2/Agent3 使用 `map_query` 时读取 `resolved_latlng`。
- promote_tool：备份、重跑 stage4–7、失败回滚；涉及 Observation 字段重命名时不得只做键替换。
- 并发：registry 文件锁与 JSONL 分片合并。

## 10. 明确的禁止事项

- 不要引入 LangChain/LangGraph 做核心流水线。
- 不要用自由文本解析代替 Pydantic。
- 不要在 Observation 里编造真实 tool 返回不了的内容。
- 不要生成含答案泄漏的 Thought；不要把 groundtruth 送入 stage5。
- 不要更改本文档定义的 schema 字段名与结构（除非先修订 SPEC）。
- 不要把 .env 或 data/ 提交到 Git。
- 不要让 Agent2 使用 sun_position_calc；不要让 Agent1 使用 reverse_image_search 和 map_query。
- 不要跳过 G 规则直接建 Draft；不要让新建 Draft 的初始 tier 为 production。
- 不要无标记硬套无法匹配的操作。
- 不要把 draft 合成 Observation 标记为 real_execution。
- 不要自动修改已有 production tool 的 schema。
- 不要多协程直接追加同一最终 JSONL。

## 11. 术语速查

- **Trajectory**：Thought-Action-Observation 序列。
- **Tool / Production / Draft / Executor / Registry**：见上文生命周期定义。
- **SubmitAnswerResult**：Agent2 终端输出与 Agent3 输入。
- **LocationHypothesis**：Agent1→Agent2 交接物。
- **VerificationResult**：Agent3 输出；含 pass/fail 与 return_to_agent。
- **RevisionContext**：返工上下文；区分 video_observed 与 system_feedback。
- **NormalizedStep / ObservationExecutionResult / TrajectoryVerificationReport**：阶段间强类型契约。
- **Loss Masking**：仅 assistant role 算 loss。
- **答案泄漏**：后见之明式 Thought；由时间戳 + stage6 程序化消除。
- **executor_ref**：Python 可导入路径，不是 HTTP 微服务 URL。
- **LoRA**：三 Agent 共享基座、分头训练。
