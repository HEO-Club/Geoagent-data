# 图片地理定位 Agent 训练数据集生成流水线 — 项目规格

版本：3.0.2 | 生成时间：2026-08-01  
修订说明：v3.0.2 `working_scope` 双端注入。v3.0.1 删除旧 stage0–7，主包收口为 `pipeline/` / `tests/`。v3.0.0 三阶段架构（字幕 → 自由 TAO → tool 树归一化 JSONL）。旧规格见 `SPEC_legacy_v2.md`。

## 0. 给 Cursor 的元指令（先读这一段）

- 这是一个数据工程流水线项目，不是 Web 应用，不是 agent 应用。
- 本项目的产物是用于 SFT 训练的**单一**地理定位 agent 轨迹数据集（JSONL）。
- 主实现位于 `pipeline/`；测试位于 `tests/`。历史规格仅见 `SPEC_legacy_v2.md`，不再保留旧 stage0–7 代码。
- 请严格按照本文档的模块划分与 data schema 实现，不要擅自更改已定契约字段名。
- 逐阶段实现，每个阶段实现后应可独立测试、独立 CLI 跑通。
- 所有 LLM 调用通过 API/SDK 完成（不涉及任何网页交互）。
- 所有结构化 LLM 输出必须用 Pydantic v2 校验；阶段2 仅对软信封做轻量校验，不对 tool 名/params 做统一 schema。
- 代码需带类型注解、docstring，关键逻辑加中文注释。
- 优先保证可测试、可调试、可断点续跑，其次才是性能。
- 模型名只能从配置读取；SDK 调用必须封装在 adapter 中；不得把具体模型名散落在业务代码中。
- 禁止引入 LangChain / LangGraph 作为核心流水线框架。
- 测试中禁止调用真实付费 API（`ALLOW_REAL_API=false` 时 adapter 必须拒绝）。
- **groundtruth 不得进入阶段2 或任何轨迹生成模型的上下文**（为日后校验预留）。
- **严禁过拟合**：不得为单条视频发明特判硬门禁/词表。

## 1. 项目目标与背景

### 1.1 最终目标

从「用图片做地理定位」的讲解视频中，蒸馏出**一条**可训练的 ReAct 轨迹，供单一地理定位 agent 做 SFT。

### 1.2 三阶段

1. **阶段1（字幕）**：根据视频生成带时间戳字幕。
2. **阶段2（自由 TAO）**：以视频 + 字幕（字幕仅蒸馏材料）先抽取外部给定工作范围，再蒸馏一条 **agent 视角**、内容准确的地理图片定位 TAO 链；thought 写「假设缺口 → 为何调 tool」，不得暴露字幕/旁白来源；去噪静默（不进链、不做删除清单）；不维护 tool 池；tool 由模型发明；无统一 tool schema。
3. **阶段3（格式化）**：维护 tool 树；归并自由 tool；将 `working_scope` 写入训练 `user_query`；输出标准 JSONL。

### 1.3 输入

- 原始视频；可选 ASR（仅阶段1 时间锚）
- groundtruth：本次不消费，不得注入阶段2/3 生成 prompt

### 1.4 质量铁律

- 阶段2 内容忠实、静默去噪、agent 视角决策链；阶段3 格式归一
- 禁止真实 Tool API；禁止 GT 进生成上下文
- 宁缺毋滥；禁止样本特化硬门禁

### 1.5 外部给定线索 / 工作范围（沿用 v2 分层，单 Agent）

- **抽取输入**：仅阶段1 字幕；**禁止**读 groundtruth。
- **`raw_given_clue`**：问题设置段外部沟通原话；角色区分 `photo_location_constraint` / `person_or_social_attribute` / `other_non_location`。
- **`working_scope`**：仅当存在拍摄地硬边界（`bound_kind=inside`）或可核验软先验（`bound_kind=near`，含「籍贯 ∧ 离家不远 ⇒ 籍贯地附近」）时规范化；`region` 为展示短语（如「河南许昌附近」）；**禁止**把软先验升格成「X内」。
- **`candidate_hypothesis`**（博主演绎候选）：可抽取供审计，**不得**写入蒸馏 prompt 的已知范围块或训练 `user_query`。
- **人物属性**：默认不另写「已知线索」段；仅当能推出合法 `working_scope` 时注入展示短语。
- **阶段2 蒸馏 prompt**：有有效 `working_scope` 时增加「Agent 已知工作范围」块（只写展示短语，禁止来源话术）；thought 须将其当先验，不得写「字幕/网友说」。
- **阶段3 `user_query`**：无 scope 时为 `Locate the place shown in the image.`；有 scope 时追加一行 `Working scope: {region}`。

## 2. 系统总体架构

```
阶段1  视频 → TranscriptSegment 列表
阶段2  字幕 → working_scope；视频 + 字幕 + working_scope → FreeFormTrajectory
阶段3  FreeFormTrajectory + tool_trees → Trajectory（user_query 含 working_scope）→ DatasetEntry JSONL

编排：run_stage{1,2,3}.py / run_one_video.py / batch_run.py
Tool 树：tool_trees.json
```

## 3. 目录结构

```
geo-agent-dataset/
├── SPEC.md
├── SPEC_legacy_v2.md
├── pipeline/
│   ├── config.py
│   ├── llm.py
│   ├── schemas/
│   ├── media/
│   ├── stage1_transcript/
│   ├── stage2_freeform_tao/
│   ├── stage3_normalize_format/
│   └── orchestrator.py
├── tool_trees.json
├── run_stage1.py
├── run_stage2.py
├── run_stage3.py
├── run_one_video.py
├── batch_run.py
├── tests/
└── data/
    ├── raw_videos/
    ├── transcripts/
    ├── intermediate/{video_id}/
    │   ├── manifest_v2.json
    │   ├── stage1_transcript.json
    │   ├── stage2_freeform_tao.json
    │   └── stage3_trajectory.json
    └── output/
        ├── shards/{video_id}.jsonl
        └── geolocate_agent.jsonl
```

## 4. 数据 Schema

见 `pipeline/schemas/`：`TranscriptSegment`、`RawGivenClue`/`WorkingScope`/`ClueExtractionResult`、`FreeFormStep`/`FreeFormTrajectory`（含可选 `working_scope`）、`ToolTree`/`ToolForest`、`Trajectory`、`DatasetEntry`（无 agent_role / 强制 GT / verified）。

## 5. 阶段接口

```python
def run_stage1(video_path: str, *, anchor_transcript_path: str | None = None,
               out_path: str | None = None) -> list[TranscriptSegment]: ...

def extract_working_scope(transcript: list[TranscriptSegment]) -> ClueExtractionResult: ...
def run_stage2(video_path: str, transcript: list[TranscriptSegment],
               *, out_path: str | None = None) -> FreeFormTrajectory: ...

def build_user_query(working_scope: WorkingScope | None = None) -> str: ...
def ensure_tool_trees(freeform: FreeFormTrajectory, trees_path: Path) -> ToolForest: ...
def remap_trajectory(...) -> Trajectory: ...
def format_dataset_entry(traj: Trajectory, *, source_video: str) -> DatasetEntry: ...
def run_stage3(freeform: FreeFormTrajectory, ...) -> DatasetEntry: ...
```

## 6. Tool 树规则

- 每棵树一个 canonical + variants
- 匹配：精确 →（可选）语义 matcher → 新建
- 并发写须文件锁 + 原子写

## 7. 测试

- 全部测试位于 `tests/`
- 禁止真实付费 API；mock adapter
- 跑通：`uv run pytest tests -q`

## 8. 依赖

见 `requirements.txt`。不得新增 LangChain/LangGraph。
