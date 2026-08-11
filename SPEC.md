# 图片地理定位 Agent 训练数据集生成流水线 — 项目规格

版本：3.3.0 | 生成时间：2026-08-12
修订说明：v3.3.0 将阶段2改为 `reasoning / tool_call / final` 三类事件，允许连续 Thought 且只为真实外部动作生成 Tool；阶段3引入执行器级 Canonical Tool 目录、operation/purpose/inputs 参数契约、上下文语义归并、受控自动新建与伪工具降级审计。v3.2.0 增加低误报条件式拆分复核、task 级答案/质量门禁、题内渐进选图、单图择优与视觉去重；单个 task 不合格不再中断同视频其他题。旧规格见 `SPEC_legacy_v2.md`。

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

从「用图片做地理定位」的讲解视频中，蒸馏出可训练的 ReAct 轨迹，供单一地理定位 agent 做 SFT。一视频可产出**多条**样本（每定位任务一条链 / 一条 JSONL）。

### 1.2 阶段

1. **阶段1（字幕）**：根据视频生成带时间戳字幕。
2. **阶段1.5（审核切分）**：结合字幕 + 稀疏抽帧做语义审核（禁止词表特判）：
   - **拒识主问句**：去掉讲解/旁白/答案后，是否仍存在需 agent 定位的图或场景？无 → `decision=reject`。
   - **切分粒度**：一个 task = 一次独立定位题 = 一条最终 `final_answer` 链。同题多图（共同支撑同一地点、或后图精化前图）合并为 **一个** task 且 `multi_target_images=true`；**同一最终地点必须合并为一个 task**；仅不同目标/不同最终地点才拆多 task。
   - **条件式复核**：首次切分由模型负责。规则只检查高精度结构矛盾（模型主动低置信、明确答案重复、resolved 却无地点、时间窗与候选严重冲突等）；无异常不重复调用模型。触发后由模型保守地双向复核，可合并过拆，也可补拆漏题；不得为了体现复核而改动正确切分。
   - **task 时间窗**：`time_start`/`time_end`（及可选 segment 索引）覆盖**整条答案链旁白**（问题设定 → 推理比对叙述 → 最终地点结论），不得裁成仅关键帧中段；时间窗管蒸馏材料完整性。
   - **关键帧 = 待定位实拍输入**（去掉讲解后 agent 仍要据之定位的现场/静帧）。候选只在归一后的 task 时间窗内产生；与边界相差很小的模型候选可先扩充时间窗，禁止把大量越界候选强行挤到边界同一帧。首次候选不合格时在该题范围内渐进补探一次，不得跨题寻找。
   - **逐帧质量验收与择优**：VLM 对候选分别输出角色、质量、答案泄露、讲解覆盖和干净原图标记。普通 `still_image` 可比较多张候选，但最终只保留质量最高的一张；工具/核验/揭晓帧不得入 `image_paths`，含最终答案泄露的帧不得入库。时间距离不得用于判断重复，使用视觉哈希去除同图变体。
   - **`video_derived` / 同题多输入**：提案须先理解定位过程角色，一次列全各独立**实拍**定位输入代表帧（张数随输入镜头走，受配置上限约束）；`task_summary` 枚举的每个实拍输入须各有时间戳；不得用工具步骤或揭晓帧充数。
   - **答案与 task 门禁**：每题分别输出 `answer_status=resolved|ambiguous|unsolved` 与 `status=accepted|needs_review|rejected`。只有答案唯一明确、图片完整干净的 `accepted` task 进入 Stage 2；模棱两可或明确无解的题单独拒绝，选图不足进入人工复核，均不得影响同视频其他题。
   - 数量：`still_image` 默认输出 1 张最佳帧；`multi_target_images` 或 `video_derived` 按 `expected_image_count` 保留各独立输入，不再硬编码为 2 张。
3. **阶段2（自由事件轨迹）**：**按 task** 以字幕切片 + `image_paths` 蒸馏 agent 视角轨迹。事件分为 `reasoning / tool_call / final`：直接看图、合并已有证据、比较、筛选、排除、排名、形成目标签名和计划转向只写 reasoning；只有真实访问外部搜索、数据库、地图/街景/卫星/天气服务或执行图像/GIS/计算程序并产生新证据时才写 tool_call + Observation。允许连续 reasoning，但每条必须完成实质认知更新；只为解释下一次调用的一句话并入该 tool_call 的 Thought。产物禁止「求助者/网友/评论区」等渠道元话语，末步严格为 `event_type=final`、`final_answer`、`params.location`、`observation=null`。
4. **阶段3（执行器级归并与格式化）**：**按 task** 从 `canonical_tool_catalog.json` 加载小型执行器目录，结合自由 Tool 的 Thought、Params、Observation 和调用上下文进行语义归并。同一执行器的不同用途通过 `operation` 区分，不得因参数或目标对象不同拆成新工具；调用参数统一为 `operation + purpose + inputs`，其中 `inputs` 宽容保留原始字段。确无同类执行器时，模型可严格生成含 description/executor/usage/operations 的新 Tool 定义；高置信伪工具可降回 reasoning。最后写入 `working_scope`、标准 JSONL 和 `stage3_tool_mapping.json` 审计指标。
### 1.3 输入

- 原始视频；可选 ASR（仅阶段1 时间锚）
- groundtruth：本次不消费，不得注入阶段2/3 生成 prompt

### 1.4 质量铁律

- 阶段2 内容忠实、静默去噪、agent 视角决策链；Observation 不得补写材料中不存在的精细数据；阶段3 格式归一
- 禁止真实 Tool API；禁止 GT 进生成上下文
- 宁缺毋滥；禁止样本特化硬门禁

### 1.5 外部给定线索 / 工作范围（沿用 v2 分层，单 Agent）

- **抽取输入**：仅该 task 的字幕切片；**禁止**读 groundtruth。
- **`raw_given_clue`**：问题设置段外部沟通原话；角色区分 `photo_location_constraint` / `person_or_social_attribute` / `other_non_location`。
- **`working_scope`**：仅当存在拍摄地硬边界（`bound_kind=inside`）或可核验软先验（`bound_kind=near`，含「籍贯 ∧ 离家不远 ⇒ 籍贯地附近」）时规范化；`region` 为展示短语；**禁止**把软先验升格成「X内」。城市级「拍摄地/拍摄城市就是 X」属硬边界；抽取看约束语义而非来源渠道（渠道无关）。
- **`candidate_hypothesis`**（博主演绎候选）：可抽取供审计，**不得**写入蒸馏 prompt 的已知范围块或训练 `user_query`。
- **阶段2 蒸馏 prompt**：有有效 `working_scope` 时增加「Agent 已知工作范围」块；thought 须将其当先验，不得写「字幕/网友说」。
- **阶段3 `user_query`**：无 scope 时为 `Locate the place shown in the image.`；有 scope 时追加一行 `Working scope: {region}`。

## 2. 系统总体架构

```
阶段1    视频 → TranscriptSegment 列表
阶段1.5  字幕 + 稀疏帧 → AuditSplitResult（视频审核 + task 级门禁/关键帧）
阶段2    每个 accepted task：字幕切片 + image_paths → reasoning/tool_call/final FreeFormTrajectory
阶段3    每 task：FreeFormTrajectory + canonical catalog/runtime trees → Trajectory + tool mapping audit → DatasetEntry JSONL

编排：run_stage{1,2,3}.py / run_stage_audit.py / run_one_video.py / batch_run.py
基础 Tool 目录：canonical_tool_catalog.json；运行时增量：tool_trees.json
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
│   ├── stage_audit_split/
│   ├── stage2_freeform_tao/
│   ├── stage3_normalize_format/
│   └── orchestrator.py
├── tool_trees.json
├── canonical_tool_catalog.json
├── run_stage1.py
├── run_stage_audit.py
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
    │   ├── stage_audit_split_draft.json
    │   ├── stage_audit_split.json
    │   └── tasks/{task_id}/
    │       ├── task_audit.json
    │       ├── candidates/{task_id}_t*.jpg
│       ├── stage2_freeform_tao.json      # 仅 accepted
│       ├── stage3_trajectory.json        # 仅 accepted
│       └── stage3_tool_mapping.json      # Tool 归并与伪工具指标
    └── output/
        ├── shards/{task_id}.jsonl
        └── geolocate_agent.jsonl
```

## 4. 数据 Schema

见 `pipeline/schemas/`：`TranscriptSegment`、`AuditSplitResult`/`GeoTaskSpec`、`RawGivenClue`/`WorkingScope`/`ClueExtractionResult`、`FreeFormStep`/`FreeFormTrajectory`、`ToolTree`/`ToolForest`、`Trajectory`（`image_paths: list[str]`）、`DatasetEntry`。

## 5. 阶段接口

```python
def run_stage1(video_path: str, *, anchor_transcript_path: str | None = None,
               out_path: str | None = None) -> list[TranscriptSegment]: ...

def run_audit_split(video_path: str, transcript: list[TranscriptSegment],
                    *, out_path: str | None = None) -> AuditSplitResult: ...

def extract_working_scope(transcript: list[TranscriptSegment]) -> ClueExtractionResult: ...
def run_stage2(video_path: str, transcript: list[TranscriptSegment],
               *, image_paths: list[str] | None = None,
               out_path: str | None = None) -> FreeFormTrajectory: ...

def build_user_query(working_scope: WorkingScope | None = None) -> str: ...
def ensure_tool_trees(freeform: FreeFormTrajectory, trees_path: Path) -> ToolForest: ...
def remap_trajectory(...) -> Trajectory: ...
def format_dataset_entry(traj: Trajectory, *, source_video: str) -> DatasetEntry: ...
def run_stage3(freeform: FreeFormTrajectory, ..., image_paths: list[str] | None = None) -> DatasetEntry: ...
```

## 6. Tool 树规则

- 每棵树一个执行器级 canonical + variants + variant_operations；Tool 以真实 API/数据库/程序边界划分，不以自然语言动词划分
- 同一执行器通过不同 `operation` 和 `inputs` 完成 query/filter/export/compare 等用途；每次调用必须额外写 `purpose`，解释本次调用补齐什么证据
- 匹配：精确 variant → 读取 Thought/Params/Observation 的上下文语义归并 → 严格自动新建；不得因输入字段略有差异直接失败或新建
- 新建定义必须包含小写下划线 name、description、executor、usage、至少一个带解释的 operation；运行时外层参数固定为 `operation/purpose/inputs`
- 高置信确认没有外部执行器的伪 Tool 可降回 reasoning，并写入 `stage3_tool_mapping.json`；低置信时不得静默丢失
- 并发写须文件锁 + 原子写

## 7. 测试

- 全部测试位于 `tests/`
- 禁止真实付费 API；mock adapter
- 跑通：`uv run pytest tests -q`

## 8. 依赖

见 `requirements.txt`。不得新增 LangChain/LangGraph。
