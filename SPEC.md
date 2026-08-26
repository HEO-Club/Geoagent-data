# 图片地理定位 Agent 训练数据集生成流水线 — 项目规格

版本：3.4.8 | 生成时间：2026-08-26
修订说明：v3.4.8 阶段3参数编译按方案 A：每个带 `input_schema` 的 `tool_call` 都用一次结构化 LLM 对照 schema，从 Thought、原始 params 与上下文编译 `inputs`（不是「仅缺必填才补洞」）；已由别名/上下文填好的键不覆盖；不得编造坐标、路径、Overpass 代码或来源中不存在的几何；编译后再合同校验；LLM 失败则保留规则结果并失败开放。可由 `STAGE3_COMPILE_PARAMS` 关闭。v3.4.7 曾写成缺必填才抽取（方案 B），已由本版纠正。v3.4.6 删除轨迹–选图一致性检查（不再跳过 Stage 3/4）；视频级 `decision=accept` 且答案非 ambiguous/unsolved 的题必须跑完 Stage 2–4 并入库；`Trajectory.image_paths` 允许空列表，无图也写出 JSONL。v3.4.5 选图质量等级（`accepted`/`needs_review`）只作标注，不再拦住 Stage 2–4；仅答案歧义/无解的 `rejected` 题跳过；每题写入程序化 `image_selection_note`，阶段4 `notes` 可追加该评价。v3.4.4 阶段4读取 `stage3_parameter_audit.json`，将 `ready / context_resolvable / repairable / invalid` 作为程序化维度 `tool_param_correctness`（按最差一次调用取分；`invalid` 记硬门槛只压分）；每条样本 `notes` 必填，弱维度须写可核对明细。阶段3仍负责参数归一并落盘审计，不写入 `quality_score`。v3.4.3 样本置信度改回阶段4（develop）：VLM 多维打分 + 程序化硬门槛只压分不拦入库，删除 Stage 3 后置 `audit_coverage` / 质量路由。v3.4.2 修正 Stage 1.5 对教程/复盘视频的系统性拒识：判断必须截在待定位原图首次出示处，后续人工解出、AI 对战或答案揭晓不能作为 reject 理由；无人机航拍、行车/步行连续现场属于有效 `video_derived` 输入，题面明确给出的时间、IP 属地和文字提示可以作为已知线索。增加 Anthropic relay 枚举包装、概率枚举、`start_s/start_sec/start_time` 等时间别名兼容。v3.4.1 将参数完整度与轨迹语义质量解耦：operation 字段增加 `requirement_level`、`acquisition_hint`、上下文来源和缺参修复动作；当前图片、前置 Tool 结果、活动区域/会话等可使用显式 `$current_image` / `$previous_tool_result` 引用，缺失参数分为 `ready / context_resolvable / repairable / invalid`。v3.4.0 新增 operation 级 `input_schema`。旧规格见 `SPEC_legacy_v2.md`。

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
- **groundtruth 不得进入阶段2、阶段3 生成或阶段4 裁判上下文**（为日后校验预留）。
- **严禁过拟合**：不得为单条视频发明特判硬门禁/词表。

## 1. 项目目标与背景

### 1.1 最终目标

从「用图片做地理定位」的讲解视频中，蒸馏出可训练的 ReAct 轨迹，供单一地理定位 agent 做 SFT。一视频可产出**多条**样本（每定位任务一条链 / 一条 JSONL）。

### 1.2 阶段

1. **阶段1（字幕）**：根据视频生成带时间戳字幕。
2. **阶段1.5（审核切分）**：结合字幕 + 稀疏抽帧做语义审核（禁止词表特判）：
   - **拒识主问句**：去掉讲解/旁白/答案后，是否仍存在需 agent 定位的图或场景？无 → `decision=reject`。
   - **切分粒度**：一个 task = 一次独立定位题 = 一条最终 `final_answer` 链。同题多图（共同支撑同一地点、或后图精化前图）合并为 **一个** task 且 `multi_target_images=true`；**同一最终地点必须合并为一个 task**；仅不同目标/不同最终地点才拆多 task。
   - **条件式复核**：首次切分由模型负责。规则只检查高精度结构矛盾（模型主动低置信、resolved 却无地点、时间窗与候选严重冲突等）以及**答案同地嫌疑**：≥2 个 `resolved` task 时，最终地点字符串全等可直接视为重复；若措辞不同，另用一次廉价 LLM（可带字幕上下文）判定是否同一最终地点或同一答案链（同题多图/后图精化），命中则记异常。无异常不重复调用双向复核模型。触发后由模型保守地双向复核，可合并过拆，也可补拆漏题；不得为了体现复核而改动正确切分；不得用脆弱地名归一化/词表代替同地判定。
   - **蒸馏窗**：`time_start`/`time_end`（及可选 segment 索引）覆盖**整条答案链旁白**（问题设定 → 推理比对叙述 → 最终地点结论），不得裁成仅关键帧中段；蒸馏窗只管 Stage 2 字幕切片完整性，**不得**作为密采样选图区间。
   - **出示窗（选图）**：审核模型仍可给单段出示粗窗（弱先验）。选图前另从蒸馏窗字幕抽取题级 **过程时间线** `process_intervals`（角色：`show_source` / `tool` / `reveal` / `other`；可用稀疏审计帧，不为时间线再密采样）。**密采样区间 = 所有 `show_source` 的并集**，裁进蒸馏窗、不得跨相邻题；`display_time_start/end` 记为该并集包络（时间线失败或无 `show_source` 时回退现行单出示粗窗 / 蒸馏窗前段封顶）。**不再报预定张数，也不再以精确关键帧秒数为主提案**。禁止把密采样扩大到整条蒸馏窗或 `tool`/`reveal` 段狂打 VLM。
   - **关键帧 = 定位链实际依赖的待定位实拍**（去掉讲解后 agent 仍要据之定位的现场/静帧）。选图目标是每题 **最小源输入集**（不是出示窗内任意干净实拍，也不是凑字幕估数/视觉簇个数）。流程：蒸馏窗字幕抽取题级 **`visual_evidence_brief`**（仅「看原图/现场时用到的视觉事实」，不含工具核验/搜索命中/揭晓）→ 抽取 `process_intervals` → 仅在 `show_source` 并集密采样 → 廉价视觉过滤（近黑/明显地图或大面积 UI 降权）→ VLM 逐帧验收（注入该时刻区间角色软先验）→ **出示连续段折叠**（`unused_broll` 与工具/揭晓一样打断连续段；每段只留质量最高代表）→ **源输入关系判定**（先做包含/放大硬合并：拼图一格的全屏再出示、裁切放大必须同组且优先留更完整帧；再两两判定是否同一张照片：`same_photo` / `same_scene` / `different_photo` / `not_input`，union-find 聚合；brief 只核验画面事实、**不得**当作张数配额；不得因「两段出示窗」硬拆成两张；不确定是否同一张照片时仍合并）→ 组间按证据支撑排序、组内按干净度择优 → 配置上限截断。找不到干净原图、有 brief 但支撑过低、或时间线给出 ≥2 段不相邻 `show_source` 却只选出 1 张且 brief 非空时标 `needs_review` 并写入 `image_selection_note`（质量等级只作标注，**不拦** Stage 2–4）；禁止用空镜/`unused_broll`/工具帧充数。
   - **逐帧验收与择优**：VLM 对候选分别输出 `kind`、质量、答案泄露、讲解覆盖、干净原图标记，以及 **`evidence_role`**（`problem_input` / `unused_broll` / `process_tool` / `reveal` / …）与 **`chain_support_score`**（画面支撑 brief 视觉事实的程度）。区间角色为软先验：画面仍是最终裁判；`tool`/`reveal` 降低「全屏建筑 = 原图」倾向。择优顺序：**证据角色/支撑分 → 干净原图 → 无讲解覆盖 → 质量分**（思维链选源，干净度选帧）。工具/核验/揭晓帧不得入 `image_paths`，含最终答案泄露的帧不得入库。不得用时间距离判断「是否新输入」；不确定是否同一张照片时不追加。画面主体场景结构明显不同（不同建筑立面/店面/路幅，非同一取景远近）必须判为不同原图。brief 为空时回退现行质量择优路径。每题须程序化写入 **`image_selection_note`**（含质量等级 `accepted`/`needs_review`、选中张数与每帧 quality/overlay/clean/support/reason；若有选图门禁原因则原样写入）。
   - **`video_derived` / 同题多输入**：默认 1 个现场代表帧；同一飞行/同一段路的换机位不加张。多张只来自被判定为不同的源输入（不同原图或另一段独立现场），受配置上限约束；不得用工具步骤或揭晓帧充数。
   - **答案与 task 门禁**：每题分别输出 `answer_status=resolved|ambiguous|unsolved` 与 `status=accepted|needs_review|rejected`。视频级 `decision=accept` 后，答案非 ambiguous/unsolved 的题（`accepted` **与** `needs_review`，含 `image_paths` 为空）**必须**跑完 Stage 2–4 并入库；仅答案模棱两可或明确无解的 `rejected` 题跳过下游；选图质量瑕疵只标等级与 note，不拦流水线；均不得影响同视频其他题。切题过粗（一题塞多个最终地点）走结构复核，**不在选图里用答案个数定张数**。
   - 数量：普通静图默认 1 张最佳帧；多张只来自源输入关系判定为不同；`expected_image_count` / `multi_target_images` 为选完后的实选结果（`len(image_paths)` / `>1`），不作预定配额。
3. **阶段2（自由事件轨迹）**：优先接收 Stage 1.5 的单题字幕切片 + `image_paths`（可为空）；若上游拆题不可用而收到整视频，则恢复字幕级保险，识别全部独立定位题，并令末步 `location` 以字符串数组按讲解顺序列出全部最终地点。事件分为 `reasoning / tool_call / final`：直接看图、合并已有证据、比较、筛选、排除、排名、形成目标签名和计划转向只写 reasoning；只有真实访问外部搜索、数据库、地图/街景/卫星/天气服务或执行图像/GIS/计算程序并产生新证据时才写 tool_call + Observation。允许连续 reasoning，但每条必须完成实质认知更新；只为解释下一次调用的一句话并入该 tool_call 的 Thought。产物禁止「求助者/网友/评论区」等渠道元话语，末步严格为 `event_type=final`、`final_answer`、`params.location`、`observation=null`。**不做**轨迹–选图一致性门禁，生成后直接进入 Stage 3。过程时间线不得写入阶段2 prompt。
4. **阶段3（执行器级归并、参数合同与格式化）**：**按 task** 从 `canonical_tool_catalog.json` 加载执行器目录，结合自由 Tool 的 Thought、Params、Observation 和调用上下文进行语义归并。同一执行器的不同用途通过 `operation` 区分，不得因参数或目标对象不同拆成新工具；调用参数统一为 `operation + purpose + inputs`。每个 operation 必须有带字段类型、必填关系、`requirement_level`、别名、取值约束、示例、解释和 `acquisition_hint` 的 `input_schema`；Stage 3 先做 operation alias 与输入字段归一，未识别字段保存在 `inputs.extensions`，不得因额外字段直接丢弃。当前图片、前置工具结果、活动区域或会话可显式写成 `$current_image`、`$previous_tool_result`、`$active_area`、`$active_session`；若所需图片尚不存在，指导 Agent 先 crop/zoom/capture 或从视频补截帧并传回图片 ID，禁止编造路径。在 `STAGE3_COMPILE_PARAMS=true`（默认）时，每个带 `input_schema` 的 `tool_call` 用一次结构化 LLM 对照该 `(tool, operation)` schema，从 Thought、原始 params、`extensions` 与上下文**编译** `inputs`（规则别名/上下文仅作种子，不覆盖已填键）；不得编造坐标、路径、Overpass 代码或来源中不存在的几何；程序化过滤后再走同一套合同校验，训练 JSONL 写入编译后的 `inputs`；LLM 失败则保留规则结果并失败开放。缺参审计输出 `ready / context_resolvable / repairable / invalid` 及结构化 repair actions，写入 `stage3_parameter_audit.json`（供阶段4记分；本阶段不据此改轨迹或拦入库）。OSM 默认接收 area/tags 等结构化条件并由执行器生成 Overpass QL；只有原链确实提供代码时才填写可选 `overpass_ql`。确无同类执行器时，模型可严格新建 Tool；高置信伪工具可降回 reasoning。`Trajectory.image_paths` **允许空列表**（无图不伪造占位路径）；JSONL 用户消息仅在有图时写入 `[Image: ...]`。最后写入标准 JSONL、`stage3_tool_mapping.json` 和 `stage3_parameter_audit.json`。阶段3 **不**写入 `quality_score`。
5. **阶段4（样本置信度评分）**：在阶段3 JSONL 落盘之后，**按 task** 综合字幕切片、选图/brief、自由轨迹、规范化轨迹、tool mapping 与 **`stage3_parameter_audit.json`**，输出多维质量置信度，供人工按分排队检查。VLM 维度分须按刻度苛评：0.95+ 仅限近乎无可指摘；能复述字幕但存在答案粒度偏粗、Observation 像讲解总结、选中图带讲解包装等瑕疵时落在中高档而非满分。**`tool_param_correctness` 由程序化规则覆盖**（读取参数审计四级 readiness，按最差一次 Tool 调用取分：无调用/全 ready→1.0、context_resolvable→0.80、repairable→0.45、invalid→0.15；审计缺失→0.50 失败开放）；任一次 `invalid` 记程序化硬门槛 `tool_params_invalid`。程序化硬门槛（缺 final / 空 location / 格式契约破坏 / 选中帧答案泄露 / 空字幕 / `tool_params_invalid` 等）与 VLM 裁判硬门槛（伪造 Observation、工具参数不可执行、无源精细数据、图轨不一致、多目标明显漏答等）命中时只将 `quality_score` 压到配置上限，**不拦入库、不改轨迹、不换图、不重蒸**。`review_priority` 默认：`<0.70` high、`<0.93` medium、否则 low（可由配置覆盖）。**每条报告 `notes` 必填**（含总分/优先级/硬门槛、参数质检四级计数与非 ready 调用明细、弱维度说明；若 Stage 1.5 有 `image_selection_note` 则追加，只作人工检查，不改打分）；弱维度（默认 score < 0.80）须写可核对证据。详细报告写入 `stage4_confidence.json`（可含 `parameter_readiness` 摘要），并回写 `DatasetEntry.quality_score`（只改该字段）。LLM 调用失败时失败开放：程序化门槛与参数分仍生效，其余维度记中性分。禁止 groundtruth 进入裁判上下文。
### 1.3 输入

- 原始视频；可选 ASR（仅阶段1 时间锚）
- groundtruth：本次不消费，不得注入阶段2/3 生成 prompt 或阶段4 裁判上下文

### 1.4 质量铁律

- 阶段2 内容忠实、静默去噪、agent 视角决策链；Observation 不得补写材料中不存在的精细数据；阶段3 格式归一；阶段4 只评分不门禁
- 禁止真实 Tool API；禁止 GT 进生成与裁判上下文
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
阶段1.5  字幕 + 稀疏帧 → AuditSplitResult（视频审核 + task 级标注/关键帧 + image_selection_note）
阶段2    每个 accepted/needs_review task：字幕切片 + image_paths → reasoning/tool_call/final FreeFormTrajectory
阶段3    每 task：FreeFormTrajectory + canonical catalog/runtime trees → 参数归一/校验 → Trajectory + tool/parameter audit → DatasetEntry JSONL
阶段4    每 task：读 parameter audit + VLM/程序化多维评分 → ConfidenceReport（notes 必填，可含选图评价）+ 回写 DatasetEntry.quality_score（不拦入库）

编排：run_stage{1,2,3,4}.py / run_stage_audit.py / run_one_video.py / batch_run.py
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
│   ├── stage4_confidence/
│   └── orchestrator.py
├── tool_trees.json
├── canonical_tool_catalog.json
├── run_stage1.py
├── run_stage_audit.py
├── run_stage2.py
├── run_stage3.py
├── run_stage4.py
├── run_one_video.py
├── batch_run.py
├── tests/
└── data/
    ├── raw_videos/                 # 原始视频（只读输入）
    ├── transcripts/                # 外部/锚点字幕
    ├── selected/{video_id}/{task_id}/   # 最终选中的待定位图（给人看/进 Stage2）
    ├── runs/                       # 跑批/重跑摘要与日志（非训练产物）
    ├── intermediate/{video_id}/    # 阶段元数据（JSON），不含密采样探测图
    │   ├── manifest_v2.json
    │   ├── stage1_transcript.json
    │   ├── stage_audit_split_draft.json
    │   ├── stage_audit_split.json
    │   └── tasks/{task_id}/
    │       ├── task_audit.json
    │       ├── candidate_assessments.partial.json  # 验收断点（可删）
    │       ├── stage2_freeform_tao.json            # accepted / needs_review
    │       ├── stage3_trajectory.json
    │       ├── stage3_tool_mapping.json
    │       ├── stage3_parameter_audit.json
    │       └── stage4_confidence.json              # 置信度报告（人工检查）
    └── output/
        ├── shards/{task_id}.jsonl
        └── geolocate_agent.jsonl
.cache/
    ├── audit_sparse/               # 审核用稀疏帧
    ├── audit_candidates/           # 出示窗密采样探测帧（可随时清空）
    └── keyframes/                  # 其他抽帧缓存
```

## 4. 数据 Schema

见 `pipeline/schemas/`：`TranscriptSegment`、`AuditSplitResult`/`GeoTaskSpec`（含 `image_selection_note`）、`RawGivenClue`/`WorkingScope`/`ClueExtractionResult`、`FreeFormStep`/`FreeFormTrajectory`、`ToolTree`/`ToolForest`、`ToolInputSchema`/`ToolParameterAudit`、`Trajectory`（`image_paths: list[str]`，允许空）、`DatasetEntry`、`DimensionScore`/`HardGateHit`/`ParameterReadinessSummary`/`ConfidenceReport`/`ConfidenceJudgeDraft`。

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

def run_stage4(*, task: GeoTaskSpec, transcript: list[TranscriptSegment],
               freeform: FreeFormTrajectory, trajectory: Trajectory,
               entry: DatasetEntry, ..., parameter_audit_path=None,
               judge=None) -> ConfidenceReport: ...
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
