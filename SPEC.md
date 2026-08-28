# 图片地理定位 Agent 训练数据集生成流水线 — 项目规格

版本：3.4.12 | 生成时间：2026-08-28
修订说明：v3.4.12 取消跨视频持久化的 `tool_trees.json`：阶段3每次只从 `canonical_tool_catalog_v2.json`（`TOOL_CATALOG_PATH`）加载内存中的 ToolForest；本 task 内可 map / 严格新建 / 降 reasoning，但不回写全局 dump、不跨视频积累 variants。`ensure_tool_trees` 签名保留，`trees_path` 仅作测试/CLI 目录覆盖。v3.4.11 阶段4选图质量（Stage 1.5 `needs_review`、包装帧/轨迹–选图错配等）不再进入软审查项或软上限压分；选图瑕疵仍可写入 `image_selection_note` / `input_quality_alignment` 维度分与人工审核卡片，但不得因选图质量单独把 `decision` 压成 `needs_review`。软审查仅保留 Observation 修复与参数 invalid 等非选图项。v3.4.10 阶段2允许注入仅 `role=tool` 的过程时间线紧凑软先验，并增加动作覆盖审核（与 Observation 审核共用三次生成预算）；旁白明确的地图/卫星/街景/量测必须写 `tool_call`，Observation 可为定性报告结果。阶段3对已有 `tool_call` 按 Thought/Params/Observation 重选执行器，不得把 reasoning 升格为工具。详见 docs/OBSERVATION_RETRY.md。v3.4.9 恢复带括号错误反馈的 Stage 2 Observation 批量审核与含首次最多三次生成；达到上限继续下游。当前 Tool v2 与融合 Stage 4 已接入，以下旧修订仅作为历史说明。v3.4.8 阶段3参数编译按方案 A：每个带 `input_schema` 的 `tool_call` 都用一次结构化 LLM 对照 schema，从 Thought、原始 params 与上下文编译 `inputs`（不是「仅缺必填才补洞」）；已由别名/上下文填好的键不覆盖；不得编造坐标、路径、Overpass 代码或来源中不存在的几何；编译后再合同校验；LLM 失败则保留规则结果并失败开放。可由 `STAGE3_COMPILE_PARAMS` 关闭。v3.4.7 曾写成缺必填才抽取（方案 B），已由本版纠正。v3.4.6 删除轨迹–选图一致性检查（不再跳过 Stage 3/4）；视频级 `decision=accept` 且答案非 ambiguous/unsolved 的题必须跑完 Stage 2–4 并入库；`Trajectory.image_paths` 允许空列表，无图也写出 JSONL。v3.4.5 选图质量等级（`accepted`/`needs_review`）只作标注，不再拦住 Stage 2–4；仅答案歧义/无解的 `rejected` 题跳过；每题写入程序化 `image_selection_note`，阶段4 `notes` 可追加该评价。v3.4.4 阶段4读取 `stage3_parameter_audit.json`，将 `ready / context_resolvable / repairable / invalid` 作为程序化维度 `tool_param_correctness`（按最差一次调用取分；`invalid` 记硬门槛只压分）；每条样本 `notes` 必填，弱维度须写可核对明细。阶段3仍负责参数归一并落盘审计，不写入 `quality_score`。v3.4.3 样本置信度改回阶段4（develop）：VLM 多维打分 + 程序化硬门槛只压分不拦入库，删除 Stage 3 后置 `audit_coverage` / 质量路由。v3.4.2 修正 Stage 1.5 对教程/复盘视频的系统性拒识：判断必须截在待定位原图首次出示处，后续人工解出、AI 对战或答案揭晓不能作为 reject 理由；无人机航拍、行车/步行连续现场属于有效 `video_derived` 输入，题面明确给出的时间、IP 属地和文字提示可以作为已知线索。增加 Anthropic relay 枚举包装、概率枚举、`start_s/start_sec/start_time` 等时间别名兼容。v3.4.1 将参数完整度与轨迹语义质量解耦：operation 字段增加 `requirement_level`、`acquisition_hint`、上下文来源和缺参修复动作；当前图片、前置 Tool 结果、活动区域/会话等可使用显式 `$current_image` / `$previous_tool_result` 引用，缺失参数分为 `ready / context_resolvable / repairable / invalid`。v3.4.0 新增 operation 级 `input_schema`。旧规格见 `SPEC_legacy_v2.md`。

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
3. **阶段2（自由事件轨迹）**：优先接收 Stage 1.5 的单题字幕切片 + `image_paths`（可为空）；若上游拆题不可用而收到整视频，则恢复字幕级保险，识别全部独立定位题，并令末步 `location` 以字符串数组按讲解顺序列出全部最终地点。事件分为 `reasoning / tool_call / final`：直接看图、合并已有证据、比较、筛选、排除、排名、形成目标签名和计划转向只写 reasoning；只有真实访问外部搜索、数据库、地图/街景/卫星/天气服务或执行图像/GIS/计算程序并产生新证据时才写 tool_call + Observation。旁白明确打开/平移/调时相的地图或卫星、在底图或街景上测量、打开街景会话，必须写 `tool_call`；Observation 可以是旁白已报告的定性结果（如「许昌几乎都是平原」「河宽约80m」），不必伪造 URL/API 载荷；禁止因「回执不够像搜索结果」把上述动作改成 reasoning，也禁止无旁白依据补做卫星核验。允许连续 reasoning，但每条必须完成实质认知更新；只为解释下一次调用的一句话并入该 tool_call 的 Thought。产物禁止「求助者/网友/评论区」等渠道元话语，末步严格为 `event_type=final`、`final_answer`、`params.location`、`observation=null`。**不做**轨迹–选图一致性门禁。允许写入**紧凑软先验**：仅 `process_intervals` 中 `role=tool` 的 `[start,end)` 列表，标明「旁白对应外部工具操作，非配额、不附工具画面、不含 `show_source`/`reveal`」；是否写 `tool_call` 仍以字幕是否明确执行并报告结果为准，先验不得迫使凑工具步。生成后批量审核 Observation，并做动作覆盖审核（字幕已明确的外部动作是否写成 `tool_call`）；明确伪造或高置信漏动作时在下一轮 prompt 的括号中列出具体错误与修正提醒（不得进入训练产物）。含首次最多生成三次；第三次仍未修复或审核服务失败时保存最近一次结构有效的轨迹、记录问题并继续 Stage 3/4，不据此删除样本。结构/措辞重试也计入同一三次预算；全部结构化生成失败仍报错。
4. **阶段3（执行器级归并、参数合同与格式化）**：**按 task** 从 `TOOL_CATALOG_PATH`（默认 `canonical_tool_catalog_v2.json`）加载执行器目录到**内存** ToolForest，结合自由 Tool 的 Thought、Params、Observation 和调用上下文进行语义归并。对**已有** `tool_call` 可按 Thought/Params/Observation 重选执行器（如误标的 `web_search` → `satellite_imagery_query` / `streetview_query`）；失败开放保留原名；**不得**把 reasoning 升格为新调用，漏抽仍由阶段2重生成解决。同一执行器的不同用途通过 `operation` 区分，不得因参数或目标对象不同拆成新工具；调用参数统一为 `operation + purpose + inputs`。每个 operation 必须有带字段类型、必填关系、`requirement_level`、别名、取值约束、示例、解释和 `acquisition_hint` 的 `input_schema`；Stage 3 先做 operation alias 与输入字段归一，未识别字段保存在 `inputs.extensions`，不得因额外字段直接丢弃。当前图片、前置工具结果、活动区域或会话可显式写成 `$current_image`、`$previous_tool_result`、`$active_area`、`$active_session`；若所需图片尚不存在，指导 Agent 先 crop/zoom/capture 或从视频补截帧并传回图片 ID，禁止编造路径。在 `STAGE3_COMPILE_PARAMS=true`（默认）时，每个带 `input_schema` 的 `tool_call` 用一次结构化 LLM 对照该 `(tool, operation)` schema，从 Thought、原始 params、`extensions` 与上下文**编译** `inputs`（规则别名/上下文仅作种子，不覆盖已填键）；不得编造坐标、路径、Overpass 代码或来源中不存在的几何；程序化过滤后再走同一套合同校验，训练 JSONL 写入编译后的 `inputs`；LLM 失败则保留规则结果并失败开放。缺参审计输出 `ready / context_resolvable / repairable / invalid` 及结构化 repair actions，写入 `stage3_parameter_audit.json`（供阶段4记分；本阶段不据此改轨迹或拦入库）。OSM 默认接收 area/tags 等结构化条件并由执行器生成 Overpass QL；只有原链确实提供代码时才填写可选 `overpass_ql`。确无同类执行器时，模型可仅在本 task 内存中严格新建 Tool；高置信伪工具可降回 reasoning。**不**把 variants 或新建工具写回全局 `tool_trees.json`，也**不**跨视频积累别名。`Trajectory.image_paths` **允许空列表**（无图不伪造占位路径）；JSONL 用户消息仅在有图时写入 `[Image: ...]`。最后写入标准 JSONL、`stage3_tool_mapping.json` 和 `stage3_parameter_audit.json`。阶段3 **不**写入 `quality_score`。
5. **阶段4（融合置信度与人工审核）**：按 task 在 Stage 3 落盘后融合 VLM 的证据、答案、逻辑、输入质量评分与程序化参数/格式审计，输出 `quality_score`、`audit_coverage`、`decision`（accept/provisional_pass/parameter_repair/needs_review/reject）及可核对原因。读取 `stage2_observation_audit.json` 的最后选定版本，已变更的 Observation 不得沿用旧审核通过结论。普通缺参、自然语言占位与可修类型问题路由修复，不等同虚假回执；明确伪造和不可执行语义等严重问题压分，但**不阻止 Stage 3/4 文件保存、不删除轨迹**。**选图质量**（Stage 1.5 `needs_review`、包装帧、轨迹–选图错配等）只反映在 `image_selection_note`、`input_quality_alignment` 与人工审核卡片，**不得**进入软审查项/软上限，也不得仅因选图瑕疵把 `decision` 压成 `needs_review`。Stage 4 本身不再发起重生成，三次封顶纠错只属于 Stage 2。审核失败降低覆盖率，不当作已经证实的低质量。额外写 `stage4_confidence.review.json/.md`：题目与选帧时间、未采用候选、前后字幕、步骤位置及修复指导；模型判断与记录事实分开，供人工核验。仅回写 JSONL 的 `quality_score`，保留 messages；禁止 groundtruth 进入生成或裁判上下文。
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
阶段3    每 task：FreeFormTrajectory + canonical catalog（内存 ToolForest）→ 参数归一/校验 → Trajectory + tool/parameter audit → DatasetEntry JSONL
阶段4    每 task：读 parameter audit + VLM/程序化多维评分 → ConfidenceReport（notes 必填，可含选图评价）+ 回写 DatasetEntry.quality_score（不拦入库）

编排：run_stage{1,2,3,4}.py / run_stage_audit.py / run_one_video.py / batch_run.py
基础 Tool 目录：canonical_tool_catalog_v2.json（无跨视频 runtime dump）
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
├── canonical_tool_catalog_v2.json
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

- 每棵树一个执行器级 canonical（本 task 内存中可临时挂 variants / variant_operations）；Tool 以真实 API/数据库/程序边界划分，不以自然语言动词划分
- 同一执行器通过不同 `operation` 和 `inputs` 完成 query/filter/export/compare 等用途；每次调用必须额外写 `purpose`，解释本次调用补齐什么证据
- 匹配：当前官方目录精确名 → 本 task 语义归并/重分类 → 必要时仅在内存严格新建；不得因输入字段略有差异直接失败或新建；**不**跨视频落盘积累别名
- 新建定义必须包含小写下划线 name、description、executor、usage、至少一个带解释的 operation；运行时外层参数固定为 `operation/purpose/inputs`
- 高置信确认没有外部执行器的伪 Tool 可降回 reasoning，并写入 `stage3_tool_mapping.json`；低置信时不得静默丢失

## 7. 测试

- 全部测试位于 `tests/`
- 禁止真实付费 API；mock adapter
- 跑通：`uv run pytest tests -q`

## 8. 依赖

见 `requirements.txt`。不得新增 LangChain/LangGraph。
