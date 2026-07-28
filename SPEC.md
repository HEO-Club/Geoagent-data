# 图片地理定位 Agent 训练数据集生成流水线 — 项目规格

版本：2.3.25 | 生成时间：2026-07-28  
修订说明：v2.3.25 假 empty 分离（合成/schema 耗尽→error）；Obs 近义字段 coerce；COARSE 投影保留 error 与有地理增益的 empty，禁止因 success 掏空中间失败步。v2.3.24 COARSE 覆盖试错+区域成功；禁中点切分；Move 去噪替代最短蒸馏硬目标。

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

- **Agent1（粗定位 LoRA / COARSE）**：识别地理/人文特征，用地理常识演绎与排除，逐步缩小到国家/地区级别
- **Agent2（精定位 LoRA / FINE）**：假设验证，从粗定位结论精确锁定到坐标级别
- **Agent3（验证 LoRA / VERIFIER）**：交叉验证，验证坐标与图像特征是否自洽，不通过则打回

每个 agent 独立执行 ReAct 循环：

观察输入 → 思考(Thought) → 调用工具(Action) → 得到结果(Observation) → 再思考 → ... → 输出结论

### 1.2 本项目要做什么

从各大自媒体平台爬取的「用图片做地理定位」的讲解视频中，结合预先提取好的带时间戳文字稿，把人类博主的定位推理过程和操作行为蒸馏成三套独立的 ReAct 轨迹，分别供三个 LoRA 训练。

### 1.3 可用输入资源（每条视频）

- 原始视频文件（mp4 等）
- 带时间戳文字稿（预先准备好，格式见第4节 TranscriptSegment schema）
  - 来源允许：`asr_raw`（平台/ASR 预提取）或 `vlm_transcript`（分窗多模态重转录，见 `prep_transcript_vlm`）
  - 直接作为 `Move.narration` 与 stage0 边界识别来源；推荐优先使用校正后的 `vlm_transcript` 以降低 ASR 错字
  - 分窗 VLM 重转录：以时间窗抽关键帧生成该窗正文；可选旧 ASR **仅作时间锚**，不采信其正文；禁止无时间结构的自由编造
  - 同时用于识别 agent 阶段切换点、精确定位答案泄漏边界
- 真实定位答案（从视频元数据或结尾提取的 groundtruth 坐标）
  - **groundtruth 只能进入 stage6**（验证、泄漏检测、过滤），不得进入 stage5 轨迹重构 LLM prompt

### 1.4 三个 Agent 的职责边界

**Agent1 粗定位（COARSE）：**

- 输入：原始图片（优先证据内容区代表图）；`user_query` 可含**工作范围**（见下）与原始沟通信息；不得提前注入博主推理出的候选地名
- 推理模式：**视频证据地理推理链**——特征识别 → 候选排除/修正（可含试错）→ 命名自然区域或同层行政区；禁止自由看图发明视频未提出的新事实；须覆盖区域级成功推进，不得只截取前半试错窗
- **线索分层**：
  - `raw_given_clue`：问题设置阶段外部沟通原话；抽取时须区分是否关于**拍摄地**（`photo_location_constraint`）还是人物籍贯/身份等（`person_or_social_attribute`）及其它非地点信息
  - `working_scope`：可直接用于候选过滤的工作范围，须与原文**边界强度**一致：
    - **硬边界**（`bound_kind=inside`）：仅当原文直接说拍摄地在「X内 / 未出X」等；`region` 写「X内」类短语
    - **软先验**（`bound_kind=near`）：聊天「拍摄地为X附近」、或籍贯地名 +「离家不远/附近」等软距离话；`region` 写「X附近」。软先验**允许**作为后续推理基础，**禁止**升格写成「X内」或「就在X市」
    - 粒度不得细于原文能支撑的界；**不要求**每步视觉再证明，也**不得**单凭它得出最终 POI
  - `candidate_hypothesis`：博主演绎候选，不可自动升级为证据，不得写入 Agent1 `user_query`
  - Agent1 `user_query` **只写入**有效 `working_scope` 的展示短语（如「河南许昌附近」），**不得**在格式化时强行追加「内」；人物属性原话默认不另写「已知线索」段
- **逐视频来源契约**：stage3 对每个视频独立抽取带 Move 索引、时间窗和原文的原子事实；不使用固定地名、地貌、设施或候选词表。Thought / Observation / 排除对象的每个事实性短语都必须被其引用的本视频事实直接蕴含；图片 **仅确认**被引用目标是否可见，不得授权发现新事实
- **空间作用域（训练契约，非程序化特判门禁）**：每条视频事实须标注 `subject_scope`（`camera_position` / `scene_region` / `location_candidate` / `unknown`）与可引用的 `spatial_anchor`（原文方位/对象短语）。不同空间对象上的观察结论可并存；`update` 步可同时引用多个不同作用域的 `video_fact_id`，联合驱动对 `location_candidate` 的 `narrow`/`exclude` 以缩小答案区间。地点排除/候选推进须标 `location_candidate`；局部空间观察只作支撑来源。`candidate_hypotheses` 仅接收 `location_candidate` 范围内的地点候选；背景地貌进入空间事实，不进入拍摄点候选池。**不**再设置针对「某两类空间对象互斥」的程序化 hard-fail（避免样本特化过拟合）
- **事实一次消费与状态增量**：Thought 写「当前候选状态 + 本步未决问题」，不得提前复述本步 Observation；Observation 仅提供相对既有状态的最小新信息；每条 `video_fact_id` 只允许产生一次完整状态更新，后续只能短承接。相同事实簇（`video_fact_ids + subject_scope + update_kind`）且无新候选增量时折叠
- **候选状态闭包**：`new_candidates` / `excluded` / `possible_regions` 须有来源事实或前序步；最终 `LocationHypothesis` 只能从已建立状态中选择，不得首次跳入地点。该性质由 stage5 逐步因果生成的上下文结构保证（第 t 步只能看到前序 T/A/O），judge 对「结论跳入」低分淘汰，不再做程序化状态重放校验
- **语义角色路由（stage3）**：答案前 Move 按粒度重路由——广域地貌/排除/自然区域 → COARSE；精确 POI/建筑/坐标 → FINE；纯 UI/故事 → NON_TRAINING
- **COARSE 时间覆盖（stage0）**：答案前的「区域试错 + 区域纠正/成功」须落入 COARSE；FINE 起点为首次 **精确 POI / 街景 / 交卷级查证** 意图（不是「打开地图/搜一下」粗排查）。**禁止**用时间轴中点作为 COARSE→FINE 默认切分；视频内 revision（区域纠正）默认仍属 COARSE，不得当作 FINE 起点
- **地理推理链去噪（stage2/3，替代最短蒸馏硬目标）**：保留全部有地理增益的 observe/correct/exclude/candidate（含失败排除与试错支线）；**删除** `stall`、纯 UI（置顶/滚动/消息列表）、纯社交开场且无地理断言的段落。不再以「蒸成最短成功链」为硬目标；禁止蒸出空 COARSE 链
- **步类型**：`observe`（建立可引用视频事实，不强制 excluded）与 `update`（须 `exclude`/`narrow`/`shift`/`correct` 之一并明确排除）。孤立 observe 不成训练步；无地理增益的 update 删除或并入后续。禁止连续两步同一候选状态
- **范围更新类型**：`narrow` / `expand` / `shift` / `correct` / `exclude`；「城市→附近」本身不算缩小
- **训练轨迹允许 tool**：
  - 固定核心：`zoom_inspect` / `ocr` / `sun_position_calc`
  - 视觉地图/卫星/地形（非地名 API）：如 `compare_images_for_geolocation`、
    `lookup_historical_map_layout` / `lookup_historical_satellite_map`、
    `find_specific_features_in_satellite_map`、`annotate_geographic_environment_on_image`、
    `detect_terrain_features`、`analyze_terrain_ambiguity`、`analyze_terrain_visual_illusion`
    及同语义动态注册 Tool（名称含 satellite/terrain/compare_images/lookup_historical 等）
- **训练轨迹禁止 tool**：`web_search` / `map_query` / `reverse_image_search` / `submit_answer`
  （`map_query` 返回解析坐标/标准地址，易把 COARSE 拉成精定位，故禁止）
- **动态 Tool**：禁止类才分解为 zoom/ocr/sun 或丢弃；允许类可原样进入 Agent1 训练链。
  地图/卫星 Tool 只服务区域级特征观察与排除，**不得**产出最终精确 POI/坐标结论
- **stage3**：匹配到允许的地图/卫星 Tool 时保留，不再一律压成 `zoom_inspect`；
  `screen_action`/`narration` 命中卫星/历史地图/标注/双图比对/地形时**优先** registry 内允许的 geo Tool，
  禁止再以低置信 `zoom_inspect` 或分解后的同质 zoom 兜底；连续同 bbox 且无新观察目标的 zoom 合并/丢弃；
  `target_features`、`source_claims` 与 `video_fact_ids` 仅来自该视频动态来源契约；
  旁白与 UI screen_action 冲突时采信旁白；fact quote 以旁白为主，过滤 UI 类 visible_clues
- 输出：`LocationHypothesis`；`reasoning_summary` 概括「工作范围 → 多位置空间事实（并存）→ 联合排除或收窄候选 → 结论」；命名自然区域进 summary/`key_clues_remaining`
- `possible_regions`：同层级规范行政区；自然/文化地带不得写入本字段
- **投影后递进可写性**：每步不同观察目标或候选状态变化；
  Obs 语义等价可折叠；**连续同 tool+params 且无新候选增量**
  （无 `exclude`/`narrow`/`shift`/`correct` 等状态更新）亦视为无增益重复并折叠（即使 Obs 字面不同）；
  无增益重复 / 闭包外事实 / 无排除 → rejected
- **投影保留失败/未命中**：不得因存在 `success` 就剔除全部 `empty`/`error`；
  须保留合成/校验失败（`error`）与有地理增益的未命中（`empty`），避免 TAO 中间环被物理删除；
  仅丢弃纯 UI / 无地理意图的 empty。`empty` 表示目标不可见或合法未命中；
  **禁止**把 schema/合成耗尽伪装成「无场景地理」的 `empty`（应标 `error`）
- handoff：`coarse_handoff=None`，`fine_handoff=None`
- 禁止：跳步；本步 Thought 偷用本步 Obs（Thought 生成**不得**注入本步 Observation）；
  最终精确 POI/坐标；Thought–Action 不对齐；反复验证 `working_scope`；用闭包外看图发现替代视频推理；视频注释/字幕/UI 当独立视觉证据；试错式无效 Action
- **judge 来源接地口径**：短语被 `video_fact_claims` 蕴含时**不得**判「无视频来源/凭空发明」；
  「前序 Obs 未建立却抢写」或「Action–Obs 模态不一致」归入因果/递进问题，勿与无来源混写
- **COARSE revision 信息隔离**：返工 prompt 仅接收抽象失败码与目标 Agent，不得注入 FINE 候选地名、自然语言 `failed_checks` 或 `suggested_recheck`。Agent1 训练 shard 的 `revision_input` 仅保留无地名的结构化审计字段（`verdict` / `return_to_agent` / failure codes）；完整验证文本留在 stage5/6 中间产物
- **质量执行机制（主链与 revision 相同）**：`reused_fact_without_delta`、`thought_observation_redundancy`、`candidate_provenance_gap` 这类因果性/增量性性质由 stage5 逐步因果生成在构造上保证，不再作为 stage6 程序化 hard-fail；stage5 仅保留轻量程序化硬校验（内部事实 ID 泄漏、本步 Observation 复述、旁白复述），其余质量判断统一由固定 rubric judge 打分做 best-of-k 拒绝采样

**Agent2 精定位（FINE）：**

- 输入：原始图片 + Agent1 的 `LocationHypothesis`（`coarse_handoff`）；可选 `user_query` 中的外部给定地名线索（见 1.6）
- 推理模式：假设验证（在粗定位基础上收窄到具体地点）；**缩小范围无人为上限**——若画面/Obs/`user_query` 线索足以一眼较精确定位，允许在较早步写出精确地点或坐标假设
- 主要 tool：web_search（`broad_discovery` 或 `precise_lookup`）/ reverse_image_search / map_query / zoom_inspect / ocr
- 输出：`SubmitAnswerResult`（最后一步必须调用 terminal tool `submit_answer`，写入 `Trajectory.fine_output`）
- handoff：`coarse_handoff` 必填；`fine_handoff=None`
- `map_query` 用法示例：可用 `params.query`（地名）和/或 `params.latlng`（待查坐标）查询；读取 Observation 时使用 `resolved_latlng`（解析后标准坐标），**不得**把输入参数名 `latlng` 当作输出字段。
- 禁止：无图像/Obs/`user_query` 依据地粘贴真值；在 Thought 中解释外部线索来源（如「网友说」）。

**Agent3 验证（VERIFIER）：**

- 输入：原始图片 + Agent2 的 `SubmitAnswerResult`（`fine_handoff`）；`coarse_handoff` 可选
- 推理模式：交叉验证（将该坐标与图像特征对照，**把 Agent2 结果当作候选答案验证**，不得看见 groundtruth）
- 主要 tool：map_query / web_search（仅 `purpose=verification`）
- 输出：`VerificationResult`（写入 `Trajectory.verifier_output`）
- handoff：`fine_handoff` 必填；`coarse_handoff` 可选
- `map_query` 用法示例：可用候选坐标作为 `params.latlng`（可同时带 `query` 消歧）；核对 Observation 的 `resolved_latlng` / `formatted_address` / `place_type` 是否与图像线索自洽。
- **主轨迹来源（方案 A）**：VERIFIER 的 TAO **以 `fine_handoff` + 原图 + 验证 Tool 执行结果为中心程序化组装**：
  - 合成脚手架时至少两步：`map_query`（核对候选 latlng）+ `web_search(purpose=verification)`（外部佐证）；
  - 视频侧可展开 Action 若不足两步，补齐缺失的验证 Tool；
  - 再逐步因果生成 Thought / 产出 `VerificationResult`。Thought 须先陈述调用动机；时序倒置（「查询结果显示…」式把本步 Observation 当已知事实）由逐步生成在构造上杜绝。
- **不是**默认把「答案宣布后到片尾」的整段旁白当作 Agent3 轨迹；片尾宣布句、历史科普、致谢/下课等无验证动作内容必须丢弃。
- 答案后视频仅作**可选补充证据**（见 1.6 / `post_answer_evidence_windows`）：仅保留含真实验证话术或可工具化验证操作的短窗；若无可选证据，仍必须合成上述主验证链。

### 1.5 四条数据质量铁律

- **防答案泄漏**：泄漏检查在 **stage6** 完成。拒绝的是**直接使用 GT / 后见之明**，不是「定位到了准确地点」。阶段时间规则见 1.6。
- **Observation 三条件**：按 schema 由 LLM 合成且风格像真实 API、格式一致（套统一 schema）、逻辑连贯（能支撑紧随其后的 Thought）。禁止调用真实外部 Tool API 生成 Observation。
- **标准地理定位 TAO**：Thought 必须且只能是图像地理推理体（植被/建筑/文字/阴影/交通等 → 为何调用本步工具）；禁止视频旁白叙事（博主/求助者/粉丝故事、片头标题复述等）。时序因果（Thought 不预知本步 Observation、不跳步）由 stage5 **逐步因果生成**在构造上保证；风格由 polish 润色 pass 统一（few-shot 见 `pipeline/tao_style_examples.py`）；旁白叙事体/非地理推理由 stage5 固定 rubric judge 低分淘汰。
- **Agent1 递进链**：COARSE 为视频接地地理推理链（可含试错+区域成功）；stage5 投影（允许的核心/地图卫星 Tool、UI 过滤、语义折叠、递进可写性）后按步携带消毒意图参考逐步生成 Thought，且 Thought 须对齐本步 Action；递进质量由 judge rubric 统一打分，不再依赖程序化门禁逐项 hard-fail。
- **Agent1 质量主目标（可训优先）**：优先稳定产出可进入 SFT 的 COARSE 轨迹（递进 + 工具有效 + Thought–Action 对齐）。judge 对 COARSE：**空转 / 无候选增量的重复同参 Action** 才是严重问题；**轻微 Obs 过写地名、ASR/字幕错字、单步不完美** 不得单独压到 ≤0.4。不降低 `STAGE5_JUDGE_THRESHOLD`；不加样本特判。宁可用、可学，再追求极致干净。
- **宁缺毋滥**：质量执行方式是**拒绝采样**而非强迫每次生成合格——stage5 对每条轨迹采样 `STAGE5_BEST_OF_K` 个候选，judge 取最高分且达阈值者，全部不达标则**该角色**轨迹废弃（不入库），**不阻断**同视频其它已成功角色（尤其 COARSE）落盘；stage6 再以 groundtruth 检查（泄漏/覆盖/距离/一致性）淘汰，rejected 不写入最终训练 JSONL。
- **返工样本是高价值数据**：区分视频内真实纠错（`video_observed`）与系统打回（`system_feedback`），优先收集，不过滤。

### 1.6 答案泄漏与时间规则

**泄漏内容规则（stage6：整链 LLM-as-judge 为主 + 窄化坐标程序化兜底）：**

- stage6 **可使用 groundtruth**（坐标与逆地理得到的最终地名）喂给泄漏 LLM judge；judge 同时可见 `user_query` 与逐步 Thought/Action/Obs。
- **泄漏 = 直接使用 GT / 后见之明**（非「出现了与 GT 一致的地点」）：
  - 证据不足时突然写出与 GT 等价的最终答案，且表现为「已知正确答案/真值/官方答案」；
  - 跳过工具链、与 Obs/`user_query` 线索无关地粘贴最终 POI/坐标；
  - COARSE 以最终精准 POI/坐标作结论（角色越界）；
  - VERIFIER 把 groundtruth 当作已知正确答案（复述 `fine_handoff` 仍合法）。
- **不算泄漏**：FINE 任意步（含非终端）基于画面/Obs/`user_query` 推出与 GT 一致的地点或坐标；终端 `submit_answer` 命中 GT；复用 `user_query` 中的外部线索；策略 B（COARSE 出现非最终答案的候选地区）。
- **外部给定线索 / 工作范围**：问题设置段外部沟通由 stage3 的逐视频结构化抽取生成带角色的 `raw_given_clue`；有拍摄地硬边界或可核验软先验时规范化为 `working_scope` 写入 Agent1 `user_query`（写入展示短语，不写来源话术）。允许「籍贯 ∧ 离家不远 ⇒ 籍贯地附近」软先验；**禁止**写成「籍贯市内 / X内」。stage5 **不再**从全部旁白自由重抽地名进 known clues。博主演绎候选属 `candidate_hypothesis`，**不得**注入 Agent1 `user_query`。`working_scope` 可直接过滤候选，但每次进一步收窄/排除须另有被引用的视频事实或 Obs 依据。抽取与改写 **不得**读 groundtruth。
- **程序化坐标兜底（窄化）**：COARSE Thought/`coarse_output` 出现坐标 → hard-fail；VERIFIER 在「正确答案/真值」话术下写出近 GT 且非 handoff 的坐标 → hard-fail。**不再**因 FINE 非终端步出现近 GT 坐标而 hard-fail。
- groundtruth、由 groundtruth 反向解析的地址 **不得进入 stage4/stage5 的任何 LLM prompt**（`user_query` 线索仅来自答案前旁白中的外部给定信息）。
- **TAO 形态（替代旁白词表扫描）**：时序倒置由 stage5 逐步因果生成在构造上杜绝；旁白叙事体与非地理推理由 stage5 judge rubric（不含 groundtruth）覆盖并低分淘汰。stage6 不再单独做形态 LLM 裁判。

**时间规则：**

- COARSE 和 FINE 默认只使用 `answer_timestamp` **之前**的证据。
- VERIFIER **主链**不依赖答案后时间窗；其 Action/Observation 来自对 `fine_handoff` 的验证 Tool 调用（Observation 由 LLM 按 schema 合成）及可选的筛选后视频证据。
- 答案宣布 **之后**的字幕/画面仅可进入 `post_answer_evidence_windows`（筛选后的可选证据），**不得**无筛选地整段划为 VERIFIER。
- 博主直接宣布答案的语句 **不能**作为验证证据。
- 答案后的纯叙事/科普/片尾（无验证话术、无屏上验证操作）**必须丢弃**，不得写入 Agent3 训练轨迹。

## 2. 系统总体架构

```
【前置地基】(先实现)
├─ Tool Registry（tool_registry.json，单一事实来源；仅 schema，无真实 executor）
├─ Tool Schema 规则体系（命名 / 语义 / 运行时注册 / Observation 合成）
├─ Observation 合成器（非 terminal：LLM 按 observation_fields 合成；terminal：skipped）
├─ 动态校验（pipeline/tools/validation.py）
└─ 轨迹 / 数据集 Schema（chat messages + role，服务 loss masking）

【主流水线】(每条视频依次经过；顺序固定)
stage0  解析带时间戳文字稿，定位答案时间戳；划分 COARSE/FINE；筛选可选答案后验证证据窗；返工区间
        → PreprocessResult（含 post_answer_evidence_windows）
stage1  按 Agent 时间区间抽帧（不按 Move），生成 TimedScreenAction
stage2  以 TimedScreenAction 会话为主轴合并旁白生成 Move（宁粗无碎；剔除纯 UI/无地理增益段）
stage3  Move → NormalizedStep（匹配 / 组合 / 注册新 Tool / fallback / thought_only）
stage4  LLM 合成 Action → ObservationExecutionResult
        （Visual：schema+params+关键帧+旁白；Retrieval：schema+params+本步旁白，无图）
stage5  生成三 Agent 主轨迹与 revision 轨迹（逐步因果生成 → polish 润色 →
        轻量硬校验 → 固定 rubric judge best-of-k 拒绝采样；禁止访问 groundtruth）
stage6  使用 groundtruth 做验证与泄漏检查（覆盖/距离/一致性）→ TrajectoryVerificationReport
stage7  生成三个 LoRA 的 DatasetEntry 与 JSONL 分片，再由单 writer 合并

【编排层】
Orchestrator（自写 Python + asyncio + tenacity），串联所有阶段，
支持并发、重试、断点续跑。批量入口 batch_run.py。

【Tool 库管理】
tool_registry.json + manage_tools.py（list / stats / register）
运行时亦可按 G 规则注册新 Tool（严格 schema 校验后立即入库可用）；无 draft/production 升档。
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
│       ├── base.py             # execute_action：LLM 合成 Observation
│       ├── registry.py         # tool_registry 读写（含文件锁）
│       └── validation.py       # 动态 params/observation 校验
├── run_one_video.py
├── batch_run.py
├── manage_tools.py
├── prep_groundtruth.py         # 离线 GT 辅助（Nominatim；非 Observation）
├── pipeline/prep_transcript_vlm.py  # 分窗 VLM 重转录 → data/transcripts_vlm/
├── tool_registry.json
├── tests/
├── data/
│   ├── raw_videos/
│   ├── transcripts/
│   ├── transcripts_vlm/        # VLM 校正字幕（可选，jobs 可指向此）
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
    params: list[ParamField]
    observation_fields: list[ObservationField]
    allowed_agents: list[AgentRole]
    is_terminal: bool = False
    # 无 tier / executor_ref：Observation 一律由 LLM 按 schema 合成（terminal 除外）

    # 注册元数据
    created_at: str  # ISO8601，系统自动注入
    source_video_timestamp: Optional[float] = None
    source_narration: Optional[str] = None
    derived_from_existing_tools: list[str] = []
    # 注意：derived_from_existing_tools 是否存在于 Registry
    # 不在 Pydantic model_validator 中检查，改在 register_tool 时用 Registry snapshot 检查

    # validator 摘要：
    # - is_terminal=False → observation_fields 不得为空；必须包含 name=status（非空枚举）
    #   与 name=error_message（nullable=True）
    # - is_terminal=True → observation_fields 必须为空列表（不产生 observation）
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
# 硬规则（注册时强制）：
# A1: 小写 snake_case；仅 [a-z0-9_]；长度 3~64
# A2: 不得以下划线开头或结尾；不得包含连续下划线
# A3: 禁止无意义名称：tool / helper / utility / new_tool / custom_tool 等
# A4: 自动生成的非种子 Tool 名称至少包含两个语义 token（至少一段下划线分隔）
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
# 仅作为生成新 Tool 名称时的 prompt 提示，不得作为硬性注册校验条件。

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
# F9: （已删除）不再区分 draft/production，无 executor_ref
# F10: Tool 级交叉字段约束（如 map_query 的 query|latlng 至少其一；
#      以及 status 条件 Observation 规则）由 validate_action_params /
#      validate_observation 实现，不能仅靠单字段 required / nullable 表达
```

### 4.6 注册新 Tool 的条件（G规则，match_or_register_tool 里检查）

样本出现次数可以统计，但 **不得**作为注册门槛。

只有 **同时满足** 以下条件才注册新 Tool（写入 registry 后立即可用于 Observation 合成）：

1. 无法匹配现有 Tool；
2. 无法用现有 Tool 组合表达；
3. 输入输出语义明确；
4. 在地理定位中具有复用可能；
5. observation schema 完整可合成（含 status / error_message 及业务字段）；
6. 不是滚动页面、移动鼠标、切换标签等纯 UI 操作；
7. 不是只对当前视频成立的一次性操作；
8. 与现有 Tool 不重复或高度相似（语义相似度阈值建议 0.85；名称编辑距离 ≤ 3 直接拒绝）。

不满足时：强制匹配现有 Tool、返回组合方案、或 `normalization_mode=fallback` / `thought_only`，**禁止无标记硬套**。

### 4.7 Observation 合成规则（H规则，LLM prompt 约束）

适用于**所有非 terminal Tool**（种子与运行时注册的新 Tool 一视同仁）。

```python
# Observation 合成分两族（语义 = 该步 Action 在视频中得到的结果，再装进 schema）：
#   VisualObs: zoom_inspect / ocr / sun_position_calc
#   RetrievalObs: web_search / map_query / reverse_image_search /
#                 find_specific_features_in_satellite_map
#
# H1: 必须按 observation_fields 每个字段逐一填写，不得增减
# H1b: validate_observation 允许校验前的近义字段 coerce（如 category→feature_category、
#      position→bbox）；不改变 registry 正式字段名；coerce 后再严格校验
# H2: nullable=true：图像/消毒后旁白中未出现对应信息时填 null，不得猜测
# H3: result_list：条目数 2~5，且每项符合 item_fields
# H4: string 禁止 "未知"/"不确定"/"N/A"；信息不可得时应为 nullable 字段填 null
# H5: 必须通过 validate_observation；Instructor 最多重试 OBS_SYNTH_MAX_RETRY 次，
#     仍失败 → generate_observations 将该 Action 标为 status=error（诚实失败，
#     附最小合法 error 载荷 + 真实 error_message），流水线继续；
#     **禁止**伪装成「无场景地理」的 empty；不得因单次合成耗尽整角色/整视频死刑
# H6: 不得从 groundtruth 获得任何信息
# H7: 合成输入必须包含 tool schema 与 Action params；
#     VisualObs 另需关键帧图像；RetrievalObs **不传图**（纯文本合成）；
#     旁白须经 sanitize_narration_for_obs(agent_role, narration) 后再传入（可为空串）；
#     COARSE+Visual：旁白清空，来源事实仅走 EvidenceIntent.source_claims；
#     FINE/VERIFIER+Retrieval：保留本步旁白地名（剥离坐标表述即可）
# H8: 地名规则按 Tool 族：
#     VisualObs：允许画面可见文字；COARSE 禁止闭包外新 POI；允许对 source_claims
#       目标做可见确认；bbox 为注意力提示（可外扩），勿因框差机械 empty
#     RetrievalObs：允许本步旁白与 Action.params（如 query）中出现的中间地名写入
#       Obs；禁止未在本步来源出现的答案级剧透；仍禁止 groundtruth
# H9: VisualObs 的 Observation 不得包含视频制作 overlay（片头/标题卡、平台/频道
#     水印与 logo、难度/星级角标、进度条、烧录字幕条、创作者标签等非场景 UI）。
#     仅描述场景内地理/建筑/自然与真实标识；ocr 仅提取场景内路牌/店招等。
#     合成疑似含 overlay → 带说明重试；耗尽 → 降 empty（真·不可用场景证据）。
#     RetrievalObs 不套用 H9 平台名启发式（避免检索摘要误杀）。
# H10: VisualObs 图像：按 Move 时间窗 + EvidenceIntent 目标感知选帧；
#     先识别内容区（primary_scene / supporting_geo_visual / interface_only），
#     interface_only 不产证据；再裁内容区；zoom/ocr 的 Action bbox 外扩 margin
#     后相对内容区二次裁剪（bbox=hint，非验尸真值）。
# H11: 视频注释/字幕可引导注意区域；Visual 目标明确不可见 → empty；
#     可见确认优先于机械 empty。schema/合成失败 → error，不得写成 empty。
```

### 4.8 初始 Tool 注册表（7 个种子 Tool）

种子 Tool **保留现有名称**。  
Registry 仅保存 schema（params / observation_fields / allowed_agents / is_terminal 等）；**不**保存 tier 或 executor_ref。  
Observation 一律由 LLM 按 H 规则合成；`submit_answer` 为 terminal，不产生 Observation。

下列清单给出目标 schema。

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
- purpose 角色硬约束（stage3 + execute_action 双处检查）：
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
    # possible_regions：同层级规范行政区域（省/州/直辖市等）；可为空（仅收窄到国家）。
    # 自然/文化地带（如「中原地区」）不得写入本字段，应进 reasoning_summary /
    # key_clues_remaining。stage5/stage6 以语义规则校验，不维护全球硬编码枚举表。
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
    # 仍含 COARSE / FINE / VERIFIER 三段以便编排层遍历：
    # - COARSE/FINE：答案前视频证据窗
    # - VERIFIER：若 post_answer_evidence_windows 非空，取其并集；否则为零长度占位
    #   （start_time == end_time），表示无视频侧验证证据可采
    revision_segments: list[tuple[float, float]]
    post_answer_evidence_windows: list[tuple[float, float]] = []
    # 答案宣布句结束之后、经筛选的可选验证证据短窗（可多段）。
    # 不得把「答案后→片尾」未筛选整段写入本字段。

class TimedScreenAction(BaseModel):
    start_time: float
    end_time: float
    description: str
    visible_clues: list[str] = []

class Move(BaseModel):
    start_time: float
    end_time: float
    narration: str  # 来自文字稿（asr_raw 或 vlm_transcript）
    screen_action: Optional[str] = None
    visible_clues: list[str] = []
    agent_role: AgentRole

class Action(BaseModel):
    tool: str  # 必须存在于 tool_registry
    params: dict

class NormalizationMode(str, Enum):
    MATCHED = "matched"
    COMPOSED = "composed"
    TOOL_REGISTERED = "tool_registered"
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
    LLM_SYNTHESIZED = "llm_synthesized"

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

    # stage5 judge 分数（best-of-k 入选候选的 rubric 得分，0~1；stage6 以其为质量基分）
    stage5_judge_score: Optional[float] = None
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

- APP_ENV, ALLOW_REAL_API（是否允许真实 LLM 调用；测试默认 false）
- GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, DASHSCOPE_API_KEY, MOONSHOT_API_KEY
- **双通道模型**（均支持多模态图片输入；禁止在代码中硬编码模型名/Key）：
  - `VLM_*`：stage3 之前（`prep_transcript_vlm`、stage1 等）视觉通道
    - `VLM_PROVIDER`（默认 `qwen`）、`VLM_BASE_URL`、`VLM_MODEL`
  - `LLM_*`：stage3 及之后（normalize / Observation 合成 / stage5–6）主通道
    - `LLM_PROVIDER`（默认 `kimi`）、`LLM_BASE_URL`、`LLM_MODEL`（默认 `kimi-k3`）
  - `KIMI_REASONING_EFFORT`（默认 `low`；仅 kimi：`low`/`high`/`max`）
  - `LLM_TIMEOUT_SEC`（默认 300；thinking 模型宜更长）
  - `GEMINI_MODEL`（仅 `*_PROVIDER=gemini` 时回退）
- MAX_CONCURRENT_VIDEOS
- ANSWER_LEAK_CHECK_ENABLED
- OBS_SYNTH_MAX_RETRY（默认 3；Observation LLM 合成校验失败重试次数）
- MAX_REVISION_ROUNDS（默认 2）
- STAGE5_BEST_OF_K（默认 2；每条轨迹逐步生成候选数，judge 择优）
- STAGE5_JUDGE_THRESHOLD（默认 0.6；judge 最低入选分，全部候选低于该值 → 该角色轨迹废弃）
- TOOL_REGISTRY_PATH, INTERMEDIATE_DIR, OUTPUT_DIR, CACHE_DIR
- DISTANCE_ERROR_THRESHOLD_KM（默认 25）

**Kimi K3 适配约束（adapter 必须遵守）：**
- OpenAI 兼容 Chat Completions；图片仅用 base64 data URL（禁止公网图片 URL）
- **不得**传 `temperature` / `top_p` / `n` / `presence_penalty` / `frequency_penalty`（服务端固定）
- 始终 thinking：用顶层 `reasoning_effort`；结构化结果只解析 `message.content`，忽略 `reasoning_content`
- 限流/断连走既有瞬断重试；失败不得静默降级为低质量 fallback

**不得**配置 Tavily / SerpAPI / 高德等外部 Tool API key：Observation 不走真实 Tool 执行。

### 5.2 llm.py

```python
def call_structured(
    prompt: str,
    response_model: type[BaseModel],
    images: Optional[list[str]] = None,
    video: Optional[str] = None,
    model: Optional[str] = None,  # None → 按 lane 从配置读取
    *,
    lane: Literal["vlm", "llm"] = "llm",
) -> BaseModel:
    """调用 LLM 并强制返回 response_model；不合法自动重试。经 adapter 封装 SDK。
    lane=vlm → stage3 前视觉通道；lane=llm → stage3+ 主通道（默认可含图片）。"""

def call_text(
    prompt: str,
    model: Optional[str] = None,
    *,
    lane: Literal["vlm", "llm"] = "llm",
) -> str:
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
    3. 再次执行名称/语义去重与 schema 校验
    4. 用 Registry snapshot 检查 derived_from_existing_tools
    5. 写临时文件 → fsync → 原子替换
    自动注入 created_at。注册成功后立即可用于 Observation 合成。
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
    校验失败 → execute_action 最多重试 OBS_SYNTH_MAX_RETRY，仍失败 → status=error；
      generate_observations 将该 Action 降为 empty 并继续（不整阶段抛错）
    """
```

### 5.5 stage0_preprocess.py

```python
def locate_answer_timestamp(transcript: list[TranscriptSegment]) -> float: ...

def segment_by_agent_role(
    transcript: list[TranscriptSegment],
    answer_timestamp: float,
    post_answer_evidence_windows: Optional[list[tuple[float, float]]] = None,
) -> list[AgentTimeSegment]:
    """
    划分 COARSE / FINE / VERIFIER 时间区间。
    COARSE/FINE 落在 answer_timestamp 之前。
    COARSE 须覆盖区域试错与区域成功；FINE 起点为精确 POI/街景/交卷级意图。
    禁止时间轴中点兜底；revision 区间不得当作 FINE 起点。
    VERIFIER：若提供非空 post_answer_evidence_windows，取其时间并集；
    否则为零长度占位（无视频验证证据可采，主链改由 stage5 合成）。
    """

def select_post_answer_evidence_windows(
    transcript: list[TranscriptSegment],
    answer_timestamp: float,
) -> list[tuple[float, float]]:
    """
    在宣布答案句结束之后筛选可选验证证据短窗。
    仅保留含验证话术（如验证/核对/对照/是否吻合）的片段；合并相邻窗。
    排除宣布句本身；不得默认吞掉答案后全部字幕。
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
    以每条 TimedScreenAction（同一次屏幕操作会话）为核：
    将该 SA 时间窗内重叠的旁白按时间序合并为 1 个 Move（宁粗无碎）。
    无 SA 覆盖的旁白按原 TranscriptSegment 保留，默认不按语气/转折细切。
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
    按 G 规则决定：匹配 / 组合 / 注册新 Tool / fallback。
    同时检查 allowed_agents 与 web_search.purpose 角色约束。
    LLM 失败、校验失败、或模型给出空 fallback 时：若 screen_action 含明显
    Tool 语义，则先尝试启发式映射到现有种子 Tool（如 web_search / zoom_inspect /
    ocr / map_query / sun_position_calc），成功则记为 matched（低 confidence），
    避免 Agent 轨迹因 Connection error 等静默坍缩为空 actions。
    纯 UI 与非 Tool 语义操作仍保持 fallback / thought_only。
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
    COARSE 路径写入 EvidenceIntent（嵌入 thought_draft），供 stage4 选帧/裁图。
    """

def normalize_all_agent_steps(
    moves_by_role: dict[AgentRole, list[Move]],
    answer_timestamp: float,
) -> dict[AgentRole, list[NormalizedStep]]:
    """
    对答案前 Move 做语义角色重路由后再 normalize：
    COARSE=广域地貌/排除/自然区域；FINE=精确 POI/建筑/坐标；
    NON_TRAINING 不进入训练脚手架。不读 groundtruth。
    逐视频 VideoContextExtraction 可分批；抽取/复核失败必须抛错，
    禁止静默降级到低质量 fallback（仅 ALLOW_REAL_API=false 时允许）。
    """
```

### 5.9 tools/base.py

```python
def sanitize_narration_for_obs(agent_role: AgentRole, narration: str) -> str:
    """按角色消毒旁白：COARSE 剥离地名/POI/坐标短语；FINE/VERIFIER 轻度剥离坐标。"""

def crop_image_by_bbox(
    image_path: str,
    bbox: list[float],
    *,
    cache_dir: Optional[str] = None,
) -> str:
    """按归一化 bbox [x,y,w,h] 或 [x1,y1,x2,y2] 裁图，返回裁剪图路径；非法框则返回原图。"""

def observation_contains_video_overlay(observation: dict) -> list[str]:
    """通用启发式：检测 Observation 是否含视频 overlay/元信息类别。"""

def execute_action(
    action: Action,
    image_path: str,
    agent_role: AgentRole,
    narration: str = "",
    *,
    evidence_intent: Optional[EvidenceIntent] = None,
) -> ObservationExecutionResult:
    """
    分发器（无真实 Tool API）：
    1. 查 registry；检查 allowed_agents 与 purpose 约束
    2. validate_action_params
    3. terminal → status=skipped，observation/source=None
    4. 非 terminal → 消毒 narration；VisualObs 内容区+bbox 外扩裁图后合成；
       RetrievalObs 纯文本合成（不传图）；interface_only → 合法 empty；
       H9 overlay（仅 Visual）→ 重试；单次耗尽 → error（上层再降 empty）
       OBS_SYNTH_MAX_RETRY
    5. diskcache：key 含图像 hash（Retrieval 用固定 text_only 标记）与 prompt_version
    """
```

### 5.10 stage4_observe.py

```python
class ObservationSynthesisExhausted(RuntimeError):
    """保留类型兼容；generate_observations 不再抛出（耗尽改降 empty）。"""

def resolve_image_for_step(
    step: NormalizedStep,
    *,
    image_path: str,
    keyframes: Optional[list[str]] = None,
) -> str:
    """按 Move 时间窗 + EvidenceIntent 目标感知选帧；无 keyframes 时回退 image_path。"""

def generate_observations(
    normalized_steps: list[NormalizedStep],
    image_path: str,
    agent_role: AgentRole,
    *,
    keyframes: Optional[list[str]] = None,
) -> list[ObservationExecutionResult]:
    """
    展开 normalized_steps 中的全部 Action，逐个 execute_action。
    每步目标感知选帧 + 内容区裁剪；传入 EvidenceIntent。
    VisualObs：首帧 empty/error 时可换近邻关键帧再合成。
    任一非 terminal 合成/schema 耗尽 → status=error（诚实失败载荷），流水线继续（全角色）；
    不得伪装成无场景地理的 empty。
    """

def pick_agent1_representative_image(
    observations: list[ObservationExecutionResult],
    steps: list[NormalizedStep],
    *,
    keyframes: list[str],
    fallback: str,
) -> str:
    """优先选择已通过内容区门禁的 primary_scene 代表图，供 stage5 轨迹生成。"""
```

### 5.11 stage5_reconstruct.py

**本阶段所有函数签名不得包含 groundtruth。**

生成范式为**逐步因果生成（teacher-forced rollout）+ polish 润色 + 轻量硬校验 + judge 拒绝采样**，
不再使用「整段改写 + 事后校验 + 失败重写」。

```python
class TrajectoryQualityRejected(RuntimeError):
    """best-of-k 全部候选低于 STAGE5_JUDGE_THRESHOLD：该角色轨迹废弃，不入库。"""

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
    1) 展开与投影（沿用既有契约）：
       Agent1：Tool 投影——保留核心三工具 ∪ 视觉地图/卫星/地形允许集；
       硬排除 web_search / map_query / reverse_image_search / submit_answer；
       允许集内 compare/卫星类原样保留；纠正非法 zoom_inspect bbox；
       UI 过滤；保留 error 与有地理增益的 empty（不得因有 success 掏空失败步）；
       语义折叠（同 tool+params 且无新候选增量亦折叠）；
       投影为空或递进可写性不足则失败。
       Agent2：保证末步 terminal submit_answer（缺失则合成）。
       Agent3：无可展开 Action 时合成验证脚手架（map_query + web_search(verification)）。
    2) 按步意图参考：每个 unit 携带其 EvidenceIntent/thought_draft 的**消毒版**
       （剥离 video_fact_id、内部标记 <<<...>>>、结论性表述），再按本步
       Action.tool 对齐改写；仅作为「观察目标」参考，禁止照抄进 Thought。
    3) 逐步生成 Thought：第 t 步的 prompt 上下文 = system + user_query +
       前 t-1 步完整 Thought/Action/Observation + 第 t 步 Action（固定，来自视频）
       + 该步消毒并对齐的意图参考；**不注入**第 t 步 Observation 与任何后续步信息。
       Thought 必须说明为何调用本步 tool，禁止提及未出现在本步 Action 的工具名；
       若异工具话术命中则轻量重试一次。由此在构造上保证：不预知、不跳步、
       每步 Observation 有信息增量、Thought 与 Action 对齐。
    4) 角色输出：COARSE 由完整 TAO 生成 LocationHypothesis
       （possible_regions 仅规范行政区）；FINE 从末步 submit_answer 抽取
       SubmitAnswerResult；VERIFIER 由完整 TAO 生成 VerificationResult
       （把 fine_handoff 当候选验证）。
    5) polish：整链一次 LLM 润色（few-shot 用 tao_style_examples 黄金示例；
       只改措辞与衔接，禁改事实/结论/步骤顺序），随后一次 LLM 忠实性对比；
       被判不忠实的步回退为润色前文本。
    6) 轻量硬校验（纯程序化，不调 LLM）：
       - 内部事实 ID（vf\d+_ 等）与脚手架标记不得出现在任何 Thought/输出；
       - 第 t 步 Thought 与第 t 步 Observation 的复述重叠率超限 → 该候选判废；
       - Thought 与旁白原文重叠率超限 → 该候选判废。
    7) 拒绝采样：步骤 3-6 为生成一个候选；共采样 STAGE5_BEST_OF_K 个候选，
       固定 rubric 的 LLM judge（无 groundtruth；rubric 覆盖：递进性/每步排除、
       Thought-Action 对齐、来源接地——video_fact 蕴含不得判凭空、链内过早归因果、
       旁白叙事体、结论跳入、整体流畅度）逐一打分，
       取最高分且 ≥ STAGE5_JUDGE_THRESHOLD 者写入 Trajectory.stage5_judge_score；
       全部低于阈值 → raise TrajectoryQualityRejected。
    COARSE 的 user_query 仅注入 stage3 `working_scope` 展示短语（不从旁白重抽地名）；
    FINE 可附带答案前旁白抽取的外部给定线索（不含来源话术）。
    禁止将 groundtruth / 真值地名 / 反向地理编码地址写入 prompt。
    system_prompt 为面向推理期的简洁角色指令（见 _SYSTEM_PROMPTS），
    不得包含面向生成器的禁令墙或内部术语。
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
    Agent3：若视频侧无任何可展开 Action，则基于 fine_handoff 合成验证脚手架
    （至少 map_query + web_search(verification)），再逐步生成 Thought / 产出 verifier_output。
    视频侧可展开步若不足两步则补齐。禁止使用 groundtruth。
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
    - video_observed：使用 video_revision_segments 生成高价值返工轨迹；
      若时段与 NormalizedStep 无重叠可展开 Action，则回退为该角色全量可展开步
      （仍标记 video_segment），避免 revision 空跑
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
    groundtruth 仅在本阶段使用。stage6 只做 **GT 相关**检查；
    轨迹质量（TAO 形态、递进链、流畅度）已由 stage5 judge 拒绝采样负责，
    本阶段不再做形态裁判 / 递进链裁判 / 合理性 soft judge。
    判定顺序：泄漏（LLM+坐标兜底）→ 角色专项。
    Agent1：缺 coarse_output → hard-fail；LocationHypothesis 是否覆盖
            真值国家/一级行政区（regions 非空时）；
            禁止轨迹含 web_search/map_query/reverse_image_search/submit_answer
            （程序化；视觉地图/卫星/比对类允许）；
            提及并使用 user_query 地点本身 ≠ 违规
    Agent2：缺 fine_output → hard-fail；距离误差超过
            DISTANCE_ERROR_THRESHOLD_KM → hard fail
    Agent3：缺 verifier_output → hard-fail；verdict 与误差一致性；
            泄漏判定允许复述 fine_handoff 候选
    泄漏：整链判断是否直接使用 GT/后见之明；命中最终地点 ≠ 泄漏
    quality_score：基分 = traj.stage5_judge_score（缺省 1.0）；
    有 hard-fail → min(quality, 0.3)；FINE 按距离误差混合；clamp [0,1]。
    返回 TrajectoryVerificationReport。
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
python manage_tools.py stats
python manage_tools.py register --from-json path/to/tool.json
```

无升档（promote）命令。register 仅做 schema 校验后写入 registry。

## 6. Tool Schema 变更与历史样本

**不再存在 draft → production 升档流程。**

若 registry 修订改变 Observation 字段名（以 map_query 为例，从旧名 `latlng` 改为 `resolved_latlng`）：

- **禁止**仅在落盘 JSON 里做键名替换后继续沿用旧 Thought。
- 必须：备份 → 更新 registry schema → 重跑 **stage4 → stage5 → stage6 → stage7**，使 Observation、Thought 与下游校验一致。
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

# Tools / 地理（仅离线 prep_groundtruth 等辅助脚本可能用到；不用于 Observation）
geopy
pillow
opencv-python

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
3. tool_registry.json — 种子 tool（仅 schema）
4. tools/registry.py — 文件锁 + 原子写
5. config.py + .env.example
6. llm.py adapters
7. execute_action — 全 Tool Observation LLM 合成

### 里程碑二：单视频 Agent1 闭环

8. stage0 → stage7（仅 Agent1）
9. run_one_video.py 人工质检

### 里程碑三：三 Agent + 运行时注册 + 返工

10. Agent2/3、G 规则注册新 Tool、revision 路径

### 里程碑四：规模化

11. 完善 schema 库与合成质量、batch_run、cache、可观测性

## 9. 测试要求

- Observation 合成路径单元测试，mock LLM；test 环境禁止真实付费调用。
- schema validator：违反 A/F/G 的样本必须被拒绝；种子名保留且可通过。
- `map_query`：F1 通过（params 含 `latlng`、observation 含 `resolved_latlng`，无重名）；若 observation 仍用 `latlng` 必须被拒绝。
- `map_query` params 交叉约束：仅 query、仅 latlng、二者同时 → 通过；二者皆缺 → 失败。
- `map_query` Observation：success / empty / error 三条 status 条件规则的正反例。
- 运行时 `tool_registered` 注册流程测试。
- stage3：LLM 失败或空 fallback 时，对含 Tool 语义的 screen_action 须能启发式落到现有种子 Tool（mock）；纯 UI 仍 fallback。
- stage0：`post_answer_evidence_windows` 仅含验证话术短窗，不得默认等于答案后→片尾；无证据时 VERIFIER 区间为零长度。
- stage5：VERIFIER 在无视频 Action 时必须能基于 `fine_handoff` 合成 `map_query`+`web_search(verification)` 验证链并产出 `verifier_output`（mock execute_action）。
- stage5：`video_revision_segments` 与 Move 无重叠时仍应产出 video_observed 返工（回退全量可展开步）。
- stage5 逐步因果生成：mock LLM 记录每步 prompt，断言第 t 步 prompt 含前 t-1 步 Observation、**不含**第 t 步 Observation 内容与任何后续步信息。
- stage5 意图参考消毒：`thought_draft` 中的 `vf\d+_*` 事实 ID 与 `<<<...>>>` 标记不得出现在任何步的 prompt 参考段与生成结果中。
- stage5 polish：忠实性对比判某步不忠实时，该步必须回退为润色前文本。
- stage5 拒绝采样：mock judge 给出低于阈值的分数时 `reconstruct_single_trajectory` 必须抛 `TrajectoryQualityRejected`；多候选时必须选择得分最高者并写入 `stage5_judge_score`。
- stage5 硬校验：Thought 含内部事实 ID / 与本步 Observation 高重叠 / 与旁白高重叠的候选必须判废。
- 并发：registry 文件锁与 JSONL 分片合并。

## 10. 明确的禁止事项

- 不要引入 LangChain/LangGraph 做核心流水线。
- 不要用自由文本解析代替 Pydantic。
- 不要调用真实外部 Tool API（搜索/地图/OCR 服务等）生成 Observation。
- 不要生成无 schema 校验的自由文本 Observation。
- 不要生成含答案泄漏的 Thought；不要把 groundtruth 送入 stage4/stage5。
- 不要更改本文档定义的 schema 字段名与结构（除非先修订 SPEC）。
- 不要把 .env 或 data/ 提交到 Git。
- 不要让 Agent2 使用 sun_position_calc；不要让 Agent1 使用 reverse_image_search 和 map_query。
- 不要跳过 G 规则直接注册新 Tool。
- 不要无标记硬套无法匹配的操作。
- 不要多协程直接追加同一最终 JSONL。

## 11. 术语速查

- **Trajectory**：Thought-Action-Observation 序列。
- **Tool / Registry**：仅 schema 库；Observation 由 LLM 按 H 规则合成。
- **SubmitAnswerResult**：Agent2 终端输出与 Agent3 输入。
- **LocationHypothesis**：Agent1→Agent2 交接物。
- **VerificationResult**：Agent3 输出；含 pass/fail 与 return_to_agent。
- **RevisionContext**：返工上下文；区分 video_observed 与 system_feedback。
- **NormalizedStep / ObservationExecutionResult / TrajectoryVerificationReport**：阶段间强类型契约。
- **Loss Masking**：仅 assistant role 算 loss。
- **答案泄漏**：直接使用 GT / 后见之明（非整链前向推理却粘贴真值）；定位到准确地点本身不算泄漏；由 stage6 整链 LLM 泄漏 judge + 窄化坐标兜底消除。
- **LoRA**：三 Agent 共享基座、分头训练。
