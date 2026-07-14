> **ARCHIVED / 历史归档**：本文档不是现行规格。现行唯一有效规格见仓库根目录 `SPEC.md`。

# SPEC Review

## Review Summary

- 检查时间：2026-07-13
- SPEC 版本：2.0（生成时间：2026-07-12）
- 阻塞问题数量：7
- 重要问题数量：10
- 一般问题数量：8
- 结论：BLOCKED

## Blocking Issues

### ISSUE-B001：初始 7 个 Tool 名称全部违反 A1 命名规则

- 严重程度：阻塞
- 涉及章节：4.4 A规则；4.8 初始 Tool 注册表
- 涉及字段或函数：`ToolDefinition.name`、`ALLOWED_VERBS`、初始 tool 清单
- 问题说明：A1 要求 `动词_名词` / `动词_名词_名词`，动词必须来自 `ALLOWED_VERBS`，长度 10~40。初始名称均不合规：`ocr`（过短且无动词）、`web_search`/`map_query`/`zoom_inspect`/`reverse_image_search`/`sun_position_calc`/`submit_answer`（首段均不在 `ALLOWED_VERBS`）。
- 为什么阻塞实现：任务1 必须实现 A 规则 validator；若严格执行，初始 `tool_registry.json` 无法通过校验，地基无法落地。
- 建议修正方案：要么将 7 个初始 tool 重命名为合规名称（如 `search_web`、`query_map`、`detect_text`/`extract_text`、`inspect_region`、`search_reverse_image`、`calculate_sun_position`、`submit_answer` 需新增动词或改为 `lookup_answer` 等），要么在 A1 中显式豁免“历史/种子 tool”并给出豁免清单。

### ISSUE-B002：Agent3 的 handoff_input 类型错误

- 严重程度：阻塞
- 涉及章节：1.4；4.11 Trajectory；4.9；5.10
- 涉及字段或函数：`Trajectory.handoff_input`、Agent3 输入
- 问题说明：1.4 规定 Agent3 输入为「原始图片 + Agent2 的 submit_answer 结论」，但 `Trajectory.handoff_input` 类型为 `Optional[LocationHypothesis]`，注释写「Agent2/3 才有」。Agent3 无法用 LocationHypothesis 表达 submit_answer。
- 为什么阻塞实现：无法正确序列化 Agent3 轨迹的交接物；stage5/stage7 类型契约断裂。
- 建议修正方案：引入独立 `SubmitAnswerResult`（或等价 schema），并将 `handoff_input` 改为可区分联合类型（如 `LocationHypothesis | SubmitAnswerResult | None`），或拆分为 `coarse_handoff` / `fine_handoff` 两个字段。

### ISSUE-B003：缺少 SubmitAnswer / Agent2 输出的独立 Schema

- 严重程度：阻塞
- 涉及章节：4.8 submit_answer；4.9；5.10 reconstruct_all_trajectories
- 涉及字段或函数：submit_answer params；Agent2→Agent3 交接
- 问题说明：Agent2 输出仅描述为 tool params（latitude/longitude/...），没有独立 Pydantic 模型。`reconstruct_all_trajectories` 写「提取 submit_answer 传给 Agent3」，但无类型定义。
- 为什么阻塞实现：Agent2→Agent3 结构化交接无法校验；与「所有结构化输出必须 Pydantic」冲突。
- 建议修正方案：新增 `SubmitAnswerResult`（字段与 submit_answer params 对齐），明确规定从 Trajectory 最后一步 Action 提取并校验的方式。

### ISSUE-B004：submit_answer 的 observation=null 与 F2 / ToolDefinition 冲突

- 严重程度：阻塞
- 涉及章节：4.3 ToolDefinition；4.5 F2；4.8 submit_answer；4.11 TrajectoryStep
- 涉及字段或函数：`ToolDefinition.observation`、F2、`TrajectoryStep.observation` / `observation_source`
- 问题说明：F2 要求 observation 中「必须有且仅有至少一个 nullable=False 字段」；submit_answer 规定 observation 为 null。`ToolDefinition.observation: list[ObservationField]` 无法表达 null。同时 `TrajectoryStep.observation_source` 在 observation 为 None 时仍必填，合法取值未定义。
- 为什么阻塞实现：无法同时满足 ToolDefinition validator、submit_answer 语义与 TrajectoryStep 建模。
- 建议修正方案：为 terminal tool 增加特例（如 `observation: list[ObservationField] = []` 且 F2 对空列表豁免；或 `is_terminal: bool`）；并将 `observation_source` 改为 `Optional[...]`，在无 observation 步骤为 None。

### ISSUE-B005：map_query 观测字段全部 nullable，违反 F2

- 严重程度：阻塞
- 涉及章节：4.5 F2；4.8 map_query
- 涉及字段或函数：`map_query.observation`、F2
- 问题说明：map_query 的 formatted_address / latlng / place_type「均 nullable」，与 F2「至少一个 nullable=False」直接冲突。
- 为什么阻塞实现：按 F2 写入注册表会被 validator 拒绝；放宽 F2 又与文档矛盾。
- 建议修正方案：至少将一个字段改为非空（例如查询失败时返回明确 status/string），或调整 F2 允许「全部可空但必须有 status 类非空字段」。

### ISSUE-B006：result_list 与 ResultItem 未建立可校验关联

- 严重程度：阻塞
- 涉及章节：4.3 ObservationField / ResultItem；4.7 H1；4.8 web_search / reverse_image_search
- 涉及字段或函数：`ObservationField.type="result_list"`、`ResultItem`
- 问题说明：`ResultItem` 定义了子结构，但 `ObservationField` / `ToolDefinition` 没有字段指向「该 result_list 使用哪套 ResultItem.fields」。运行时无法按 schema 校验 results/matches 条目。
- 为什么阻塞实现：H1「按 observation schema 逐字段填写」对 result_list 无法程序化落地；动态 Draft Tool 的 observation 校验不完整。
- 建议修正方案：在 ObservationField 增加 `item_fields: Optional[list[ObservationField]]`（或嵌入 `ResultItem`），仅当 type=result_list 时必填。

### ISSUE-B007：ParamField 无法表达 default，且与 4.8 示例冲突

- 严重程度：阻塞
- 涉及章节：4.3 ParamField；4.8 web_search
- 涉及字段或函数：`ParamField`、`web_search.top_k(int,default=3)`
- 问题说明：4.8 写明 top_k 默认 3，但 ParamField 无 `default` 字段，无法在注册表中表达默认值；Action.params 运行时校验也缺少默认值注入规则。
- 为什么阻塞实现：初始 tool 注册表无法按文档完整建模；params 校验语义不完整。
- 建议修正方案：为 ParamField 增加 `default: str | int | float | bool | None = None`，并规定 `required=False` 时允许 default；在 Action 校验时补齐缺省参数。

## Important Issues

### ISSUE-I001：stage5 函数签名接收 groundtruth，存在答案泄漏风险

- 严重程度：重要
- 涉及章节：5.10；1.5；10 禁止事项；项目规则第21条
- 涉及字段或函数：`reconstruct_single_trajectory(..., groundtruth=...)`
- 问题说明：轨迹重构签名显式传入 groundtruth，但未规定「仅用于程序化泄漏检测，禁止进入 LLM prompt」。实现者极易把真值送入改写模型。
- 为什么阻塞实现：不阻塞编译，但会系统性污染训练数据。
- 建议修正方案：拆分参数用途：泄漏检测使用独立 `forbidden_answer_tokens`（由 groundtruth/地名派生，在模型外生成）；明确禁止将 raw groundtruth 传入 `call_structured`/`call_text`。

### ISSUE-I002：返工（revision）生成路径未闭合

- 严重程度：重要
- 涉及章节：5.4 detect_revision_segments；5.10；4.11；1.5
- 涉及字段或函数：`revision_segments`、`is_revision`、`revision_input`、`return_to_agent`
- 问题说明：stage0 可检测返工片段，但 `reconstruct_all_trajectories` 不接收 revision_segments，也未说明如何把 Agent3 的 `return_to_agent` 映射为新的 Agent1/2 返工轨迹；视频内返工与系统内打回是两套概念，未定义衔接。
- 为什么阻塞实现：任务6 无法按契约实现完整返工数据路径。
- 建议修正方案：增加 `reconstruct_revision_trajectories(...)` 或扩展 `reconstruct_all_trajectories` 签名；明确 revision 样本的输入（VerificationResult 或视频片段）、输出 Agent、与主轨迹的关联 id。

### ISSUE-I003：架构描述“按 move 抽帧”与阶段顺序/签名矛盾

- 严重程度：重要
- 涉及章节：第2节主流水线；5.5；5.6
- 涉及字段或函数：stage1 `extract_keyframes(time_range)`；stage2 `build_moves`
- 问题说明：架构写阶段1「按 move 时间段抽关键帧」，但 move 在阶段2才产生；函数签名实际按 agent 阶段 `time_range` 抽帧。
- 为什么阻塞实现：实现顺序与接口依赖关系不清晰，易做错数据流。
- 建议修正方案：将架构文案改为「按 agent 阶段时间区间抽帧」；若确需按 move，则调整阶段顺序与签名。

### ISSUE-I004：production + pending 与“初始均为 production”语义不清

- 严重程度：重要
- 涉及章节：4.3；4.8；5.8 execute_action；第6节
- 涉及字段或函数：`ToolTier`、`implementation_status`、`execute_action`
- 问题说明：初始 tool「均为 tier=production」，但 `implementation_status` 默认 `pending`；execute_action 对 pending 走 VLM 合成。种子 tool 究竟应 pending 还是 deployed 未写清。
- 为什么阻塞实现：Executor 行为与注册表初始状态无法确定。
- 建议修正方案：为每个初始 tool 明确 status；区分「有本地实现但未部署」与「已可真实执行」；统一 status 字段命名（文档混用 status/implementation_status）。

### ISSUE-I005：allowed_agents 无法表达“同一 Tool 对不同 Agent 的使用限制”

- 严重程度：重要
- 涉及章节：1.4；4.8 web_search；10
- 涉及字段或函数：`allowed_agents`、COARSE「仅宽泛查询」、VERIFIER「仅验证性查询」
- 问题说明：权限模型只有允许/不允许，无法编码查询宽度、用途约束；只能靠 prompt 软约束，与「硬约束」表述冲突。
- 为什么阻塞实现：stage3/训练数据无法自动校验“宽泛/验证性”违规。
- 建议修正方案：增加 `usage_policy: dict[AgentRole, Policy]`（如 query_scope=broad|precise|verify_only），或拆成不同 tool。

### ISSUE-I006：Action.params 与动态 Observation 缺少明确运行时校验入口

- 严重程度：重要
- 涉及章节：4.11 Action；5.8；5.9
- 涉及字段或函数：`Action.params: dict`、observation `dict`
- 问题说明：文档要求套统一 schema，但未指定校验函数签名（例如 `validate_action_params(tool, params)` / `validate_observation(tool, obs)`），也未说明非法 params 时硬套/丢弃/重试策略。
- 为什么阻塞实现：stage3/4 错误处理无法统一。
- 建议修正方案：在 schemas 或 tools/base 中规定校验 API、异常类型与失败处置。

### ISSUE-I007：stage3 返回结构与 stage4 输入未直接对齐

- 严重程度：重要
- 涉及章节：5.7；5.9；5.10
- 涉及字段或函数：`normalize_to_actions` → `list[tuple[str, Action]]`；`generate_observations(actions: list[Action])`
- 问题说明：stage3 产出 thought_draft+action，stage4 只收 actions；thought_draft 如何进入 stage5 未定义。另：`screen_action is None` 的 move 是否生成 action 未说明。
- 为什么阻塞实现：阶段串联需自行猜测中间结构。
- 建议修正方案：定义中间结构（如 `NormalizedStep`），明确空 screen_action 的处理（跳过 / 纯 narration thought / 丢弃）。

### ISSUE-I008：stage4 返回类型无法表达 submit_answer 的空 observation

- 严重程度：重要
- 涉及章节：5.9；4.11
- 涉及字段或函数：`generate_observations` → `list[tuple[dict, str]]`
- 问题说明：返回类型写死 `dict`，与 submit_answer observation=None 冲突；与 TrajectoryStep 不一致。
- 为什么阻塞实现：类型标注与真实数据流矛盾。
- 建议修正方案：改为 `list[tuple[Optional[dict], Optional[str]]]` 或专用 `ObservationResult` 模型。

### ISSUE-I009：answer_timestamp 与 VERIFIER 阶段边界可能冲突

- 严重程度：重要
- 涉及章节：1.5；5.4；5.10
- 涉及字段或函数：`answer_timestamp`、`agent_segments[VERIFIER]`
- 问题说明：防泄漏要求 answer_timestamp 之前 Thought 不得含最终地名；Agent3/验证总结通常发生在宣布答案之后。未规定：VERIFIER 是否允许出现在 timestamp 之后、泄漏检查是否对 VERIFIER 豁免、宣布答案后的片段如何合法进入 Agent3。
- 为什么阻塞实现：stage0 切分与 stage5 泄漏规则可能互相否定。
- 建议修正方案：明确规定 COARSE/FINE 必须结束于 answer_timestamp 之前（或允许的重叠规则）；VERIFIER 默认在其后且泄漏检查策略不同。

### ISSUE-I010：Tool endpoint 语义与升档回滚/健康检查缺失

- 严重程度：重要
- 涉及章节：5.3 promote_tool；5.15；第6节
- 涉及字段或函数：`endpoint`、`promote_tool`
- 问题说明：示例为 Python 导入路径 `tools.<name>`，正文又写「部署到云服务器」，未明确是 import path 还是 HTTP URL；无健康检查、无升档失败回滚、无历史数据替换失败时的备份策略。
- 为什么阻塞实现：任务2/升档流程无法安全实现。
- 建议修正方案：明确 endpoint 类型；规定 promote 前 smoke test；升档采用写时复制/备份 JSONL，失败回滚注册表与数据。

## Minor Issues

### ISSUE-M001：F6 在 model_validator 中查注册表存在加载循环依赖

- 严重程度：一般
- 涉及章节：4.5 F6；5.3 load_registry
- 涉及字段或函数：`derived_from_existing_tools`
- 问题说明：加载注册表时逐条构造 ToolDefinition 会触发 F6，但注册表尚未完全加载。
- 建议修正方案：F6 改为注册写入时的外部校验，或允许传入 registry snapshot。

### ISSUE-M002：sun_position_calc 的 possible_latitude_range 类型标注为 latlng 不合理

- 严重程度：一般
- 涉及章节：4.8 sun_position_calc；4.3 latlng
- 涉及字段或函数：`possible_latitude_range`
- 问题说明：latlng 定义为 (lat, lng)，纬度范围不是坐标对。
- 建议修正方案：改为 `(float, float)` 语义为 (min_lat, max_lat)，或新增 `lat_range` 类型。

### ISSUE-M003：ParamField.example 对所有类型强制，可选参数也必须给 example

- 严重程度：一般
- 涉及章节：4.3 ParamField
- 涉及字段或函数：`example`
- 问题说明：可实现，但与 optional/default 组合时文档未说明 example 是否仍必填。
- 建议修正方案：明确始终必填，或 `required=False` 时可空。

### ISSUE-M004：并发写 tool_registry.json / 共享 JSONL 无文件锁约定

- 严重程度：一般
- 涉及章节：5.3；5.14；第2节编排层
- 涉及字段或函数：`register_tool`、`batch_run`、`format_all_and_save`
- 问题说明：多视频并发可能损坏注册表或交错写 JSONL。
- 建议修正方案：规定文件锁、单 writer 队列、或按 video 分片再合并。

### ISSUE-M005：diskcache key 未包含图像指纹与 tool schema 版本

- 严重程度：一般
- 涉及章节：5.8
- 涉及字段或函数：cache key = tool_name + params hash
- 问题说明：zoom_inspect/ocr 等依赖图像；schema 变更后可能命中脏缓存。
- 建议修正方案：key 增加 image hash（或 path+mtime）与 tool schema hash/version。

### ISSUE-M006：verify_and_score 对 Agent1/3 的 distance_error_km 语义未定义

- 严重程度：一般
- 涉及章节：5.11；4.12
- 涉及字段或函数：`tuple[bool, float, float]`、`distance_error_km`
- 问题说明：三元组对所有 Agent 统一返回，但 Agent1/3 无坐标误差时第三值含义不明（NaN？-1？None？类型却是 float）。
- 建议修正方案：改为 `Optional[float]` 或按 Agent 返回专用结果模型。

### ISSUE-M007：checkpoint 中间产物文件名/schema 未标准化

- 严重程度：一般
- 涉及章节：5.13；第2节
- 涉及字段或函数：`data/intermediate/{video_id}/`
- 问题说明：仅给目录，未规定每阶段文件名与完成标记，断点续跑难以统一实现。
- 建议修正方案：规定如 `stage0.json`…`stage7.json` 与 `DONE`/`manifest.json`。

### ISSUE-M008：依赖与模型名存在不确定性

- 严重程度：一般
- 涉及章节：第7节；5.1；5.2
- 涉及字段或函数：`gemini-2.5-pro`、`google-genai`、各第三方 SDK
- 问题说明：模型名与 SDK API 可能随时间变化；未钉版本号。
- 建议修正方案：requirements 钉版本；配置允许覆盖模型名；在适配层隔离 SDK。

## Interface Closure Check

1. Transcript → Preprocess — **PARTIAL**  
   `VideoInput.transcript` 可进入 `preprocess`，但 `preprocess` 返回裸 `dict`，无 Pydantic 模型，键类型（AgentRole 作 key）序列化约定缺失。

2. Preprocess → Keyframes — **PARTIAL**  
   可用 `agent_segments` 的 time_range 调 `extract_keyframes`；与架构“按 move 抽帧”表述冲突（见 ISSUE-I003）。

3. Transcript + Screen Actions → Moves — **PARTIAL**  
   `build_moves` / `build_all_agent_moves` 基本闭合，但 screen_actions 与 transcript 分段的时间对齐算法未定义；列表长度关系未规定。

4. Moves → Actions — **PARTIAL**  
   `normalize_to_actions` 可消费 Move，但空 screen_action、硬套标记、组合 tool 方案的返回结构未定义（见 ISSUE-I007）。

5. Actions → Observations — **PARTIAL**  
   数量上“对每个 action 调用”隐含 1:1，但 submit_answer 空 observation 与返回类型冲突（ISSUE-B004/I008）。

6. Agent1 Trajectory → LocationHypothesis — **PASS**  
   Agent1 输出类型在 4.9/5.10 有明确定义，可从轨迹提取。

7. LocationHypothesis → Agent2 Trajectory — **PASS**  
   `handoff_input: LocationHypothesis` 对 Agent2 匹配。

8. Agent2 Result → Agent3 Trajectory — **FAIL**  
   缺少 SubmitAnswerResult；Agent3 handoff 类型错误（ISSUE-B002/B003）。

9. VerificationResult → Revision Trajectory — **FAIL**  
   VerificationResult 有 return_to_agent，但重构/格式化路径未闭合（ISSUE-I002）。

10. Trajectory → DatasetEntry — **PARTIAL**  
    字段大体够用（含 draft/revision/agent_role），但 Agent3 handoff 类型错误会污染 DatasetEntry.handoff_input。

11. DatasetEntry → Three LoRA JSONL Files — **PASS**  
    按 `agent_role` 分写三份 JSONL 的路径明确；loss mask 约定明确。

12. Draft Tool → Promotion → Historical Data Replacement — **PARTIAL**  
    主流程有文字描述，但 endpoint 语义、并发安全、失败回滚、健康检查缺失（ISSUE-I010/M004）。

## Recommended SPEC Changes

1. **建议编号：R001**  
   - 修改章节：4.4 / 4.8  
   - 当前问题：初始 tool 名违反 A1  
   - 建议的新定义：统一重命名种子 tool，或增加 `SEED_TOOL_NAME_ALLOWLIST` 豁免并写明不再新增豁免  
   - 是否属于开始任务1前必须修复：是

2. **建议编号：R002**  
   - 修改章节：4.9 / 4.11 / 5.10  
   - 当前问题：Agent2/3 交接类型错误且缺 schema  
   - 建议的新定义：新增 `SubmitAnswerResult`；修正 `Trajectory.handoff_input` 联合类型或双字段  
   - 是否属于开始任务1前必须修复：是

3. **建议编号：R003**  
   - 修改章节：4.3 / 4.5 / 4.8 / 4.11  
   - 当前问题：submit_answer 无 observation 与 F2、observation_source 冲突  
   - 建议的新定义：terminal tool 豁免 F2；`observation_source` 改为 Optional  
   - 是否属于开始任务1前必须修复：是

4. **建议编号：R004**  
   - 修改章节：4.8 map_query / 4.5 F2  
   - 当前问题：全部 nullable  
   - 建议的新定义：增加非空 `status` 或将 `formatted_address` 在成功路径设为必填策略  
   - 是否属于开始任务1前必须修复：是

5. **建议编号：R005**  
   - 修改章节：4.3 ObservationField  
   - 当前问题：result_list 无子 schema  
   - 建议的新定义：`item_fields: list[ObservationField] | None = None`，type=result_list 时必填，禁止嵌套 result_list  
   - 是否属于开始任务1前必须修复：是

6. **建议编号：R006**  
   - 修改章节：4.3 ParamField  
   - 当前问题：无 default  
   - 建议的新定义：增加 `default` 字段与补齐规则  
   - 是否属于开始任务1前必须修复：是

7. **建议编号：R007**  
   - 修改章节：5.10 / 5.4 / 新修订节  
   - 当前问题：返工路径未闭合；groundtruth 进入重构签名  
   - 建议的新定义：补 revision API；限制 groundtruth 仅用于模型外检查  
   - 是否属于开始任务1前必须修复：否（建议任务1前至少澄清 groundtruth；返工可在任务6前修复）

8. **建议编号：R008**  
   - 修改章节：第2节 / 5.5 / 5.7 / 5.9 / 5.8 / 第6节  
   - 当前问题：阶段文案、中间结构、endpoint、pending/deployed  
   - 建议的新定义：统一抽帧粒度；定义 NormalizedStep；明确 endpoint 与初始 status  
   - 是否属于开始任务1前必须修复：否（任务2前必须修复 endpoint/status；任务3前修复抽帧文案）

## Final Decision

### BLOCKED

存在阻塞问题，必须先修改 SPEC.md，不能开始任务1。
