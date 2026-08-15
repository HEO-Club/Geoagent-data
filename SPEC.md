# 图片地理定位 Agent 训练数据集生成流水线 — 项目规格

版本：3.3.7 | 生成时间：2026-08-15
修订说明：v3.3.7 条件式复核的「明确答案重复」改为：多 resolved task 时用 LLM 判定最终地点是否同地/同答案链（字符串全等仅作零成本正例短路），命中后再走双向复核；不得靠措辞归一化猜同地。v3.3.6 源输入归并改为「包含/放大硬合并 + 是否同一张照片」两两判定（union-find）；brief 只核验画面事实、不作张数配额；拼图中已出现的一格全屏再出示必须同组。v3.3.5 选图密采样改为过程时间线：蒸馏窗字幕抽取多段 `process_intervals`（`show_source`/`tool`/`reveal`/`other`），只在 `show_source` 并集内密采样；逐帧注入区间角色软先验；时间线失败回退单出示粗窗；多段不相邻出示却只选出 1 张且 brief 非空 → `needs_review`；Stage 2 后置门禁可拦「轨迹明确依赖第二份输入却只选 1 张」。v3.3.4 选图改为定位链证据对齐：题级 `visual_evidence_brief` + 逐帧 `evidence_role`/`chain_support_score`；组间按证据支撑择优、组内仍按干净度；Stage 2 后可选轨迹–选图一致性门禁（只拦入库、不换图）。v3.3.3 取消审核模型预定选图张数；选图改为每题「最小源输入集」（出示连续段折叠 + 源输入关系判定），`expected_image_count` 仅记录实选张数。v3.3.2 将数据落盘三分：`data/selected/` 仅最终选图，`data/intermediate/` 仅阶段 JSON/门禁元数据，密采样探测帧进 `.cache/audit_candidates/`；跑批摘要进 `data/runs/`。v3.3.1 将阶段1.5 选图改为蒸馏窗与出示窗分离。v3.3.0 将阶段2改为 `reasoning / tool_call / final` 三类事件；阶段3引入执行器级 Canonical Tool 目录。旧规格见 `SPEC_legacy_v2.md`。

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
   - **条件式复核**：首次切分由模型负责。规则只检查高精度结构矛盾（模型主动低置信、resolved 却无地点、时间窗与候选严重冲突等）以及**答案同地嫌疑**：≥2 个 `resolved` task 时，最终地点字符串全等可直接视为重复；若措辞不同，另用一次廉价 LLM（可带字幕上下文）判定是否同一最终地点或同一答案链（同题多图/后图精化），命中则记异常。无异常不重复调用双向复核模型。触发后由模型保守地双向复核，可合并过拆，也可补拆漏题；不得为了体现复核而改动正确切分；不得用脆弱地名归一化/词表代替同地判定。
   - **蒸馏窗**：`time_start`/`time_end`（及可选 segment 索引）覆盖**整条答案链旁白**（问题设定 → 推理比对叙述 → 最终地点结论），不得裁成仅关键帧中段；蒸馏窗只管 Stage 2 字幕切片完整性，**不得**作为密采样选图区间。
   - **出示窗（选图）**：审核模型仍可给单段出示粗窗（弱先验）。选图前另从蒸馏窗字幕抽取题级 **过程时间线** `process_intervals`（角色：`show_source` / `tool` / `reveal` / `other`；可用稀疏审计帧，不为时间线再密采样）。**密采样区间 = 所有 `show_source` 的并集**，裁进蒸馏窗、不得跨相邻题；`display_time_start/end` 记为该并集包络（时间线失败或无 `show_source` 时回退现行单出示粗窗 / 蒸馏窗前段封顶）。**不再报预定张数，也不再以精确关键帧秒数为主提案**。禁止把密采样扩大到整条蒸馏窗或 `tool`/`reveal` 段狂打 VLM。
   - **关键帧 = 定位链实际依赖的待定位实拍**（去掉讲解后 agent 仍要据之定位的现场/静帧）。选图目标是每题 **最小源输入集**（不是出示窗内任意干净实拍，也不是凑字幕估数/视觉簇个数）。流程：蒸馏窗字幕抽取题级 **`visual_evidence_brief`**（仅「看原图/现场时用到的视觉事实」，不含工具核验/搜索命中/揭晓）→ 抽取 `process_intervals` → 仅在 `show_source` 并集密采样 → 廉价视觉过滤（近黑/明显地图或大面积 UI 降权）→ VLM 逐帧验收（注入该时刻区间角色软先验）→ **出示连续段折叠**（`unused_broll` 与工具/揭晓一样打断连续段；每段只留质量最高代表）→ **源输入关系判定**（先做包含/放大硬合并：拼图一格的全屏再出示、裁切放大必须同组且优先留更完整帧；再两两判定是否同一张照片：`same_photo` / `same_scene` / `different_photo` / `not_input`，union-find 聚合；brief 只核验画面事实、**不得**当作张数配额；不得因「两段出示窗」硬拆成两张；不确定是否同一张照片时仍合并）→ 组间按证据支撑排序、组内按干净度择优 → 配置上限截断。找不到干净 (A)、有 brief 但支撑过低、或时间线给出 ≥2 段不相邻 `show_source` 却只选出 1 张且 brief 非空时 `needs_review`；禁止用空镜/`unused_broll`/工具帧充数。
   - **逐帧验收与择优**：VLM 对候选分别输出 `kind`、质量、答案泄露、讲解覆盖、干净原图标记，以及 **`evidence_role`**（`problem_input` / `unused_broll` / `process_tool` / `reveal` / …）与 **`chain_support_score`**（画面支撑 brief 视觉事实的程度）。区间角色为软先验：画面仍是最终裁判；`tool`/`reveal` 降低「全屏建筑 = 原图」倾向。择优顺序：**证据角色/支撑分 → 干净原图 → 无讲解覆盖 → 质量分**（思维链选源，干净度选帧）。工具/核验/揭晓帧不得入 `image_paths`，含最终答案泄露的帧不得入库。不得用时间距离判断「是否新输入」；不确定是否同一张照片时不追加。画面主体场景结构明显不同（不同建筑立面/店面/路幅，非同一取景远近）必须判为不同原图。brief 为空时回退现行质量择优路径。
   - **`video_derived` / 同题多输入**：默认 1 个现场代表帧；同一飞行/同一段路的换机位不加张。多张只来自被判定为不同的源输入（不同原图或另一段独立现场），受配置上限约束；不得用工具步骤或揭晓帧充数。
   - **答案与 task 门禁**：每题分别输出 `answer_status=resolved|ambiguous|unsolved` 与 `status=accepted|needs_review|rejected`。只有答案唯一明确、图片完整干净的 `accepted` task 进入 Stage 2；模棱两可或明确无解的题单独拒绝，选图失败进入人工复核，均不得影响同视频其他题。切题过粗（一题塞多个最终地点）走结构复核，**不在选图里用答案个数定张数**。
   - 数量：普通静图默认 1 张最佳帧；多张只来自源输入关系判定为不同；`expected_image_count` / `multi_target_images` 为选完后的实选结果（`len(image_paths)` / `>1`），不作预定配额。
3. **阶段2（自由事件轨迹）**：优先接收 Stage 1.5 的单题字幕切片 + `image_paths`；若上游拆题不可用而收到整视频，则恢复字幕级保险，识别全部独立定位题，并令末步 `location` 以字符串数组按讲解顺序列出全部最终地点。事件分为 `reasoning / tool_call / final`：直接看图、合并已有证据、比较、筛选、排除、排名、形成目标签名和计划转向只写 reasoning；只有真实访问外部搜索、数据库、地图/街景/卫星/天气服务或执行图像/GIS/计算程序并产生新证据时才写 tool_call + Observation。允许连续 reasoning，但每条必须完成实质认知更新；只为解释下一次调用的一句话并入该 tool_call 的 Thought。产物禁止「求助者/网友/评论区」等渠道元话语，末步严格为 `event_type=final`、`final_answer`、`params.location`、`observation=null`。生成成功后可选 **轨迹–选图–brief 一致性门禁**（高精度冲突才拦）：含「开篇 reasoning 明确依赖第二份独立输入但 `image_paths` 只有 1 张」；冲突则跳过 Stage 3 入库，**不换图、不重蒸、不改轨迹**。过程时间线不得写入阶段2 prompt。
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
    │       ├── stage2_freeform_tao.json            # 仅 accepted
    │       ├── stage3_trajectory.json
    │       └── stage3_tool_mapping.json
    └── output/
        ├── shards/{task_id}.jsonl
        └── geolocate_agent.jsonl
.cache/
    ├── audit_sparse/               # 审核用稀疏帧
    ├── audit_candidates/           # 出示窗密采样探测帧（可随时清空）
    └── keyframes/                  # 其他抽帧缓存
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
