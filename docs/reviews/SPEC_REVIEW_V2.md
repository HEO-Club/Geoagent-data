> **ARCHIVED / 历史归档**：本文档不是现行规格。现行唯一有效规格见仓库根目录 `SPEC.md`。

# SPEC Review V2

## Review Summary

- 检查时间：2026-07-13
- SPEC 版本：2.1
- 对照基线：SPEC_REVIEW.md（针对 SPEC v2.0）
- 阻塞问题数量：0
- 重要问题数量：0
- 一般问题数量：0
- 结论：READY FOR TASK 1

## Re-check of Original Blocking Issues

| 原编号 | 原问题 | V2 状态 | 解决方式（SPEC 2.1） |
|--------|--------|---------|----------------------|
| ISSUE-B001 | 种子 Tool 名违反 A1 动词规则 | **RESOLVED** | 删除「必须以 ALLOWED_VERBS 动词开头」硬规则；新 A 规则为 snake_case/长度/禁用词等；保留 7 个种子名；ALLOWED_VERBS 仅作 LLM 提示 |
| ISSUE-B002 | Agent3 handoff_input 类型错误 | **RESOLVED** | 删除统一 `handoff_input`；改为 `coarse_handoff` / `fine_handoff`，VERIFIER 必填 `fine_handoff` |
| ISSUE-B003 | 缺少 SubmitAnswer 独立 Schema | **RESOLVED** | 新增 `SubmitAnswerResult`；Agent2 终端步 params 必须解析为此模型并写入 `fine_output` |
| ISSUE-B004 | submit_answer 与 F2 / observation_source 冲突 | **RESOLVED** | 增加 `is_terminal`；terminal 的 `observation_fields=[]`；`observation` 与 `observation_source` 均为 Optional/None |
| ISSUE-B005 | map_query 全 nullable 违反 F2 | **RESOLVED** | 普通 Tool 强制非空 `status` + 可空 `error_message`；业务字段可继续 nullable；删除旧 F2 |
| ISSUE-B006 | result_list 未关联 ResultItem | **RESOLVED** | `ObservationField.item_fields`；type=result_list 时必填；删除独立 ResultItem |
| ISSUE-B007 | ParamField 无法表达 default | **RESOLVED** | 增加 `default`、`enum_values`；补齐规则与类型支持（含 lat_range） |

## Re-check of Original Important Issues

| 原编号 | V2 状态 | 解决要点 |
|--------|---------|----------|
| ISSUE-I001 | **RESOLVED** | stage5 全部签名删除 groundtruth；仅 stage6 使用 |
| ISSUE-I002 | **RESOLVED** | `RevisionContext` + `reconstruct_revision_trajectories`；区分 video_observed / system_feedback；`MAX_REVISION_ROUNDS` |
| ISSUE-I003 | **RESOLVED** | 架构与 stage1 统一为按 Agent 时间区间抽帧；Move 在 stage2 |
| ISSUE-I004 | **RESOLVED** | 仅 draft/production；删除 pending/deployed；`executor_ref` |
| ISSUE-I005 | **RESOLVED** | web_search 增加 `purpose` 枚举与角色硬约束；allowed_agents 含 VERIFIER |
| ISSUE-I006 | **RESOLVED** | 规定 `pipeline/tools/validation.py` 三个校验 API |
| ISSUE-I007 | **RESOLVED** | `NormalizedStep` + `normalization_mode`（含 thought_only） |
| ISSUE-I008 | **RESOLVED** | `ObservationExecutionResult`；terminal 为 status=skipped |
| ISSUE-I009 | **RESOLVED** | 1.6 节时间规则：COARSE/FINE 用答案前证据；VERIFIER 可用答案后验证片段 |
| ISSUE-I010 | **RESOLVED** | executor_ref 为 Python 导入路径；升档 smoke test + 备份/回滚；历史数据重跑 stage4→7 |

## Re-check of Original Minor Issues

| 原编号 | V2 状态 | 解决要点 |
|--------|---------|----------|
| ISSUE-M001 | **RESOLVED** | derived_from 存在性改在 `register_tool` 用 snapshot 检查 |
| ISSUE-M002 | **RESOLVED** | 新增 `lat_range`；sun_position_calc 使用该类型 |
| ISSUE-M003 | **RESOLVED** | example 始终必填且符合类型；default 规则写明 |
| ISSUE-M004 | **RESOLVED** | registry 文件锁+原子写；JSONL 分片+单 writer 合并 |
| ISSUE-M005 | **RESOLVED** | cache key 含 schema/executor/params/image/(model/prompt) |
| ISSUE-M006 | **RESOLVED** | `TrajectoryVerificationReport.distance_error_km: Optional[float]` |
| ISSUE-M007 | **RESOLVED** | 统一 intermediate 文件名与 `VideoManifest` / `StageStatus` |
| ISSUE-M008 | **RESOLVED** | 配置读模型名、adapter 封装、requirements 版本范围/lock |

## Blocking Issues

无。

## Important Issues

无。

## Minor Issues

无。

## Interface Closure Check

1. Transcript → Preprocess — **PASS**  
   `VideoInput.transcript` → `preprocess() -> PreprocessResult`；`agent_segments` 为对象列表，不以 AgentRole 作 JSON key。

2. Preprocess → Keyframes — **PASS**  
   使用 `AgentTimeSegment` 的时间区间调用 `extract_keyframes`；明确不按 Move 抽帧。

3. Transcript + Screen Actions → Moves — **PASS**  
   `TimedScreenAction` + 时间重叠对齐；禁止按下标配对；产出 `Move`。

4. Moves → Actions — **PASS**  
   `normalize_to_steps() -> list[NormalizedStep]`；支持多 Action、thought_only、draft/composed/fallback。

5. Actions → Observations — **PASS**  
   `generate_observations() -> list[ObservationExecutionResult]`；与 Action 一一展开；terminal 为 skipped/None。

6. Agent1 Trajectory → LocationHypothesis — **PASS**  
   `coarse_output: LocationHypothesis` 明确。

7. LocationHypothesis → Agent2 Trajectory — **PASS**  
   `coarse_handoff` 必填；与 `reconstruct_all_trajectories` 传递一致。

8. Agent2 Result → Agent3 Trajectory — **PASS**  
   `SubmitAnswerResult` + `fine_output` / `fine_handoff` 闭合。

9. VerificationResult → Revision Trajectory — **PASS**  
   `return_to_agent` 映射 + `RevisionContext` + `reconstruct_revision_trajectories` + 轮数上限。

10. Trajectory → DatasetEntry — **PASS**  
    handoff/output/revision 字段对齐；terminal 与 loss mask 约定明确。

11. DatasetEntry → Three LoRA JSONL Files — **PASS**  
    按 `agent_role` 分片再合并三份 JSONL。

12. Draft Tool → Promotion → Historical Data Replacement — **PASS**  
    显式触发；smoke test；备份/原子更新/回滚；历史样本重跑 stage4→5→6→7。

## Recommended SPEC Changes

无必须项。当前 SPEC 2.1 可进入任务1。

## Final Decision

### READY FOR TASK 1

Blocking Issue 数量为 0；原 SPEC_REVIEW.md 全部问题已复核关闭；Interface Closure Check 12/12 为 PASS。可以开始任务1：基础设施与数据契约。
