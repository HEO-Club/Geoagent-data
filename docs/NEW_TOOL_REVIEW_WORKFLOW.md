# 临时 Tool 人工/Codex 复核与正式接入规范

## 1. 原则

Stage 3 不得修改 `canonical_tool_catalog_v2.json`，也不得生成可自动入库的新 Tool。目录外调用一律以临时 Tool 或临时 operation 完成本 task，并进入 `temporary_review` 池。临时项只表示“当前归并没有可靠完成”，不等于已经证明需要新增 Tool。

## 2. 两个数据池

- `canonical_only`：所有外部动作均已匹配到正式 Canonical Tool 和正式 operation。
- `temporary_review`：至少出现一个临时 Tool、临时 operation 或模糊未映射项。`stage3_tool_review_candidates.json` 保存步骤、Thought、原始参数、Observation、模型理由、最接近的正式 Tool 和置信度，供人工或 Codex 复核。

使用 `scripts/partition_tool_review_pool.py` 可对批量结果生成两个池子的清单。该脚本只分流，不修改轨迹和工具目录。

## 3. Codex 复核输入与允许结论

复核时必须同时读取：

1. 当前 task 的 `stage3_tool_review_candidates.json`；
2. 当前 task 的 `stage2_freeform_tao.json` 与字幕切片；
3. `docs/tool_specs_v2/` 下 31 份正式 Tool 的完整 schema 和实现边界；
4. 至少一个独立样本中的同类候选；
5. 如主张新后端，必须提供其官方文档、数据许可和认证入口。

复核只能输出以下五类结论：

- `map_existing`：可归入现有 Tool/operation，手工修正当前说明书；
- `use_existing_operation_with_repaired_inputs`：Tool/operation 已存在，只需修改参数；
- `rewrite_as_reasoning`：没有真实外部执行动作，不应保留 Tool；
- `candidate_new_operation`：真实执行器已存在，但执行阶段与现有 operation 都不同；
- `candidate_new_tool`：存在全新的真实执行器/API/数据库/本地程序边界。

## 4. 新 Tool 的严格门槛

只有同时满足以下条件，才允许进入正式设计阶段：

1. 已逐一排除 31 类 Tool 及其全部 operation，并给出逐项可核对理由；
2. 差异来自独立执行器边界，而不是对象、地区、查询词、参数、输出格式或自然语言名称不同；
3. 至少两个独立 task 反复出现同一能力缺口。单例默认不新增；极少数单例若对应明确且不可替代的官方执行器，也必须由人工负责人书面批准；
4. 能指出真实可实现后端，并提供官方文档、认证方式、数据覆盖、许可、成本和失败语义；
5. 能定义稳定的 name、description、executor、usage、operations 与完整 `input_schema`；
6. 能实现最小只读 executor，禁止以“让模型写任意代码”代替工具实现；
7. 有正例、近邻 Tool 反例、缺参、无结果、无覆盖、权限错误和 provider 错误测试；
8. 通过人工审核和 PR，不允许脚本直接覆盖正式目录。

任一项不满足，结论必须回到 `map_existing`、`rewrite_as_reasoning` 或继续保留临时项，不能为追求覆盖率而新增。

## 5. 真正确认需要新增时的步骤

1. 新建独立候选设计文件，先不改正式目录；
2. 按现有 v2 schema 写明每个 operation 的字段类型、必填级别、别名、范围、示例、解释、上下文来源和 `acquisition_hint`；
3. 完成官方资料调研与许可评估，明确获取 Key/账户的位置；
4. 实现只读 executor 和统一返回合同，区分 `no_result`、`no_coverage`、`permission_denied` 与 `provider_error`；
5. 运行最小真实调用测试，并证明 Observation 来自真实返回；
6. 用历史临时样本回放，比较新增前后的映射准确率和误归并率；
7. 人工审核通过后单独提 PR 更新 `canonical_tool_catalog_v2.json`、实现代码、测试和文档；
8. 合并后重新跑受影响样本，将临时调用迁移到新 Tool。不得修改原始字幕或伪造执行记录。

## 6. 评审输出建议

Codex/人工评审记录至少包含：`candidate_id`、`decision`、`existing_tools_checked`、`execution_boundary`、`evidence_tasks`、`proposed_schema`（仅在确认进入设计阶段后填写）、`implementation_options`、`license_risks`、`test_plan`、`reviewer`、`reviewed_at`。`decision` 不是 `candidate_new_tool` 时，`proposed_schema` 必须为空。
