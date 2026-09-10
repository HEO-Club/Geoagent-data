# Tool Runtime Harness

## 定位

Tool Runtime Harness 是 Stage 3 之后的可选执行/复核层，不是新的数据蒸馏 Stage，也不会被 Stage 1–4 默认调用。Stage 2/3 仍负责从视频讲解中恢复“当时如何推理、调用了什么工具”；Harness 负责在获得明确授权和 Provider 配置后，尝试用当前真实执行器重放 Canonical Action，并输出独立报告。

Harness 永远不覆盖 `stage3_trajectory.json`，也不会用当前重放结果改写原视频中的 Observation。历史网页、OSM、街景或卫星数据可能已变化，因此重放差异只能进入审核报告，不能自动断定讲解为虚假。

## 输入与输出

输入是标准 `Trajectory`：

```text
stage3_trajectory.json
  ├── reasoning：只记录，不执行
  ├── tool_call：校验并调用真实执行器
  └── final：校验 location，不访问外部服务
```

默认输出目录：

```text
data/tool_runs/{safe_trajectory_id}_{id_hash}/
  ├── harness_report.json   # 完整执行报告和检查点
  ├── events.jsonl          # append-only 生命周期事件
  ├── results/              # 持久化结构化 result_id
  ├── images/               # 可恢复的派生 image_id
  ├── artifacts/            # 其他工具产物
  └── .harness.lock         # 同一轨迹单 writer 锁
```

## 执行顺序

每个 `tool_call` 依次经过：

1. 校验外层合同必须且只能包含 `operation / purpose / inputs`。
2. 拦截 `api_key / token / client_secret / password / authorization` 等凭证字段，并对报告脱敏。
3. 解析受限运行时引用；不使用 `eval` 或任意 JSONPath。
4. 使用正式 Canonical Tool 目录重新归一参数并计算 readiness；归一化器异常也只隔离当前步骤。
5. `ready / context_resolvable` 才调用执行器；`repairable / invalid` 分别记录为 blocked / invalid。
6. 调用前写入 `running` 检查点和 `step_started` 事件。
7. 把稳定 `call_id` 注入 `RuntimeContext.extras`，未来 Provider 支持幂等键时可直接复用。
8. 把任意第三方异常收成 `executor_exception`，不让单个 Provider 崩溃整个批次。
9. 将 Provider 回执严格收敛为有限、可 JSON 序列化的数据；非法对象、NaN 和 Infinity 记为 `invalid_observation`。
10. 成功后保存真实 Observation、artifacts、result_id、依赖步骤和参数审计。
11. 原子更新检查点，并写入 `step_finished` 事件。

默认失败开放：一个独立 Tool 失败后仍执行后续不依赖它的步骤；引用失败结果的步骤会自然进入 blocked。只有显式传 `--stop-on-error` 时才阻止后续 Tool。Final 仍记录原轨迹答案，便于把“答案”和“当前重放是否成功”分开审核。

## 支持的引用

只允许完整字符串形式的引用：

```text
$current_image
$current_images
$previous_tool_result
$active_area
$active_session
$step_2_tool_result
$step_2_tool_result.candidates[0].bbox
$step_3_tool_result.result_id
```

引用可以出现在 inputs 的任意嵌套对象或数组值中，但不做字符串插值。当前或未来步骤引用、未知引用、缺失字段、数组越界和凭证字段引用都会产生结构化错误。

每个成功解析的引用会记录：输入路径、引用原文、来源、依赖步骤、结果类型和脱敏预览。恢复时，如果某一步因产物丢失而重跑，所有显式或上下文依赖该步骤的旧结果都会被保守地重新执行。

## 断点恢复边界

`harness_report.json` 同时保存：

- Trajectory 内容指纹；
- Canonical Tool 目录指纹；
- `tool/**/*.py` 执行器代码以及运行前参数归一化/Pydantic 合同指纹；
- 初始运行上下文和 Provider 类型指纹；
- Trajectory、上下文和 Tool inputs 中实际存在的本地输入文件内容指纹；
- `stop_on_error` 执行策略；
- 每一步 call_id、状态、参数审计和真实回执。

任一关键指纹变化时，Harness 拒绝复用旧检查点，避免“代码已变但旧结果被当成新结果”。如果 Provider 的配置或测试数据改变但 Python 类型不变，调用方应设置：

```python
ctx.extras["harness_provider_revision"] = "provider-config-v2"
```

CLI 可用等价参数：

```powershell
--provider-revision "provider-config-v2"
```

如果进程上次停在 `running`，恢复时会重跑该步骤并写明可能产生重复计费。目前执行器都是只读查询或确定性本地处理；未来若接入有副作用的 Tool，必须在 Provider adapter 中增加 idempotency key，不能依靠 Harness 猜测 exactly-once。

成功步骤产生的本地 artifact 会保存路径、大小和 SHA-256。恢复时，输入文件内容变化会拒绝整个旧检查点；派生产物缺失或内容变化会重跑生产步骤，并向下传播使依赖步骤失效。

## CLI

```powershell
.\.venv\Scripts\python.exe run_tool_harness.py `
  --trajectory "data/intermediate/VIDEO/tasks/TASK/stage3_trajectory.json" `
  --workspace "data/tool_runs" `
  --catalog "canonical_tool_catalog_v2.json"
```

常用参数：

- `--no-resume`：不复用同 ID 检查点，另建运行目录。
- `--no-retry-failed`：恢复时保留旧失败，不重新调用。
- `--stop-on-error`：失败后停止后续 Tool；默认关闭。
- `--allow-root PATH`：增加允许读取的本地目录；可重复传入。
- `--provider-revision TEXT`：声明非密钥 Provider/数据配置版本；变化时拒绝复用旧检查点。

真实 API 仍由 `.env` 中的独立闸门控制，CLI 不提供把密钥放进参数的选项。

## 当前限制

1. Harness 当前按步骤顺序执行，尚未并行调度无依赖分支；这是为了先确保审计和恢复正确。
2. `FilesystemResultStore` 每个结果一个 JSON，适合当前规模；大批量运行后应迁移到 SQLite/DuckDB 索引，同时保留不可变 payload。
3. Provider 对象的内部配置无法安全通用序列化；应用需用 `harness_provider_revision` 显式声明配置版本。
4. 当前不自动比较“蒸馏 Observation”和“真实 Observation”。两者需要证据时间、数据版本和语义容差后才能进入验证 Agent。
5. Harness 是运行/复核基础设施，不负责自动修复 Thought、选择 Tool 或创建新 Tool。
6. 当前对输入文件和 artifact 计算完整 SHA-256；超大视频批量运行前应增加受校验的内容哈希缓存，避免每次恢复重复扫描大文件。
