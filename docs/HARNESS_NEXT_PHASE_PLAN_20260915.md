# GeoAgent Harness 下一阶段实施说明

日期：2026-09-15  
基线：`develop@a577c1f881bc4f64f41904792b8bc39f6c72ae70`

## 1. “下一步”到底是什么意思

是继续深入完善 Harness，但不是继续堆一批 Tool 壳子，也不是重写已经完成的 31 类 Tool。

当前 Harness 更准确地说是一个**可靠执行器**：给它一条已经写好的 Tool trajectory，它可以做参数校验、真实调用、结果存储、错误记录、有限重试、checkpoint、恢复和报告。它已经能把“计划好的调用”可靠执行。

它暂时还不是一个完整的**任务规划与控制系统**：给它一句自然语言任务时，它不会自动决定该调用哪些 Tool、依赖顺序是什么、如何把上一步结果字段绑定到下一步参数、证据不足时如何重新规划。

所以后续目标是：

```text
现状：自然语言任务 → 人/已有轨迹先写调用序列 → Harness 可靠执行

目标：自然语言任务
      → Planner 生成可验证计划
      → Preflight 检查能力/凭证/输入
      → Harness 按依赖执行 Tool DAG
      → Controller 根据真实 Observation 重规划
      → Evidence Gate 判断是否足以回答
      → final_answer
```

## 2. 目前已经具备什么

- 31 类 Tool、57 个 operation 有真实 wrapper；
- Tool schema、执行输入和 Observation 合同；
- 共享 ResultStore，支持跨 Tool 使用 result ID；
- Harness checkpoint、resume、报告和调用审计；
- 有界重试、指数退避和尝试记录；
- 会话型 Tool 的 checkpoint 重建；
- 外网闸门和 SSRF 防护；
- OSM → filter/export → distance/bearing → spatial_filter 的真实本地链；
- 760 项测试通过；
- 未配置外部服务时诚实返回错误，不伪造结果。

## 3. 为什么 Tool 全部实现后仍不能“一句话自动跑完”

### 3.1 缺少计划器

例如用户说“比较上海某区域 2020 和 2025 年的卫星图”。卫星执行器要求 bbox 或中心点加半径；自然语言只给了地名。现在 Harness 会把原参数直接交给 Tool，得到 `missing_input`。完整系统应显式规划：

```text
geocode("上海某区域")
  → 选择真实候选坐标
  → 构造 bbox
  → satellite.retrieve(2020)
  → satellite.change_time(2025)
  → satellite.compare_time
```

这不是卫星 Tool 的 bug，而是缺少“前置依赖规划”。

### 3.2 缺少能力与凭证预检

系统目前要实际调用后才知道 Brave Key、Mapillary token 或 Copernicus OAuth 是否可用。完整系统应在运行前生成 capability snapshot：

- 哪些 Tool 是纯本地；
- 哪些 provider 已配置；
- 哪些 operation 需要联网；
- 是否有凭证、额度和允许的调用范围；
- 哪些调用有成本、隐私或许可风险；
- 当前是否允许真实外网。

### 3.3 缺少结果绑定编译器

上一步返回的是结构化 Observation，下一个 Tool 要的是特定字段。不能靠字符串 `$previous_tool_result` 解决所有问题。需要显式绑定，例如：

```text
node_2.inputs.area.center.lon <- node_1.result.candidates[0].location.lon
node_2.inputs.area.center.lat <- node_1.result.candidates[0].location.lat
```

绑定在执行前必须做类型检查；候选有多个时不能擅自选择，应由规则或模型基于证据选择，并记录理由。

### 3.4 缺少证据驱动的控制循环

真实 API 会空结果、超时、限流、时间覆盖不足。下一阶段需要 Controller：

- 成功且证据足够：继续；
- 成功但证据不足：扩大/缩小范围或换 provider；
- 可重试错误：按策略重试；
- 不可重试错误：改计划；
- 缺少凭证：在开始前阻止对应节点；
- 预算不足：选择较便宜路径或终止；
- 所有真实证据仍不足：明确回答无法确定，不能编造地点。

## 4. 推荐实施顺序

### P0-A：Tool Capability Manifest 与 Preflight

为 57 个 operation 建立机器可读能力清单，至少包含：

- provider / executor；
- local 或 network；
- 所需环境变量；
- 输入/输出 schema；
- 是否只读；
- 是否幂等、可否安全重试；
- 是否有会话状态；
- 是否产生成本；
- 是否上传用户数据；
- 许可与署名要求；
- 可用 provider、替代 provider；
- 常见错误与修复提示。

Preflight 在运行前输出：`ready / degraded / blocked`，并说明原因。

**验收标准**

- 57 个 operation 全覆盖；
- 缺一个 Key 时能在执行前定位到具体 Tool；
- 纯本地计划不因外部 Key 缺失而被阻断；
- `ALLOW_REAL_TOOL_API=false` 时所有网络节点被标为 blocked；
- 不读取或输出明文 secret。

### P0-B：Planner 与显式 Tool DAG

输入自然语言任务与 capability snapshot，输出显式 DAG：

- 节点：具体 `tool.operation`；
- 边：前置依赖与结果绑定；
- 每个节点的 purpose；
- 每项输入来自用户、上下文还是前置 Observation；
- 失败替代路径；
- 预算与最多调用次数；
- 完成判据。

Planner 不允许隐藏调用。例如卫星节点缺 bbox 时必须在计划里插入 geocode，而不能由卫星 Tool 偷偷联网地理编码。

**验收标准**

- 地名→卫星任务能自动插入 geocode；
- OSM 结果→filter→distance/spatial 能正确绑定 result ID；
- 不存在的文件、坐标、session ID 不会被编造；
- DAG 无循环，所有必填输入都有来源；
- 计划可序列化、重放和审计。

### P0-C：Observation Binding / Resolver

实现安全的结构化字段选择和转换：

- JSONPath/声明式路径；
- 类型校验；
- CRS 与单位转换；
- 候选列表选择；
- bbox/center+radius 构造；
- result ID 与 session ID 区分；
- 缺值时返回可规划错误，而不是传 `None` 继续。

**验收标准**

- 绑定错误在调用前发现；
- 每个实际输入能追溯到来源 Observation；
- 坐标转换只发生一次；
- checkpoint 恢复后绑定结果一致。

### P1-A：统一 Provider 错误分类与重试策略

当前部分 adapter 会把认证失败、网络失败和 provider 5xx 都归入较宽泛的 `engine_unavailable`。应统一为：

- `authentication_failed`
- `permission_denied`
- `quota_exhausted`
- `rate_limited`
- `timeout`
- `network_error`
- `provider_error`
- `invalid_provider_response`
- `no_results`
- `missing_input`
- `unsupported_operation`

Harness 再根据错误和幂等性决定是否重试、换 provider 或重规划。

### P1-B：Controller / Replanner

Controller 每个节点后检查：

- Observation 是否真实且 schema 合格；
- 是否满足当前证据缺口；
- 是否要执行备用节点；
- 是否需要修改范围、时间或 top_k；
- 是否达到工具调用、时间和费用预算；
- 是否可以进入 final_answer。

需要严格限制重规划轮数，防止搜索循环和账单失控。

### P1-C：成本、配额和许可策略

在 capability manifest 和 run report 中加入：

- 预计/实际请求数；
- 预计/实际模型 token；
- provider 费率版本；
- 单 Tool 和单 run 预算；
- 数据上传标记；
- 缓存/训练许可标签；
- 429、额度耗尽和熔断状态。

### P2：黄金端到端评测集

建立至少 20 条可重复任务，覆盖：

1. 纯本地图像/OCR；
2. 网页搜索→页面读取；
3. 地名→geocode→POI/OSM；
4. OSM→filter/export→距离/方位/空间过滤；
5. 街景 open→navigate→change_time→capture；
6. 地名→geocode→卫星时相→比较；
7. bbox→DEM→terrain→visibility；
8. 历史天气→时间细化；
9. 行政/建设/航班外部档案；
10. 多路径失败与诚实退出。

每条任务记录：预期计划结构、必须/禁止调用的 Tool、最大调用数、关键 Observation 字段、最终答案判定和费用上限。

## 5. 最小可用版本的定义

完成 P0-A、P0-B、P0-C 后，可以称为“会合理组织 Tool 的 Harness MVP”：

- 用户给自然语言任务；
- 系统先检查环境与能力；
- 自动生成显式、可审计的 Tool DAG；
- 自动补齐必要的前置 Tool；
- 把真实 Observation 绑定到下游；
- Harness 按计划可靠执行与恢复；
- 证据不足时不假装成功。

完成 P1-A、P1-B、P1-C 后，才是“面对真实 API 波动能够稳定控制流程”的版本。

完成 P2 并用真实凭证跑通后，才有证据说整个系统在代表性任务上可以合理运行。

## 6. 建议下一次直接开始的工作

第一项编码任务应是：

> 实现 `ToolCapabilityManifest + PreflightReport`，自动扫描/登记 57 个 operation 的本地/联网属性、所需凭证、幂等性、会话、成本和错误策略；接入 Harness CLI，并补齐全量测试。

原因是 Planner 必须先知道当前到底有哪些能力可用。没有 Preflight，Planner 很容易生成一个需要 Brave、Mapillary 或 Copernicus 的计划，但运行环境根本没有对应凭证，直到中途才失败。

完成它后，第二项才是 `Planner + Tool DAG`；第三项是 Observation binding；第四项是 Controller/Replanner；最后用真实凭证跑黄金端到端评测。
