# 轨迹质量置信度与 Tool 参数合同

## 1. 设计目标

本机制评估的是“一条轨迹达到高质量 SFT 标准的可信程度”，不是生成模型自报的主观概率。报告同时给出 `quality_score` 和 `audit_coverage`：前者表示当前证据下的质量估计，后者表示多少关键维度真正经过了可复现审核。没有发现错误但缺少审核证据时使用中性分并降低 coverage，不能获得高置信 accept。

质量报告由六个维度组成：证据落地度 30%、最终答案支撑度 20%、Tool 与参数正确性 20%、逻辑一致性 15%、输入质量与对齐 10%、SFT 格式完整性 5%。权重体现当前项目最重要的风险排序：虚假 Observation 和没有证据支撑的最终答案会直接污染 Agent 行为，因此合计占 50%；Tool/参数决定轨迹能否复现，占 20%；逻辑、输入与格式负责保证整条样本可学习。

决策采用分数、覆盖率、参数准备度和硬错误共同判断：无硬错误、`quality_score >= 0.85` 且 `audit_coverage >= 0.70` 才是 `accept`；证据与格式可靠但语义/输入覆盖尚不足可标 `provisional_pass`；只有参数可补时标 `parameter_repair`；Tool 不存在或语义冲突进入 `needs_review`；明确虚假 Observation、最终 location 缺失、task 答案不一致、答案泄露图或轨迹—图片冲突直接 `reject`。

独立审核 Agent 位于 `pipeline/quality/reviewer.py`。它只输出结构化维度分数、问题、step 序号和证据引用，不得重写轨迹。模型自评可以作为附加信号，但不能替代严格 Observation 审计、参数 Schema、Stage 1.5 选图门禁或确定性格式检查。

## 2. 实际校准结果

校准使用了三组既有产物和一组合成对照：

| 数据组 | 数量 | 平均分 | 平均覆盖率 | 解释 |
|---|---:|---:|---:|---|
| 事件化 Stage 2/3 旧结果 | 10 | 0.650 | 0.448 | 缺严格 Observation、Stage 1.5 和语义审核；只能进入 review |
| 严格 Observation 门禁后（上下文参数化） | 9 | 0.815 | 0.666 | 1 provisional、7 parameter repair、1 review；不再把所有缺参都送人工 |
| 旧 Stage 1.5 单条结果 | 1 | 0.653 | 0.570 | 有选图和最终答案，但大量伪 Tool/旧参数结构拉低分数 |
| 完整合成高质量正例 | 1 | >=0.90 | >=0.90 | 具备 task、选图、Observation、参数和语义审核，正确 accept |

严格 Observation 门禁后的均分比旧结果提高约 0.13，方向与人工复核一致；旧 Stage 1.5 样例不会因为答案正确和有图片就获得高分，说明机制能识别“答案对但轨迹不可作为高质量 SFT”的情况。最终规则特意避免低 coverage 自动 reject，因为旧结果中的未知项不能被当作负证据。

离线报告位于 `data/runs/quality_stage23_events_final.json`、`data/runs/quality_observation_gate_final.json` 和 `data/runs/quality_old_stage15_final.json`。这些运行产物默认被 `.gitignore` 排除。

## 3. operation 级输入参数合同

外层调用仍保持：

```json
{
  "operation": "query",
  "purpose": "查询候选区域内的桥梁要素",
  "inputs": {
    "area": "郑州市",
    "tags": {"bridge": "yes"},
    "return_geometry": true
  }
}
```

`operation` 决定执行器当前提供哪种能力，`purpose` 记录本次调用要补齐的证据缺口，`inputs` 才是实际执行参数。每个 Canonical Tool 的每个 operation 都附加 `ToolInputSchema`，其中包含字段名、类型、是否必填、中文解释、历史别名、允许值、数值范围和示例；同时支持“至少提供一项”和互斥字段约束。

参数归一不会因为原始字段名称不同就直接丢弃样本。例如 `区域/region/search_area` 可归一为 `area`，`检索对象/关键词/q` 可归一为 `query`。无法识别的字段保存在 `inputs.extensions` 并产生 warning。字段同时声明 `semantic / execution / optional` requirement level 和缺参获取方式；当前图片与前置结果可显式引用，真正缺失时输出 repair action，指导 Agent 先截图/crop、先 query、提取真实时间或引用活动会话。

OSM 的默认输入是结构化 `area/tags/feature_types/spatial_relation`，执行器内部负责生成 Overpass QL。`overpass_ql` 是可选高级字段，只在原始轨迹确实编写代码时传递。`query/filter/export/count` 使用不同合同，从而避免把查询条件误当成导出参数。卫星影像明确区分 `compare_time` 和 `compare_candidates`：前者要求同一区域的多个时间点，后者要求多个真实候选地点和统一比对模板，不能只填写候选数量。

基础目录从 15 个执行器扩展为 17 个，新增 `flight_data_query` 和 `metadata_read`。`field_site_visit` 没有可供部署 Agent 调用的执行器，仍保持未登记并进入人工复核，不能为了提高通过率而伪装成合法 Tool。

## 4. 参数验证结果

对严格 Observation 门禁后的 9 条轨迹共 79 次 Tool 调用进行了离线参数审计。初始宽泛 Schema 只有 13/79 次可直接通过；别名和 operation 语义校准后为 45/79（57.0%）；加入上下文引用和补参指导后为 45 `ready`、8 `context_resolvable`、25 `repairable`、1 `invalid`，可直接继续执行的比例为 53/79（67.1%）。唯一 invalid 是部署 Agent 无法真实执行的 `field_site_visit`。repairable 主要是候选列表、真实时间、图层、查询词等必须从 Thought/字幕或前置工具取得的内容，不能通过放松规则掩盖。

模型理解探针使用 10 个完全合成场景，不包含项目视频、字幕或用户数据。当前模型 10/10 正确完成原有七类操作，并能在局部反向搜图前先 crop 当前图片、在 OSM export 前先执行 query、完全没有图片时返回 `needs_input` 而不伪造文件。结果见 `data/runs/tool_contract_agent_probe.json`。

## 5. 使用方式

```powershell
# 只读计算既有轨迹置信度
.\.venv\Scripts\python.exe scripts\score_trajectories.py <结果目录> --out data\runs\quality.json

# 只读审计既有 Stage 3 参数
.\.venv\Scripts\python.exe scripts\audit_tool_parameters.py <结果目录> --out data\runs\parameter_audit.json

# 合成场景测试 Agent 是否理解参数合同（需要显式允许真实 API）
.\.venv\Scripts\python.exe scripts\probe_tool_contract_agent.py
```

新运行的 Stage 3 会自动生成 `stage3_tool_mapping.json`、`stage3_parameter_audit.json` 和 `stage3_quality_report.json`，并把总分写入 JSONL 的 `quality_score`。完整语义审核 Agent 是可选增强项；如果没有相应外部传输授权，离线评分会把语义维度标记为未覆盖，而不是伪造审核完成。
