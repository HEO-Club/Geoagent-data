# Tool 参数合同

样本置信度由阶段4负责（`pipeline/stage4_confidence/`）：VLM 多维打分 + 程序化硬门槛，只回写 `quality_score` 与 `review_priority`，不拦入库。Stage 3 负责 operation 级输入合同归一，并写入 `stage3_parameter_audit.json`；**阶段4读取该审计**，将 `ready / context_resolvable / repairable / invalid` 映射为程序化维度 `tool_param_correctness`（按最差一次调用取分），并写入报告 `parameter_readiness` 与必填 `notes`。

## 1. operation 级输入参数合同

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

## 2. 参数验证结果

对严格 Observation 门禁后的 9 条轨迹共 79 次 Tool 调用进行了离线参数审计。初始宽泛 Schema 只有 13/79 次可直接通过；别名和 operation 语义校准后为 45/79（57.0%）；加入上下文引用和补参指导后为 45 `ready`、8 `context_resolvable`、25 `repairable`、1 `invalid`，可直接继续执行的比例为 53/79（67.1%）。唯一 invalid 是部署 Agent 无法真实执行的 `field_site_visit`。repairable 主要是候选列表、真实时间、图层、查询词等必须从 Thought/字幕或前置工具取得的内容，不能通过放松规则掩盖。

模型理解探针使用 10 个完全合成场景，不包含项目视频、字幕或用户数据。当前模型 10/10 正确完成原有七类操作，并能在局部反向搜图前先 crop 当前图片、在 OSM export 前先执行 query、完全没有图片时返回 `needs_input` 而不伪造文件。结果见 `data/runs/tool_contract_agent_probe.json`。

## 3. 阶段4如何记分

- 读取 `stage3_parameter_audit.json` 的 `calls[].readiness`
- 按最差一次调用：全 ready / 无 Tool → 1.00；`context_resolvable` → 0.80；`repairable` → 0.45；`invalid` → 0.15；审计缺失 → 0.50（失败开放）
- 任一次 `invalid` → 硬门槛 `tool_params_invalid`（只压 `quality_score` 到 cap）
- 每条 `stage4_confidence.json` 的 `notes` 必填，含四级计数与非 ready 调用明细；弱维度（默认 < 0.80）须写可核对 reason

## 4. 使用方式

```powershell
# 只读审计既有 Stage 3 参数
.\.venv\Scripts\python.exe scripts\audit_tool_parameters.py <结果目录> --out data\runs\parameter_audit.json

# 合成场景测试 Agent 是否理解参数合同（需要显式允许真实 API）
.\.venv\Scripts\python.exe scripts\probe_tool_contract_agent.py
```

新运行的 Stage 3 会自动生成 `stage3_tool_mapping.json` 和 `stage3_parameter_audit.json`。Stage 4 读取后者记入 `tool_param_correctness`，写入 `stage4_confidence.json` 并回写 JSONL 的 `quality_score`。
