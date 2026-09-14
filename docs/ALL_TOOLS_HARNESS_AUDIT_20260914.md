# 全量 Tool 与 Runtime Harness 审计报告

日期：2026-09-14
审计对象：远程 `develop` 提交 `bb259bd` 与本地 Harness/地理后处理提交 `3f7d7ba` 的合并结果

## 1. 结论

学生提交确实把目录中的 31 个 canonical tool、57 个 operation 全部接入了非占位执行器。它不是“空壳”，每类工具都有输入校验、结构化 Observation 和对应测试；在安装新增地理依赖后，远程原始版本达到 741 passed、1 failed。唯一失败是旧 Harness 测试仍期待 `map_layer_query` 返回 `not_implemented`，实际新执行器正确返回 `engine_unavailable`。

但“每个工具单测通过”不等于“工具链能跑通”。合并后实地执行 `geocode → osm_query → osm_result_process → distance/bearing → spatial_filter → export` 时，先后发现共享结果仓库不兼容、旧 Overpass 几何格式不兼容、导出不产生文件、Harness 丢失 extensions 和会话恢复状态等链路级问题。上述问题已修复，最终全量测试为：

```text
760 passed, 6 warnings in 19.24s
```

当前结论是：

- 31 tool / 57 operation 的本地、注入 Provider 和错误路径均可分发、可测试。
- 本地地理链已真实跑通，并验证了重启后的检查点复用。
- 没有密钥时，外部 Tool 会明确失败，不会伪造 Observation，也不会偷偷访问网络。
- 还不能把任意 Stage 3 轨迹自动改写成“补 geocode、再查卫星”的新计划；当前 Harness 是安全、可恢复的执行器，不是 Planner。这是下一阶段最重要的能力缺口。

## 2. 测试过程与证据

### 2.1 远程提交基线

| 阶段 | 结果 | 说明 |
|---|---:|---|
| 未安装新依赖 | 723 passed / 17 failed / 2 skipped | 15 个 solar/shadow、1 个网页正文读取依赖失败，另 1 个旧 Harness 断言失效 |
| 安装新增依赖后 | 741 passed / 1 failed | 仅剩旧 `not_implemented` 断言，与新实现状态不符 |
| `pip install -r requirements.txt` | 失败 | 未使用的 `editdistance` 在 Windows + CPython 3.13 无 wheel，源码编译失败 |

### 2.2 合并与修复后的验证

| 验证 | 结果 |
|---|---:|
| 闸门、SSRF、网页、视频、三类地理 Tool、Harness、目录对齐 | 125 passed |
| Harness 重试、扩展字段、会话恢复、共享结果仓库 | 73 passed |
| 本地真实地理链首次执行 + 检查点恢复 | 1 passed，恢复时 Provider 调用数为 0 |
| Python 编译 | `compileall` 通过 |
| 依赖一致性 | `pip check`：No broken requirements found |
| 最终全量 pytest | 760 passed / 6 warnings |

6 个 warning 均为非本次引入：4 个 Pillow `getdata` 未来弃用提醒，2 个 rasterio 仿射乘法 PendingDeprecationWarning。

### 2.3 真实 Stage 3 轨迹体验

使用已有 `BV13m61BJEQC__t01/stage3_trajectory.json`，显式关闭两个真实 API 闸门并设置 `--max-attempts 2`：

```text
total=16, skipped_reasoning=8, failed=7, terminal=1
```

结果是正确且诚实的：5 个外部查询因 `ALLOW_REAL_TOOL_API=false` 返回 `engine_unavailable`；2 个卫星查询把自然语言区域直接交给需要 bbox 的执行器，返回 `missing_input`；最终答案仍按轨迹合同保留。没有出现伪造地图、影像或搜索结果。

这次体验同时证明了下一阶段 Planner 的必要性：`area="郑州黄河文化公园"` 在语义 schema 中是合法 location，但 Copernicus 执行需要 WGS84 bbox。正确流程应由 Planner 显式插入 `geocode`，而不是由卫星 Tool 暗中发起另一个 Tool 调用。

实测报告位置：`data/tool_runs/postmerge_real_20260914/BV13m61BJEQC__t01_251caa8d/harness_report.json`。

## 3. 已发现并修复的问题

| 严重度 | 问题 | 影响 | 修复 |
|---|---|---|---|
| 高 | 新工具使用 `ALLOW_REAL_API`，旧工具使用 `ALLOW_REAL_TOOL_API` | 打开 Stage 2/3 LLM 后可能意外放开图片上传、搜索和位置 API | 所有 Tool 统一只认 `ALLOW_REAL_TOOL_API`；`ALLOW_REAL_API` 只管数据流水线模型调用 |
| 高 | `web_page_read` 与远程视频下载接受任意 URL | 可访问 localhost、内网或云元数据地址；重定向可绕过首个 URL 校验 | 新增统一公网 URL 校验，拒绝私网/保留地址、URL 凭证和混合 DNS，并逐次校验重定向 |
| 高 | 学生版 OSM 后处理只查自己的内存 registry | 无法读取 `osm_query` 写入的共享 `ResultStore`，真实链第三步即失败 | OSM 与空间筛选统一接入共享 ResultStore，并保留进程内 registry 兼容 |
| 高 | 旧 Overpass 返回坐标数组和 `latitude/longitude`，新后处理只接受 GeoJSON | 距离、方位、空间筛选连续失败 | 统一转成 Point/LineString GeoJSON，再交给测地与 Shapely 算法 |
| 中 | 会话型 Tool 只把 dataclass 放在内存 extras | Harness 重启后即使首步检查点复用，weather/streetview/satellite 后续操作也找不到会话 | 新增 Tool 级会话恢复钩子，从持久化 Observation 重建运行时状态并推进计数器 |
| 中 | Harness 忽略 `Observation.extensions` | 未声明输入元数据在报告中丢失，凭证扫描也有盲区 | 模型、快照、指纹和凭证扫描全部纳入 extensions |
| 中 | Harness 只有“恢复时重跑失败步骤”，没有单步物理尝试审计 | 瞬时 Provider 故障无法有界重试，也看不到每次耗时/错误 | 新增 `max_attempts`、可重试错误码、指数退避、确定性抖动和 `ToolAttemptRecord`；默认仍为 1，避免意外重复计费 |
| 中 | `osm_result_process.export` 只返回大字符串，不写导出文件 | 下游拿不到实际 GeoJSON/CSV/KML/OSM artifact | 原子写入 artifact，Observation 返回 `export_path` 与格式 |
| 中 | CSV 标签可用 `=`, `+`, `-`, `@` 开头 | 用 Excel 打开可能触发公式注入 | 表头与单元格统一转义，并输出 UTF-8 BOM |
| 中 | 合成 OSM XML 丢失 relation members/node refs 却无提示 | 用户可能误当成可无损回灌的原始 OSM | 保留格式支持，但在结果中明确标注为从规范化几何重建的交换文件 |
| 中 | `editdistance` 未被代码使用且阻断 Python 3.13 安装 | 新环境无法一键安装 | 从 requirements 移除；项目已有其他相似度实现，不影响运行 |
| 低 | 根目录 pytest 会误收集 `tmp` 中的隔离 worktree | 出现 `ImportPathMismatchError`，本地测试不稳定 | 新增 `pytest.ini`，收集范围固定为正式 `tests/` |
| 低 | Tool 包存在大量安全可自动修复的 lint | 可读性和后续维护变差 | Ruff 安全修复 63 项；余项主要是冗余类型转换与代码风格建议，不为追求零告警而改写已测算法 |

## 4. 算法审计

### 4.1 结论良好的实现

- `distance_bearing_calculator` 使用 `pyproj.Geod` 的 WGS84 椭球反解，而非经纬度平面距离；多点按相邻段累加；Polygon 宽度使用局部 AEQD 投影后的最小外接矩形短边。
- `spatial_filter` 使用 Shapely 处理 within/intersects/contains/crosses；near 在局部 AEQD 中按米计算；EPSG/GCJ-02 输入先统一到 WGS84。
- `terrain_analysis` 的坡度/坡向采用本地 Horn DEM 算法，地形剖面和高程统计不依赖外部 LLM。
- `visibility_analysis` 沿观察者—目标线采样 DEM，可选地球曲率修正；明确不是完整 viewshed。
- `solar_ephemeris` 使用 pvlib NREL SPA，并显式处理时区、UTC 与极昼极夜。
- `shadow_analysis` 使用太阳方位/高度与局部平面的射线求交；无高度时只给方向，不编造长度。
- 影像编辑、OCR、EXIF、帧提取、卫星比较等均把真实 artifact 与结构化结果分开返回，失败不会生成“看似成功”的结果。

### 4.2 已保留的算法边界

这些不是当前实现错误，因此没有强行修改：

- AEQD 最近距离和宽度适合局部范围；跨洲、跨日期变更线或极区大几何需要分块/椭球最近点算法。
- Horn 坡度高度依赖 DEM 分辨率、投影和 nodata 质量；不能把 30 m DEM 的结果描述成建筑级精度。
- sightline 是一条剖面，不等于二维/三维 viewshed；DSM 与 DEM 的语义必须区分。
- shadow 是局部平面模型，不包含建筑遮挡和复杂地形 ray tracing。
- 卫星 `oblique_view` 只是已有正射预览加观察参数，不是真斜摄影；实现已明确标注。
- OSM 公共实例和 Nominatim/Overpass 不适合批量生产；生产应自建、使用授权提取或商业服务。

## 5. Harness 当前设计

```text
Stage 3 Trajectory
  → 轨迹/目录/运行时指纹
  → 精确引用解析与参数审计
  → 独立 Tool 外网闸门与文件根限制
  → 逻辑 Tool step
      → 1..N 次受控 physical attempts
      → Observation/extension 凭证扫描
      → ResultStore / ImageStore / artifacts
  → 每步原子检查点 + append-only events.jsonl
  → 会话状态重建与断点恢复
  → HarnessReport（不改写源轨迹）
```

当前实现覆盖：真实分发、失败隔离、依赖步骤、精确 `$step` 引用、结果/图片持久化、artifact 指纹、恢复重跑、并发运行锁、凭证脱敏、Provider/代码版本失效、单步重试审计和会话恢复。

## 6. 最新设计资料对照

- OpenAI 官方模型指南建议把“何时使用、必需输入、副作用、重试安全、常见错误”放进工具描述；工具量大时按需加载相关工具，并为长流程保留已完成动作、ID、工具结果和未解决阻塞。当前目录 schema 已覆盖输入与 acquisition hint，但还缺运行时副作用/成本元数据：[OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.5)。
- MCP 2025-11-25 Tool 规范支持 `inputSchema`、`outputSchema`、结构化结果和 `readOnlyHint`/`destructiveHint`/`idempotentHint`/`openWorldHint`。这些 annotation 应作为未来 MCP 暴露层的提示，不能替代本地信任策略：[MCP Tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)。
- Temporal 把不确定 API 调用视为可失败 Activity，只重试瞬时/间歇错误，不重试永久输入错误；工作流状态要确定性，副作用需幂等。当前 Harness 的错误码白名单与有界重试按此原则实现：[Temporal Retry Policies](https://github.com/temporalio/documentation/blob/main/docs/encyclopedia/retry-policies.mdx)。
- OpenTelemetry 建议区分一次逻辑调用与每次物理 HTTP 尝试，并记录 resend count、低基数 error type、耗时和状态。当前 `ToolAttemptRecord` 已具备数据基础，下一步可接 OTel exporter：[OpenTelemetry HTTP spans](https://opentelemetry.io/docs/specs/semconv/http/http-spans/)。
- LangGraph 的 checkpointer/store 分层、节点边界恢复值得借鉴；本项目按 SPEC 不引入 LangChain/LangGraph，继续使用自有轻量实现：[LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)。

## 7. 下一阶段规划

### P0：Planner / Preflight（下一步应立即做）

1. 在执行前生成显式 Tool DAG：把自然语言 location 转为 `geocode` 前置节点，再把真实 bbox/result_id 传给卫星、地图、天气、地形等执行器。
2. 禁止隐藏 Tool 调用：每次自动补充都必须成为独立、可见、可审计的 Tool step。
3. 为 57 个 operation 增加运行时 capability manifest：Provider、所需环境变量、是否联网、是否只读、是否可重试、是否有会话、费用等级与许可限制。
4. 统一 Provider 错误分类：`engine_unavailable` 只表示未配置/未安装；网络、429、5xx、解析失败分别使用 `network_error`、`rate_limited`、`provider_error`、`invalid_response`，让重试策略真正可靠。

### P1：可观测性与预算

1. 把 run/step/attempt 映射到 OpenTelemetry spans，默认不记录敏感输入和完整网页正文。
2. 按 Provider 记录请求次数、估算费用、429、延迟和命中率；加入 run 级最大调用次数与预算上限。
3. 为付费 API 增加每 Provider 并发限制和熔断器，不让单个轨迹耗尽账户配额。

### P2：安全并行与 MCP 暴露

1. 只并行无依赖、无共享会话写入的 DAG 分支；同一 session 的 navigate/change_time/capture 保持串行。
2. 导出 MCP `outputSchema` 与结构化错误；annotation 由本地可信 manifest 生成。
3. 支持 tool-list 版本/缓存失效和按任务加载工具子集，避免一次把 57 个 operation 全塞给模型。

### P3：真实 Provider 验收

1. 申请 API 后建立 `tests/integration/`，默认跳过，只在独立测试账户和显式闸门下运行。
2. 为每个 Provider 保存脱敏回执 fixture，覆盖 200、空结果、429、401/403、5xx、超时和 schema 漂移。
3. 建立 20–50 条黄金地理任务，评价 Tool 选择、参数获取、链路完成率、证据充分性、成本与恢复成功率，而不只看单元测试数量。
