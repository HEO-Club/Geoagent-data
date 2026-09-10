# Tool Runtime Harness v1 最终设计与验收报告

日期：2026-09-10  
范围：Stage 3 后可选的真实 Tool 重放与复核层  
结论：可以作为 v1 运行骨架冻结，并进入本地地理后处理 Tool 阶段；不能把未实现 Tool 或历史数据差异包装成成功。

## 1. Harness 的定位

Stage 2/3 负责从讲解材料恢复“当时如何思考、调用了什么工具、材料报告了什么结果”。Harness 不重新生成 Thought，也不覆盖蒸馏 Observation；它读取标准 `Trajectory`，在当前时间和当前 Provider 条件下执行 canonical Action，并把真实运行结果写入独立报告。

```text
Stage 3 Trajectory
    → 外层 Action 合同
    → 受限运行时引用
    → Canonical operation/input schema
    → RuntimeContext
    → Tool dispatcher
    → Observation/result_id/artifact
    → 检查点、事件日志与恢复
```

这一区分保证了两类证据不混淆：

- 原视频/字幕中的 Observation 是历史蒸馏证据；
- Harness Observation 是当前重放证据；
- 当前 OSM、网页、街景或卫星数据与历史不同，不能自动判定原讲解虚假。

## 2. 核心设计

### 2.1 明确状态机

每个 Trajectory event 都有稳定 `call_id` 和显式状态：

```text
reasoning → skipped_reasoning
tool_call → running → succeeded | failed
                    → blocked | invalid
final     → terminal
```

Tool 调用前先原子写入 `running`，再调用执行器。进程中断后，恢复逻辑能识别未完成步骤并保守重跑。第三方异常、参数编译异常和非法回执都只影响当前步骤，不击穿整条轨迹。

### 2.2 运行时引用而非字符串猜测

解析器支持 `$current_image`、`$current_images`、`$previous_tool_result`、`$active_area`、`$active_session` 和 `$step_N_tool_result.path[0]`。引用必须占据完整字符串，不进行字符串插值，也不使用 `eval` 或通用 JSONPath。

解析记录包含输入路径、来源、依赖步骤、值类型和脱敏预览。当前/未来步骤引用、缺失结果、字段不存在、下标越界和凭证字段引用均返回结构化错误。

### 2.3 依赖传播

Harness 记录显式 `$step_N...` 依赖，也记录由 previous result、active area、active session 和参数上下文注入形成的隐式依赖。

恢复时，如果上游结果文件、图片或 artifact 丢失/变化，上游步骤会重跑，所有下游旧结果也按依赖闭包失效。不会出现“新上游配旧下游”的混合状态。

### 2.4 持久化与完整性

默认目录：

```text
data/tool_runs/{safe_trajectory_id}_{id_hash}/
├── harness_report.json
├── events.jsonl
├── results/
├── images/
├── artifacts/
└── .harness.lock
```

`FilesystemResultStore` 每个 result_id 对应一个原子写入 JSON，保存 payload SHA-256。派生图片 ID 可在重启后恢复。报告和结果写入均执行 flush/fsync，并通过单 writer 锁避免同一轨迹并发覆盖。

### 2.5 检查点复用条件

恢复前同时验证：

- Trajectory 内容；
- Canonical Tool 目录；
- `tool/**/*.py` 执行器代码；
- 参数归一化器和 Pydantic 合同代码；
- 初始 RuntimeContext；
- Provider 类型和显式 provider revision；
- 允许读取的文件根；
- 本地输入文件内容；
- stop-on-error 执行策略；
- Observation 与 result payload 指纹；
- artifact 路径、大小和内容哈希。

关键身份发生变化时拒绝旧检查点；artifact 损坏时重跑相应依赖分支。CLI 可用 `--provider-revision` 显式声明非密钥 Provider/数据配置版本。

### 2.6 安全边界

密钥不允许进入 inputs、Thought、purpose、运行时引用、Observation、artifact、session 或 ResultStore。疑似 `api_key`、access token、client secret、password、Authorization、OpenAI/Google 风格密钥和 Bearer token 会被阻止或替换为 `***REDACTED***`。

输入图片先登记为受控 image_id。运行目录、配置 allow-root、RuntimeContext allow-root 和 Trajectory 精确输入文件采用加法合并；不会因为一张图片位于某个目录，就默认放开整个父目录。

## 3. 本轮最终审计发现并修复的问题

### 3.1 执行器指纹漏掉参数编译代码

旧实现只哈希 Tool 包和目录 JSON。Stage 3 参数归一化/Pydantic 合同变化后可能错误复用旧检查点。现在把 `params.py`、Tool Schema 和 Trajectory Schema 纳入执行器指纹。

### 3.2 CLI 无法声明 Provider 配置变化

文档要求使用 `harness_provider_revision`，但 CLI 没有入口。现在新增 `--provider-revision`，同一 Provider 类型更换后端、数据快照或非密钥配置时可主动使旧检查点失效。

### 3.3 参数归一化器异常可能击穿批次

Canonical 参数编译器若因目录或未知值抛异常，旧实现会终止整条 Harness。现在收为 `parameter_validation_exception`，当前步骤 invalid，依赖步骤 blocked，独立分支和 final 继续。

### 3.4 Provider 非 JSON/非有限回执可能留下 running

执行器返回 object、NaN 或 Infinity 时，旧实现可能在指纹/落盘阶段崩溃或产生内存与文件不一致。现在回执必须先 canonicalize 为有限 JSON；不合格回执统一为 `invalid_observation`。

### 3.5 Context 与 CLI allow-root 没有合并

调用方显式提供 `RuntimeContext.allowed_file_roots` 后，CLI/config 的额外根目录会被忽略。现在采用并集合并，同时加入运行目录与精确输入图片路径。

### 3.6 同路径输入内容变化未被检测

旧指纹只包含路径字符串。同一路径替换图片后可能复用旧结果。现在对 Trajectory、Context 和 Tool inputs 中实际存在的本地文件保存路径、大小和完整 SHA-256；内容变化会拒绝旧 run。

### 3.7 artifact 只检查存在性

旧逻辑无法发现“文件仍在但内容被覆盖”。现在成功步骤保存 artifact 内容指纹；缺失或篡改会使生产步骤及其下游依赖重跑。

## 4. 实际操控体验

### 4.1 混合 Tool 验收链

构造并真实执行了 11 步轨迹：

```text
reasoning
→ image_edit.crop
→ image_measure.measure
→ image_compare.compare
→ media_metadata_read.file
→ geocode.geocode（Fake Provider，真实执行器）
→ osm_query.query（Fake Provider，真实执行器）
→ osm_query.count（持久化 result_id）
→ map_layer_query.load_layer（尚未实现）
→ 依赖失败分支的 osm_query.count
→ final_answer
```

结果：

```text
total               11
skipped_reasoning    1
succeeded            7
failed               1
blocked              1
terminal             1
```

验证点：

- 裁剪图被登记为派生 image_id；
- 测量步骤正确引用裁剪图并得到 32 px；
- 比较步骤同时解析 current image 和步骤输出；
- 元数据读取返回 64×48；
- geocode bbox 被 OSM 查询引用；
- OSM query payload 通过持久化 result_id 进入 count；
- 未实现 map layer 明确返回 `not_implemented`；
- 引用失败分支的步骤返回 `unresolved_reference`；
- final 仍保留原轨迹地点；
- Stage 3 蒸馏 Observation 未被复用。

结论：Harness 能控制成功分支、失败分支、依赖分支、本地 artifact 和结构化地理结果，不会把“执行失败”扩大成“整个任务崩溃”。

### 4.2 真实 Stage 3 轨迹

使用 `BV13m61BJEQC__t01/stage3_trajectory.json`，强制 `ALLOW_REAL_TOOL_API=false` 和 `ALLOW_REAL_API=false`：

```text
total               15
skipped_reasoning    7
failed               7
terminal             1
final                郑州黄河文化公园依山亭
```

7 个 Tool 均诚实返回 `not_implemented`：map layer 2 次、satellite imagery 3 次、media search 2 次。CLI 以退出码 1 表示 `completed_with_errors`，没有把 final 丢失，也没有覆盖源 Trajectory。使用 `--no-retry-failed` 恢复后 15/15 条记录均从检查点复用。

### 4.3 故障与恢复

测试覆盖：

- Provider 普通异常；
- 模拟进程级 KeyboardInterrupt；
- 参数编译异常；
- 非 JSON object、NaN、Infinity；
- 前向/未知/缺失引用；
- result payload 篡改；
- artifact 删除和内容覆盖；
- 输入文件同路径替换；
- Trajectory/catalog/runtime/policy 指纹变化；
- 两进程竞争同一 run；
- Provider 输入、输出和持久化密钥泄漏。

进程中断会留下可审计的 running 检查点；使用同一稳定 call_id 恢复后重跑该步骤，并保留重复远程请求可能计费的 warning。

## 5. 测试结论

```text
Harness 定向测试     35 passed
Ruff                  passed
Mypy                  passed
整仓全量测试          326 passed
既有 warning          4（Pillow getdata 弃用，与本轮无关）
```

目录/执行器对齐测试同时确认：31 类 Canonical Tool、57 个 operation 均存在可导入分发入口；当前真正实现 9 类 Tool、16 个 operation，其余返回统一 `not_implemented`。

## 6. 能力判断

Harness v1 已经可以可靠控制当前 Tool 系统：它能解析 Stage 3 Action、调度本地和 Provider 型执行器、传递结果、管理图片、隔离错误、恢复中断并生成独立审计证据。它已经不是函数分发器，而是可恢复、可追踪、默认安全的 Agent 行动内核。

它尚不能解决：

- 尚未实现的 41 个 operation；
- Provider 的 429、配额、真实网络抖动和数据许可；
- 有副作用 Tool 的 exactly-once；
- 跨任务批量预算、并发和统一缓存；
- 数千任务下每文件 ResultStore 的索引性能；
- 当前重放结果与历史 Observation 的时间语义比较。

因此正确转向是继续实现低风险、高复用的地理后处理 Tool，而不是继续扩大 Harness 核心复杂度或立即并行化所有步骤。

## 7. 下一阶段路线

### 阶段 A：三类本地地理后处理 Tool

1. `osm_result_process.filter/export`
   - 输入只接受真实 result_id 或结构化 OSM payload；
   - 支持 element type、tag、name、geometry presence、bbox 等筛选；
   - 导出 JSON、GeoJSON、CSV；
   - 保留 source_result、CRS、过滤条件、输入/输出数量和丢弃原因。

2. `distance_bearing_calculator.distance/bearing`
   - 使用 `pyproj.Geod` 计算 WGS84 测地距离和初始/反向方位；
   - 支持坐标对、候选列表和 result 引用；
   - 明确米/千米/度，不允许从“远处”“附近”等语言生成精确值；
   - 坐标缺失时返回 `needs_acquisition`。

3. `spatial_filter.geometry_filter`
   - 使用 Shapely 对真实 geometry 执行 within/intersects/contains/crosses；
   - near/buffer 等米制运算先转换合适的本地投影，不直接在经纬度上按米计算；
   - 输出通过/拒绝要素及其空间关系依据。

这三类完成后形成第一条完整地理链：

```text
geocode
→ osm_query
→ osm_result_process
→ distance/bearing
→ spatial_filter
→ final
```

### 阶段 B：本地媒体获取

优先实现 `video_frame_extract.frame_retrieve/frame_sample`。它不需要外部 API，可以直接用 FFmpeg，把 Stage 1.5 时间戳、视频文件和 Tool Runtime image_id 串起来，并验证大文件输入哈希缓存需求。

### 阶段 C：高频 Provider Tool

在凭证配置完成后依次实现：

1. `map_layer_query`、`route_query`；
2. `web_search`、`web_page_read`；
3. `media_search`；
4. `streetview_query`；
5. `satellite_imagery_query`。

每个 Provider adapter 必须增加超时、限流、指数退避、配额/成本记录、缓存键、署名/许可、provider revision 和稳定 idempotency key。中国地图还必须保留 GCJ-02 原始坐标及 WGS84 转换方法。

### 阶段 D：遥感和高级计算

数据获取稳定后再进入 satellite compare、terrain、visibility、weather archive、solar ephemeris 和 shadow analysis。DEM/DSM、空间分辨率、时间戳和误差传播必须成为输出合同，不能把低分辨率数据解释成建筑级事实。

### 阶段 E：批量 Harness

在本地后处理链稳定后增加：

- 多 Trajectory 队列和无依赖分支并发；
- Provider 级速率/费用预算与熔断；
- SQLite/DuckDB result 索引；
- 大文件内容哈希缓存；
- 当前 replay 与历史 Observation 的人工审核差异报告；
- Stage 4 可选读取 HarnessReport，但不让重放失败自动删除训练样本。

## 8. 最终建议

冻结当前 Harness v1 的核心合同。下一次开发直接从 `osm_result_process.filter/export` 开始，同时复用当前 ResultStore、引用解析、失败分类和恢复能力；完成三类本地地理 Tool 后再做第一条完整真实地理链验收。这样可以用最少外部依赖验证 Harness 的长期接口是否真正适合后续 31 类 Tool。
