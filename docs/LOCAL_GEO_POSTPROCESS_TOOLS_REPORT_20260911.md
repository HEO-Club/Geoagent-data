# 第一优先级本地地理后处理 Tool 实现报告

日期：2026-09-11

## 一、完成范围

本轮实现 3 类 Tool、5 个 operation：

```text
osm_result_process.filter
osm_result_process.export
distance_bearing_calculator.distance
distance_bearing_calculator.bearing
spatial_filter.geometry_filter
```

当前累计落地 12 类 Tool、21 个 operation。三个新 Tool 均为纯本地确定性执行器，不调用外部 API；它们只消费已有真实 result_id、坐标或几何，不从自然语言编造精确空间数据。

## 二、osm_result_process

### filter

支持：

- `osm_type`、`osm_id`；
- 名称精确、包含和正则筛选；
- OSM tag 精确值、候选值数组和 `*` 存在性筛选；
- `has_geometry`；
- WGS84 bbox 相交；
- 可选 `spatial_filter`；
- 最终结果 `limit`。

执行顺序固定为：属性过滤 → 空间过滤 → 最终 limit。未知 filter 字段直接返回 `unsupported_filter`，不会静默忽略。结果保存输入/输出/剔除数量、条件、CRS、署名和新 `result_id`。

### export

支持：

- JSON；
- GeoJSON FeatureCollection；
- UTF-8 BOM CSV；
- KML。

文件写入 Harness artifact 目录，使用临时文件、flush/fsync 和原子替换，并返回字节数与 SHA-256。`include_geometry=false` 时只导出属性。

没有实现 `osm_xml`。规范化 OSM payload 可能已经失去 relation members 和 way node refs，强行重建会产生看似合法但不可回灌的伪 OSM XML。因此 Canonical Schema 已移除该允许值；直接调用仍返回 `lossy_format_unsupported`。

## 三、distance_bearing_calculator

### distance

- `geodesic`：使用 `pyproj.Geod.inv` 计算相邻点 WGS84 椭球距离；
- `geometry`：使用 `pyproj.Geod.geometry_length` 计算线/面边界测地长度；
- `width`：将 Polygon 投影到以对象中心为原点的局部 AEQD，再用 minimum rotated rectangle 的短边作为明确标注的近似宽度；
- 单位：m、km、mile；
- 输出每段距离、总距离、正反方位、方法、椭球、来源和假设；
- 支持结构化点、GeoJSON Point、OSM element 和 result_id。

`route` 不在本地伪算。没有真实路线结果时返回 `needs_acquisition`，要求先调用 `route_query`。

### bearing

使用 `pyproj.Geod.inv` 返回：

- 相对真北、顺时针的初始方位角；
- 反向方位角；
- 16 方位罗盘方向；
- 两点测地距离。

`magnetic_north` 需要日期、高度和磁偏角模型，`image_axis` 需要相机方向/图像轴标定；缺少这些证据时返回 `needs_acquisition`。重合点返回 `coincident_points`。

## 四、spatial_filter

支持：

```text
within
intersects
contains
crosses
near
```

输入可以是 OSM elements 或 GeoJSON features。参考 geometry 支持 GeoJSON、Feature/FeatureCollection、bbox、坐标对象和 result_id。

`near` 不在经纬度上直接按米做欧氏距离，而是每次调用创建一个以参照几何中心为原点的局部 AEQD 投影，将 reference 和全部候选统一转换后计算最短距离。每次调用只创建一次 Transformer，避免按要素重复构建投影。

输出记录 matched/rejected/missing-geometry 数量、每个匹配的几何来源和 near 距离，以及实际使用的投影定义。

## 五、共同安全和正确性边界

- 只接受 EPSG:4326、WGS84、OGC:CRS84 或无 CRS 的规范化 OSM/GeoJSON；其他 CRS 返回 `unsupported_crs`。
- `[lat,lon]` 用于普通点数组；GeoJSON `coordinates` 严格使用 `[lon,lat]`。
- bbox 为 `[west,south,east,north]`。
- 候选 result 含多个点时，bearing 要求先明确选一个，返回 `ambiguous_coordinate` 而不是自动挑选。
- 缺失 geometry 不猜测；统计到 `missing_geometry_count`。
- 属性、几何和上游坐标不真实时，本地计算可能“数值精确但事实错误”，因此结果始终保留来源和假设。
- source result 最多本地处理/导出 100,000 个对象，防止异常输入耗尽内存。

## 六、开发中发现并修复的问题

1. 未传 `source_result` 与“传了但无法解析”原先混为 `needs_acquisition`；现分别为 `missing_input` 和 `needs_acquisition`。
2. `limit` 原先早于空间筛选，会截断真正匹配项；现改为最后应用。
3. bbox 和 AEQD Transformer 原先可能按要素重复创建；现每次调用只创建一次。
4. 非 WGS84 source 若坐标数值碰巧较小，可能被静默误读；现增加 CRS 门禁。
5. 规范化 payload 无法忠实恢复 OSM XML；现拒绝伪造并收紧 Schema。
6. 路线距离、磁北和图像方位缺少必要数据；现显式转向前置 Tool，而不是输出假精度。
7. CSV 中外部名称可能触发表格公式；现对 `= + - @` 开头字段增加公式注入防护。
8. 模型生成的 `name_regex` 可能触发高成本回溯；现拒绝 lookaround、反向引用和嵌套量词，并且每次调用只编译一次。
9. `near` 不应把超全球尺度阈值包装成局部投影精度；现限制距离为 2,000 km、参考几何跨度为 30°，超过范围要求改用适合的大尺度方法。

## 七、完整 Harness 链验证

执行链：

```text
geocode.geocode
→ osm_query.query
→ osm_result_process.filter
→ distance_bearing_calculator.distance
→ distance_bearing_calculator.bearing
→ spatial_filter.geometry_filter
→ osm_result_process.export
→ final_answer
```

结果：

```text
首次执行：8 succeeded/terminal，0 failed
持久化结果：6
导出文件：GeoJSON，2 个 Point feature
恢复执行：8/8 reused_from_checkpoint
恢复时 Provider 调用：geocode=0，Overpass=0
```

测试使用离线 Fake Provider 驱动真实 geocode/Overpass 执行器，三个后处理 Tool 和 Harness 均为真实代码；没有访问网络。

## 八、测试结果

```text
新增 Tool/链路阶段测试：33 passed
基础地理 + Harness 集成：86 passed
整仓全量：348 passed
Ruff：passed
Mypy：passed
```

4 条 warning 来自 Stage 1.5 的 Pillow `getdata` 未来弃用提示，与本轮无关。

## 九、下一步

下一优先级建议实现 `video_frame_extract.frame_retrieve/frame_sample`，因为它是纯本地 FFmpeg Tool，可以验证 Harness 对大视频路径、时间戳、派生 image_id 和内容哈希缓存的控制能力。

外部证据 Tool 则按凭证准备情况推进：map/route → web/page → media → streetview → satellite。完整账号、申请步骤和变量名见 `docs/API_CREDENTIAL_APPLICATION_GUIDE_20260911.md`。
