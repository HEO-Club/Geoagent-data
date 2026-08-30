# Canonical Tool v2 细分与生产接入

状态：`production_wired`。生产目录为 `canonical_tool_catalog_v2.json`，已接入 Stage 2 工具合同提示、Stage 3 语义映射与参数编译，并由融合 Stage 4 审核 Tool/operation/input_schema 的可执行性。旧 `canonical_tool_catalog.json` 仅保留为 v1 迁移来源。

## 设计原则

- 按实际执行函数、API 或会话边界拆分，不按目标对象拆分；
- OSM 中查桥、查道路、查电塔仍属于同一个 `osm_query`，不制造对象级工具；
- 搜索与读取网页、获取卫星图与比较卫星图、查询 OSM 与处理查询结果分别归入不同工具；
- 每个 operation 同时提供精简参数清单和完整 `input_schema`，Agent 默认只读 required/optional/context/one_of，出现问题时再读取字段解释与 acquisition hint。

## 31 类工具

| 类别 | Tool | Operations |
|---|---|---|
| 图片 | `image_edit` | enhance, crop, zoom |
| 图片 | `image_measure` | measure |
| 图片 | `image_compare` | compare |
| 图片 | `ocr_read` | recognize, decode |
| 图片 | `reverse_image_search` | search, search_crop |
| 图片 | `media_metadata_read` | exif, file |
| Web/媒体 | `web_search` | keyword_search, site_search |
| Web/媒体 | `web_page_read` | open_result |
| Web/媒体 | `media_search` | video_search, photo_search |
| Web/媒体 | `video_frame_extract` | frame_retrieve |
| 地图 | `poi_search` | poi_search, browse |
| 地图 | `geocode` | geocode |
| 地图 | `route_query` | route |
| 地图 | `map_layer_query` | load_layer |
| OSM | `osm_query` | query, count |
| OSM | `osm_result_process` | filter, export |
| 街景 | `streetview_query` | open, navigate, change_time, capture |
| 遥感 | `satellite_imagery_query` | retrieve, change_time, oblique_view |
| 遥感 | `satellite_imagery_compare` | compare_time, compare_candidates |
| GIS | `distance_bearing_calculator` | distance, bearing |
| GIS | `visibility_analysis` | sightline |
| GIS | `terrain_analysis` | terrain |
| GIS | `spatial_filter` | geometry_filter |
| 气象 | `weather_archive_query` | cloud_cover, weather, snow_cover, refine_range |
| 天文 | `solar_ephemeris` | sunset_time, sun_position |
| 天文 | `shadow_analysis` | shadow_model |
| 档案 | `administrative_registry` | administrative, directory |
| 档案 | `infrastructure_registry` | construction, permit |
| 航班 | `flight_data_query` | search, track, nearby_traffic |
| 外部模型 | `llm_query` | consult, enumerate |
| 终端 | `final_answer` | submit |

## 参数与迁移覆盖

```text
原工具：17
提案工具：31
operation：57
参数字段：223
缺失迁移：0
重复迁移：0
```

对既有 9 条轨迹、79 次调用进行离线迁移：78 次成功，旧版实际使用 13 类工具，v2 展开为 20 类。唯一无法迁移的是不可部署的 `field_site_visit`。

主要拆分效果：

```text
map_query            → poi_search / map_layer_query
image_process        → image_edit / image_measure / image_compare
geospatial_analysis  → distance_bearing / visibility / terrain / spatial_filter
astronomy_query      → solar_ephemeris / shadow_analysis
satellite query      → satellite_imagery_query / satellite_imagery_compare
osm_query            → osm_query / osm_result_process
registry_lookup      → administrative_registry / infrastructure_registry
media_search         → media_search / video_frame_extract
web_search           → web_search / web_page_read
```

完整迁移说明：`docs/tool_catalog_v2_proposed.json`；生产 ToolForest：`canonical_tool_catalog_v2.json`。

真实结果分析：`data/runs/tool_catalog_v2_observation_gate_analysis.json`。
