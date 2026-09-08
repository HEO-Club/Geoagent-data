# Canonical Tool v2：31 份 Schema 与实现调研

本目录的 31 个 JSON 与 `canonical_tool_catalog_v2.json` 顺序一致，共覆盖 57 个 operation、223 个参数字段。每个文件包括正式 Tool 定义、逐 operation 参数用途与获取方式、推荐实现后端、认证/数据来源、执行步骤、失败语义、证据边界、许可风险、MCP 建议和新增 Tool 复核提示。

`research_status=design_researched_not_runtime_verified` 表示该部分是实现调研，不声称服务已经安装、鉴权、付费、取得训练许可或真实执行成功。

## 图片与媒体

1. [`image_edit`](01_image_edit.json)
2. [`image_measure`](02_image_measure.json)
3. [`image_compare`](03_image_compare.json)
4. [`ocr_read`](04_ocr_read.json)
5. [`reverse_image_search`](05_reverse_image_search.json)
6. [`media_metadata_read`](06_media_metadata_read.json)
7. [`web_search`](07_web_search.json)
8. [`web_page_read`](08_web_page_read.json)
9. [`media_search`](09_media_search.json)
10. [`video_frame_extract`](10_video_frame_extract.json)

## 地图、OSM、街景与遥感

11. [`poi_search`](11_poi_search.json)
12. [`geocode`](12_geocode.json)
13. [`route_query`](13_route_query.json)
14. [`map_layer_query`](14_map_layer_query.json)
15. [`osm_query`](15_osm_query.json)
16. [`osm_result_process`](16_osm_result_process.json)
17. [`streetview_query`](17_streetview_query.json)
18. [`satellite_imagery_query`](18_satellite_imagery_query.json)
19. [`satellite_imagery_compare`](19_satellite_imagery_compare.json)

## GIS、气象与天文

20. [`distance_bearing_calculator`](20_distance_bearing_calculator.json)
21. [`visibility_analysis`](21_visibility_analysis.json)
22. [`terrain_analysis`](22_terrain_analysis.json)
23. [`spatial_filter`](23_spatial_filter.json)
24. [`weather_archive_query`](24_weather_archive_query.json)
25. [`solar_ephemeris`](25_solar_ephemeris.json)
26. [`shadow_analysis`](26_shadow_analysis.json)

## 档案、航班、外部模型与终端

27. [`administrative_registry`](27_administrative_registry.json)
28. [`infrastructure_registry`](28_infrastructure_registry.json)
29. [`flight_data_query`](29_flight_data_query.json)
30. [`llm_query`](30_llm_query.json)
31. [`final_answer`](31_final_answer.json)

生成命令：

```powershell
.\.venv\Scripts\python.exe scripts\export_tool_implementation_specs.py
```

新增 Tool 不得直接修改这里或正式目录。先按 [`NEW_TOOL_REVIEW_WORKFLOW.md`](../NEW_TOOL_REVIEW_WORKFLOW.md) 完成临时项复核，再单独设计、实现、测试并提 PR。
