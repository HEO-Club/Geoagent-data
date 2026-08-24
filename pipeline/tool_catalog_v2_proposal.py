"""更细 Canonical Tool v2 提案；不接入生产目录。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.schemas.tools import ToolForest, ToolInputSchema, ToolOperation, ToolTree
from pipeline.stage3_normalize_format.params import attach_operation_input_schemas
from pipeline.stage3_normalize_format.trees import load_forest

# 新工具按真实执行函数/会话边界拆分；同一数据库中仅目标对象不同不拆。
TOOL_SPLITS: dict[str, dict[str, Any]] = {
    "image_edit": {"executor": "local_image_transform", "description": "裁剪、缩放或增强输入图片。", "sources": {"image_process": ["enhance", "crop", "zoom"]}},
    "image_measure": {"executor": "local_image_measurement", "description": "测量图片中的距离、角度、比例或颜色量。", "sources": {"image_process": ["measure"]}},
    "image_compare": {"executor": "local_image_comparison", "description": "对多张图片执行特征、像素或几何比较。", "sources": {"image_process": ["compare"]}},
    "ocr_read": {"executor": "ocr_engine", "description": "识别或解码图中文字和编码。", "sources": {"ocr_read": ["recognize", "decode"]}},
    "reverse_image_search": {"executor": "reverse_image_search_engine", "description": "提交整图或裁剪图进行相似图片检索。", "sources": {"reverse_image_search": ["search", "search_crop"]}},
    "media_metadata_read": {"executor": "local_media_metadata_reader", "description": "读取 EXIF、GPS 和通用媒体元数据。", "sources": {"metadata_read": ["exif", "file"]}},
    "web_search": {"executor": "web_search_engine", "description": "执行开放网页或指定站点搜索。", "sources": {"web_search": ["keyword_search", "site_search"]}},
    "web_page_read": {"executor": "web_page_reader", "description": "打开并抽取已找到网页的内容。", "sources": {"web_search": ["open_result"]}},
    "media_search": {"executor": "media_search_service", "description": "搜索外部视频或照片。", "sources": {"media_search": ["video_search", "photo_search"]}},
    "video_frame_extract": {"executor": "video_frame_extractor", "description": "从已找到视频中提取指定时间或场景帧。", "sources": {"media_search": ["frame_retrieve"]}},
    "poi_search": {"executor": "map_poi_service", "description": "搜索或浏览地图 POI。", "sources": {"map_query": ["poi_search", "browse"]}},
    "geocode": {"executor": "geocoding_service", "description": "在地名、地址和坐标之间转换。", "sources": {"map_query": ["geocode"]}},
    "route_query": {"executor": "route_service", "description": "查询道路连接和路线关系。", "sources": {"map_query": ["route"]}},
    "map_layer_query": {"executor": "map_layer_service", "description": "加载水系、地形、行政或历史地图图层。", "sources": {"map_query": ["load_layer"]}},
    "osm_query": {"executor": "openstreetmap_overpass", "description": "在 OSM/Overpass 中查询或统计要素。", "sources": {"osm_query": ["query", "count"]}},
    "osm_result_process": {"executor": "local_vector_result_processor", "description": "筛选或导出已有 OSM 结果。", "sources": {"osm_query": ["filter", "export"]}},
    "streetview_query": {"executor": "streetview_service", "description": "打开、导航、切换或截取街景会话。", "sources": {"streetview_query": ["open", "navigate", "change_time", "capture"]}},
    "satellite_imagery_query": {"executor": "satellite_imagery_service", "description": "获取、切换或查看卫星/航片影像。", "sources": {"satellite_imagery_query": ["retrieve", "change_time", "oblique_view"]}},
    "satellite_imagery_compare": {"executor": "satellite_comparison_service", "description": "执行多时相或多候选影像比较。", "sources": {"satellite_imagery_query": ["compare_time", "compare_candidates"]}},
    "distance_bearing_calculator": {"executor": "geodesic_calculator", "description": "计算距离或方位角。", "sources": {"geospatial_analysis": ["distance", "bearing"]}},
    "visibility_analysis": {"executor": "visibility_engine", "description": "计算视线、遮挡和可视域。", "sources": {"geospatial_analysis": ["sightline"]}},
    "terrain_analysis": {"executor": "terrain_analysis_engine", "description": "计算高程、坡度和地形剖面。", "sources": {"geospatial_analysis": ["terrain"]}},
    "spatial_filter": {"executor": "spatial_relation_engine", "description": "按几何和空间关系过滤要素。", "sources": {"geospatial_analysis": ["geometry_filter"]}},
    "weather_archive_query": {"executor": "weather_archive_service", "description": "查询历史天气、云量或积雪。", "sources": {"weather_archive_query": ["cloud_cover", "weather", "snow_cover", "refine_range"]}},
    "solar_ephemeris": {"executor": "solar_ephemeris_calculator", "description": "计算日落时间和太阳位置。", "sources": {"astronomy_query": ["sunset_time", "sun_position"]}},
    "shadow_analysis": {"executor": "shadow_model_engine", "description": "根据太阳和物体参数计算理论阴影。", "sources": {"astronomy_query": ["shadow_model"]}},
    "administrative_registry": {"executor": "administrative_registry", "description": "查询行政归属、标准地名和对象名录。", "sources": {"registry_lookup": ["administrative", "directory"]}},
    "infrastructure_registry": {"executor": "infrastructure_registry", "description": "查询建设历史、许可和登记记录。", "sources": {"registry_lookup": ["construction", "permit"]}},
    "flight_data_query": {"executor": "flight_tracking_or_archive_service", "description": "查询航班档案、航迹和附近航空活动。", "sources": {"flight_data_query": ["search", "track", "nearby_traffic"]}},
    "llm_query": {"executor": "external_llm_service", "description": "向外部模型咨询或生成候选清单。", "sources": {"llm_query": ["consult", "enumerate"]}},
    "final_answer": {"executor": "agent_runtime", "description": "提交最终地点。", "sources": {"final_answer": ["submit"]}},
}


def _find_tree(forest: ToolForest, name: str) -> ToolTree:
    for tree in forest.trees:
        if tree.canonical.name == name:
            return tree
    raise KeyError(name)


def _find_operation(tree: ToolTree, name: str) -> ToolOperation:
    for operation in tree.canonical.operations:
        if operation.name == name:
            return operation
    raise KeyError(f"{tree.canonical.name}.{name}")


def _compact_contract(schema: ToolInputSchema | None) -> dict[str, Any]:
    if schema is None:
        return {"required": [], "optional": [], "context": [], "one_of": []}
    return {
        "required": [field.name for field in schema.fields if field.required],
        "optional": [field.name for field in schema.fields if not field.required],
        "context": [
            field.name for field in schema.fields if field.context_sources
        ],
        "one_of": schema.required_any,
    }


def build_tool_catalog_v2(
    catalog_path: str | Path = "canonical_tool_catalog.json",
) -> dict[str, Any]:
    source = attach_operation_input_schemas(load_forest(Path(catalog_path)))
    tools = []
    migration = []
    mapped: set[tuple[str, str]] = set()
    for new_name, definition in TOOL_SPLITS.items():
        operations = []
        variants: list[str] = []
        for old_name, operation_names in definition["sources"].items():
            tree = _find_tree(source, old_name)
            for operation_name in operation_names:
                operation = _find_operation(tree, operation_name)
                operations.append(
                    {
                        "name": operation.name,
                        "description": operation.description,
                        "compact_params": _compact_contract(operation.input_schema),
                        "input_schema": (
                            operation.input_schema.model_dump()
                            if operation.input_schema
                            else None
                        ),
                    }
                )
                migration.append(
                    {
                        "from_tool": old_name,
                        "from_operation": operation_name,
                        "to_tool": new_name,
                        "to_operation": operation_name,
                    }
                )
                mapped.add((old_name, operation_name))
            for variant, variant_op in tree.variant_operations.items():
                if variant_op in operation_names and variant not in variants:
                    variants.append(variant)
        tools.append(
            {
                "name": new_name,
                "description": definition["description"],
                "executor": definition["executor"],
                "operations": operations,
                "variants": variants,
            }
        )

    expected = {
        (tree.canonical.name, operation.name)
        for tree in source.trees
        for operation in tree.canonical.operations
    }
    missing = sorted(expected - mapped)
    duplicates = len(migration) - len(mapped)
    return {
        "schema_version": "canonical_tool_catalog_v2_proposal",
        "status": "proposal_only_not_wired",
        "stats": {
            "source_tools": len(source.trees),
            "proposed_tools": len(tools),
            "operations": len(migration),
            "parameter_fields": sum(
                len(operation["input_schema"]["fields"])
                for tool in tools
                for operation in tool["operations"]
                if operation["input_schema"]
            ),
            "missing_mappings": len(missing),
            "duplicate_mappings": duplicates,
        },
        "tools": tools,
        "migration": migration,
        "missing_mappings": missing,
    }
