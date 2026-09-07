"""按 canonical_tool_catalog_v2.json 落地的真实 Tool 执行器包。

每个子目录对应一个 canonical tool；每个 operation 一个模块，暴露 `execute`。
`image_edit`、`image_measure`、`image_compare`、`ocr_read`、`reverse_image_search` 与 `media_metadata_read` 已接入执行器；其余 tool 仍为占位，不调用外部付费 API。
"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext

from tool import image_edit
from tool import image_measure
from tool import image_compare
from tool import ocr_read
from tool import reverse_image_search
from tool import media_metadata_read
from tool import web_search
from tool import web_page_read
from tool import media_search
from tool import video_frame_extract
from tool import poi_search
from tool import geocode
from tool import route_query
from tool import map_layer_query
from tool import osm_query
from tool import osm_result_process
from tool import streetview_query
from tool import satellite_imagery_query
from tool import satellite_imagery_compare
from tool import distance_bearing_calculator
from tool import visibility_analysis
from tool import terrain_analysis
from tool import spatial_filter
from tool import weather_archive_query
from tool import solar_ephemeris
from tool import shadow_analysis
from tool import administrative_registry
from tool import infrastructure_registry
from tool import flight_data_query
from tool import llm_query
from tool import final_answer

TOOLS = {
    'image_edit': image_edit,
    'image_measure': image_measure,
    'image_compare': image_compare,
    'ocr_read': ocr_read,
    'reverse_image_search': reverse_image_search,
    'media_metadata_read': media_metadata_read,
    'web_search': web_search,
    'web_page_read': web_page_read,
    'media_search': media_search,
    'video_frame_extract': video_frame_extract,
    'poi_search': poi_search,
    'geocode': geocode,
    'route_query': route_query,
    'map_layer_query': map_layer_query,
    'osm_query': osm_query,
    'osm_result_process': osm_result_process,
    'streetview_query': streetview_query,
    'satellite_imagery_query': satellite_imagery_query,
    'satellite_imagery_compare': satellite_imagery_compare,
    'distance_bearing_calculator': distance_bearing_calculator,
    'visibility_analysis': visibility_analysis,
    'terrain_analysis': terrain_analysis,
    'spatial_filter': spatial_filter,
    'weather_archive_query': weather_archive_query,
    'solar_ephemeris': solar_ephemeris,
    'shadow_analysis': shadow_analysis,
    'administrative_registry': administrative_registry,
    'infrastructure_registry': infrastructure_registry,
    'flight_data_query': flight_data_query,
    'llm_query': llm_query,
    'final_answer': final_answer,
}


def execute(
    tool: str,
    operation: str,
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按 (tool, operation) 分发给对应模块。"""

    package = TOOLS.get(tool)
    if package is None:
        return Observation(
            ok=False,
            error=f"未知 tool: {tool}",
            error_code="unknown_tool",
        )
    handler = package.OPERATIONS.get(operation)
    if handler is None:
        return Observation(
            ok=False,
            error=f"未知 operation: {tool}.{operation}",
            error_code="unknown_operation",
        )
    return handler(purpose=purpose, inputs=inputs, ctx=ctx)


__all__ = ["TOOLS", "Observation", "RuntimeContext", "execute"]
