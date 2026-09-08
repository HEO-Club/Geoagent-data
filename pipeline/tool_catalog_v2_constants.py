"""Dependency-light Tool v2 identity constants used during catalog loading."""

from __future__ import annotations

from pipeline.schemas.tools import ToolForest

LEGACY_V1_TOOL_NAMES = frozenset(
    {
        "image_process",
        "ocr_read",
        "reverse_image_search",
        "web_search",
        "map_query",
        "osm_query",
        "streetview_query",
        "satellite_imagery_query",
        "weather_archive_query",
        "astronomy_query",
        "geospatial_analysis",
        "media_search",
        "registry_lookup",
        "llm_query",
        "flight_data_query",
        "metadata_read",
        "final_answer",
    }
)

V2_SENTINEL_TOOLS = frozenset(
    {"image_edit", "poi_search", "osm_result_process", "terrain_analysis"}
)


def is_v2_forest(forest: ToolForest) -> bool:
    """Return whether a forest contains the production v2 split catalog."""

    names = {tree.canonical.name for tree in forest.trees}
    return V2_SENTINEL_TOOLS.issubset(names)
