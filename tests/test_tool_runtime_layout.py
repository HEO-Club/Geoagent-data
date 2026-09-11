"""真实 Tool 包目录必须与 canonical_tool_catalog_v2.json 对齐。"""

from __future__ import annotations

import json
from pathlib import Path

from tool import TOOLS, execute
from tool._catalog import catalog_input_fields
from tool.administrative_registry._registry import _INPUT_FIELDS as _ADMIN_FIELDS
from tool.contract import Observation, declared_inputs
from tool.final_answer._submit import _LOCATION_FIELD
from tool.flight_data_query._query import _NEARBY_FIELDS, _SEARCH_FIELDS, _TRACK_FIELDS
from tool.image_measure._measure import _MEASURE_FIELDS
from tool.infrastructure_registry._registry import _INPUT_FIELDS as _INFRA_FIELDS
from tool.llm_query._query import _CONSULT_FIELDS, _ENUMERATE_FIELDS
from tool.osm_query._query import _COUNT_INPUT_FIELDS, _QUERY_INPUT_FIELDS
from tool.route_query._route import _ROUTE_FIELDS
from tool.weather_archive_query._archive import (
    _QUERY_INPUT_FIELDS as _WEATHER_QUERY_FIELDS,
    _REFINE_INPUT_FIELDS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "canonical_tool_catalog_v2.json"
TOOL_ROOT = REPO_ROOT / "tool"


def _catalog_tools() -> list[tuple[str, list[str]]]:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    items: list[tuple[str, list[str]]] = []
    for tree in payload["trees"]:
        canonical = tree["canonical"]
        name = str(canonical["name"])
        operations = [str(op["name"]) for op in canonical.get("operations") or []]
        items.append((name, operations))
    return items


def test_tool_package_matches_catalog_layout() -> None:
    catalog = _catalog_tools()
    assert len(catalog) == 31
    assert sum(len(operations) for _, operations in catalog) == 57
    assert set(TOOLS) == {name for name, _ in catalog}
    for tool_name, operations in catalog:
        package_dir = TOOL_ROOT / tool_name
        assert package_dir.is_dir(), tool_name
        assert (package_dir / "__init__.py").is_file(), tool_name
        assert set(TOOLS[tool_name].OPERATIONS) == set(operations)
        for operation in operations:
            assert (package_dir / f"{operation}.py").is_file(), (tool_name, operation)


IMPLEMENTED_OPERATIONS = {
    ("image_edit", "crop"),
    ("image_edit", "zoom"),
    ("image_edit", "enhance"),
    ("image_measure", "measure"),
    ("image_compare", "compare"),
    ("ocr_read", "recognize"),
    ("ocr_read", "decode"),
    ("reverse_image_search", "search"),
    ("reverse_image_search", "search_crop"),
    ("media_metadata_read", "exif"),
    ("media_metadata_read", "file"),
    ("web_search", "keyword_search"),
    ("web_search", "site_search"),
    ("web_page_read", "open_result"),
    ("media_search", "video_search"),
    ("media_search", "photo_search"),
    ("video_frame_extract", "frame_retrieve"),
    ("poi_search", "poi_search"),
    ("poi_search", "browse"),
    ("geocode", "geocode"),
    ("route_query", "route"),
    ("map_layer_query", "load_layer"),
    ("osm_query", "query"),
    ("osm_query", "count"),
    ("osm_result_process", "filter"),
    ("osm_result_process", "export"),
    ("streetview_query", "open"),
    ("streetview_query", "navigate"),
    ("streetview_query", "change_time"),
    ("streetview_query", "capture"),
    ("satellite_imagery_query", "retrieve"),
    ("satellite_imagery_query", "change_time"),
    ("satellite_imagery_query", "oblique_view"),
    ("satellite_imagery_compare", "compare_time"),
    ("satellite_imagery_compare", "compare_candidates"),
    ("distance_bearing_calculator", "distance"),
    ("distance_bearing_calculator", "bearing"),
    ("visibility_analysis", "sightline"),
    ("terrain_analysis", "terrain"),
    ("spatial_filter", "geometry_filter"),
    ("weather_archive_query", "weather"),
    ("weather_archive_query", "cloud_cover"),
    ("weather_archive_query", "snow_cover"),
    ("weather_archive_query", "refine_range"),
    ("solar_ephemeris", "sun_position"),
    ("solar_ephemeris", "sunset_time"),
    ("shadow_analysis", "shadow_model"),
    ("administrative_registry", "administrative"),
    ("administrative_registry", "directory"),
    ("infrastructure_registry", "construction"),
    ("infrastructure_registry", "permit"),
    ("flight_data_query", "search"),
    ("flight_data_query", "track"),
    ("flight_data_query", "nearby_traffic"),
    ("llm_query", "consult"),
    ("llm_query", "enumerate"),
    ("final_answer", "submit"),
}


def test_unimplemented_catalog_operations_return_placeholder() -> None:
    for tool_name, operations in _catalog_tools():
        for operation in operations:
            observation = execute(
                tool_name,
                operation,
                purpose="scaffold",
                inputs={},
            )
            assert isinstance(observation, Observation)
            assert observation.ok is False
            if (tool_name, operation) in IMPLEMENTED_OPERATIONS:
                assert observation.error_code != "not_implemented"
                assert observation.error_code == "missing_input"
            else:
                assert observation.error_code == "not_implemented"


def test_unknown_tool_and_operation_are_structured_errors() -> None:
    unknown_tool = execute("not_a_tool", "query", purpose="x", inputs={})
    assert unknown_tool.error_code == "unknown_tool"
    unknown_op = execute("osm_query", "not_an_op", purpose="x", inputs={})
    assert unknown_op.error_code == "unknown_operation"


def _executor_input_fields() -> dict[tuple[str, str], set[str]]:
    return {
        ("image_edit", "enhance"): {"image", "region", "adjustments", "output_format"},
        ("image_edit", "crop"): {"image", "region", "padding", "padding_mode", "output_format"},
        ("image_edit", "zoom"): {"image", "region", "scale", "output_format"},
        ("image_measure", "measure"): set(_MEASURE_FIELDS),
        ("image_compare", "compare"): {"images", "method", "region"},
        ("ocr_read", "recognize"): {"image", "region", "languages", "text_kind"},
        ("ocr_read", "decode"): {"image", "region", "code_types"},
        ("reverse_image_search", "search"): {"image", "engines", "top_k"},
        ("reverse_image_search", "search_crop"): {"image", "region", "engines", "top_k"},
        ("media_metadata_read", "exif"): {"file", "fields"},
        ("media_metadata_read", "file"): {"file", "fields"},
        ("web_search", "keyword_search"): {"query", "domains", "language", "time_range", "top_k"},
        ("web_search", "site_search"): {"query", "site", "time_range", "top_k"},
        ("web_page_read", "open_result"): {"url", "result_id", "extract"},
        ("media_search", "video_search"): {"query", "area", "time_range", "platforms", "top_k"},
        ("media_search", "photo_search"): {"query", "area", "time_range", "sources", "top_k"},
        ("video_frame_extract", "frame_retrieve"): {"video", "timestamps", "view"},
        ("poi_search", "poi_search"): {
            "area",
            "query",
            "bbox",
            "center",
            "categories",
            "filters",
            "radius_m",
            "language",
            "top_k",
        },
        ("poi_search", "browse"): {
            "area",
            "bbox",
            "center",
            "query",
            "categories",
            "filters",
            "radius_m",
            "top_k",
        },
        ("geocode", "geocode"): {"query", "area", "direction", "language", "top_k"},
        ("route_query", "route"): set(_ROUTE_FIELDS),
        ("map_layer_query", "load_layer"): {"area", "layers", "time_range", "provider"},
        ("osm_query", "query"): set(_QUERY_INPUT_FIELDS),
        ("osm_query", "count"): set(_COUNT_INPUT_FIELDS),
        ("osm_result_process", "filter"): {"source_result", "filters", "spatial_filter"},
        ("osm_result_process", "export"): {"source_result", "format", "include_geometry"},
        ("streetview_query", "open"): {"area", "coordinates", "provider"},
        ("streetview_query", "navigate"): {"session", "area", "direction", "distance_m"},
        ("streetview_query", "change_time"): {"session", "area", "time_range"},
        ("streetview_query", "capture"): {"session", "heading", "pitch", "fov"},
        ("satellite_imagery_query", "retrieve"): {
            "area",
            "coordinates",
            "source_result",
            "time_range",
            "provider",
            "layer",
            "cloud_cover_max",
            "resolution_m",
        },
        ("satellite_imagery_query", "change_time"): {
            "area",
            "time_range",
            "source_result",
            "provider",
        },
        ("satellite_imagery_query", "oblique_view"): {"area", "heading", "tilt", "time_range"},
        ("satellite_imagery_compare", "compare_time"): {"area", "times", "comparison"},
        ("satellite_imagery_compare", "compare_candidates"): {
            "candidates",
            "template",
            "time_range",
            "provider",
        },
        ("distance_bearing_calculator", "distance"): {"points", "features", "unit", "mode"},
        ("distance_bearing_calculator", "bearing"): {"origin", "target", "reference"},
        ("visibility_analysis", "sightline"): {
            "observer",
            "target",
            "through_points",
            "terrain",
        },
        ("terrain_analysis", "terrain"): {"area", "path", "metrics"},
        ("spatial_filter", "geometry_filter"): {
            "source_result",
            "relation",
            "geometry",
            "distance_m",
        },
        ("weather_archive_query", "weather"): set(_WEATHER_QUERY_FIELDS),
        ("weather_archive_query", "cloud_cover"): set(_WEATHER_QUERY_FIELDS),
        ("weather_archive_query", "snow_cover"): set(_WEATHER_QUERY_FIELDS),
        ("weather_archive_query", "refine_range"): set(_REFINE_INPUT_FIELDS),
        ("solar_ephemeris", "sun_position"): {"area", "datetime", "timezone"},
        ("solar_ephemeris", "sunset_time"): {"locations", "time_range", "timezone"},
        ("shadow_analysis", "shadow_model"): {
            "area",
            "datetime",
            "object_height_m",
            "surface",
        },
        ("administrative_registry", "administrative"): set(_ADMIN_FIELDS),
        ("administrative_registry", "directory"): set(_ADMIN_FIELDS),
        ("infrastructure_registry", "construction"): set(_INFRA_FIELDS),
        ("infrastructure_registry", "permit"): set(_INFRA_FIELDS),
        ("flight_data_query", "search"): set(_SEARCH_FIELDS),
        ("flight_data_query", "track"): set(_TRACK_FIELDS),
        ("flight_data_query", "nearby_traffic"): set(_NEARBY_FIELDS),
        ("llm_query", "consult"): set(_CONSULT_FIELDS),
        ("llm_query", "enumerate"): set(_ENUMERATE_FIELDS),
        ("final_answer", "submit"): {_LOCATION_FIELD},
    }


def test_executor_declared_fields_match_catalog() -> None:
    catalog = catalog_input_fields()
    executor = _executor_input_fields()
    assert set(executor) == set(catalog)
    mismatches = {
        f"{tool}.{operation}": {
            "only_executor": sorted(executor[(tool, operation)] - catalog[(tool, operation)]),
            "only_catalog": sorted(catalog[(tool, operation)] - executor[(tool, operation)]),
        }
        for tool, operation in catalog
        if executor[(tool, operation)] != catalog[(tool, operation)]
    }
    assert mismatches == {}


def test_declared_inputs_keep_extras_in_extensions() -> None:
    parsed = declared_inputs(
        {
            "image": "img_1",
            "region": [0, 0, 10, 10],
            "measurement": "distance",
            "axis": "horizontal",
            "note": "横着量",
            "extensions": {"site": "桥塔"},
        },
        "image",
        "region",
        "measurement",
        "axis",
        "reference",
    )
    assert parsed["image"] == "img_1"
    assert parsed["axis"] == "horizontal"
    assert parsed["extensions"] == {"site": "桥塔", "note": "横着量"}
    assert "note" not in parsed or parsed["extensions"]["note"] == "横着量"
