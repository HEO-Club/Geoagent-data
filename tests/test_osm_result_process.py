"""osm_result_process 执行器测试；禁止真实付费 API 与网络。"""

from __future__ import annotations

import json
from typing import Any
from xml.etree import ElementTree as ET

from tool import execute
from tool.contract import Observation, RuntimeContext


def _sample_elements() -> list[dict[str, Any]]:
    return [
        {
            "result_id": "osm_1",
            "osm_type": "way",
            "osm_id": 123,
            "tags": {
                "bridge": "yes",
                "name": "黄河铁路桥",
                "railway": "rail",
                "confirmed_location": "MUST NOT LEAK",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[113.67, 34.89], [113.68, 34.90]],
            },
        },
        {
            "result_id": "osm_2",
            "osm_type": "node",
            "osm_id": 456,
            "tags": {"power": "tower", "bridge": "no"},
            "geometry": {"type": "Point", "coordinates": [113.66, 34.88]},
        },
        {
            "result_id": "osm_3",
            "osm_type": "way",
            "osm_id": 123,
            "tags": {"bridge": "yes", "name": "黄河铁路桥"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[113.67, 34.89], [113.68, 34.90]],
            },
        },
    ]


def _source(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation": "query",
        "elements": _sample_elements(),
        "count": 3,
        "data_timestamp": "2026-09-10T00:00:00Z",
        "confirmed_location": "MUST NOT LEAK",
    }
    if extra:
        payload.update(extra)
    return payload


def _nested_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            found.update(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_nested_keys(item))
    return found


def _filter(
    *,
    inputs: dict[str, Any] | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, Any] = {
        "source_result": _source(),
        "filters": {"tags": {"bridge": "yes"}},
    }
    if inputs:
        payload.update(inputs)
    return execute(
        "osm_result_process",
        "filter",
        purpose="筛选 OSM 结果",
        inputs=payload,
        ctx=ctx if ctx is not None else RuntimeContext(),
    )


def _export(
    *,
    inputs: dict[str, Any] | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, Any] = {
        "source_result": _source(),
        "format": "geojson",
    }
    if inputs:
        payload.update(inputs)
    return execute(
        "osm_result_process",
        "export",
        purpose="导出 OSM 结果",
        inputs=payload,
        ctx=ctx if ctx is not None else RuntimeContext(),
    )


def test_empty_inputs_are_missing_input() -> None:
    observation = execute("osm_result_process", "filter", purpose="scaffold", inputs={})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_natural_language_source_result_is_invalid() -> None:
    observation = _filter(inputs={"source_result": "刚查到的那些桥"})
    assert observation.ok is False
    assert observation.error_code == "invalid_input"
    assert observation.error is not None
    assert "自然语言" in observation.error


def test_unknown_result_id_is_missing_input() -> None:
    observation = _filter(inputs={"source_result": "osm_result_01"}, ctx=RuntimeContext())
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_previous_tool_result_filters_tags() -> None:
    ctx = RuntimeContext(previous_tool_result=_source())
    observation = _filter(
        inputs={"source_result": "$previous_tool_result", "filters": {"bridge": "yes"}},
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["kept"] == 1
    assert observation.result["removed"] == 2
    assert observation.result["count"] == 1
    assert observation.result["elements"][0]["osm_id"] == 123
    assert observation.result["applied"]["provider"] == "local"
    assert observation.result["applied"]["filters"]["tags"]["bridge"] == "yes"
    assert observation.result["applied"]["filters"]["dedupe"] is True
    assert observation.result["data_timestamp"] == "2026-09-10T00:00:00Z"
    assert observation.result["result_id"] == "osm_processed_1"
    assert any("未回源 OSM" in item for item in observation.result["assumptions"])


def test_inline_elements_and_shorthand_tags() -> None:
    observation = _filter(inputs={"filters": {"bridge": "yes", "osm_type": "way"}})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["kept"] == 1
    assert observation.result["elements"][0]["osm_type"] == "way"


def test_missing_attribute_is_unsupported_filter() -> None:
    observation = _filter(inputs={"filters": {"tags": {"layer": "1"}}})
    assert observation.ok is False
    assert observation.error_code == "unsupported_filter"
    assert observation.error is not None
    assert "layer" in observation.error


def test_star_tag_means_key_exists() -> None:
    observation = _filter(inputs={"filters": {"tags": {"railway": "*"}}})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["kept"] == 1
    assert observation.result["elements"][0]["tags"]["railway"] == "rail"


def test_dedupe_can_be_disabled() -> None:
    observation = _filter(inputs={"filters": {"tags": {"bridge": "yes"}, "dedupe": False}})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["kept"] == 2
    assert [item["osm_id"] for item in observation.result["elements"]] == [123, 123]


def test_empty_filters_without_spatial_are_missing_input() -> None:
    observation = _filter(inputs={"filters": {}})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_spatial_intersects_keeps_crossing_way() -> None:
    river = {
        "type": "LineString",
        "coordinates": [[113.65, 34.895], [113.70, 34.895]],
    }
    observation = _filter(
        inputs={
            "filters": {"osm_type": "way"},
            "spatial_filter": {"relation": "intersects", "geometry": river},
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["kept"] == 1
    assert observation.result["applied"]["spatial_filter"]["relation"] == "intersects"


def test_spatial_near_keeps_nearby_point() -> None:
    river = {
        "type": "LineString",
        "coordinates": [[113.65, 34.89], [113.70, 34.89]],
    }
    observation = _filter(
        inputs={
            "filters": {"osm_type": "node"},
            "spatial_filter": {"relation": "near", "geometry": river, "distance_m": 2000},
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["kept"] == 1
    assert observation.result["elements"][0]["osm_id"] == 456


def test_spatial_place_name_is_invalid() -> None:
    observation = _filter(
        inputs={
            "filters": {"osm_type": "way"},
            "spatial_filter": {"relation": "near", "geometry": "黄河", "distance_m": 100},
        }
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"
    assert observation.error is not None
    assert "地名" in observation.error


def test_spatial_without_geometry_on_elements_is_unsupported() -> None:
    source = {
        "elements": [
            {"osm_type": "way", "osm_id": 1, "tags": {"bridge": "yes"}},
        ]
    }
    observation = _filter(
        inputs={
            "source_result": source,
            "filters": {"bridge": "yes"},
            "spatial_filter": {
                "relation": "intersects",
                "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
            },
        }
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_filter"


def test_export_formats_and_include_geometry_false() -> None:
    geojson = _export(inputs={"format": "geojson"})
    assert geojson.ok is True
    assert geojson.result is not None
    assert geojson.result["format"] == "geojson"
    assert geojson.result["geojson"]["type"] == "FeatureCollection"
    assert len(geojson.result["geojson"]["features"]) == 3
    parsed = json.loads(geojson.result["payload"])
    assert parsed["features"][0]["geometry"]["type"] == "LineString"

    as_json = _export(inputs={"format": "json", "include_geometry": False})
    assert as_json.ok is True
    assert as_json.result is not None
    rows = json.loads(as_json.result["payload"])
    assert "geometry" not in rows[0]
    assert as_json.result["applied"]["include_geometry"] is False

    csv_obs = _export(inputs={"format": "csv"})
    assert csv_obs.ok is True
    assert csv_obs.result is not None
    assert "bridge" in csv_obs.result["payload"]
    assert "osm_id" in csv_obs.result["payload"]

    kml_obs = _export(inputs={"format": "kml"})
    assert kml_obs.ok is True
    assert kml_obs.result is not None
    root = ET.fromstring(kml_obs.result["payload"])
    assert root.tag.endswith("kml")

    osm_obs = _export(inputs={"format": "osm_xml"})
    assert osm_obs.ok is True
    assert osm_obs.result is not None
    osm_root = ET.fromstring(osm_obs.result["payload"])
    assert osm_root.tag == "osm"
    assert osm_root.find("way") is not None


def test_unsupported_format_is_invalid_input() -> None:
    observation = _export(inputs={"format": "gpkg"})
    assert observation.ok is False
    assert observation.error_code == "invalid_input"


def test_allow_real_api_false_still_runs_locally(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    observation = _filter()
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["provider"] == "local"


def test_forbidden_keys_are_stripped() -> None:
    observation = _filter()
    assert observation.ok is True
    assert observation.result is not None
    dumped = str(observation.result)
    assert "MUST NOT LEAK" not in dumped
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys


def test_filter_result_id_can_be_exported() -> None:
    ctx = RuntimeContext()
    filtered = _filter(ctx=ctx)
    assert filtered.ok is True
    assert filtered.result is not None
    result_id = filtered.result["result_id"]
    exported = _export(inputs={"source_result": result_id, "format": "json"}, ctx=ctx)
    assert exported.ok is True
    assert exported.result is not None
    rows = json.loads(exported.result["payload"])
    assert len(rows) == 1
    assert rows[0]["osm_id"] == 123


def test_nested_result_unwrap() -> None:
    observation = _filter(inputs={"source_result": {"result": _source()}})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["kept"] == 1
