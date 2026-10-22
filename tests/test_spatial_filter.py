"""spatial_filter.geometry_filter 本地空间筛选测试；禁止真实付费 API 与网络。"""

from __future__ import annotations

from typing import Any

import pytest
from pyproj import CRS, Transformer

from tool import execute
from tool._crs import wgs84_to_gcj02
from tool.contract import Observation, RuntimeContext

_RIVER = {
    "type": "LineString",
    "coordinates": [[0.0, 0.0], [1.0, 0.0]],
}
_MID_NEAR = {"type": "Point", "coordinates": [0.5, 0.001]}
_FAR = {"type": "Point", "coordinates": [0.5, 0.05]}
_CROSSING = {
    "type": "LineString",
    "coordinates": [[0.5, -0.1], [0.5, 0.1]],
}
_BOX = {
    "type": "Polygon",
    "coordinates": [[[0.0, -0.01], [1.0, -0.01], [1.0, 0.01], [0.0, 0.01], [0.0, -0.01]]],
}


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


def _feature(geometry: dict[str, Any], *, fid: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {"id": fid, "confirmed_location": "MUST NOT LEAK"}
    if extra:
        properties.update(extra)
    return {"type": "Feature", "id": fid, "properties": properties, "geometry": geometry}


def _geojson(*features: dict[str, Any]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def _filter(*, inputs: dict[str, Any], ctx: RuntimeContext | None = None) -> Observation:
    return execute(
        "spatial_filter",
        "geometry_filter",
        purpose="筛选",
        inputs=inputs,
        ctx=ctx,
    )


def test_empty_inputs_are_missing_input() -> None:
    observation = _filter(inputs={})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_near_without_distance_is_missing_input() -> None:
    observation = _filter(
        inputs={
            "source_result": _geojson(_feature(_MID_NEAR, fid="a")),
            "relation": "near",
            "geometry": _RIVER,
        }
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "distance_m" in observation.error


def test_intersects_keeps_crossing_line() -> None:
    observation = _filter(
        inputs={
            "source_result": _geojson(
                _feature(_CROSSING, fid="bridge"),
                _feature(_FAR, fid="tower"),
            ),
            "relation": "intersects",
            "geometry": _RIVER,
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["kept"] == 1
    assert observation.result["removed"] == 1
    assert observation.result["features"][0]["properties"]["id"] == "bridge"
    assert observation.result["features"][0]["relation"] == "intersects"
    assert observation.result["applied"]["relation"] == "intersects"
    assert observation.result["applied"]["provider"] == "local"
    assert "confirmed_location" not in _nested_keys(observation.result)


def test_within_contains_and_crosses() -> None:
    inside = _feature({"type": "Point", "coordinates": [0.5, 0.0]}, fid="in")
    outside = _feature(_FAR, fid="out")
    within = _filter(
        inputs={
            "source_result": _geojson(inside, outside),
            "relation": "within",
            "geometry": _BOX,
        }
    )
    assert within.ok is True and within.result is not None
    assert [item["properties"]["id"] for item in within.result["features"]] == ["in"]

    contains = _filter(
        inputs={
            "source_result": _geojson(_feature(_BOX, fid="poly")),
            "relation": "contains",
            "geometry": {"type": "Point", "coordinates": [0.5, 0.0]},
        }
    )
    assert contains.ok is True and contains.result is not None
    assert contains.result["kept"] == 1

    crosses = _filter(
        inputs={
            "source_result": _geojson(_feature(_CROSSING, fid="cross"), _feature(_FAR, fid="miss")),
            "relation": "crosses",
            "geometry": _RIVER,
        }
    )
    assert crosses.ok is True and crosses.result is not None
    assert [item["properties"]["id"] for item in crosses.result["features"]] == ["cross"]


def test_near_uses_line_geometry_not_endpoint_label() -> None:
    source = _geojson(_feature(_MID_NEAR, fid="mid"), _feature(_FAR, fid="far"))
    near_line = _filter(
        inputs={
            "source_result": source,
            "relation": "near",
            "geometry": _RIVER,
            "distance_m": 200,
        }
    )
    assert near_line.ok is True and near_line.result is not None
    assert near_line.result["kept"] == 1
    kept = near_line.result["features"][0]
    assert kept["properties"]["id"] == "mid"
    assert kept["relation"] == "near"
    assert kept["distance_m"] == pytest.approx(111.32, rel=2e-2)
    assert any("参照几何" in item for item in near_line.result["assumptions"])

    near_label = _filter(
        inputs={
            "source_result": source,
            "relation": "near",
            "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
            "distance_m": 200,
        }
    )
    assert near_label.ok is True and near_label.result is not None
    assert near_label.result["kept"] == 0
    assert near_label.result["count"] == 0


def test_place_name_geometry_is_invalid() -> None:
    observation = _filter(
        inputs={
            "source_result": _geojson(_feature(_MID_NEAR, fid="a")),
            "relation": "near",
            "geometry": "黄河",
            "distance_m": 100,
        }
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"
    assert observation.error is not None
    assert "地名" in observation.error


def test_source_without_geometry_is_unsupported() -> None:
    observation = _filter(
        inputs={
            "source_result": {
                "elements": [{"osm_type": "way", "osm_id": 1, "tags": {"bridge": "yes"}}],
            },
            "relation": "intersects",
            "geometry": _RIVER,
        }
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_filter"


def test_invalid_relation_and_non_positive_distance() -> None:
    bad_rel = _filter(
        inputs={
            "source_result": _geojson(_feature(_MID_NEAR, fid="a")),
            "relation": "buffer",
            "geometry": _RIVER,
        }
    )
    assert bad_rel.ok is False
    assert bad_rel.error_code == "invalid_input"

    bad_dist = _filter(
        inputs={
            "source_result": _geojson(_feature(_MID_NEAR, fid="a")),
            "relation": "near",
            "geometry": _RIVER,
            "distance_m": 0,
        }
    )
    assert bad_dist.ok is False
    assert bad_dist.error_code == "invalid_input"


def test_osm_elements_and_layer_features_are_accepted() -> None:
    osm = _filter(
        inputs={
            "source_result": {
                "operation": "query",
                "elements": [
                    {
                        "osm_type": "way",
                        "osm_id": 123,
                        "tags": {"bridge": "yes", "confirmed_location": "MUST NOT LEAK"},
                        "geometry": _CROSSING,
                    },
                    {
                        "osm_type": "node",
                        "osm_id": 456,
                        "tags": {"power": "tower"},
                        "geometry": _FAR,
                    },
                ],
            },
            "relation": "intersects",
            "geometry": _RIVER,
        }
    )
    assert osm.ok is True and osm.result is not None
    assert osm.result["kept"] == 1
    assert osm.result["features"][0]["properties"]["osm_id"] == 123
    assert "confirmed_location" not in _nested_keys(osm.result)

    layer = _filter(
        inputs={
            "source_result": {"features": [_feature(_CROSSING, fid="road")]},
            "relation": "intersects",
            "geometry": _RIVER,
        }
    )
    assert layer.ok is True and layer.result is not None
    assert layer.result["kept"] == 1


def test_previous_tool_result_and_registry_id() -> None:
    source = _geojson(_feature(_CROSSING, fid="bridge"))
    previous = _filter(
        inputs={
            "source_result": "$previous_tool_result",
            "relation": "intersects",
            "geometry": _RIVER,
        },
        ctx=RuntimeContext(previous_tool_result=source),
    )
    assert previous.ok is True and previous.result is not None
    assert previous.result["kept"] == 1

    ctx = RuntimeContext(
        extras={
            "osm_results": {
                "osm_result_01": {
                    "elements": [
                        {"osm_id": 9, "osm_type": "way", "geometry": _CROSSING, "tags": {"bridge": "yes"}},
                    ],
                }
            }
        }
    )
    by_id = _filter(
        inputs={
            "source_result": "osm_result_01",
            "relation": "intersects",
            "geometry": _RIVER,
        },
        ctx=ctx,
    )
    assert by_id.ok is True and by_id.result is not None
    assert by_id.result["kept"] == 1
    stored = ctx.extras["spatial_results"][by_id.result["result_id"]]
    assert stored["kept"] == 1


def test_crs_mismatch_is_transformed_to_wgs84() -> None:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True)
    x, y = transformer.transform(0.5, 0.0)
    observation = _filter(
        inputs={
            "source_result": {
                "type": "FeatureCollection",
                "crs": "epsg:3857",
                "features": [
                    _feature({"type": "Point", "coordinates": [x, y]}, fid="proj"),
                ],
            },
            "relation": "intersects",
            "geometry": _BOX,
        }
    )
    assert observation.ok is True and observation.result is not None
    assert observation.result["kept"] == 1
    assert observation.result["applied"]["crs"] == "wgs84"


def test_gcj02_is_transformed_to_wgs84() -> None:
    lon, lat = 113.67, 34.89
    gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
    delta = 0.0001
    observation = _filter(
        inputs={
            "source_result": {
                "type": "FeatureCollection",
                "crs": "gcj02",
                "features": [
                    _feature({"type": "Point", "coordinates": [gcj_lon, gcj_lat]}, fid="zhengzhou"),
                ],
            },
            "relation": "intersects",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [lon - delta, lat - delta],
                        [lon + delta, lat - delta],
                        [lon + delta, lat + delta],
                        [lon - delta, lat + delta],
                        [lon - delta, lat - delta],
                    ]
                ],
            },
        }
    )
    assert observation.ok is True and observation.result is not None
    assert observation.result["kept"] == 1
    assert observation.result["applied"]["crs"] == "wgs84"


def test_empty_keep_is_success() -> None:
    observation = _filter(
        inputs={
            "source_result": _geojson(_feature(_FAR, fid="far")),
            "relation": "intersects",
            "geometry": _RIVER,
        }
    )
    assert observation.ok is True and observation.result is not None
    assert observation.result["kept"] == 0
    assert observation.result["removed"] == 1
    assert observation.result["features"] == []
    assert observation.result["operation"] == "geometry_filter"
