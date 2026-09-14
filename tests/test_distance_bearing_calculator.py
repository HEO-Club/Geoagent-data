"""distance_bearing_calculator 本地测地执行器测试；禁止真实付费 API 与网络。"""

from __future__ import annotations

from typing import Any

import pytest

from tool import execute
from tool.contract import Observation, RuntimeContext

# WGS84 赤道 1° 经度差（GeographicLib / pyproj.Geod）
_EQUATOR_1DEG_M = 111319.49079327357


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


def _distance(*, inputs: dict[str, Any], ctx: RuntimeContext | None = None) -> Observation:
    return execute(
        "distance_bearing_calculator",
        "distance",
        purpose="测距",
        inputs=inputs,
        ctx=ctx,
    )


def _bearing(*, inputs: dict[str, Any], ctx: RuntimeContext | None = None) -> Observation:
    return execute(
        "distance_bearing_calculator",
        "bearing",
        purpose="方位",
        inputs=inputs,
        ctx=ctx,
    )


def test_empty_inputs_are_missing_input() -> None:
    distance = _distance(inputs={})
    bearing = _bearing(inputs={})
    assert distance.ok is False
    assert distance.error_code == "missing_input"
    assert bearing.ok is False
    assert bearing.error_code == "missing_input"


def test_equator_one_degree_longitude_is_geodesic_meters() -> None:
    observation = _distance(inputs={"points": [[0.0, 0.0], [1.0, 0.0]]})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["method"] == "geodesic"
    assert observation.result["unit"] == "m"
    assert observation.result["ellipsoid"] == "WGS84"
    assert observation.result["crs"] == "wgs84"
    assert observation.result["value"] == pytest.approx(_EQUATOR_1DEG_M, rel=1e-8)
    assert observation.result["distance_m"] == pytest.approx(_EQUATOR_1DEG_M, rel=1e-8)
    assert "applied" in observation.result
    assert any("测地" in item for item in observation.result["assumptions"])
    assert "confirmed_location" not in _nested_keys(observation.result)


def test_unit_conversion_km_and_mile() -> None:
    km = _distance(inputs={"points": [[0.0, 0.0], [1.0, 0.0]], "unit": "km"})
    mile = _distance(inputs={"points": [[0.0, 0.0], [1.0, 0.0]], "unit": "mile"})
    assert km.ok is True and km.result is not None
    assert mile.ok is True and mile.result is not None
    assert km.result["unit"] == "km"
    assert km.result["value"] == pytest.approx(_EQUATOR_1DEG_M / 1000.0, rel=1e-8)
    assert km.result["distance_m"] == pytest.approx(_EQUATOR_1DEG_M, rel=1e-8)
    assert mile.result["unit"] == "mile"
    assert mile.result["value"] == pytest.approx(_EQUATOR_1DEG_M / 1609.344, rel=1e-8)


def test_multi_point_path_is_not_start_end_straight() -> None:
    path = _distance(inputs={"points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]})
    straight = _distance(inputs={"points": [[0.0, 0.0], [1.0, 1.0]]})
    assert path.ok is True and path.result is not None
    assert straight.ok is True and straight.result is not None
    assert path.result["method"] == "geodesic_path"
    assert "segments" in path.result
    assert len(path.result["segments"]) == 2
    assert path.result["distance_m"] > straight.result["distance_m"]
    assert any("相邻测地段" in item for item in path.result["assumptions"])


def test_place_name_is_rejected() -> None:
    observation = _distance(inputs={"points": "郑州黄河铁路桥"})
    assert observation.ok is False
    assert observation.error_code == "invalid_input"
    assert observation.error is not None
    assert "地名" in observation.error


def test_mode_route_is_unsupported() -> None:
    observation = _distance(
        inputs={"points": [[0.0, 0.0], [1.0, 0.0]], "mode": "route"},
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_mode"
    assert observation.error is not None
    assert "route_query" in observation.error


def test_geometry_linestring_matches_vertex_path() -> None:
    geometry = _distance(
        inputs={
            "mode": "geometry",
            "features": {
                "type": "LineString",
                "coordinates": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        }
    )
    path = _distance(inputs={"points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]})
    assert geometry.ok is True and geometry.result is not None
    assert path.ok is True and path.result is not None
    assert geometry.result["method"] == "geometry_length"
    assert geometry.result["distance_m"] == pytest.approx(path.result["distance_m"], rel=1e-8)


def test_width_two_points_is_nearest_geodesic() -> None:
    observation = _distance(
        inputs={"points": [[0.0, 0.0], [1.0, 0.0]], "mode": "width"},
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["method"] == "nearest_geodesic"
    assert observation.result["value"] == pytest.approx(_EQUATOR_1DEG_M, rel=1e-8)


def test_width_single_linestring_is_invalid() -> None:
    observation = _distance(
        inputs={
            "mode": "width",
            "features": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 0.0]]},
        }
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"


def test_width_polygon_uses_local_aeqd_mrr() -> None:
    observation = _distance(
        inputs={
            "mode": "width",
            "features": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [0.0, 0.0],
                        [0.01, 0.0],
                        [0.01, 0.002],
                        [0.0, 0.002],
                        [0.0, 0.0],
                    ]
                ],
            },
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["method"] == "local_aeqd_mrr_width"
    assert observation.result["unit"] == "m"
    width_m = float(observation.result["distance_m"])
    assert 180.0 < width_m < 260.0


def test_bearing_true_north_east_is_90() -> None:
    observation = _bearing(
        inputs={"origin": {"lon": 0.0, "lat": 0.0}, "target": {"lon": 1.0, "lat": 0.0}},
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["method"] == "geodesic"
    assert observation.result["reference"] == "true_north"
    assert observation.result["unit"] == "deg"
    assert observation.result["azimuth_deg"] == pytest.approx(90.0, abs=1e-6)
    assert observation.result["value"] == observation.result["azimuth_deg"]
    assert observation.result["back_azimuth_deg"] == pytest.approx(270.0, abs=1e-6)
    assert observation.result["compass"] == "E"
    assert observation.result["crs"] == "wgs84"


def test_bearing_magnetic_and_image_axis_are_unsupported() -> None:
    magnetic = _bearing(
        inputs={
            "origin": "0,0",
            "target": "1,0",
            "reference": "magnetic_north",
        }
    )
    image_axis = _bearing(
        inputs={
            "origin": "0,0",
            "target": "1,0",
            "reference": "image_axis",
        }
    )
    assert magnetic.ok is False
    assert magnetic.error_code == "unsupported_reference"
    assert image_axis.ok is False
    assert image_axis.error_code == "unsupported_reference"


def test_previous_tool_result_points_and_strips_forbidden() -> None:
    ctx = RuntimeContext(
        previous_tool_result={
            "operation": "geocode",
            "results": [
                {
                    "location": {"lon": 0.0, "lat": 0.0, "crs": "wgs84"},
                    "confirmed_location": "MUST NOT LEAK",
                },
                {
                    "location": {"lon": 1.0, "lat": 0.0, "crs": "wgs84"},
                    "raw_content": "MUST NOT LEAK",
                },
            ],
        }
    )
    observation = _distance(inputs={"points": "$previous_tool_result"}, ctx=ctx)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["value"] == pytest.approx(_EQUATOR_1DEG_M, rel=1e-8)
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys


def test_gcj02_points_are_converted_to_wgs84_before_geodesic() -> None:
    observation = _distance(
        inputs={
            "points": [
                {"lon": 113.67, "lat": 34.89, "crs": "gcj02"},
                {"lon": 113.68, "lat": 34.89, "crs": "gcj02"},
            ]
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["crs"] == "wgs84"
    assert observation.result["applied"]["points"][0]["crs"] == "wgs84"
    assert observation.result["applied"]["points"][0]["lon"] != 113.67
    assert any("GCJ-02" in item for item in observation.result["assumptions"])
    assert observation.result["distance_m"] > 0
