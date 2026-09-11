"""shadow_analysis.shadow_model 正向阴影测试；禁止真实付费 API 与网络。"""

from __future__ import annotations

import math
from typing import Any

import pytest

from tool import execute
from tool.contract import Observation, RuntimeContext

_MID_LAT = {"lat": 40.0, "lon": 116.0}
_HEIGHT_M = 10.0


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


def _shadow(
    *,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    return execute(
        "shadow_analysis",
        "shadow_model",
        purpose="算阴影",
        inputs=inputs,
        ctx=ctx,
    )


def _horizontal_length(height_m: float, elevation_deg: float) -> float:
    return height_m / math.tan(math.radians(elevation_deg))


def test_empty_inputs_are_missing_input() -> None:
    observation = _shadow(inputs={})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_missing_datetime_is_missing_input() -> None:
    observation = _shadow(inputs={"area": _MID_LAT})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "datetime" in observation.error


def test_place_name_area_is_missing_input() -> None:
    observation = _shadow(
        inputs={
            "area": "郑州附近黄河沿线",
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
        }
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "geocode" in observation.error


def test_mid_latitude_summer_noon_shadow_points_north() -> None:
    observation = _shadow(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
            "object_height_m": _HEIGHT_M,
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    result = observation.result
    assert result["method"] == "vertical_object_plane_intersection"
    assert result["library"] == "pvlib"
    assert result["status"] == "computed_under_inputs"
    assert result["surface"]["mode"] == "horizontal"
    elevation = result["sun"]["apparent_elevation_deg"]
    assert elevation > 20.0
    expected = _horizontal_length(_HEIGHT_M, elevation)
    assert result["shadow_length_m"] == pytest.approx(expected, rel=1e-6)
    azimuth = result["shadow_azimuth_deg"]
    assert azimuth < 20.0 or azimuth > 340.0
    assert result["offset"]["north_m"] > 0.0
    assert "hypothesis" not in result
    joined = "".join(result["assumptions"])
    assert "水平" in joined
    assert "NREL SPA" in joined
    assert "confirmed_location" not in _nested_keys(result)


def test_missing_height_is_direction_only() -> None:
    observation = _shadow(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    result = observation.result
    assert result["status"] == "direction_only"
    assert result["shadow_length_m"] is None
    assert result["object_height_m"] is None
    assert "offset" not in result
    azimuth = result["shadow_azimuth_deg"]
    assert azimuth < 20.0 or azimuth > 340.0
    joined = "".join(result["assumptions"])
    assert "未编造阴影长度" in joined


def test_zero_height_has_zero_length() -> None:
    observation = _shadow(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
            "object_height_m": 0,
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["status"] == "computed_under_inputs"
    assert observation.result["shadow_length_m"] == 0.0
    assert observation.result["offset"] == {"east_m": 0.0, "north_m": 0.0}


def test_night_is_sun_below_horizon() -> None:
    observation = _shadow(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T00:00:00",
            "timezone": "Asia/Shanghai",
            "object_height_m": _HEIGHT_M,
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    result = observation.result
    assert result["status"] == "sun_below_horizon"
    assert result["shadow_length_m"] is None
    assert "offset" not in result
    assert result["sun"]["apparent_elevation_deg"] <= 0.0
    joined = "".join(result["assumptions"])
    assert "地平线" in joined


def test_inclined_surface_changes_length() -> None:
    shared = {
        "area": _MID_LAT,
        "datetime": "2024-06-21T12:00:00",
        "timezone": "Asia/Shanghai",
        "object_height_m": _HEIGHT_M,
    }
    flat = _shadow(inputs=shared)
    slope = _shadow(
        inputs={
            **shared,
            "surface": {"slope_deg": 20.0, "aspect_deg": 180.0},
        }
    )
    assert flat.ok is True and flat.result is not None
    assert slope.ok is True and slope.result is not None
    assert slope.result["surface"]["mode"] == "inclined"
    assert slope.result["shadow_length_m"] != pytest.approx(
        flat.result["shadow_length_m"], rel=1e-4
    )
    assert slope.result["shadow_length_m"] is not None
    assert slope.result["shadow_length_m"] < flat.result["shadow_length_m"]


def test_previous_tool_result_extracts_geocode_location() -> None:
    ctx = RuntimeContext(
        previous_tool_result={
            "ok": True,
            "result": {
                "results": [
                    {
                        "location": {"lat": 34.75, "lon": 113.67, "crs": "wgs84"},
                        "confirmed_location": "MUST NOT LEAK",
                    }
                ]
            },
        }
    )
    observation = _shadow(
        inputs={
            "area": "$previous_tool_result",
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
            "object_height_m": _HEIGHT_M,
            "confirmed_location": "MUST NOT LEAK",
        },
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["lat"] == pytest.approx(34.75)
    assert observation.result["applied"]["lon"] == pytest.approx(113.67)
    assert "confirmed_location" not in _nested_keys(observation.result)


def test_bbox_uses_centroid_assumption() -> None:
    observation = _shadow(
        inputs={
            "area": {"west": 115.0, "south": 39.0, "east": 117.0, "north": 41.0},
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
            "object_height_m": _HEIGHT_M,
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["lon"] == pytest.approx(116.0)
    assert observation.result["applied"]["lat"] == pytest.approx(40.0)
    assert observation.result["applied"]["location_mode"] == "centroid"
    assert any("质心" in item for item in observation.result["assumptions"])


def test_extra_timezone_matches_sun_position_split() -> None:
    shanghai = _shadow(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
            "object_height_m": _HEIGHT_M,
        }
    )
    utc = _shadow(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "UTC",
            "object_height_m": _HEIGHT_M,
        }
    )
    zulu = _shadow(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00Z",
            "timezone": "Asia/Shanghai",
            "object_height_m": _HEIGHT_M,
        }
    )
    assert shanghai.ok is True and shanghai.result is not None
    assert utc.ok is True and utc.result is not None
    assert zulu.ok is True and zulu.result is not None
    assert shanghai.result["sun"]["apparent_elevation_deg"] != pytest.approx(
        utc.result["sun"]["apparent_elevation_deg"], abs=0.5
    )
    assert shanghai.result["applied"]["timezone"] == "Asia/Shanghai"
    assert utc.result["applied"]["timezone"] == "UTC"
    assert shanghai.result["applied"]["datetime_utc"] != utc.result["applied"]["datetime_utc"]
    assert zulu.result["applied"]["datetime_utc"].startswith("2024-06-21T12:00:00")
    assert shanghai.result["shadow_length_m"] != pytest.approx(
        utc.result["shadow_length_m"], rel=1e-3
    )


def test_negative_height_is_invalid() -> None:
    observation = _shadow(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
            "object_height_m": -1,
        }
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"


def test_slope_without_aspect_is_missing_input() -> None:
    observation = _shadow(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
            "object_height_m": _HEIGHT_M,
            "surface": {"slope_deg": 15.0},
        }
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_dem_surface_is_invalid() -> None:
    observation = _shadow(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
            "object_height_m": _HEIGHT_M,
            "surface": {"dem": "local.tif", "path": "terrain.tif"},
        }
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"
