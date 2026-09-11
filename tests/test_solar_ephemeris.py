"""solar_ephemeris 本地 pvlib 历算测试；禁止真实付费 API 与网络。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from tool import execute
from tool.contract import Observation, RuntimeContext

_EQUATOR = {"lat": 0.0, "lon": 0.0}
_MID_LAT = {"lat": 40.0, "lon": 116.0}


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


def _sun_position(
    *,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    return execute(
        "solar_ephemeris",
        "sun_position",
        purpose="算太阳位置",
        inputs=inputs,
        ctx=ctx,
    )


def _sunset(
    *,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    return execute(
        "solar_ephemeris",
        "sunset_time",
        purpose="算日落",
        inputs=inputs,
        ctx=ctx,
    )


def _hour(iso_text: str | None) -> float:
    assert iso_text is not None
    parsed = datetime.fromisoformat(iso_text)
    return parsed.hour + parsed.minute / 60.0 + parsed.second / 3600.0


def test_empty_inputs_are_missing_input() -> None:
    position = _sun_position(inputs={})
    sunset = _sunset(inputs={})
    assert position.ok is False
    assert position.error_code == "missing_input"
    assert sunset.ok is False
    assert sunset.error_code == "missing_input"


def test_missing_datetime_is_missing_input() -> None:
    observation = _sun_position(inputs={"area": _EQUATOR})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "datetime" in observation.error


def test_date_only_datetime_is_missing_input() -> None:
    observation = _sun_position(
        inputs={"area": _EQUATOR, "datetime": "2024-03-20", "timezone": "UTC"}
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "时刻" in observation.error


def test_place_name_area_is_missing_input() -> None:
    observation = _sun_position(
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


def test_place_name_locations_is_missing_input() -> None:
    observation = _sunset(inputs={"locations": "昆明", "time_range": "2024-06-21"})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "geocode" in observation.error


def test_ambiguous_timezone_abbreviation_is_invalid() -> None:
    observation = _sun_position(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "CST",
        }
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_timezone"


def test_year_only_time_range_is_missing_input() -> None:
    observation = _sunset(inputs={"locations": _MID_LAT, "time_range": "2024"})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "过宽" in observation.error


def test_wide_year_range_is_missing_input() -> None:
    observation = _sunset(inputs={"locations": _MID_LAT, "time_range": "1980-1995"})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_equator_equinox_noon_elevation_is_near_zenith() -> None:
    observation = _sun_position(
        inputs={
            "area": _EQUATOR,
            "datetime": "2024-03-20T12:00:00Z",
            "timezone": "UTC",
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["method"] == "nrel_spa"
    assert observation.result["library"] == "pvlib"
    assert observation.result["status"] == "computed_under_inputs"
    assert 80.0 < observation.result["elevation_deg"] < 90.0
    assert observation.result["azimuth_deg"] == pytest.approx(
        observation.result["azimuth_deg"] % 360.0
    )
    assert "hypothesis" not in observation.result
    assert any("NREL SPA" in item for item in observation.result["assumptions"])
    assert "confirmed_location" not in _nested_keys(observation.result)


def test_same_wall_clock_shanghai_and_utc_differ() -> None:
    shanghai = _sun_position(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
        }
    )
    utc = _sun_position(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "UTC",
        }
    )
    assert shanghai.ok is True and shanghai.result is not None
    assert utc.ok is True and utc.result is not None
    assert shanghai.result["elevation_deg"] != pytest.approx(utc.result["elevation_deg"], abs=0.5)
    assert shanghai.result["applied"]["timezone"] == "Asia/Shanghai"
    assert utc.result["applied"]["timezone"] == "UTC"
    assert shanghai.result["applied"]["datetime_utc"] != utc.result["applied"]["datetime_utc"]


def test_naive_local_and_zulu_same_digits_differ() -> None:
    naive = _sun_position(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
        }
    )
    zulu = _sun_position(
        inputs={
            "area": _MID_LAT,
            "datetime": "2024-06-21T12:00:00Z",
            "timezone": "Asia/Shanghai",
        }
    )
    assert naive.ok is True and naive.result is not None
    assert zulu.ok is True and zulu.result is not None
    assert naive.result["elevation_deg"] != pytest.approx(zulu.result["elevation_deg"], abs=0.5)
    assert naive.result["applied"]["datetime_utc"].endswith("+00:00")
    assert zulu.result["applied"]["datetime_utc"].startswith("2024-06-21T12:00:00")
    assert naive.result["applied"]["datetime_local"].startswith("2024-06-21T12:00:00")
    assert "Z" not in naive.result["applied"]["datetime_local"]
    joined = "".join(naive.result["assumptions"])
    assert "UTC" in joined


def test_june_sunset_later_than_december_at_mid_latitude() -> None:
    june = _sunset(
        inputs={
            "locations": _MID_LAT,
            "time_range": "2024-06-21",
            "timezone": "Asia/Shanghai",
        }
    )
    december = _sunset(
        inputs={
            "locations": _MID_LAT,
            "time_range": "2024-12-21",
            "timezone": "Asia/Shanghai",
        }
    )
    assert june.ok is True and june.result is not None
    assert december.ok is True and december.result is not None
    june_set = june.result["results"][0]["sunset"]
    dec_set = december.result["results"][0]["sunset"]
    assert _hour(june_set) > _hour(dec_set)
    assert june.result["results"][0]["civil_dusk"] is not None
    assert _hour(june.result["results"][0]["civil_dusk"]) >= _hour(june_set)


def test_multiple_locations_are_hypothesis_conditions() -> None:
    observation = _sunset(
        inputs={
            "locations": [
                {"lat": 40.0, "lon": 116.0, "name": "候选北"},
                {"lat": 25.0, "lon": 102.0, "name": "候选南"},
            ],
            "time_range": "2024-06-21",
            "timezone": "Asia/Shanghai",
            "confirmed_location": "MUST NOT LEAK",
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["hypothesis"] is True
    assert len(observation.result["results"]) == 2
    labels = {item["location"].get("label") for item in observation.result["results"]}
    assert labels == {"候选北", "候选南"}
    assert all(item["hypothesis"] is True for item in observation.result["results"])
    assert all(item["status"] == "computed_under_inputs" for item in observation.result["results"])
    assert "confirmed_location" not in _nested_keys(observation.result)
    joined = "".join(observation.result["assumptions"])
    assert "不表示该地点或拍摄时间已被确认" in joined


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
    observation = _sun_position(
        inputs={
            "area": "$previous_tool_result",
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
        },
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["lat"] == pytest.approx(34.75)
    assert observation.result["applied"]["lon"] == pytest.approx(113.67)
    assert "confirmed_location" not in _nested_keys(observation.result)


def test_bbox_uses_centroid_assumption() -> None:
    observation = _sun_position(
        inputs={
            "area": {"west": 115.0, "south": 39.0, "east": 117.0, "north": 41.0},
            "datetime": "2024-06-21T12:00:00",
            "timezone": "Asia/Shanghai",
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["lon"] == pytest.approx(116.0)
    assert observation.result["applied"]["lat"] == pytest.approx(40.0)
    assert observation.result["applied"]["location_mode"] == "centroid"
    assert any("质心" in item for item in observation.result["assumptions"])
