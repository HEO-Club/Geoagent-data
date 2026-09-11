"""weather_archive_query 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

import json
from datetime import timedelta
from io import BytesIO
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.weather_archive_query import _archive as archive_mod
from tool.weather_archive_query._archive import (
    LayerFetchRequest,
    SeriesRequest,
)


class FakeTimeseriesProvider:
    """测试替身：记录时序请求并返回 Open-Meteo 风格 JSON。"""

    name = "fake-open-meteo"

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload
        self.calls: list[SeriesRequest] = []

    def fetch_series(self, request: SeriesRequest) -> dict[str, Any]:
        self.calls.append(request)
        if self.payload is not None:
            return dict(self.payload)
        return _sample_series(request)


class FakeLayerProvider:
    """测试替身：记录图层请求并返回 PNG。"""

    name = "fake-gibs"

    def __init__(self, png: bytes | None = None) -> None:
        self.png = png if png is not None else _png_bytes()
        self.calls: list[LayerFetchRequest] = []

    def fetch_map(self, request: LayerFetchRequest) -> bytes:
        self.calls.append(request)
        return self.png


def _png_bytes(color: tuple[int, int, int] = (180, 190, 210)) -> bytes:
    image = Image.new("RGB", (8, 8), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _sample_series(request: SeriesRequest) -> dict[str, Any]:
    start = request.start
    end = request.end
    days = (end - start).days + 1
    iso = [(start + timedelta(days=offset)).isoformat() for offset in range(days)]
    daily: dict[str, Any] = {"time": iso}
    daily_units: dict[str, str] = {"time": "iso8601"}
    for name in request.daily:
        if name == "precipitation_sum":
            daily[name] = [0.0 if index % 2 else 3.2 for index in range(days)]
            daily_units[name] = "mm"
        elif name == "snowfall_sum":
            daily[name] = [0.0 if index % 3 else 1.5 for index in range(days)]
            daily_units[name] = "cm"
        elif name == "weather_code":
            daily[name] = [1 if index % 2 else 71 for index in range(days)]
            daily_units[name] = "wmo code"
        else:
            daily[name] = [2.0 + index for index in range(days)]
            daily_units[name] = "°C"
    hourly: dict[str, Any] = {"time": [f"{day}T00:00" for day in iso]}
    hourly_units: dict[str, str] = {"time": "iso8601"}
    for name in request.hourly:
        if name == "cloud_cover":
            hourly[name] = [80.0 if index % 2 else 12.0 for index in range(days)]
            hourly_units[name] = "%"
        elif name == "snow_depth":
            hourly[name] = [0.05 if index % 3 else 0.0 for index in range(days)]
            hourly_units[name] = "m"
        elif name == "snowfall":
            hourly[name] = [0.4 if index % 3 == 0 else 0.0 for index in range(days)]
            hourly_units[name] = "cm"
        elif name == "precipitation":
            hourly[name] = [1.0 if index % 2 else 0.0 for index in range(days)]
            hourly_units[name] = "mm"
        else:
            hourly[name] = [10.0 + index for index in range(days)]
            hourly_units[name] = "m"
    payload: dict[str, Any] = {
        "latitude": request.latitude,
        "longitude": request.longitude,
        "confirmed_location": "MUST NOT LEAK",
        "raw_content": "MUST NOT LEAK",
    }
    if request.daily:
        payload["daily"] = daily
        payload["daily_units"] = daily_units
    if request.hourly:
        payload["hourly"] = hourly
        payload["hourly_units"] = hourly_units
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


def _call(
    operation: str,
    *,
    inputs: dict[str, Any] | None = None,
    timeseries: FakeTimeseriesProvider | None = None,
    layers: FakeLayerProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, Any] = {
        "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
        "time_range": "2020-01-01/2020-01-03",
    }
    if inputs:
        payload.update(inputs)
    runtime = ctx if ctx is not None else RuntimeContext()
    series_engine = timeseries if timeseries is not None else FakeTimeseriesProvider()
    layer_engine = layers if layers is not None else FakeLayerProvider()
    runtime.extras.setdefault("weather_archive_timeseries_provider", series_engine)
    runtime.extras.setdefault("weather_archive_layer_provider", layer_engine)
    return execute(
        "weather_archive_query",
        operation,
        purpose="查询气象档案",
        inputs=payload,
        ctx=runtime,
    )


class _FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._body = payload

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_missing_area_is_missing_input() -> None:
    observation = _call("weather", inputs={"area": None})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_missing_time_range_is_missing_input() -> None:
    observation = _call("weather", inputs={"time_range": None})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_empty_inputs_are_missing_input() -> None:
    observation = execute(
        "weather_archive_query",
        "weather",
        purpose="scaffold",
        inputs={},
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_place_name_without_geometry_is_missing_input() -> None:
    observation = _call("weather", inputs={"area": "郑州市"})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "geocode" in observation.error


def test_weather_default_only_hits_timeseries() -> None:
    series = FakeTimeseriesProvider()
    layers = FakeLayerProvider()
    observation = _call("weather", timeseries=series, layers=layers)
    assert observation.ok is True
    assert observation.result is not None
    assert series.calls
    assert layers.calls == []
    block = observation.result["series"]
    assert block["provider"] == "open-meteo"
    assert block["data_source"] == "ERA5"
    assert block["data_type"] == "reanalysis_grid"
    assert block["evidence_kind"] == "time_series"
    assert "temperature_2m_mean" in block["values"]
    assert "precipitation_sum" in block["units"]
    assert observation.result["time_range"] == {"start": "2020-01-01", "end": "2020-01-03"}
    assert "layers" not in observation.result
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys
    assert any("再分析格点" in item for item in observation.result["assumptions"])


def test_cloud_cover_default_hits_both_backends() -> None:
    series = FakeTimeseriesProvider()
    layers = FakeLayerProvider()
    observation = _call("cloud_cover", timeseries=series, layers=layers)
    assert observation.ok is True
    assert observation.result is not None
    assert series.calls
    assert layers.calls
    assert observation.result["series"]["values"]["cloud_cover"]
    layer = observation.result["layers"][0]
    assert layer["provider"] == "nasa_gibs"
    assert layer["data_type"] == "remote_sensing_layer"
    assert layer["evidence_kind"] == "imagery"
    assert layer["layer"] == "MODIS_Terra_CorrectedReflectance_TrueColor"
    assert layer["image_id"]
    assert layer["variable_kind"] == "cloud_cover"
    assert "image_ids" in observation.artifacts
    assert layers.calls[0].time.isoformat() == "2020-01-01"


def test_open_meteo_provider_skips_gibs() -> None:
    series = FakeTimeseriesProvider()
    layers = FakeLayerProvider()
    observation = _call(
        "cloud_cover",
        inputs={"provider": "open-meteo"},
        timeseries=series,
        layers=layers,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert series.calls
    assert layers.calls == []
    assert "layers" not in observation.result


def test_snow_cover_keeps_depth_and_ndsi_separate() -> None:
    series = FakeTimeseriesProvider()
    layers = FakeLayerProvider()
    observation = _call("snow_cover", timeseries=series, layers=layers)
    assert observation.ok is True
    assert observation.result is not None
    values = observation.result["series"]["values"]
    kinds = {item["kind"] for item in observation.result["series"]["variables"]}
    assert "snow_depth" in values
    assert "snowfall_sum" in values
    assert "snow_depth" in kinds
    assert "snowfall" in kinds
    assert "snow_cover_fraction" not in values
    layer = observation.result["layers"][0]
    assert layer["layer"] == "MODIS_Terra_NDSI_Snow_Cover"
    assert layer["variable_kind"] == "snow_cover_fraction"
    assert any("积雪深度" in item for item in observation.result["assumptions"])


def test_refine_range_filters_dates_without_network() -> None:
    series = FakeTimeseriesProvider()
    layers = FakeLayerProvider()
    ctx = RuntimeContext()
    opened = _call("weather", timeseries=series, layers=layers, ctx=ctx)
    assert opened.ok is True
    assert opened.result is not None
    series_calls = len(series.calls)
    layer_calls = len(layers.calls)
    refined = execute(
        "weather_archive_query",
        "refine_range",
        purpose="筛出有降水的日期",
        inputs={
            "source_result": opened.session,
            "time_range": "2020-01-01/2020-01-03",
            "condition": "precipitation_sum > 0",
        },
        ctx=ctx,
    )
    assert refined.ok is True
    assert refined.result is not None
    assert len(series.calls) == series_calls
    assert len(layers.calls) == layer_calls
    assert refined.result["matched_dates"]
    assert all(
        value is None or value > 0
        for value in refined.result["series"]["values"]["precipitation_sum"]
    )
    assert any("不下" in item for item in refined.result["assumptions"])


def test_refine_range_no_match_is_ok_empty() -> None:
    series = FakeTimeseriesProvider()
    ctx = RuntimeContext()
    opened = _call("weather", timeseries=series, ctx=ctx)
    assert opened.result is not None
    refined = execute(
        "weather_archive_query",
        "refine_range",
        purpose="筛不到的条件",
        inputs={
            "source_result": "$previous_tool_result",
            "time_range": "2020-01-01/2020-01-03",
            "condition": {"variable": "precipitation_sum", "op": ">", "value": 1000},
        },
        ctx=ctx,
    )
    assert refined.ok is True
    assert refined.result is not None
    assert refined.result["matched_dates"] == []
    assert refined.result["series"]["times"] == []


def test_refine_range_rejects_natural_language_source() -> None:
    observation = execute(
        "weather_archive_query",
        "refine_range",
        purpose="假结果",
        inputs={
            "source_result": "上一次查到的下雪天",
            "time_range": "2020-01-01",
        },
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"


def test_unknown_provider_is_unsupported() -> None:
    observation = _call("weather", inputs={"provider": "noaa-stations"})
    assert observation.ok is False
    assert observation.error_code == "unsupported_provider"


def test_allow_real_api_false_without_injection_is_unavailable() -> None:
    observation = execute(
        "weather_archive_query",
        "weather",
        purpose="无注入",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "time_range": "2020-01-01",
        },
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"


def test_openmeteo_http_query_shape(monkeypatch: Any) -> None:
    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(str(request.full_url))
        body = {
            "daily": {
                "time": ["2020-01-01"],
                "temperature_2m_mean": [1.2],
                "precipitation_sum": [0.0],
                "weather_code": [1],
            },
            "daily_units": {
                "time": "iso8601",
                "temperature_2m_mean": "°C",
                "precipitation_sum": "mm",
                "weather_code": "wmo code",
            },
        }
        return _FakeHttpResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setattr(archive_mod.urllib.request, "urlopen", fake_urlopen)
    observation = execute(
        "weather_archive_query",
        "weather",
        purpose="组装 Open-Meteo",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "time_range": "2020-01-01",
            "provider": "open-meteo",
        },
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls
    parsed = urlparse(calls[0])
    params = parse_qs(parsed.query)
    assert "archive-api.open-meteo.com" in parsed.netloc
    assert "latitude" in params
    assert "longitude" in params
    assert params["start_date"] == ["2020-01-01"]
    assert "temperature_2m_mean" in params["daily"][0]


def test_gibs_http_query_shape(monkeypatch: Any) -> None:
    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        url = str(request.full_url)
        calls.append(url)
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if params.get("REQUEST") == ["GetMap"]:
            return _FakeHttpResponse(_png_bytes())
        body = {
            "hourly": {"time": ["2020-01-01T00:00"], "cloud_cover": [40.0]},
            "hourly_units": {"time": "iso8601", "cloud_cover": "%"},
        }
        return _FakeHttpResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setattr(archive_mod.urllib.request, "urlopen", fake_urlopen)
    observation = execute(
        "weather_archive_query",
        "cloud_cover",
        purpose="组装 GIBS",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "time_range": "2020-01-01",
        },
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    gibs = [url for url in calls if "gibs.earthdata.nasa.gov" in url]
    assert gibs
    params = parse_qs(urlparse(gibs[0]).query)
    assert params["REQUEST"] == ["GetMap"]
    assert params["LAYERS"] == ["MODIS_Terra_CorrectedReflectance_TrueColor"]
    assert params["TIME"] == ["2020-01-01"]
    assert params["CRS"] == ["EPSG:4326"]
    south, west, north, east = params["BBOX"][0].split(",")
    assert float(south) < float(north)
    assert float(west) < float(east)
