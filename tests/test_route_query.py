"""route_query 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from tool import execute
from tool._crs import format_gcj02_lonlat, gcj02_to_wgs84
from tool.contract import Observation, RuntimeContext
from tool.route_query import _route as route_mod
from tool.route_query._route import RouteRequest


class FakeRouteProvider:
    """测试替身：记录请求并返回高德或归一化载荷。"""

    name = "fake"
    crs = "gcj02"

    def __init__(self, payload: dict[str, Any] | None = None, *, name: str | None = None) -> None:
        self.payload = payload if payload is not None else _sample_amap_driving()
        self.calls: list[RouteRequest] = []
        if name is not None:
            self.name = name

    def route(self, request: RouteRequest) -> dict[str, Any]:
        self.calls.append(request)
        return self.payload


def _sample_amap_driving() -> dict[str, Any]:
    return {
        "status": "1",
        "route": {
            "paths": [
                {
                    "distance": "25000",
                    "duration": "1800",
                    "polyline": "113.670000,34.890000;113.720000,34.820000;113.800000,34.750000",
                    "steps": [
                        {
                            "instruction": "沿中州大道向南",
                            "road": "中州大道",
                            "distance": "14000",
                            "duration": "1000",
                            "polyline": "113.670000,34.890000;113.720000,34.820000",
                        },
                        {
                            "instruction": "沿航海路向西",
                            "road": "航海路",
                            "distance": "11000",
                            "duration": "800",
                            "polyline": "113.720000,34.820000;113.800000,34.750000",
                        },
                    ],
                    "confirmed_location": "MUST NOT LEAK",
                    "raw_content": "FULL ROUTE JSON MUST NOT LEAK",
                }
            ]
        },
    }


def _sample_injected() -> dict[str, Any]:
    return {
        "routes": [
            {
                "distance_m": 25000,
                "duration_s": 1800,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[113.67, 34.89], [113.80, 34.75]],
                    "crs": "gcj02",
                },
                "legs": [{"distance_m": 25000, "duration_s": 1800, "summary": "G107"}],
                "source": "fake",
                "confirmed_location": "MUST NOT LEAK",
            }
        ]
    }


def _sample_osrm() -> dict[str, Any]:
    return {
        "code": "Ok",
        "routes": [
            {
                "distance": 24880.4,
                "duration": 1760.2,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[113.67, 34.89], [113.80, 34.75]],
                },
                "legs": [{"distance": 24880.4, "duration": 1760.2, "summary": "G107"}],
                "raw_content": "MUST NOT LEAK",
            }
        ],
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


def _lookup(
    *,
    inputs: dict[str, object] | None = None,
    provider: FakeRouteProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {
        "origin": "113.670000,34.890000",
        "destination": "113.800000,34.750000",
    }
    if inputs:
        payload.update(inputs)
    runtime = ctx
    engine = provider if provider is not None else FakeRouteProvider()
    if runtime is None:
        runtime = RuntimeContext(extras={"route_query_provider": engine})
    elif provider is not None:
        runtime.extras["route_query_provider"] = engine
    return execute(
        "route_query",
        "route",
        purpose="查询路线",
        inputs=payload,
        ctx=runtime,
    )


class _FakeHttpResponse:
    def __init__(self, payload: Any) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_drive_returns_route_and_straight_line_distance() -> None:
    provider = FakeRouteProvider()
    observation = _lookup(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert provider.calls and provider.calls[0].travel_mode == "drive"
    route = observation.result["routes"][0]
    assert route["route_id"] == "route_1"
    assert route["distance_m"] == 25000.0
    assert route["duration_s"] == 1800.0
    assert "straight_line_distance_m" in route
    assert "distance" not in route
    assert route["straight_line_distance_m"] != route["distance_m"]
    assert route["geometry"]["type"] == "LineString"
    assert route["geometry"]["crs"] == "wgs84"
    lon, lat = gcj02_to_wgs84(113.67, 34.89)
    assert route["geometry"]["coordinates"][0] == [lon, lat]
    assert route["legs"][0]["summary"] == "中州大道"
    assert observation.result["applied"]["provider"] == "fake"
    assert observation.result["applied"]["travel_mode"] == "drive"
    dumped = str(observation.result)
    assert "MUST NOT LEAK" not in dumped
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys
    assert "confirmed_location" not in keys
    assert any("straight_line_distance_m" in item for item in observation.result["assumptions"])
    assert any("几十年前" in item for item in observation.result["assumptions"])


def test_missing_origin_is_missing_input() -> None:
    observation = execute(
        "route_query",
        "route",
        purpose="缺输入",
        inputs={"destination": "113.80,34.75"},
        ctx=RuntimeContext(extras={"route_query_provider": FakeRouteProvider()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_missing_destination_is_missing_input() -> None:
    observation = execute(
        "route_query",
        "route",
        purpose="缺输入",
        inputs={"origin": "113.67,34.89"},
        ctx=RuntimeContext(extras={"route_query_provider": FakeRouteProvider()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_place_name_origin_is_invalid_origin() -> None:
    observation = _lookup(inputs={"origin": "郑州站"})
    assert observation.ok is False
    assert observation.error_code == "invalid_origin"


def test_place_name_destination_is_invalid_destination() -> None:
    observation = _lookup(inputs={"destination": "开封站"})
    assert observation.ok is False
    assert observation.error_code == "invalid_destination"


def test_invalid_travel_mode_is_rejected() -> None:
    observation = _lookup(inputs={"travel_mode": "hover"})
    assert observation.ok is False
    assert observation.error_code == "invalid_travel_mode"


def test_lat_lon_when_second_exceeds_90() -> None:
    provider = FakeRouteProvider()
    observation = _lookup(
        inputs={"origin": "34.89,113.67", "destination": {"lat": 34.75, "lon": 113.80}},
        provider=provider,
    )
    assert observation.ok is True
    origin = provider.calls[0].origin
    assert origin.lat == 34.89
    assert origin.lon == 113.67


def test_waypoints_are_parsed() -> None:
    provider = FakeRouteProvider()
    observation = _lookup(
        inputs={"waypoints": [{"lon": 113.72, "lat": 34.82}]},
        provider=provider,
    )
    assert observation.ok is True
    assert len(provider.calls[0].waypoints) == 1
    assert provider.calls[0].waypoints[0].lon == 113.72
    applied = observation.result["applied"]["waypoints"]
    assert applied[0]["lon"] == 113.72


def test_osrm_transit_is_unsupported_travel_mode() -> None:
    provider = FakeRouteProvider(name="osrm")
    provider.crs = "wgs84"
    observation = _lookup(inputs={"travel_mode": "transit", "city": "郑州"}, provider=provider)
    assert observation.ok is False
    assert observation.error_code == "unsupported_travel_mode"
    assert provider.calls == []


def test_amap_walk_with_waypoints_is_unsupported() -> None:
    provider = FakeRouteProvider(name="amap")
    observation = _lookup(
        inputs={
            "travel_mode": "walk",
            "waypoints": [[113.72, 34.82]],
        },
        provider=provider,
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_waypoints"
    assert provider.calls == []


def test_amap_bike_with_waypoints_is_unsupported() -> None:
    provider = FakeRouteProvider(name="amap")
    observation = _lookup(
        inputs={"travel_mode": "bike", "waypoints": ["113.72,34.82"]},
        provider=provider,
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_waypoints"


def test_amap_transit_requires_city() -> None:
    provider = FakeRouteProvider(name="amap")
    observation = _lookup(inputs={"travel_mode": "transit"}, provider=provider)
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "city" in observation.error
    assert provider.calls == []


def test_amap_transit_uses_city_and_cityd() -> None:
    provider = FakeRouteProvider(name="amap")
    observation = _lookup(
        inputs={"travel_mode": "transit", "city": "郑州", "cityd": "开封"},
        provider=provider,
    )
    assert observation.ok is True
    assert provider.calls
    request = provider.calls[0]
    assert request.travel_mode == "transit"
    assert request.city == "郑州"
    assert request.cityd == "开封"
    assert observation.result is not None
    assert observation.result["applied"]["city"] == "郑州"
    assert observation.result["applied"]["cityd"] == "开封"


def test_any_travel_mode_is_drive() -> None:
    provider = FakeRouteProvider()
    observation = _lookup(inputs={"travel_mode": "any"}, provider=provider)
    assert observation.ok is True
    assert provider.calls[0].travel_mode == "drive"
    assert observation.result["applied"]["travel_mode"] == "drive"
    assert observation.result["applied"]["requested_travel_mode"] == "any"
    assert any("any" in item for item in observation.result["assumptions"])


def test_default_travel_mode_is_drive() -> None:
    provider = FakeRouteProvider()
    observation = _lookup(provider=provider)
    assert observation.ok is True
    assert provider.calls[0].travel_mode == "drive"
    assert any("未指定" in item for item in observation.result["assumptions"])


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "route_query",
        "route",
        purpose="闸门",
        inputs={"origin": "113.67,34.89", "destination": "113.80,34.75"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_allow_real_api_true_missing_key_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("AMAP_WEB_KEY", "")
    monkeypatch.setenv("AMAP_API_KEY", "")
    monkeypatch.setenv("ROUTE_QUERY_PROVIDER", "amap")
    observation = execute(
        "route_query",
        "route",
        purpose="缺钥匙",
        inputs={"origin": "113.67,34.89", "destination": "113.80,34.75"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "AMAP" in observation.error


def test_amap_driving_sends_key_and_lon_lat(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        parsed = urlparse(str(request.full_url))
        assert parsed.path.endswith("/driving")
        params = parse_qs(parsed.query)
        assert params["key"] == ["test-amap-key"]
        assert params["origin"] == [format_gcj02_lonlat(113.67, 34.89)]
        assert params["destination"] == [format_gcj02_lonlat(113.80, 34.75)]
        return _FakeHttpResponse(_sample_amap_driving())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("AMAP_WEB_KEY", "test-amap-key")
    monkeypatch.setenv("AMAP_API_KEY", "")
    monkeypatch.setenv("ROUTE_QUERY_PROVIDER", "amap")
    monkeypatch.setattr(route_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "route_query",
        "route",
        purpose="mock amap",
        inputs={"origin": "113.670000,34.890000", "destination": "113.800000,34.750000"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls, "应发起一次高德 HTTP 请求"
    assert observation.result["applied"]["provider"] == "amap"
    assert observation.result["routes"][0]["geometry"]["crs"] == "wgs84"
    assert any("已近似转换" in item for item in observation.result["assumptions"])
    assert any("训练" in item for item in observation.result["assumptions"])
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys
    assert "MUST NOT LEAK" not in str(observation.result)


def test_amap_driving_sends_waypoints(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        params = parse_qs(urlparse(str(request.full_url)).query)
        assert params["waypoints"] == [format_gcj02_lonlat(113.72, 34.82)]
        return _FakeHttpResponse(_sample_amap_driving())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("AMAP_WEB_KEY", "test-amap-key")
    monkeypatch.setenv("ROUTE_QUERY_PROVIDER", "amap")
    monkeypatch.setattr(route_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "route_query",
        "route",
        purpose="mock waypoints",
        inputs={
            "origin": "113.670000,34.890000",
            "destination": "113.800000,34.750000",
            "waypoints": ["113.720000,34.820000"],
        },
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls


def test_osrm_missing_endpoint_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("ROUTE_QUERY_PROVIDER", "osrm")
    monkeypatch.setenv("ROUTE_OSRM_ENDPOINT", "")
    observation = execute(
        "route_query",
        "route",
        purpose="缺端点",
        inputs={"origin": "113.67,34.89", "destination": "113.80,34.75"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ROUTE_OSRM_ENDPOINT" in observation.error


def test_public_osrm_is_rejected_by_default(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("ROUTE_QUERY_PROVIDER", "osrm")
    monkeypatch.setenv("ROUTE_OSRM_ENDPOINT", "https://router.project-osrm.org")
    monkeypatch.setenv("ROUTE_ALLOW_PUBLIC_OSRM", "")
    observation = execute(
        "route_query",
        "route",
        purpose="公共实例",
        inputs={"origin": "113.67,34.89", "destination": "113.80,34.75"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "公共 OSRM" in observation.error


def test_self_hosted_osrm_sends_profile_and_user_agent(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        url = str(request.full_url)
        assert "/route/v1/driving/" in url
        assert "113.670000,34.890000;113.800000,34.750000" in url
        params = parse_qs(urlparse(url).query)
        assert params["geometries"] == ["geojson"]
        assert params["overview"] == ["simplified"]
        assert params["steps"] == ["true"]
        assert request.get_header("User-agent") == "geoagent-dataset/1.0 (route_query; test)"
        return _FakeHttpResponse(_sample_osrm())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("ROUTE_QUERY_PROVIDER", "osrm")
    monkeypatch.setenv("ROUTE_OSRM_ENDPOINT", "http://127.0.0.1:5000")
    monkeypatch.setenv("ROUTE_OSRM_USER_AGENT", "geoagent-dataset/1.0 (route_query; test)")
    monkeypatch.setenv("ROUTE_ALLOW_PUBLIC_OSRM", "")
    monkeypatch.setattr(route_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "route_query",
        "route",
        purpose="mock osrm",
        inputs={"origin": "113.670000,34.890000", "destination": "113.800000,34.750000"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls
    route = observation.result["routes"][0]
    assert route["distance_m"] == 24880.4
    assert route["geometry"]["crs"] == "wgs84"
    assert observation.result["applied"]["provider"] == "osrm"
    assert any("WGS84" in item for item in observation.result["assumptions"])
    assert "MUST NOT LEAK" not in str(observation.result)


def test_injected_results_strip_forbidden_keys() -> None:
    observation = _lookup(provider=FakeRouteProvider(_sample_injected()))
    assert observation.ok is True
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert observation.result["routes"][0]["distance_m"] == 25000


def test_invalid_waypoints_is_rejected() -> None:
    observation = _lookup(inputs={"waypoints": "郑州站"})
    assert observation.ok is False
    assert observation.error_code == "invalid_waypoints"
