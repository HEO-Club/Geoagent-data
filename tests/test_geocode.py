"""geocode 执行器测试；所有网络请求均使用注入客户端。"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from tool import execute
from tool._crs import format_gcj02_lonlat, gcj02_to_wgs84
from tool.contract import Observation, RuntimeContext
from tool.geocode import _geocode as geocode_mod
from tool.geocode._geocode import GeocodeRequest
from tool.runtime import InMemoryResultStore


class FakeGeocodeClient:
    name = "fake_geocoder"

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.reverse_calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        limit: int,
        language: str | None,
        viewbox: tuple[float, float, float, float] | None,
    ) -> list[dict[str, Any]]:
        self.search_calls.append(
            {"query": query, "limit": limit, "language": language, "viewbox": viewbox}
        )
        return [
            {
                "place_id": 1,
                "osm_type": "relation",
                "osm_id": 123,
                "lat": "34.951",
                "lon": "113.501",
                "display_name": "郑州黄河文化公园, 惠济区, 郑州市",
                "class": "leisure",
                "type": "park",
                "boundingbox": ["34.94", "34.96", "113.49", "113.52"],
                "address": {"city": "郑州市"},
            }
        ]

    def reverse(self, lat: float, lon: float, *, language: str | None) -> dict[str, Any]:
        self.reverse_calls.append({"lat": lat, "lon": lon, "language": language})
        return {
            "osm_type": "node",
            "osm_id": 456,
            "lat": str(lat),
            "lon": str(lon),
            "display_name": "测试坐标",
            "type": "address",
            "address": {"country": "中国"},
        }


def _ctx(client: FakeGeocodeClient) -> RuntimeContext:
    return RuntimeContext(
        extras={"geocode_client": client},
        result_store=InMemoryResultStore(),
    )


def test_forward_geocode_returns_candidates_and_result_id() -> None:
    client = FakeGeocodeClient()
    ctx = _ctx(client)
    observation = execute(
        "geocode",
        "geocode",
        purpose="解析公园坐标",
        inputs={
            "query": "郑州黄河文化公园",
            "area": "郑州市",
            "language": "zh",
            "top_k": 3,
        },
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert client.search_calls == [
        {
            "query": "郑州黄河文化公园, 郑州市",
            "limit": 3,
            "language": "zh",
            "viewbox": None,
        }
    ]
    candidate = observation.result["candidates"][0]
    assert candidate["latitude"] == 34.951
    assert candidate["longitude"] == 113.501
    assert candidate["bbox"] == [113.49, 34.94, 113.52, 34.96]
    assert observation.result["crs"] == "EPSG:4326"
    assert observation.result["result_id"].startswith("geocode_")
    stored = ctx.result_store.get(observation.result["result_id"])
    assert stored["candidate_count"] == 1


def test_reverse_geocode_auto_detects_coordinates() -> None:
    client = FakeGeocodeClient()
    observation = execute(
        "geocode",
        "geocode",
        purpose="反查坐标",
        inputs={"query": [34.95, 113.5], "direction": "auto"},
        ctx=_ctx(client),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["direction"] == "reverse"
    assert client.reverse_calls[0]["lat"] == 34.95
    assert client.reverse_calls[0]["lon"] == 113.5


def test_geocode_validates_before_network_gate() -> None:
    missing = execute("geocode", "geocode", purpose="缺参数", inputs={})
    assert missing.ok is False
    assert missing.error_code == "missing_input"

    invalid = execute(
        "geocode",
        "geocode",
        purpose="非法坐标",
        inputs={"query": [95, 200], "direction": "reverse"},
        ctx=_ctx(FakeGeocodeClient()),
    )
    assert invalid.ok is False
    assert invalid.error_code == "invalid_coordinates"

    gated = execute(
        "geocode",
        "geocode",
        purpose="禁止联网",
        inputs={"query": "郑州市"},
    )
    assert gated.ok is False
    assert gated.error_code == "engine_unavailable"


def test_forward_geocode_limits_batch_size() -> None:
    observation = execute(
        "geocode",
        "geocode",
        purpose="过多查询",
        inputs={"query": [f"place-{index}" for index in range(6)]},
        ctx=_ctx(FakeGeocodeClient()),
    )
    assert observation.ok is False
    assert observation.error_code == "too_many_queries"


def test_forward_geocode_uses_bbox_as_bounded_viewbox() -> None:
    client = FakeGeocodeClient()
    observation = execute(
        "geocode",
        "geocode",
        purpose="限定候选范围",
        inputs={
            "query": "文化公园",
            "area": {"name": "郑州市", "bbox": [113.4, 34.9, 113.6, 35.0]},
        },
        ctx=_ctx(client),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert client.search_calls[0]["query"] == "文化公园, 郑州市"
    assert client.search_calls[0]["viewbox"] == (113.4, 34.9, 113.6, 35.0)
    assert observation.result["viewbox"] == [113.4, 34.9, 113.6, 35.0]


# --- Amap / gated executor coverage ---

class FakeGeocodeProvider:
    """测试替身：记录请求并返回高德或归一化载荷。"""

    name = "fake"
    crs = "gcj02"

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else _sample_geocodes()
        self.calls: list[GeocodeRequest] = []

    def lookup(self, request: GeocodeRequest) -> dict[str, Any]:
        self.calls.append(request)
        return self.payload


def _sample_geocodes() -> dict[str, Any]:
    return {
        "status": "1",
        "geocodes": [
            {
                "formatted_address": "河南省郑州市惠济区黄河铁路桥",
                "district": "惠济区",
                "location": "113.670000,34.890000",
                "level": "村庄",
                "adcode": "410103",
                "confirmed_location": "MUST NOT LEAK",
                "raw_content": "FULL GEOCODE JSON MUST NOT LEAK",
            },
            {
                "formatted_address": "河南省郑州市惠济区花园口",
                "location": [],
                "level": "乡镇",
            },
        ],
    }


def _sample_regeocode() -> dict[str, Any]:
    return {
        "status": "1",
        "regeocode": {
            "formatted_address": "河南省郑州市惠济区花园口镇",
            "addressComponent": {
                "country": "中国",
                "province": "河南省",
                "city": "郑州市",
                "district": "惠济区",
                "township": "花园口镇",
                "adcode": "410103",
                "streetNumber": {"street": [], "number": []},
            },
            "raw_content": "MUST NOT LEAK",
        },
    }


def _sample_injected() -> dict[str, Any]:
    return {
        "results": [
            {
                "formatted_address": "河南省郑州市惠济区黄河铁路桥",
                "name": "黄河铁路桥",
                "match_level": "兴趣点",
                "location": {"lon": 113.67, "lat": 34.89, "crs": "gcj02"},
                "provider_id": "410103",
                "source": "fake",
                "confirmed_location": "MUST NOT LEAK",
            }
        ]
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
    provider: FakeGeocodeProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {"query": "郑州黄河铁路桥"}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    engine = provider if provider is not None else FakeGeocodeProvider()
    if runtime is None:
        runtime = RuntimeContext(extras={"geocode_provider": engine})
    elif provider is not None:
        runtime.extras["geocode_provider"] = engine
    return execute(
        "geocode",
        "geocode",
        purpose="地理编码",
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


def test_forward_returns_address_match_level_and_wgs84() -> None:
    provider = FakeGeocodeProvider()
    observation = _lookup(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert provider.calls and provider.calls[0].direction == "forward"
    assert provider.calls[0].query == "郑州黄河铁路桥"
    results = observation.result["results"]
    assert results[0]["result_id"] == "geocode_1"
    assert results[0]["formatted_address"] == "河南省郑州市惠济区黄河铁路桥"
    assert results[0]["match_level"] == "村庄"
    lon, lat = gcj02_to_wgs84(113.67, 34.89)
    assert results[0]["location"]["lon"] == lon
    assert results[0]["location"]["lat"] == lat
    assert results[0]["location"]["crs"] == "wgs84"
    assert results[0]["location"]["lon"] != 113.67
    assert "location" not in results[1]
    assert observation.result["applied"]["provider"] == "fake"
    assert observation.result["applied"]["crs"] == "wgs84"
    assert observation.result["applied"]["direction"] == "forward"
    dumped = str(observation.result)
    assert "MUST NOT LEAK" not in dumped
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys
    assert "confirmed_location" not in keys
    assert any("拍摄点" in item for item in observation.result["assumptions"])
    assert any("村庄" in item or "行政区" in item for item in observation.result["assumptions"])


def test_auto_coordinate_query_is_reverse() -> None:
    provider = FakeGeocodeProvider(_sample_regeocode())
    observation = _lookup(inputs={"query": "113.670000,34.890000"}, provider=provider)
    assert observation.ok is True
    assert provider.calls[0].direction == "reverse"
    assert provider.calls[0].location is not None
    assert provider.calls[0].location.lon == 113.67
    assert provider.calls[0].location.lat == 34.89
    results = observation.result["results"]
    assert results[0]["formatted_address"] == "河南省郑州市惠济区花园口镇"
    assert results[0]["match_level"] == "乡镇"
    assert results[0]["location"]["crs"] == "wgs84"
    assert results[0]["location"]["lon"] == 113.67
    assert results[0]["location"]["lat"] == 34.89
    assert observation.result["applied"]["direction"] == "reverse"


def test_lat_lon_when_second_exceeds_90() -> None:
    provider = FakeGeocodeProvider(_sample_regeocode())
    observation = _lookup(inputs={"query": "34.89,113.67"}, provider=provider)
    assert observation.ok is True
    point = provider.calls[0].location
    assert point is not None
    assert point.lat == 34.89
    assert point.lon == 113.67


def test_area_text_becomes_amap_city() -> None:
    provider = FakeGeocodeProvider()
    observation = _lookup(inputs={"area": "郑州市"}, provider=provider)
    assert observation.ok is True
    assert provider.calls[0].city == "郑州市"
    assert observation.result["applied"]["city"] == "郑州市"
    assert observation.result["applied"]["area"] == "郑州市"


def test_active_area_context_default() -> None:
    provider = FakeGeocodeProvider()
    observation = _lookup(
        inputs={"area": "$active_area"},
        provider=provider,
        ctx=RuntimeContext(active_area="郑州市", extras={}),
    )
    assert observation.ok is True
    assert provider.calls[0].city == "郑州市"


def test_amap_bbox_is_unsupported() -> None:
    provider = FakeGeocodeProvider()
    observation = _lookup(
        inputs={"area": [113.5, 34.7, 113.8, 35.0]},
        provider=provider,
    )
    assert observation.ok is True
    assert provider.calls[0].city is None
    assert provider.calls[0].bbox is None
    unsupported = observation.result["applied"]["unsupported"]
    assert "bbox" in unsupported


def test_center_radius_is_unsupported_not_photo_point() -> None:
    provider = FakeGeocodeProvider()
    observation = _lookup(
        inputs={"area": {"lat": 34.75, "lon": 113.65, "radius_m": 2000}},
        provider=provider,
    )
    assert observation.ok is True
    assert provider.calls[0].location is None
    unsupported = observation.result["applied"]["unsupported"]
    assert unsupported["center"] == {"lat": 34.75, "lon": 113.65}
    assert unsupported["radius_m"] == 2000


def test_query_array_is_joined_with_space() -> None:
    provider = FakeGeocodeProvider()
    observation = _lookup(inputs={"query": ["郑州", "黄河铁路桥"]}, provider=provider)
    assert observation.ok is True
    assert provider.calls[0].query == "郑州 黄河铁路桥"


def test_missing_query_is_missing_input() -> None:
    observation = execute(
        "geocode",
        "geocode",
        purpose="缺输入",
        inputs={},
        ctx=RuntimeContext(extras={"geocode_provider": FakeGeocodeProvider()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_invalid_direction_is_rejected() -> None:
    observation = _lookup(inputs={"direction": "sideways"})
    assert observation.ok is False
    assert observation.error_code == "invalid_direction"


def test_explicit_reverse_without_coords_is_invalid_query() -> None:
    observation = _lookup(inputs={"query": "郑州黄河铁路桥", "direction": "reverse"})
    assert observation.ok is False
    assert observation.error_code == "invalid_query"


def test_explicit_forward_keeps_coordinate_string() -> None:
    provider = FakeGeocodeProvider()
    observation = _lookup(
        inputs={"query": "113.67,34.89", "direction": "forward"},
        provider=provider,
    )
    assert observation.ok is True
    assert provider.calls[0].direction == "forward"
    assert provider.calls[0].location is None


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "geocode",
        "geocode",
        purpose="闸门",
        inputs={"query": "郑州"},
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
    monkeypatch.setenv("GEOCODE_PROVIDER", "amap")
    observation = execute(
        "geocode",
        "geocode",
        purpose="缺钥匙",
        inputs={"query": "郑州"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "AMAP" in observation.error


def test_amap_forward_sends_key_address_and_city(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        parsed = urlparse(str(request.full_url))
        assert parsed.path.endswith("/geo")
        params = parse_qs(parsed.query)
        assert params["key"] == ["test-amap-key"]
        assert params["address"] == ["郑州黄河铁路桥"]
        assert params["city"] == ["郑州市"]
        return _FakeHttpResponse(_sample_geocodes())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("AMAP_WEB_KEY", "test-amap-key")
    monkeypatch.setenv("AMAP_API_KEY", "")
    monkeypatch.setenv("GEOCODE_PROVIDER", "amap")
    monkeypatch.setattr(geocode_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "geocode",
        "geocode",
        purpose="mock amap",
        inputs={"query": "郑州黄河铁路桥", "area": "郑州市", "direction": "forward"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls, "应发起一次高德 HTTP 请求"
    assert observation.result["applied"]["provider"] == "amap"
    assert observation.result["results"][0]["location"]["crs"] == "wgs84"
    assert any("已近似转换" in item for item in observation.result["assumptions"])
    assert any("训练" in item for item in observation.result["assumptions"])
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys
    assert "MUST NOT LEAK" not in str(observation.result)


def test_amap_reverse_sends_location(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        parsed = urlparse(str(request.full_url))
        assert parsed.path.endswith("/regeo")
        params = parse_qs(parsed.query)
        assert params["key"] == ["test-amap-key"]
        assert params["location"] == [format_gcj02_lonlat(113.67, 34.89)]
        return _FakeHttpResponse(_sample_regeocode())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("AMAP_WEB_KEY", "test-amap-key")
    monkeypatch.setenv("GEOCODE_PROVIDER", "amap")
    monkeypatch.setattr(geocode_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "geocode",
        "geocode",
        purpose="mock regeo",
        inputs={"query": "113.670000,34.890000", "direction": "reverse"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls
    assert observation.result["results"][0]["match_level"] == "乡镇"


def test_nominatim_missing_endpoint_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("GEOCODE_PROVIDER", "nominatim")
    monkeypatch.setenv("GEOCODE_NOMINATIM_ENDPOINT", "")
    observation = execute(
        "geocode",
        "geocode",
        purpose="缺端点",
        inputs={"query": "Zhengzhou"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "GEOCODE_NOMINATIM_ENDPOINT" in observation.error


def test_public_nominatim_is_rejected_by_default(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("GEOCODE_PROVIDER", "nominatim")
    monkeypatch.setenv("GEOCODE_NOMINATIM_ENDPOINT", "https://nominatim.openstreetmap.org")
    monkeypatch.setenv("GEOCODE_ALLOW_PUBLIC_NOMINATIM", "")
    observation = execute(
        "geocode",
        "geocode",
        purpose="公共实例",
        inputs={"query": "Zhengzhou"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "公共 Nominatim" in observation.error


def test_self_hosted_nominatim_sends_user_agent(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        parsed = urlparse(str(request.full_url))
        assert parsed.path.endswith("/search")
        params = parse_qs(parsed.query)
        assert params["q"] == ["Zhengzhou railway bridge"]
        assert params["format"] == ["jsonv2"]
        assert request.get_header("User-agent") == "geoagent-dataset/1.0 (geocode; test)"
        return _FakeHttpResponse(
            [
                {
                    "display_name": "Zhengzhou Yellow River Railway Bridge",
                    "lat": "34.890000",
                    "lon": "113.670000",
                    "addresstype": "bridge",
                    "osm_type": "way",
                    "osm_id": 123,
                    "raw_content": "MUST NOT LEAK",
                }
            ]
        )

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("GEOCODE_PROVIDER", "nominatim")
    monkeypatch.setenv("GEOCODE_NOMINATIM_ENDPOINT", "http://127.0.0.1:8080")
    monkeypatch.setenv("GEOCODE_NOMINATIM_USER_AGENT", "geoagent-dataset/1.0 (geocode; test)")
    monkeypatch.setenv("GEOCODE_ALLOW_PUBLIC_NOMINATIM", "")
    monkeypatch.setattr(geocode_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "geocode",
        "geocode",
        purpose="mock nominatim",
        inputs={"query": "Zhengzhou railway bridge", "direction": "forward"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls
    hit = observation.result["results"][0]
    assert hit["formatted_address"] == "Zhengzhou Yellow River Railway Bridge"
    assert hit["match_level"] == "bridge"
    assert hit["location"]["crs"] == "wgs84"
    assert hit["provider_id"] == "way/123"
    assert observation.result["applied"]["provider"] == "nominatim"
    assert any("WGS84" in item for item in observation.result["assumptions"])
    assert "MUST NOT LEAK" not in str(observation.result)


def test_nominatim_bbox_becomes_viewbox(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        params = parse_qs(urlparse(str(request.full_url)).query)
        assert params["viewbox"] == ["113.500000,35.000000,113.800000,34.700000"]
        assert params["bounded"] == ["1"]
        return _FakeHttpResponse([])

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("GEOCODE_PROVIDER", "nominatim")
    monkeypatch.setenv("GEOCODE_NOMINATIM_ENDPOINT", "http://127.0.0.1:8080")
    monkeypatch.setattr(geocode_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "geocode",
        "geocode",
        purpose="viewbox",
        inputs={"query": "bridge", "area": [113.5, 34.7, 113.8, 35.0]},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls
    assert observation.result["results"] == []


def test_injected_results_strip_forbidden_keys() -> None:
    observation = _lookup(provider=FakeGeocodeProvider(_sample_injected()))
    assert observation.ok is True
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert observation.result["results"][0]["formatted_address"] == "河南省郑州市惠济区黄河铁路桥"


def test_undeclared_top_k_is_ignored() -> None:
    observation = _lookup(inputs={"top_k": "many"})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["top_k"] == 10
