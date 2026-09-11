"""poi_search 执行器测试；不访问真实 Nominatim/Overpass。"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from tool import execute
from tool._crs import format_gcj02_lonlat, gcj02_to_wgs84
from tool.contract import Observation, RuntimeContext
from tool.poi_search import _amap as search_mod
from tool.poi_search._amap import PoiSearchRequest
from tool.runtime import InMemoryResultStore


class FakeGeocodeClient:
    name = "fake_geocoder"

    def search(
        self,
        query: str,
        *,
        limit: int,
        language: str | None,
        viewbox: tuple[float, float, float, float] | None,
    ) -> list[dict[str, Any]]:
        del limit, language, viewbox
        return [
            {
                "osm_type": "relation",
                "osm_id": 1,
                "lat": "34.95",
                "lon": "113.50",
                "display_name": query,
                "class": "leisure",
                "type": "park",
            }
        ]

    def reverse(self, lat: float, lon: float, *, language: str | None) -> dict[str, Any]:
        raise AssertionError("not used")


class FakeOverpassClient:
    name = "fake_overpass"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def query(self, overpass_ql: str) -> dict[str, Any]:
        self.calls.append(overpass_ql)
        return {
            "elements": [
                {
                    "type": "node",
                    "id": 10,
                    "lat": 34.95,
                    "lon": 113.50,
                    "tags": {"name": "示例公园", "leisure": "park"},
                }
            ]
        }


def test_query_only_uses_name_geocoder() -> None:
    store = InMemoryResultStore()
    observation = execute(
        "poi_search",
        "poi_search",
        purpose="名称查找",
        inputs={"query": "郑州黄河文化公园", "top_k": 5},
        ctx=RuntimeContext(
            extras={"geocode_client": FakeGeocodeClient()},
            result_store=store,
        ),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["strategy"] == "name_geocode"
    assert observation.result["candidate_count"] == 1
    assert observation.result["candidates"][0]["category"] == "leisure"
    assert observation.result["result_id"].startswith("poi_")


def test_category_search_uses_overpass_and_context_area() -> None:
    client = FakeOverpassClient()
    observation = execute(
        "poi_search",
        "poi_search",
        purpose="公园候选",
        inputs={"categories": ["公园"], "top_k": 20},
        ctx=RuntimeContext(
            active_area="郑州市",
            extras={"overpass_client": client},
            result_store=InMemoryResultStore(),
        ),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["strategy"] == "overpass"
    assert '["leisure"="park"]' in client.calls[0]
    assert 'area["name"="郑州市"]' in client.calls[0]
    candidate = observation.result["candidates"][0]
    assert candidate["name"] == "示例公园"
    assert candidate["category"] == "leisure=park"


def test_browse_combines_name_and_tag_filters() -> None:
    client = FakeOverpassClient()
    observation = execute(
        "poi_search",
        "browse",
        purpose="浏览学校",
        inputs={
            "bbox": [113.0, 34.0, 114.0, 35.0],
            "query": "武林中学",
            "filters": {"amenity": "school"},
        },
        ctx=RuntimeContext(extras={"overpass_client": client}),
    )
    assert observation.ok is True
    ql = client.calls[0]
    assert '["amenity"="school"]' in ql
    assert '["name"~"武林中学",i]' in ql


def test_category_search_without_scope_requests_acquisition() -> None:
    observation = execute(
        "poi_search",
        "poi_search",
        purpose="缺范围",
        inputs={"categories": ["学校"]},
        ctx=RuntimeContext(extras={"overpass_client": FakeOverpassClient()}),
    )
    assert observation.ok is False
    assert observation.error_code == "needs_acquisition"


def test_browse_rejects_unfiltered_scan() -> None:
    observation = execute(
        "poi_search",
        "browse",
        purpose="无过滤",
        inputs={"area": "郑州市"},
        ctx=RuntimeContext(extras={"overpass_client": FakeOverpassClient()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_filter"


# --- Amap / gated executor coverage ---

class FakePoiProvider:
    """测试替身：记录请求并返回高德或归一化载荷。"""

    name = "fake"
    max_top_k = 500

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else _sample_pois()
        self.calls: list[PoiSearchRequest] = []

    def search(self, request: PoiSearchRequest) -> dict[str, Any]:
        self.calls.append(request)
        return self.payload


def _sample_pois() -> dict[str, Any]:
    return {
        "status": "1",
        "pois": [
            {
                "id": "B000A83M61",
                "name": "郑州黄河铁路桥",
                "address": "惠济区花园口",
                "location": "113.670000,34.890000",
                "type": "交通设施服务;桥梁;桥梁",
                "confirmed_location": "MUST NOT LEAK",
                "raw_content": "FULL POI JSON MUST NOT LEAK",
            },
            {
                "id": "B000A7BD6C",
                "name": "花园口景区",
                "address": [],
                "location": [],
                "type": "风景名胜;风景名胜;风景名胜",
            },
        ],
    }


def _sample_injected() -> dict[str, Any]:
    return {
        "results": [
            {
                "name": "郑州黄河铁路桥",
                "address": "惠济区花园口",
                "location": {"lon": 113.67, "lat": 34.89, "crs": "gcj02"},
                "category": "桥梁",
                "provider_id": "B000A83M61",
                "source": "fake",
                "taken_at": "MUST NOT LEAK",
            },
            {
                "name": "花园口景区",
                "provider_id": "B000A7BD6C",
                "category": "风景名胜",
                "location": {"lon": 113.68, "lat": 34.90, "crs": "gcj02"},
                "source": "fake",
            },
            {
                "name": "黄河博物馆",
                "provider_id": "B000A11111",
                "category": "博物馆",
                "location": {"lon": 113.69, "lat": 34.91, "crs": "gcj02"},
                "source": "fake",
            },
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


def _search(
    *,
    operation: str = "poi_search",
    inputs: dict[str, object] | None = None,
    provider: FakePoiProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {"query": "铁路桥"}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    engine = provider if provider is not None else FakePoiProvider()
    if runtime is None:
        runtime = RuntimeContext(extras={"poi_search_provider": engine})
    elif provider is not None:
        runtime.extras["poi_search_provider"] = engine
    return execute(
        "poi_search",
        operation,
        purpose="检索 POI",
        inputs=payload,
        ctx=runtime,
    )


def test_poi_search_returns_name_address_wgs84_and_provider_id() -> None:
    provider = FakePoiProvider()
    observation = _search(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert provider.calls and provider.calls[0].mode == "text"
    assert provider.calls[0].keywords == "铁路桥"
    results = observation.result["results"]
    assert results[0]["result_id"] == "poi_1"
    assert results[0]["name"] == "郑州黄河铁路桥"
    assert results[0]["address"] == "惠济区花园口"
    assert results[0]["provider_id"] == "B000A83M61"
    assert results[0]["source"] == "fake"
    lon, lat = gcj02_to_wgs84(113.67, 34.89)
    assert results[0]["location"]["lon"] == lon
    assert results[0]["location"]["lat"] == lat
    assert results[0]["location"]["crs"] == "wgs84"
    assert results[0]["location"]["lon"] != 113.67
    assert "address" not in results[1]
    assert "location" not in results[1]
    assert observation.result["applied"]["provider"] == "fake"
    assert observation.result["applied"]["crs"] == "wgs84"
    dumped = str(observation.result)
    assert "MUST NOT LEAK" not in dumped
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys
    assert "confirmed_location" not in keys
    assert any("已近似转换" in item for item in observation.result["assumptions"])
    assert any("训练" in item for item in observation.result["assumptions"])


def test_missing_required_any_is_missing_input() -> None:
    observation = execute(
        "poi_search",
        "poi_search",
        purpose="缺输入",
        inputs={},
        ctx=RuntimeContext(extras={"poi_search_provider": FakePoiProvider()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_browse_missing_area_is_missing_input() -> None:
    observation = execute(
        "poi_search",
        "browse",
        purpose="缺区域",
        inputs={},
        ctx=RuntimeContext(extras={"poi_search_provider": FakePoiProvider()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_query_array_is_joined_for_amap_keywords() -> None:
    provider = FakePoiProvider()
    observation = _search(inputs={"query": ["铁路桥", "石栏"]}, provider=provider)
    assert observation.ok is True
    assert provider.calls[0].keywords == "铁路桥|石栏"


def test_place_name_area_uses_text_search() -> None:
    provider = FakePoiProvider()
    observation = _search(
        inputs={"area": "郑州市", "categories": ["桥梁", "风景名胜"]},
        provider=provider,
    )
    assert observation.ok is True
    request = provider.calls[0]
    assert request.mode == "text"
    assert request.city == "郑州市"
    assert request.types == "桥梁|风景名胜"
    assert request.radius_m is None
    assert observation.result is not None
    assert observation.result["applied"]["search_mode"] == "text"
    assert observation.result["applied"]["area"] == "郑州市"
    assert observation.result["applied"]["city"] == "郑州市"


def test_center_and_radius_use_around_search() -> None:
    provider = FakePoiProvider()
    observation = _search(
        inputs={
            "area": {"lat": 34.75, "lon": 113.65},
            "radius_m": 1500,
            "query": "铁路桥",
        },
        provider=provider,
    )
    assert observation.ok is True
    request = provider.calls[0]
    assert request.mode == "around"
    assert request.location == "113.650000,34.750000"
    assert request.radius_m == 1500
    assert observation.result is not None
    assert observation.result["applied"]["search_mode"] == "around"
    assert observation.result["applied"]["radius_m"] == 1500


def test_bbox_uses_polygon_search() -> None:
    provider = FakePoiProvider()
    observation = _search(
        inputs={"area": [113.60, 34.70, 113.80, 34.90], "query": "铁路桥"},
        provider=provider,
    )
    assert observation.ok is True
    request = provider.calls[0]
    assert request.mode == "polygon"
    assert request.polygon == "113.600000,34.900000|113.800000,34.700000"
    assert observation.result is not None
    assert observation.result["applied"]["search_mode"] == "polygon"


def test_radius_without_coords_is_unsupported_and_not_sent() -> None:
    provider = FakePoiProvider()
    observation = _search(
        inputs={"area": "郑州市", "query": "铁路桥", "radius_m": 2000},
        provider=provider,
    )
    assert observation.ok is True
    assert provider.calls[0].mode == "text"
    assert provider.calls[0].radius_m is None
    assert observation.result is not None
    assert "radius_m" not in observation.result["applied"]
    assert observation.result["applied"]["unsupported"]["radius_m"] == 2000


def test_active_area_context_is_resolved() -> None:
    provider = FakePoiProvider()
    observation = _search(
        inputs={"area": "$active_area", "query": "铁路桥"},
        provider=provider,
        ctx=RuntimeContext(
            active_area="郑州市",
            extras={"poi_search_provider": provider},
        ),
    )
    assert observation.ok is True
    assert provider.calls[0].city == "郑州市"


def test_browse_paginates_previous_results_without_search() -> None:
    provider = FakePoiProvider()
    previous = _sample_injected()
    observation = execute(
        "poi_search",
        "browse",
        purpose="翻页",
        inputs={"area": "郑州市", "filters": {"page": 2}, "top_k": 1},
        ctx=RuntimeContext(
            previous_tool_result=previous,
            extras={"poi_search_provider": provider},
        ),
    )
    assert observation.ok is True
    assert provider.calls == []
    assert observation.result is not None
    assert observation.result["applied"]["search_mode"] == "local_browse"
    assert len(observation.result["results"]) == 1
    assert observation.result["results"][0]["name"] == "花园口景区"


def test_browse_expand_calls_detail() -> None:
    provider = FakePoiProvider()
    observation = execute(
        "poi_search",
        "browse",
        purpose="展开",
        inputs={"area": "郑州市", "filters": {"expand": True, "poi_id": "B000A83M61"}},
        ctx=RuntimeContext(
            previous_tool_result=_sample_injected(),
            extras={"poi_search_provider": provider},
        ),
    )
    assert observation.ok is True
    assert provider.calls
    assert provider.calls[0].mode == "detail"
    assert provider.calls[0].poi_ids == "B000A83M61"


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "poi_search",
        "poi_search",
        purpose="闸门",
        inputs={"query": "铁路桥"},
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
    observation = execute(
        "poi_search",
        "poi_search",
        purpose="缺钥匙",
        inputs={"query": "铁路桥"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "AMAP" in observation.error


def test_amap_engine_sends_key_and_strips_empty_lists(monkeypatch: Any) -> None:
    calls: list[Any] = []

    class _FakeHttpResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._body = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> _FakeHttpResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        parsed = urlparse(str(request.full_url))
        assert parsed.path.endswith("/text")
        params = parse_qs(parsed.query)
        assert params["key"] == ["test-amap-key"]
        assert params["keywords"] == ["铁路桥"]
        assert params["types"] == ["桥梁"]
        assert params["city"] == ["郑州市"]
        assert params["citylimit"] == ["true"]
        return _FakeHttpResponse(_sample_pois())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("AMAP_WEB_KEY", "test-amap-key")
    monkeypatch.setenv("AMAP_API_KEY", "")
    monkeypatch.setattr(search_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "poi_search",
        "poi_search",
        purpose="mock amap",
        inputs={"query": "铁路桥", "area": "郑州市", "categories": "桥梁"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert calls, "应发起一次高德 HTTP 请求"
    assert observation.result["applied"]["provider"] == "amap"
    assert observation.result["results"][0]["location"]["crs"] == "wgs84"
    assert "address" not in observation.result["results"][1]
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys
    dumped = str(observation.result)
    assert "MUST NOT LEAK" not in dumped


def test_amap_around_sends_location_and_radius(monkeypatch: Any) -> None:
    calls: list[Any] = []

    class _FakeHttpResponse:
        def read(self) -> bytes:
            return json.dumps({"status": "1", "pois": []}).encode("utf-8")

        def __enter__(self) -> _FakeHttpResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        parsed = urlparse(str(request.full_url))
        assert parsed.path.endswith("/around")
        params = parse_qs(parsed.query)
        assert params["key"] == ["test-amap-key"]
        assert params["location"] == [format_gcj02_lonlat(113.65, 34.75)]
        assert params["radius"] == ["2000"]
        assert params["keywords"] == ["铁路桥"]
        return _FakeHttpResponse()

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("AMAP_WEB_KEY", "test-amap-key")
    monkeypatch.setattr(search_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "poi_search",
        "poi_search",
        purpose="around",
        inputs={
            "query": "铁路桥",
            "area": {"lat": 34.75, "lon": 113.65},
            "radius_m": 2000,
        },
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls


def test_invalid_top_k_is_rejected() -> None:
    observation = _search(inputs={"top_k": "many"})
    assert observation.ok is False
    assert observation.error_code == "invalid_top_k"
