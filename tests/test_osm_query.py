"""osm_query 执行器测试；不访问公共 Overpass。"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.osm_query import _query as osm_mod
from tool.osm_query._query import OsmQueryRequest
from tool.runtime import InMemoryResultStore


class FakeOverpassClient:
    name = "fake_overpass"

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "elements": [
                {
                    "type": "way",
                    "id": 101,
                    "center": {"lat": 34.95, "lon": 113.50},
                    "bounds": {
                        "minlat": 34.94,
                        "minlon": 113.49,
                        "maxlat": 34.96,
                        "maxlon": 113.51,
                    },
                    "tags": {"name": "测试桥", "bridge": "yes", "railway": "rail"},
                    "geometry": [
                        {"lat": 34.94, "lon": 113.49},
                        {"lat": 34.96, "lon": 113.51},
                    ],
                }
            ]
        }
        self.calls: list[str] = []

    def query(self, overpass_ql: str) -> dict[str, Any]:
        self.calls.append(overpass_ql)
        return self.payload


def _ctx(client: FakeOverpassClient) -> RuntimeContext:
    return RuntimeContext(
        extras={"overpass_client": client},
        result_store=InMemoryResultStore(),
    )


def test_structured_query_generates_safe_ql_and_normalizes_elements() -> None:
    client = FakeOverpassClient()
    ctx = _ctx(client)
    observation = execute(
        "osm_query",
        "query",
        purpose="查询铁路桥",
        inputs={
            "area": "郑州市",
            "tags": {"bridge": "yes", "railway": "rail"},
            "element_types": ["way"],
            "return_geometry": True,
            "limit": 20,
        },
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result is not None
    ql = client.calls[0]
    assert 'area["name"="郑州市"]' in ql
    assert '["bridge"="yes"]' in ql
    assert '["railway"="rail"]' in ql
    assert "out tags geom qt 20;" in ql
    element = observation.result["elements"][0]
    assert element["osm_type"] == "way"
    assert element["name"] == "测试桥"
    assert element["geometry"] == [[113.49, 34.94], [113.51, 34.96]]
    assert observation.result["result_id"].startswith("osm_")


def test_feature_mapping_bbox_and_center_scope() -> None:
    bbox_client = FakeOverpassClient()
    bbox = execute(
        "osm_query",
        "query",
        purpose="查桥",
        inputs={"bbox": [113.0, 34.0, 114.0, 35.0], "feature_types": ["桥梁"]},
        ctx=_ctx(bbox_client),
    )
    assert bbox.ok is True
    assert "(34,113,35,114)" in bbox_client.calls[0]
    assert '["bridge"]' in bbox_client.calls[0]

    center_client = FakeOverpassClient()
    center = execute(
        "osm_query",
        "query",
        purpose="附近学校",
        inputs={
            "center": {"lat": 34.95, "lon": 113.5},
            "radius_m": 1000,
            "feature_types": ["学校"],
            "spatial_relation": "near",
        },
        ctx=_ctx(center_client),
    )
    assert center.ok is True
    assert "(around:1000,34.95,113.5)" in center_client.calls[0]
    assert '["amenity"="school"]' in center_client.calls[0]


def test_count_uses_real_stored_result_without_network() -> None:
    client = FakeOverpassClient()
    ctx = _ctx(client)
    queried = execute(
        "osm_query",
        "query",
        purpose="查询",
        inputs={"area": "郑州市", "feature_types": ["桥梁"]},
        ctx=ctx,
    )
    assert queried.result is not None
    counted = execute(
        "osm_query",
        "count",
        purpose="统计",
        inputs={"source_result": queried.result["result_id"], "group_by": "bridge"},
        ctx=ctx,
    )
    assert counted.ok is True
    assert counted.result is not None
    assert counted.result["count"] == 1
    assert counted.result["groups"] == {"bridge=yes": 1}
    assert len(client.calls) == 1


def test_provider_count_response() -> None:
    client = FakeOverpassClient(
        {"elements": [{"type": "count", "tags": {"nodes": "2", "ways": "3", "relations": "1", "total": "6"}}]}
    )
    observation = execute(
        "osm_query",
        "count",
        purpose="远程统计",
        inputs={"area": "郑州市", "feature_types": ["桥梁"]},
        ctx=_ctx(client),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["count"] == 6
    assert client.calls[0].endswith("out count;")


def test_remote_group_by_requires_full_source_result() -> None:
    observation = execute(
        "osm_query",
        "count",
        purpose="不能静默忽略分组",
        inputs={
            "area": "郑州市",
            "feature_types": ["桥梁"],
            "group_by": "bridge",
        },
        ctx=_ctx(FakeOverpassClient()),
    )
    assert observation.ok is False
    assert observation.error_code == "needs_acquisition"


def test_osm_rejects_unsafe_or_unexecutable_inputs_before_network() -> None:
    missing = execute("osm_query", "query", purpose="缺参数", inputs={})
    assert missing.ok is False
    assert missing.error_code == "missing_input"

    no_filter = execute(
        "osm_query",
        "query",
        purpose="无过滤",
        inputs={"area": "郑州市"},
        ctx=_ctx(FakeOverpassClient()),
    )
    assert no_filter.ok is False
    assert no_filter.error_code == "missing_filter"

    unknown = execute(
        "osm_query",
        "query",
        purpose="未知类型",
        inputs={"area": "郑州市", "feature_types": ["神秘设施"]},
        ctx=_ctx(FakeOverpassClient()),
    )
    assert unknown.ok is False
    assert unknown.error_code == "unsupported_feature_type"

    conflicting = execute(
        "osm_query",
        "query",
        purpose="冲突条件",
        inputs={
            "area": "郑州市",
            "feature_types": ["桥梁"],
            "tags": {"bridge": "no"},
        },
        ctx=_ctx(FakeOverpassClient()),
    )
    assert conflicting.ok is False
    assert conflicting.error_code == "conflicting_filters"

    raw = execute(
        "osm_query",
        "query",
        purpose="原始代码",
        inputs={"overpass_ql": "node[amenity=school];out;"},
        ctx=_ctx(FakeOverpassClient()),
    )
    assert raw.ok is False
    assert raw.error_code == "raw_query_disabled"


def test_raw_query_requires_explicit_trusted_opt_in() -> None:
    client = FakeOverpassClient()
    ctx = _ctx(client)
    ctx.extras["allow_raw_overpass_ql"] = True
    observation = execute(
        "osm_query",
        "query",
        purpose="受信代码",
        inputs={"overpass_ql": 'way["bridge"];out center 10;'},
        ctx=ctx,
    )
    assert observation.ok is True
    assert client.calls[0].startswith("[out:json];\n[timeout:25];")


# --- Amap / gated executor coverage ---

class FakeOsmQueryProvider:
    """测试替身：记录 Overpass QL 并返回归一化载荷。"""

    name = "fake"

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else _sample_elements()
        self.calls: list[OsmQueryRequest] = []

    def query(self, request: OsmQueryRequest) -> dict[str, Any]:
        self.calls.append(request)
        return self.payload


def _sample_elements() -> dict[str, Any]:
    return {
        "version": 0.6,
        "generator": "Overpass API",
        "osm3s": {"timestamp_osm_base": "2026-09-10T00:00:00Z"},
        "elements": [
            {
                "type": "way",
                "id": 123,
                "tags": {
                    "bridge": "yes",
                    "name": "黄河铁路桥",
                    "confirmed_location": "MUST NOT LEAK",
                },
                "geometry": [
                    {"lat": 34.89, "lon": 113.67},
                    {"lat": 34.90, "lon": 113.68},
                ],
                "raw_content": "FULL OSM JSON MUST NOT LEAK",
            },
            {
                "type": "node",
                "id": 456,
                "lat": 34.88,
                "lon": 113.66,
                "tags": {"power": "tower", "bridge": "no"},
            },
        ],
    }


def _sample_count() -> dict[str, Any]:
    return {
        "osm3s": {"timestamp_osm_base": "2026-09-10T00:00:00Z"},
        "elements": [
            {
                "type": "count",
                "id": 0,
                "tags": {"nodes": "2", "ways": "1", "relations": "0", "total": "3"},
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


def _query(
    *,
    inputs: dict[str, Any] | None = None,
    provider: FakeOsmQueryProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, Any] = {"area": "郑州市", "tags": {"bridge": "yes"}}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    engine = provider if provider is not None else FakeOsmQueryProvider()
    if runtime is None:
        runtime = RuntimeContext(extras={"osm_query_provider": engine})
    elif provider is not None:
        runtime.extras["osm_query_provider"] = engine
    return execute(
        "osm_query",
        "query",
        purpose="查询 OSM 要素",
        inputs=payload,
        ctx=runtime,
    )


def _count(
    *,
    inputs: dict[str, Any] | None = None,
    provider: FakeOsmQueryProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, Any] = {"area": "郑州市", "tags": {"bridge": "yes"}}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    engine = provider if provider is not None else FakeOsmQueryProvider(_sample_count())
    if runtime is None:
        runtime = RuntimeContext(extras={"osm_query_provider": engine})
    elif provider is not None:
        runtime.extras["osm_query_provider"] = engine
    return execute(
        "osm_query",
        "count",
        purpose="统计 OSM 要素",
        inputs=payload,
        ctx=runtime,
    )


class _FakeHttpResponse:
    def __init__(self, payload: Any) -> None:
        self._body = (
            payload if isinstance(payload, bytes) else __import__("json").dumps(payload).encode("utf-8")
        )

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_empty_inputs_are_missing_input() -> None:
    observation = execute("osm_query", "query", purpose="scaffold", inputs={})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_image_only_is_missing_input() -> None:
    observation = execute(
        "osm_query",
        "query",
        purpose="看图",
        inputs={"current_image": "$current_image"},
        ctx=RuntimeContext(current_image="img_1"),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "不会看图" in observation.error


def test_structured_area_and_tags_generate_ql_without_agent_code() -> None:
    provider = FakeOsmQueryProvider()
    observation = _query(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert provider.calls
    ql = provider.calls[0].ql
    assert 'area["name"="郑州市"]->.searchArea' in ql
    assert 'nwr["bridge"="yes"](area.searchArea)' in ql
    assert "out geom 200" in ql
    assert "{{" not in ql
    assert observation.result["applied"]["overpass_ql"] == ql
    assert "overpass_ql" not in observation.result["applied"].get("unused", {})


def test_bbox_uses_overpass_south_west_north_east() -> None:
    provider = FakeOsmQueryProvider()
    observation = _query(
        inputs={"area": None, "bbox": [113.5, 34.7, 113.8, 34.9], "tags": {"bridge": "yes"}},
        provider=provider,
    )
    assert observation.ok is True
    ql = provider.calls[0].ql
    assert "(34.7,113.5,34.9,113.8)" in ql
    assert observation.result["applied"]["bbox"]["west"] == 113.5
    assert observation.result["applied"]["bbox"]["crs"] == "wgs84"


def test_center_radius_writes_around_filter() -> None:
    provider = FakeOsmQueryProvider()
    observation = _query(
        inputs={
            "area": None,
            "center": {"lat": 34.8, "lon": 113.65},
            "radius_m": 1000,
            "tags": {"bridge": "yes"},
        },
        provider=provider,
    )
    assert observation.ok is True
    ql = provider.calls[0].ql
    assert "(around:1000,34.8,113.65)" in ql


def test_feature_types_bridge_maps_to_bridge_yes() -> None:
    provider = FakeOsmQueryProvider()
    observation = _query(
        inputs={"tags": None, "feature_types": ["桥梁"]},
        provider=provider,
    )
    assert observation.ok is True
    ql = provider.calls[0].ql
    assert '["bridge"="yes"]' in ql
    assert observation.result["applied"]["feature_types"] == ["桥梁"]


def test_unmapped_feature_types_are_missing_input() -> None:
    observation = _query(inputs={"feature_types": ["不存在的地物类型xyz"]})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "tags" in observation.error


def test_overpass_ql_is_executed_and_returned() -> None:
    provider = FakeOsmQueryProvider()
    ql = '[out:json][timeout:10];nwr["bridge"="yes"](34.7,113.5,34.9,113.8);out geom;'
    observation = _query(
        inputs={"area": None, "tags": None, "overpass_ql": ql},
        provider=provider,
    )
    assert observation.ok is True
    executed = provider.calls[0].ql
    assert 'nwr["bridge"="yes"](34.7,113.5,34.9,113.8)' in executed
    assert "[out:json]" in executed
    assert observation.result["applied"]["overpass_ql"] == executed


def test_overpass_ql_without_spatial_is_unsupported() -> None:
    observation = _query(
        inputs={"area": None, "overpass_ql": '[out:json];nwr["bridge"="yes"];out;'},
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_query"


def test_overpass_turbo_macro_is_unsupported() -> None:
    observation = _query(
        inputs={"area": None, "overpass_ql": "{{bbox}};nwr[bridge];out;"},
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_query"


def test_query_normalizes_ids_tags_geometry_and_strips_forbidden() -> None:
    provider = FakeOsmQueryProvider()
    observation = _query(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    elements = observation.result["elements"]
    assert elements[0]["result_id"] == "osm_1"
    assert elements[0]["osm_type"] == "way"
    assert elements[0]["osm_id"] == 123
    assert elements[0]["tags"]["bridge"] == "yes"
    assert elements[0]["geometry"]["type"] == "LineString"
    assert elements[0]["geometry"]["coordinates"][0] == [113.67, 34.89]
    assert elements[1]["geometry"]["type"] == "Point"
    assert observation.result["count"] == 2
    assert observation.result["data_timestamp"] == "2026-09-10T00:00:00Z"
    dumped = str(observation.result)
    assert "MUST NOT LEAK" not in dumped
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys
    assert "confirmed_location" not in keys


def test_count_ignores_undeclared_overpass_ql() -> None:
    observation = execute(
        "osm_query",
        "count",
        purpose="未声明字段",
        inputs={"overpass_ql": '[out:json];nwr["bridge"="yes"](34.7,113.5,34.9,113.8);out count;'},
        ctx=RuntimeContext(extras={"osm_query_provider": FakeOsmQueryProvider(_sample_count())}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_count_uses_out_count() -> None:
    provider = FakeOsmQueryProvider(_sample_count())
    observation = _count(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    ql = provider.calls[0].ql
    assert "out count;" in ql
    assert observation.result["count"] == 3
    assert observation.result["data_timestamp"] == "2026-09-10T00:00:00Z"
    assert any("不代表现实世界完整数量" in item for item in observation.result["assumptions"])


def test_count_source_result_is_local_and_group_by_works() -> None:
    source = {
        "operation": "query",
        "elements": [
            {"osm_type": "way", "osm_id": 1, "tags": {"bridge": "yes"}},
            {"osm_type": "way", "osm_id": 2, "tags": {"bridge": "yes"}},
            {"osm_type": "node", "osm_id": 3, "tags": {"bridge": "no"}},
        ],
        "data_timestamp": "2026-09-10T00:00:00Z",
    }
    observation = execute(
        "osm_query",
        "count",
        purpose="本地统计",
        inputs={"source_result": source, "group_by": "bridge"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["count"] == 3
    assert observation.result["applied"]["provider"] == "local"
    groups = {item["bridge"]: item["count"] for item in observation.result["groups"]}
    assert groups == {"yes": 2, "no": 1}
    assert observation.result["data_timestamp"] == "2026-09-10T00:00:00Z"


def test_count_previous_tool_result_reference() -> None:
    previous = {
        "elements": [
            {"osm_type": "way", "osm_id": 1, "tags": {"bridge": "yes"}},
            {"osm_type": "way", "osm_id": 2, "tags": {"bridge": "yes"}},
        ]
    }
    observation = execute(
        "osm_query",
        "count",
        purpose="引用上一轮",
        inputs={"source_result": "$previous_tool_result"},
        ctx=RuntimeContext(previous_tool_result=previous),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["count"] == 2
    assert observation.result["applied"]["provider"] == "local"


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "osm_query",
        "query",
        purpose="闸门",
        inputs={"area": "郑州市", "tags": {"bridge": "yes"}},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_overpass_http_posts_ql(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        body = request.data.decode("utf-8") if isinstance(request.data, bytes) else ""
        params = parse_qs(body)
        assert "data" in params
        assert '["bridge"="yes"]' in params["data"][0]
        assert request.get_header("User-agent") == "geoagent-dataset/1.0 (osm_query; test)"
        return _FakeHttpResponse(_sample_elements())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("OSM_QUERY_OVERPASS_ENDPOINT", "http://127.0.0.1:12345/api/interpreter")
    monkeypatch.setenv("OSM_QUERY_OVERPASS_USER_AGENT", "geoagent-dataset/1.0 (osm_query; test)")
    monkeypatch.setattr(osm_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "osm_query",
        "query",
        purpose="HTTP",
        inputs={"area": "郑州市", "tags": {"bridge": "yes"}},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls
    assert observation.result["applied"]["provider"] == "overpass"
    assert observation.result["elements"][0]["osm_id"] == 123
    assert any("WGS84" in item for item in observation.result["assumptions"])


def test_active_area_context_default() -> None:
    provider = FakeOsmQueryProvider()
    observation = _query(
        inputs={"area": "$active_area"},
        provider=provider,
        ctx=RuntimeContext(active_area="郑州市", extras={}),
    )
    assert observation.ok is True
    ql = provider.calls[0].ql
    assert 'area["name"="郑州市"]' in ql


def test_return_geometry_false_uses_out_tags() -> None:
    provider = FakeOsmQueryProvider()
    observation = _query(inputs={"return_geometry": False}, provider=provider)
    assert observation.ok is True
    assert "out tags 200" in provider.calls[0].ql
    assert "geometry" not in observation.result["elements"][0]


def test_overpass_ql_caps_timeout(monkeypatch: Any) -> None:
    monkeypatch.setenv("OSM_QUERY_TIMEOUT_SEC", "25")
    provider = FakeOsmQueryProvider()
    observation = _query(
        inputs={
            "area": None,
            "overpass_ql": '[out:json][timeout:90];nwr["bridge"="yes"](34.7,113.5,34.9,113.8);out;',
        },
        provider=provider,
    )
    assert observation.ok is True
    assert "[timeout:25]" in provider.calls[0].ql
    assert "[timeout:90]" not in provider.calls[0].ql


def test_structured_unused_when_overpass_ql_present() -> None:
    provider = FakeOsmQueryProvider()
    observation = _query(
        inputs={
            "area": "郑州市",
            "tags": {"bridge": "yes"},
            "overpass_ql": '[out:json];nwr["railway"](34.7,113.5,34.9,113.8);out geom;',
        },
        provider=provider,
    )
    assert observation.ok is True
    unused = observation.result["applied"]["unused"]
    assert unused["area"] == "郑州市"
    assert unused["tags"] == {"bridge": "yes"}
    assert '["railway"]' in provider.calls[0].ql
