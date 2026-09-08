"""osm_query 执行器测试；不访问公共 Overpass。"""

from __future__ import annotations

from typing import Any

from tool import execute
from tool.contract import RuntimeContext
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
