"""poi_search 执行器测试；不访问真实 Nominatim/Overpass。"""

from __future__ import annotations

from typing import Any

from tool import execute
from tool.contract import RuntimeContext
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
