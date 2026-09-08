"""基础地理 Tool 的会话级闭环：地理编码 → OSM/POI → 结果统计。"""

from __future__ import annotations

from typing import Any

from tool import execute
from tool.contract import RuntimeContext
from tool.runtime import InMemoryResultStore


class ChainGeocoder:
    name = "chain_geocoder"

    def search(
        self,
        query: str,
        *,
        limit: int,
        language: str | None,
        viewbox: tuple[float, float, float, float] | None,
    ) -> list[dict[str, Any]]:
        del query, limit, language, viewbox
        return [
            {
                "osm_type": "relation",
                "osm_id": 1,
                "lat": "34.95",
                "lon": "113.50",
                "display_name": "候选行政区",
                "boundingbox": ["34.90", "35.00", "113.40", "113.60"],
            }
        ]

    def reverse(self, lat: float, lon: float, *, language: str | None) -> dict[str, Any]:
        raise AssertionError((lat, lon, language))


class ChainOverpass:
    name = "chain_overpass"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def query(self, overpass_ql: str) -> dict[str, Any]:
        self.calls.append(overpass_ql)
        if '["leisure"="park"]' in overpass_ql:
            return {
                "elements": [
                    {
                        "type": "node",
                        "id": 2,
                        "lat": 34.96,
                        "lon": 113.51,
                        "tags": {"name": "候选公园", "leisure": "park"},
                    }
                ]
            }
        return {
            "elements": [
                {
                    "type": "way",
                    "id": 3,
                    "center": {"lat": 34.97, "lon": 113.52},
                    "tags": {"name": "候选桥梁", "bridge": "yes"},
                }
            ]
        }


def test_basic_geo_chain_uses_grounded_bbox_and_result_ids() -> None:
    overpass = ChainOverpass()
    ctx = RuntimeContext(
        extras={
            "geocode_client": ChainGeocoder(),
            "overpass_client": overpass,
        },
        result_store=InMemoryResultStore(),
    )
    geocoded = execute(
        "geocode",
        "geocode",
        purpose="把讲解中的行政区变成候选范围",
        inputs={"query": "候选行政区"},
        ctx=ctx,
    )
    assert geocoded.ok is True and geocoded.result is not None
    bbox = geocoded.result["candidates"][0]["bbox"]

    bridges = execute(
        "osm_query",
        "query",
        purpose="在真实候选范围内查询桥梁",
        inputs={"bbox": bbox, "feature_types": ["桥梁"]},
        ctx=ctx,
    )
    assert bridges.ok is True and bridges.result is not None
    counted = execute(
        "osm_query",
        "count",
        purpose="统计真实查询回执",
        inputs={"source_result": bridges.result["result_id"]},
        ctx=ctx,
    )
    assert counted.ok is True and counted.result is not None
    assert counted.result["count"] == 1

    parks = execute(
        "poi_search",
        "poi_search",
        purpose="在同一范围内查询公园",
        inputs={"bbox": bbox, "categories": ["公园"]},
        ctx=ctx,
    )
    assert parks.ok is True and parks.result is not None
    assert parks.result["candidates"][0]["name"] == "候选公园"
    assert len(overpass.calls) == 2
