"""geocode 执行器测试；所有网络请求均使用注入客户端。"""

from __future__ import annotations

from typing import Any

from tool import execute
from tool.contract import RuntimeContext
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
    assert gated.error_code == "provider_unavailable"


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
