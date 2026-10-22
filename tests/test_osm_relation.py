"""osm_query 的 near、crossings、intersect。不访问公共 Overpass。"""

from __future__ import annotations

from typing import Any

from tool import execute
from tool.contract import RuntimeContext
from tool.osm_query._query import EngineUnavailableError, OsmQueryRequest


class _Provider:
    name = "fake"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[OsmQueryRequest] = []

    def query(self, request: OsmQueryRequest) -> dict[str, Any]:
        self.calls.append(request)
        if '["boundary"="administrative"]' in request.ql:
            return {
                "elements": [
                    {
                        "type": "area",
                        "id": 3600123456,
                        "tags": {
                            "name": "郑州市",
                            "boundary": "administrative",
                            "admin_level": "5",
                        },
                    }
                ]
            }
        return self.payload


def _ctx(provider: Any) -> RuntimeContext:
    return RuntimeContext(extras={"osm_query_provider": provider})


def test_near_uses_around_and_returns_geometry() -> None:
    provider = _Provider(
        {
            "elements": [
                _way(9, {"bridge": "yes", "name": "测试桥"}, [(34.801, 113.603), (34.799, 113.605)]),
                _way(8, {"waterway": "river", "name": "测试河"}, [(34.800, 113.600), (34.804, 113.604)]),
            ]
        }
    )
    observation = execute(
        "osm_query",
        "near",
        purpose="桥在河附近",
        inputs={
            "bbox": [113.60, 34.79, 113.64, 34.82],
            "subject": "桥梁",
            "reference": "河流",
            "within_m": 700,
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    ql = provider.calls[-1].ql
    assert "(around.ref:" not in ql
    assert '["bridge"="yes"]' in ql
    assert '["waterway"="river"]' in ql
    assert "out geom" in ql
    assert len(provider.calls) == 1
    assert observation.result is not None
    assert observation.result["count"] == 1
    assert observation.result["partial"] is False
    assert observation.result["elements"][0]["geometry"]["type"] == "Point"


def test_near_downloads_each_type_when_the_bundle_times_out() -> None:
    bridge = _way(9, {"bridge": "yes", "name": "测试桥"}, [(34.801, 113.603), (34.799, 113.605)])
    river = _way(8, {"waterway": "river", "name": "测试河"}, [(34.800, 113.600), (34.804, 113.604)])

    class _TimeoutThenSplit:
        name = "fake"

        def __init__(self) -> None:
            self.calls: list[OsmQueryRequest] = []

        def query(self, request: OsmQueryRequest) -> dict[str, Any]:
            self.calls.append(request)
            ql = request.ql
            if '["bridge"="yes"]' in ql and '["waterway"="river"]' in ql:
                raise EngineUnavailableError("Overpass 超时: timed out", "timeout")
            if '["bridge"="yes"]' in ql:
                return {"elements": [bridge]}
            if '["waterway"="river"]' in ql:
                return {"elements": [river]}
            return {"elements": []}

    provider = _TimeoutThenSplit()
    observation = execute(
        "osm_query",
        "near",
        purpose="整包超时后分别下载",
        inputs={
            "bbox": [113.60, 34.79, 113.64, 34.82],
            "subject": "桥梁",
            "reference": "河流",
            "within_m": 700,
            "layout": [{"feature": "桥梁", "bearing": "西北-东南"}],
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    assert len(provider.calls) == 3
    assert observation.result is not None
    assert observation.result["count"] == 1
    assert observation.result["elements"][0]["subject"]["name"] == "测试桥"


def test_near_resolves_admin_area_before_around() -> None:
    provider = _Provider({"elements": []})
    observation = execute(
        "osm_query",
        "near",
        purpose="市内相邻",
        inputs={"area": "郑州市", "subject": "桥梁", "reference": "河流"},
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    assert any("out bb;" in call.ql for call in provider.calls)
    feature_ql = provider.calls[-1].ql
    assert "area(3600123456)->.searchArea;" in feature_ql
    assert "(around.ref:" not in feature_ql
    assert observation.result is not None
    assert observation.result["count"] == 0


def test_unknown_subject_is_rejected() -> None:
    provider = _Provider({"elements": []})
    observation = execute(
        "osm_query",
        "near",
        purpose="山地",
        inputs={"bbox": [113.5, 34.7, 113.8, 34.9], "subject": "山地", "reference": "河流"},
        ctx=_ctx(provider),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert provider.calls == []


def test_crossings_query_follows_the_line() -> None:
    provider = _Provider({"elements": []})
    observation = execute(
        "osm_query",
        "crossings",
        purpose="河上的桥",
        inputs={"bbox": [113.5, 34.7, 113.8, 34.9], "line": "河流", "kinds": ["bridge"]},
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    ql = provider.calls[-1].ql
    assert "->.line;" in ql
    assert "(around.line:40)" in ql
    assert '["bridge"]' in ql
    assert '["waterway"="river"]' in ql


def test_intersect_returns_crossing_point() -> None:
    provider = _Provider(
        {
            "elements": [
                {
                    "type": "way",
                    "id": 1,
                    "tags": {"railway": "rail"},
                    "geometry": [
                        {"lat": 34.0, "lon": 113.0},
                        {"lat": 34.0, "lon": 113.02},
                    ],
                },
                {
                    "type": "way",
                    "id": 2,
                    "tags": {"highway": "primary"},
                    "geometry": [
                        {"lat": 33.99, "lon": 113.01},
                        {"lat": 34.01, "lon": 113.01},
                    ],
                },
            ]
        }
    )
    observation = execute(
        "osm_query",
        "intersect",
        purpose="铁路与道路交叉",
        inputs={
            "bbox": [112.9, 33.9, 113.1, 34.1],
            "subject": "铁路",
            "reference": "道路",
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["count"] == 1
    point = observation.result["candidates"][0]
    assert abs(point["latitude"] - 34.0) < 0.0001
    assert abs(point["longitude"] - 113.01) < 0.0001
    assert point["geometry"]["type"] == "Point"


def _way(osm_id: int, tags: dict[str, str], points: list[tuple[float, float]]) -> dict[str, Any]:
    return {
        "type": "way",
        "id": osm_id,
        "tags": tags,
        "geometry": [{"lat": lat, "lon": lon} for lat, lon in points],
    }


def _passing_pair() -> list[dict[str, Any]]:
    return [
        _way(9, {"bridge": "yes", "name": "测试桥"}, [(34.801, 113.603), (34.799, 113.605)]),
        _way(8, {"waterway": "river", "name": "测试河"}, [(34.800, 113.600), (34.804, 113.604)]),
    ]


def test_near_drops_a_place_that_fails_bearing() -> None:
    provider = _Provider(
        {
            "elements": [
                _way(9, {"bridge": "yes", "name": "测试桥"}, [(34.801, 113.600), (34.801, 113.602)]),
                _way(8, {"waterway": "river", "name": "东西河"}, [(34.800, 113.600), (34.800, 113.604)]),
            ]
        }
    )
    observation = execute(
        "osm_query",
        "near",
        purpose="走向不符",
        inputs={
            "bbox": [113.60, 34.79, 113.64, 34.82],
            "subject": "桥梁",
            "reference": "河流",
            "within_m": 700,
            "layout": [{"feature": "河流", "bearing": "东北-西南"}],
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["count"] == 0
    assert len(provider.calls) == 1


def test_near_keeps_a_place_that_passes_layout() -> None:
    provider = _Provider({"elements": _passing_pair()})
    observation = execute(
        "osm_query",
        "near",
        purpose="走向和方位都符合",
        inputs={
            "bbox": [113.60, 34.79, 113.64, 34.82],
            "subject": "桥梁",
            "reference": "河流",
            "within_m": 700,
            "layout": [
                {"feature": "河流", "bearing": "东北-西南"},
                {"feature": "桥梁", "bearing": "西北-东南"},
                {"subject": "桥梁", "reference": "河流", "direction": "东南"},
            ],
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["count"] == 1
    assert observation.result["places"][0]["subject"]["name"] == "测试桥"


def test_near_tiles_a_large_bbox() -> None:
    provider = _Provider({"elements": []})
    observation = execute(
        "osm_query",
        "near",
        purpose="大范围切块",
        inputs={
            "bbox": [113.0, 34.0, 114.0, 35.0],
            "subject": "桥梁",
            "reference": "河流",
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    assert len(provider.calls) >= 4
    spans = [_bbox_span(call.ql) for call in provider.calls]
    assert all(span is not None and span[0] < 0.4 for span in spans)
    assert observation.result is not None
    assert observation.result["count"] == 0


def test_near_subdivides_a_truncated_tile() -> None:
    class _Truncated:
        name = "fake"

        def __init__(self) -> None:
            self.calls: list[OsmQueryRequest] = []

        def query(self, request: OsmQueryRequest) -> dict[str, Any]:
            self.calls.append(request)
            span = _bbox_span(request.ql)
            if span is not None and max(span) > 0.08:
                return {
                    "elements": [
                        _way(1, {"bridge": "yes"}, [(34.75, 113.55), (34.76, 113.56)]),
                        _way(2, {"bridge": "yes"}, [(34.74, 113.54), (34.73, 113.53)]),
                        _way(3, {"waterway": "river"}, [(34.70, 113.50), (34.71, 113.51)]),
                    ]
                }
            return {"elements": _passing_pair()}

    provider = _Truncated()
    observation = execute(
        "osm_query",
        "near",
        purpose="截断后再切",
        inputs={
            "bbox": [113.50, 34.70, 113.62, 34.80],
            "subject": "桥梁",
            "reference": "河流",
            "limit": 3,
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    assert len(provider.calls) > 1
    assert observation.result is not None
    assert observation.result["count"] == 1
    assert observation.result["places"][0]["subject"]["name"] == "测试桥"


class _Dem:
    def fetch(self, bbox: Any, *, demtype: str) -> "_Dem":
        del bbox, demtype
        return self

    def sample(self, lon: float, lat: float) -> float:
        if lon > 113.602 and lat < 34.802:
            return 200.0
        return 100.0


def test_near_keeps_a_place_when_elevation_rises() -> None:
    provider = _Provider({"elements": _passing_pair()})
    observation = execute(
        "osm_query",
        "near",
        purpose="河的东南有山",
        inputs={
            "bbox": [113.60, 34.79, 113.64, 34.82],
            "subject": "桥梁",
            "reference": "河流",
            "within_m": 700,
            "layout": [{"subject": "山地", "reference": "河流", "direction": "东南"}],
        },
        ctx=RuntimeContext(extras={"osm_query_provider": provider, "terrain_analysis_provider": _Dem()}),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["count"] == 1


def test_near_is_unavailable_without_dem() -> None:
    provider = _Provider({"elements": _passing_pair()})
    observation = execute(
        "osm_query",
        "near",
        purpose="没有高程",
        inputs={
            "bbox": [113.60, 34.79, 113.64, 34.82],
            "subject": "桥梁",
            "reference": "河流",
            "layout": [{"subject": "山地", "reference": "河流", "direction": "东南"}],
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_TOOL_API" in observation.error
    assert provider.calls == []


def test_near_uses_the_terrain_dem_provider(monkeypatch: Any) -> None:
    class _Grid:
        def sample(self, lon: float, lat: float) -> float:
            if lon > 113.602 and lat < 34.802:
                return 200.0
            return 100.0

    class _Terrain:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def fetch(self, bbox: Any, *, demtype: str) -> _Grid:
            del bbox
            self.calls.append(demtype)
            return _Grid()

    terrain = _Terrain()
    monkeypatch.setattr("tool.osm_query._place._resolve_dem_provider", lambda ctx: terrain)
    provider = _Provider({"elements": _passing_pair()})
    observation = execute(
        "osm_query",
        "near",
        purpose="走地形高程",
        inputs={
            "bbox": [113.60, 34.79, 113.64, 34.82],
            "subject": "桥梁",
            "reference": "河流",
            "within_m": 700,
            "layout": [{"subject": "山地", "reference": "河流", "direction": "东南"}],
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["count"] == 1
    assert terrain.calls == ["COP30"]


def test_near_fetches_dem_by_tile_when_the_range_is_large(monkeypatch: Any) -> None:
    spans: list[tuple[float, float]] = []

    class _Grid:
        def sample(self, lon: float, lat: float) -> float:
            del lon, lat
            return 100.0

    class _Terrain:
        def fetch(self, bbox: Any, *, demtype: str) -> _Grid:
            del demtype
            spans.append((bbox.east - bbox.west, bbox.north - bbox.south))
            return _Grid()

    monkeypatch.setattr("tool.osm_query._place._resolve_dem_provider", lambda ctx: _Terrain())
    provider = _Provider({"elements": []})
    observation = execute(
        "osm_query",
        "near",
        purpose="大范围按块取高程",
        inputs={
            "bbox": [113.0, 34.0, 116.0, 37.0],
            "subject": "桥梁",
            "reference": "河流",
            "layout": [{"subject": "山地", "reference": "河流", "direction": "东南"}],
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    assert spans
    assert all(width <= 2.0 and height <= 2.0 for width, height in spans)
    provider = _Provider({"elements": _passing_pair()})
    observation = execute(
        "osm_query",
        "near",
        purpose="没有建成区面",
        inputs={
            "bbox": [113.60, 34.79, 113.64, 34.82],
            "subject": "桥梁",
            "reference": "河流",
            "within_m": 700,
            "layout": [{"subject": "城区", "reference": "河流", "direction": "东南"}],
        },
        ctx=_ctx(provider),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["count"] == 0
    assert observation.result["applied"]["landuse_missing"] >= 1
    assert any("建成区" in item for item in observation.result["assumptions"])


def _bbox_span(ql: str) -> tuple[float, float] | None:
    import re

    match = re.search(r"\(([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)\)", ql)
    if match is None:
        return None
    south, _west, north, east = (float(item) for item in match.groups())
    west = float(match.group(2))
    return north - south, east - west
