"""本地 Shapely 空间关系与 AEQD 近邻筛选测试。"""

from __future__ import annotations

from tool import execute
from tool.contract import RuntimeContext
from tool.runtime import InMemoryResultStore


def _ctx() -> tuple[RuntimeContext, str]:
    store = InMemoryResultStore()
    result_id = store.put(
        "osm",
        {
            "crs": "EPSG:4326",
            "elements": [
                {"osm_type": "node", "osm_id": 1, "latitude": 0, "longitude": 0},
                {"osm_type": "node", "osm_id": 2, "latitude": 0, "longitude": 0.01},
                {
                    "osm_type": "way",
                    "osm_id": 3,
                    "geometry": [[-1, 0], [1, 0]],
                },
                {"osm_type": "relation", "osm_id": 4, "tags": {"type": "site"}},
            ],
        },
    )
    return RuntimeContext(result_store=store), result_id


def test_within_and_missing_geometry_count() -> None:
    ctx, result_id = _ctx()
    observation = execute(
        "spatial_filter",
        "geometry_filter",
        purpose="范围筛选",
        inputs={
            "source_result": result_id,
            "relation": "within",
            "geometry": {"bbox": [-0.1, -0.1, 0.1, 0.1]},
        },
        ctx=ctx,
    )
    assert observation.ok is True
    assert {item["osm_id"] for item in observation.result["elements"]} == {1, 2}
    assert observation.result["missing_geometry_count"] == 1
    assert observation.result["result_id"].startswith("spatial_")


def test_crosses_and_intersects() -> None:
    ctx, result_id = _ctx()
    reference = {
        "type": "Polygon",
        "coordinates": [[[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1]]],
    }
    crossed = execute(
        "spatial_filter",
        "geometry_filter",
        purpose="穿越关系",
        inputs={"source_result": result_id, "relation": "crosses", "geometry": reference},
        ctx=ctx,
    )
    assert {item["osm_id"] for item in crossed.result["elements"]} == {3}

    intersects = execute(
        "spatial_filter",
        "geometry_filter",
        purpose="相交关系",
        inputs={"source_result": result_id, "relation": "intersects", "geometry": reference},
        ctx=ctx,
    )
    assert {item["osm_id"] for item in intersects.result["elements"]} == {1, 2, 3}


def test_near_uses_meter_threshold() -> None:
    ctx, result_id = _ctx()
    near = execute(
        "spatial_filter",
        "geometry_filter",
        purpose="一公里内",
        inputs={
            "source_result": result_id,
            "relation": "near",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "distance_m": 1000,
        },
        ctx=ctx,
    )
    assert {item["osm_id"] for item in near.result["elements"]} == {1, 3}
    assert near.result["metric_projection"].startswith("+proj=aeqd")

    wider = execute(
        "spatial_filter",
        "geometry_filter",
        purpose="两公里内",
        inputs={
            "source_result": result_id,
            "relation": "near",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "distance_m": 1200,
        },
        ctx=ctx,
    )
    assert {item["osm_id"] for item in wider.result["elements"]} == {1, 2, 3}


def test_near_requires_distance_and_rejects_bad_geometry() -> None:
    ctx, result_id = _ctx()
    missing = execute(
        "spatial_filter",
        "geometry_filter",
        purpose="缺距离",
        inputs={
            "source_result": result_id,
            "relation": "near",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
        },
        ctx=ctx,
    )
    assert missing.error_code == "missing_input"

    invalid = execute(
        "spatial_filter",
        "geometry_filter",
        purpose="非法几何",
        inputs={
            "source_result": result_id,
            "relation": "within",
            "geometry": {"bbox": [10, 10, 5, 5]},
        },
        ctx=ctx,
    )
    assert invalid.error_code == "invalid_geometry"

    excessive = execute(
        "spatial_filter",
        "geometry_filter",
        purpose="拒绝伪局部超大距离",
        inputs={
            "source_result": result_id,
            "relation": "near",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "distance_m": 3_000_000,
        },
        ctx=ctx,
    )
    assert excessive.error_code == "distance_too_large"
