"""本地 WGS84 距离、宽度和真北方位测试。"""

from __future__ import annotations

import pytest

from tool import execute
from tool.contract import RuntimeContext
from tool.runtime import InMemoryResultStore


def _ctx() -> RuntimeContext:
    return RuntimeContext(result_store=InMemoryResultStore())


def test_geodesic_distance_and_bearing() -> None:
    ctx = _ctx()
    distance = execute(
        "distance_bearing_calculator",
        "distance",
        purpose="计算两点距离",
        inputs={"points": [[0, 0], [0, 1]], "unit": "km"},
        ctx=ctx,
    )
    assert distance.ok is True
    assert distance.result["total"] == pytest.approx(111.31949, rel=1e-6)
    assert distance.result["pairs"][0]["forward_azimuth_deg"] == pytest.approx(90)
    assert distance.result["result_id"].startswith("distance_")

    bearing = execute(
        "distance_bearing_calculator",
        "bearing",
        purpose="计算真北方位",
        inputs={"origin": [0, 0], "target": [1, 0]},
        ctx=ctx,
    )
    assert bearing.ok is True
    assert bearing.result["forward_azimuth_deg"] == pytest.approx(0)
    assert bearing.result["compass_direction"] == "N"
    assert bearing.result["reference"] == "true_north"


def test_distance_resolves_points_from_result_store() -> None:
    ctx = _ctx()
    result_id = ctx.result_store.put(
        "geo",
        {
            "candidates": [
                {"name": "A", "latitude": 34.95, "longitude": 113.50},
                {"name": "B", "latitude": 34.96, "longitude": 113.51},
            ]
        },
    )
    observation = execute(
        "distance_bearing_calculator",
        "distance",
        purpose="引用候选点",
        inputs={"points": result_id},
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result["point_count"] == 2
    assert observation.result["pairs"][0]["origin"]["label"] == "A"


def test_geometry_length_and_polygon_width() -> None:
    line = {
        "osm_type": "way",
        "osm_id": 1,
        "geometry": [[0, 0], [0.01, 0]],
    }
    length = execute(
        "distance_bearing_calculator",
        "distance",
        purpose="线长度",
        inputs={"features": [line], "mode": "geometry", "unit": "m"},
    )
    assert length.ok is True
    assert length.result["measurements"][0]["value_m"] == pytest.approx(1113.19, rel=2e-3)

    polygon = {
        "osm_type": "way",
        "osm_id": 2,
        "geometry": [[0, 0], [0.01, 0], [0.01, 0.005], [0, 0.005], [0, 0]],
    }
    width = execute(
        "distance_bearing_calculator",
        "distance",
        purpose="面宽度",
        inputs={"features": [polygon], "mode": "width", "unit": "m"},
    )
    assert width.ok is True
    assert width.result["measurements"][0]["value_m"] == pytest.approx(552.87, rel=5e-3)


@pytest.mark.parametrize("reference", ["magnetic_north", "image_axis"])
def test_bearing_requires_missing_reference_data(reference: str) -> None:
    observation = execute(
        "distance_bearing_calculator",
        "bearing",
        purpose="不可推断参考系",
        inputs={"origin": [0, 0], "target": [1, 1], "reference": reference},
    )
    assert observation.ok is False
    assert observation.error_code == "needs_acquisition"


def test_route_ambiguity_coincident_and_invalid_coordinates() -> None:
    route = execute(
        "distance_bearing_calculator",
        "distance",
        purpose="沿路距离",
        inputs={"points": [[0, 0], [1, 1]], "mode": "route"},
    )
    assert route.error_code == "needs_acquisition"

    coincident = execute(
        "distance_bearing_calculator",
        "bearing",
        purpose="重合点",
        inputs={"origin": [1, 1], "target": [1, 1]},
    )
    assert coincident.error_code == "coincident_points"

    invalid = execute(
        "distance_bearing_calculator",
        "distance",
        purpose="非法坐标",
        inputs={"points": [[91, 0], [0, 0]]},
    )
    assert invalid.error_code == "invalid_coordinate"


def test_bearing_rejects_ambiguous_candidate_result() -> None:
    ctx = _ctx()
    result_id = ctx.result_store.put(
        "geo",
        {"candidates": [{"latitude": 0, "longitude": 0}, {"latitude": 1, "longitude": 1}]},
    )
    observation = execute(
        "distance_bearing_calculator",
        "bearing",
        purpose="候选未选择",
        inputs={"origin": result_id, "target": [2, 2]},
        ctx=ctx,
    )
    assert observation.error_code == "ambiguous_coordinate"
