"""Harness 下的完整本地地理后处理链；Provider 使用离线假响应。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from tool.contract import RuntimeContext
from tool.runtime import ToolHarness, ToolHarnessConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeGeocoder:
    name = "fake_geocoder"

    def __init__(self) -> None:
        self.calls = 0

    def search(
        self,
        query: str,
        *,
        limit: int,
        language: str | None,
        viewbox: tuple[float, float, float, float] | None,
    ) -> list[dict[str, Any]]:
        del query, limit, language, viewbox
        self.calls += 1
        return [
            {
                "osm_type": "relation",
                "osm_id": 1,
                "lat": "34.95",
                "lon": "113.50",
                "display_name": "郑州市",
                "boundingbox": ["34.90", "35.00", "113.40", "113.60"],
            }
        ]

    def reverse(self, lat: float, lon: float, *, language: str | None) -> dict[str, Any]:
        raise AssertionError((lat, lon, language))


class FakeOverpass:
    name = "fake_overpass"

    def __init__(self) -> None:
        self.calls = 0

    def query(self, overpass_ql: str) -> dict[str, Any]:
        del overpass_ql
        self.calls += 1
        return {
            "elements": [
                {
                    "type": "node",
                    "id": 101,
                    "lat": 34.9500,
                    "lon": 113.5000,
                    "tags": {"name": "黄河桥 A", "bridge": "yes", "material": "stone"},
                },
                {
                    "type": "node",
                    "id": 102,
                    "lat": 34.9600,
                    "lon": 113.5100,
                    "tags": {"name": "黄河桥 B", "bridge": "yes", "material": "steel"},
                },
                {
                    "type": "way",
                    "id": 103,
                    "center": {"lat": 34.97, "lon": 113.52},
                    "geometry": [
                        {"lat": 34.96, "lon": 113.51},
                        {"lat": 34.98, "lon": 113.53},
                    ],
                    "tags": {"name": "普通道路", "highway": "primary"},
                },
            ]
        }


def _trajectory() -> Trajectory:
    return Trajectory(
        id="local-geo-postprocess-chain",
        system_prompt="system",
        user_query="Locate the bridge candidates.",
        steps=[
            TrajectoryStep(
                event_type="tool_call",
                thought="解析城市范围",
                action=Action(
                    tool="geocode",
                    params={
                        "operation": "geocode",
                        "purpose": "取得郑州 bbox",
                        "inputs": {"query": "郑州市", "top_k": 1},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="查询范围内桥梁",
                action=Action(
                    tool="osm_query",
                    params={
                        "operation": "query",
                        "purpose": "获取桥梁候选及几何",
                        "inputs": {
                            "bbox": "$step_1_tool_result.candidates[0].bbox",
                            "feature_types": ["桥梁"],
                            "return_geometry": True,
                            "limit": 10,
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="保留带 bridge 标签的对象",
                action=Action(
                    tool="osm_result_process",
                    params={
                        "operation": "filter",
                        "purpose": "过滤桥梁候选",
                        "inputs": {
                            "source_result": "$step_2_tool_result.result_id",
                            "filters": {"tags": {"bridge": "*"}},
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="计算两个桥梁候选之间的距离",
                action=Action(
                    tool="distance_bearing_calculator",
                    params={
                        "operation": "distance",
                        "purpose": "比较候选间距",
                        "inputs": {
                            "points": [
                                "$step_3_tool_result.elements[0]",
                                "$step_3_tool_result.elements[1]",
                            ],
                            "unit": "km",
                            "mode": "geodesic",
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="计算候选方向",
                action=Action(
                    tool="distance_bearing_calculator",
                    params={
                        "operation": "bearing",
                        "purpose": "确定 A 到 B 的真北方位",
                        "inputs": {
                            "origin": "$step_3_tool_result.elements[0]",
                            "target": "$step_3_tool_result.elements[1]",
                            "reference": "true_north",
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="筛选两公里内的候选",
                action=Action(
                    tool="spatial_filter",
                    params={
                        "operation": "geometry_filter",
                        "purpose": "保留靠近桥 A 的候选",
                        "inputs": {
                            "source_result": "$step_3_tool_result.result_id",
                            "relation": "near",
                            "geometry": "$step_3_tool_result.elements[0]",
                            "distance_m": 2000,
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="导出过滤后的 GeoJSON",
                action=Action(
                    tool="osm_result_process",
                    params={
                        "operation": "export",
                        "purpose": "保存复核图层",
                        "inputs": {
                            "source_result": "$step_3_tool_result.result_id",
                            "format": "geojson",
                            "include_geometry": True,
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="final",
                thought="提交地点",
                action=Action(tool="final_answer", params={"location": "郑州市"}),
            ),
        ],
    )


def test_complete_local_geo_chain_and_resume(tmp_path: Path) -> None:
    geocoder = FakeGeocoder()
    overpass = FakeOverpass()
    context = RuntimeContext(
        extras={"geocode_client": geocoder, "overpass_client": overpass}
    )
    config = ToolHarnessConfig(
        workspace_dir=tmp_path / "runs",
        catalog_path=REPO_ROOT / "canonical_tool_catalog_v2.json",
        provider_revision="offline-fixture-v1",
    )
    first_harness = ToolHarness(config, context=context)
    first = first_harness.run(_trajectory())
    assert first.status == "completed"
    assert first.counts == {"total": 8, "succeeded": 7, "terminal": 1}
    assert first.records[3].observation.result["total"] == pytest.approx(1.437, rel=1e-3)
    assert first.records[4].observation.result["compass_direction"] == "NE"
    assert first.records[5].observation.result["matched_count"] == 2
    export_path = Path(first.records[6].observation.artifacts["export_path"])
    assert export_path.is_file()
    assert len(list((first_harness.last_run_dir / "results").glob("*.json"))) == 6
    assert geocoder.calls == 1
    assert overpass.calls == 1

    resumed_geocoder = FakeGeocoder()
    resumed_overpass = FakeOverpass()
    resumed = ToolHarness(
        config,
        context=RuntimeContext(
            extras={
                "geocode_client": resumed_geocoder,
                "overpass_client": resumed_overpass,
            }
        ),
    ).run(_trajectory())
    assert resumed.status == "completed"
    assert all(record.reused_from_checkpoint for record in resumed.records)
    assert resumed_geocoder.calls == 0
    assert resumed_overpass.calls == 0
