"""terrain_analysis -> visibility_analysis 的真实本地 artifact 链。"""

from __future__ import annotations

from pathlib import Path

from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from tool.contract import RuntimeContext
from tool.runtime import ToolHarness, ToolHarnessConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _trajectory() -> Trajectory:
    values = [[100.0] * 31 for _ in range(21)]
    return Trajectory(
        id="terrain-visibility-artifact-chain",
        system_prompt="system",
        user_query="Check whether two points have terrain line of sight.",
        steps=[
            TrajectoryStep(
                event_type="tool_call",
                thought="准备真实地形栅格",
                action=Action(
                    tool="terrain_analysis",
                    params={
                        "operation": "terrain",
                        "purpose": "产生可复用 DEM artifact",
                        "inputs": {
                            "area": {
                                "west": 0.0,
                                "south": 0.0,
                                "east": 0.02,
                                "north": 0.01,
                            },
                            "metrics": ["elevation"],
                            "terrain": {
                                "values": values,
                                "bbox": [0.0, 0.0, 0.02, 0.01],
                                "kind": "dem",
                                "crs": "wgs84",
                            },
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="复用上一步 DEM 检查视线",
                action=Action(
                    tool="visibility_analysis",
                    params={
                        "operation": "sightline",
                        "purpose": "计算点到点视线",
                        "inputs": {
                            "observer": {"lon": 0.002, "lat": 0.005, "height_m": 2},
                            "target": {"lon": 0.018, "lat": 0.005, "height_m": 2},
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="final",
                thought="提交验证结论",
                action=Action(tool="final_answer", params={"location": "local-dem-test"}),
            ),
        ],
    )


def test_terrain_visibility_artifact_chain(tmp_path: Path) -> None:
    harness = ToolHarness(
        ToolHarnessConfig(
            workspace_dir=tmp_path / "runs",
            catalog_path=REPO_ROOT / "canonical_tool_catalog_v2.json",
            provider_revision="local-terrain-v1",
        ),
        context=RuntimeContext(),
    )
    report = harness.run(_trajectory())
    details = "\n".join(
        f"{record.step_index}: {record.tool}.{record.operation} "
        f"{record.status} {record.error_code}: {record.error}"
        for record in report.records
    )
    assert report.status == "completed", details
    assert report.counts == {"total": 3, "succeeded": 2, "terminal": 1}

    terrain = report.records[0].observation
    assert terrain is not None and terrain.result is not None
    terrain_path = Path(terrain.artifacts["terrain_path"])
    assert terrain_path.is_file()
    assert terrain.result["terrain_ref"] == str(terrain_path)

    visibility = report.records[1].observation
    assert visibility is not None and visibility.result is not None
    assert visibility.result["visible"] is True
    assert visibility.result["applied"]["terrain_source"] == "geotiff"
