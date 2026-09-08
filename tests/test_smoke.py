"""骨架冒烟：schemas / config。"""

from __future__ import annotations

import importlib

import pytest

from pipeline.config import Settings, clear_settings_cache, get_settings
from pipeline.schemas import (
    Action,
    ChatMessage,
    DatasetEntry,
    FreeFormStep,
    FreeFormTrajectory,
    ToolDefinition,
    ToolForest,
    ToolTree,
    Trajectory,
    TrajectoryStep,
    TranscriptSegment,
)


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    clear_settings_cache()
    s = get_settings()
    assert s.ALLOW_REAL_API is False
    assert s.TOOL_TREES_PATH == "tool_trees.json"
    assert isinstance(s, Settings)


def test_soft_envelope_and_dataset_schemas() -> None:
    seg = TranscriptSegment(start=0.0, end=1.0, text="hello")
    assert seg.text == "hello"
    ff = FreeFormTrajectory(
        source_video="vid1",
        steps=[
            FreeFormStep(
                thought="look",
                tool="weird_zoom",
                params={"bbox": [0, 0, 1, 1]},
                observation={"ok": True},
            )
        ],
    )
    assert ff.steps[0].tool == "weird_zoom"
    traj = Trajectory(
        id="t1",
        system_prompt="sys",
        user_query="q",
        image_paths=["img.jpg"],
        steps=[
            TrajectoryStep(
                thought="t",
                action=Action(tool="zoom_inspect", params={}),
                observation={"result": "x"},
            )
        ],
    )
    entry = DatasetEntry(
        id="t1",
        source_video="vid1",
        messages=[ChatMessage(role="system", content="sys")],
    )
    assert entry.messages[0].role == "system"
    forest = ToolForest(
        trees=[
            ToolTree(
                canonical=ToolDefinition(name="zoom_inspect", description="z"),
                variants=["weird_zoom"],
            )
        ]
    )
    assert forest.trees[0].variants == ["weird_zoom"]
    assert traj.steps[0].action.tool == "zoom_inspect"


def test_orchestrator_stage_order() -> None:
    """编排含审核切分，不依赖已删除的旧 stage 模块。"""
    import pipeline.orchestrator as orch

    assert orch.STAGE_ORDER == (
        "stage1",
        "stage_audit_split",
        "stage2",
        "stage3",
        "stage4",
    )
    mod = importlib.import_module("pipeline.orchestrator")
    for name in (
        "stage0_preprocess",
        "stage5_reconstruct",
        "stage6_verify",
        "stage7_format",
    ):
        assert name not in getattr(mod, "__dict__", {})
