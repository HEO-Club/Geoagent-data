"""轨迹–选图一致性门禁单测。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.stage_audit_split import trajectory_image_check as tic


def test_check_skips_without_images(tmp_path: Path) -> None:
    traj = FreeFormTrajectory(
        source_video="x",
        steps=[
            FreeFormStep(event_type="reasoning", thought="看到红瓦屋顶"),
        ],
    )
    result = tic.check_trajectory_image_consistency(
        image_paths=[str(tmp_path / "missing.jpg")],
        visual_evidence_brief="红瓦屋顶",
        trajectory=traj,
    )
    assert result.conflict is False
    assert "无可用选中图" in result.reason


def test_check_high_confidence_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = tmp_path / "a.jpg"
    frame.write_bytes(b"jpg")
    traj = FreeFormTrajectory(
        source_video="x",
        steps=[
            FreeFormStep(event_type="reasoning", thought="图中有哥特尖顶"),
            FreeFormStep(
                event_type="final",
                thought="结束",
                tool="final_answer",
                params={"location": "某地"},
            ),
        ],
    )

    def fake_call(prompt: str, schema: Any, **kwargs: Any) -> Any:
        assert schema is tic.TrajectoryImageConsistencyResult
        assert "哥特尖顶" in prompt
        assert kwargs.get("images")
        return tic.TrajectoryImageConsistencyResult(
            conflict=True,
            confidence=0.9,
            reason="选中图是海滩，与尖顶 brief 冲突",
        )

    monkeypatch.setattr(tic, "call_structured", fake_call)
    result = tic.check_trajectory_image_consistency(
        image_paths=[str(frame)],
        visual_evidence_brief="哥特尖顶与尖拱门廊",
        trajectory=traj,
    )
    assert result.conflict is True
    assert result.confidence >= 0.8


def test_check_low_confidence_not_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = tmp_path / "a.jpg"
    frame.write_bytes(b"jpg")
    traj = FreeFormTrajectory(
        source_video="x",
        steps=[FreeFormStep(event_type="reasoning", thought="临水木栈道")],
    )

    def fake_call(prompt: str, schema: Any, **_k: Any) -> Any:
        return tic.TrajectoryImageConsistencyResult(
            conflict=True,
            confidence=0.4,
            reason="不太确定",
        )

    monkeypatch.setattr(tic, "call_structured", fake_call)
    result = tic.check_trajectory_image_consistency(
        image_paths=[str(frame)],
        visual_evidence_brief="临水木栈道",
        trajectory=traj,
    )
    assert result.conflict is False


def test_multi_input_reasoning_with_one_image_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """开篇 reasoning 写图1与图2，但只选 1 张 → 高精度冲突。"""
    frame = tmp_path / "only.jpg"
    frame.write_bytes(b"jpg")
    traj = FreeFormTrajectory(
        source_video="x",
        steps=[
            FreeFormStep(
                event_type="reasoning",
                thought="图1是临街立面，图2是路口全景，两者共同缩小范围",
            ),
            FreeFormStep(
                event_type="final",
                thought="结束",
                tool="final_answer",
                params={"location": "某地"},
            ),
        ],
    )

    def should_not_call(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("多输入结构冲突不应再调 VLM")

    monkeypatch.setattr(tic, "call_structured", should_not_call)
    result = tic.check_trajectory_image_consistency(
        image_paths=[str(frame)],
        visual_evidence_brief="临街立面与路口",
        trajectory=traj,
    )
    assert result.conflict is True
    assert result.confidence >= 0.8
    assert "第二份独立输入" in result.reason


def test_single_image_reasoning_does_not_false_alarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """单图题只写「图中」不得误报多输入冲突。"""
    frame = tmp_path / "a.jpg"
    frame.write_bytes(b"jpg")
    traj = FreeFormTrajectory(
        source_video="x",
        steps=[FreeFormStep(event_type="reasoning", thought="图中有红瓦屋顶与宽阔水面")],
    )

    def fake_call(prompt: str, schema: Any, **_k: Any) -> Any:
        return tic.TrajectoryImageConsistencyResult(
            conflict=False,
            confidence=0.9,
            reason="一致",
        )

    monkeypatch.setattr(tic, "call_structured", fake_call)
    result = tic.check_trajectory_image_consistency(
        image_paths=[str(frame)],
        visual_evidence_brief="红瓦屋顶与宽阔水面",
        trajectory=traj,
    )
    assert result.conflict is False
