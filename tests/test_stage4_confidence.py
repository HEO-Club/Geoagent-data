"""阶段4：置信度评分单测（全 mock，禁止真实 API）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas.audit import GeoTaskSpec, KeyframeAssessment, TargetKind
from pipeline.schemas.confidence import (
    ConfidenceJudgeDraft,
    HardGateHit,
)
from pipeline.schemas.dataset import ChatMessage, DatasetEntry
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage4_confidence import merge_confidence, rewrite_entry_quality_score, run_stage4
from pipeline.stage4_confidence.rules import evaluate_programmatic_gates


def _task(**kwargs: Any) -> GeoTaskSpec:
    base = dict(
        task_id="vid__t01",
        time_start=0.0,
        time_end=10.0,
        target_kind=TargetKind.still_image,
        image_paths=["img.jpg"],
    )
    base.update(kwargs)
    return GeoTaskSpec(**base)


def _traj_ok() -> Trajectory:
    return Trajectory(
        id="vid__t01",
        system_prompt="sys",
        user_query="Locate the place shown in the image.",
        image_paths=["img.jpg"],
        steps=[
            TrajectoryStep(
                event_type="reasoning",
                thought="看图有红瓦屋顶",
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="查地图",
                action=Action(
                    tool="map_query",
                    params={
                        "operation": "query",
                        "purpose": "查地图",
                        "inputs": {"q": "红瓦"},
                    },
                ),
                observation={"result": "候选若干"},
            ),
            TrajectoryStep(
                event_type="final",
                thought="确认地点",
                action=Action(tool="final_answer", params={"location": "某市某镇"}),
                observation=None,
            ),
        ],
    )


def _freeform() -> FreeFormTrajectory:
    return FreeFormTrajectory(
        source_video="vid",
        steps=[
            FreeFormStep(event_type="reasoning", thought="看图有红瓦屋顶"),
            FreeFormStep(
                event_type="tool_call",
                thought="查地图",
                tool="map_query",
                params={"q": "红瓦"},
                observation={"result": "候选若干"},
            ),
            FreeFormStep(
                event_type="final",
                thought="确认地点",
                tool="final_answer",
                params={"location": "某市某镇"},
                observation=None,
            ),
        ],
    )


def _entry() -> DatasetEntry:
    return DatasetEntry(
        id="vid__t01",
        source_video="vid",
        messages=[
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="q\n[Image: img.jpg]"),
            ChatMessage(role="assistant", content="Thought: ok"),
        ],
        quality_score=None,
    )


def test_weighted_merge_and_hard_gate_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE4_HARD_GATE_CAP", "0.3")
    clear_settings_cache()
    draft = ConfidenceJudgeDraft(
        evidence_grounding=0.9,
        final_answer_support=0.9,
        tool_param_correctness=0.9,
        logical_consistency=0.9,
        input_quality_alignment=0.9,
        dimension_reasons={"evidence_grounding": "扎实"},
        hard_gates=[
            HardGateHit(code="fabricated_observation", evidence="观测无来源")
        ],
        notes="模型门槛",
    )
    report = merge_confidence(
        task_id="vid__t01",
        format_score=1.0,
        format_reason="ok",
        programmatic_gates=[],
        draft=draft,
        judge_call_failed=False,
    )
    assert report.base_score >= 0.85
    assert report.quality_score == pytest.approx(0.3)
    assert report.review_priority == "high"
    assert any(g.code == "fabricated_observation" for g in report.hard_gates)


def test_review_priority_bands_after_strict_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STAGE4_PRIORITY_HIGH_BELOW", "0.70")
    monkeypatch.setenv("STAGE4_PRIORITY_MEDIUM_BELOW", "0.93")
    clear_settings_cache()

    def _report(score: float, *, gates: list[HardGateHit] | None = None) -> str:
        draft = ConfidenceJudgeDraft(
            evidence_grounding=score,
            final_answer_support=score,
            tool_param_correctness=score,
            logical_consistency=score,
            input_quality_alignment=score,
            hard_gates=gates or [],
        )
        return merge_confidence(
            task_id="t",
            format_score=score,
            format_reason="ok",
            programmatic_gates=[],
            draft=draft,
            judge_call_failed=False,
        ).review_priority

    assert _report(0.96) == "low"
    assert _report(0.90) == "medium"
    assert _report(0.75) == "medium"
    assert _report(0.60) == "high"


def test_programmatic_missing_final_and_empty_location() -> None:
    traj = Trajectory(
        id="t",
        system_prompt="s",
        user_query="q",
        image_paths=["a.jpg"],
        steps=[
            TrajectoryStep(
                event_type="reasoning",
                thought="只有思考",
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="误作末步",
                action=Action(tool="map_query", params={"operation": "x", "purpose": "p", "inputs": {}}),
                observation={"result": None},
            ),
        ],
    )
    gates, fmt, _ = evaluate_programmatic_gates(
        traj,
        _task(),
        [TranscriptSegment(start=0, end=1, text="旁白")],
    )
    codes = {g.code for g in gates}
    assert "last_not_final" in codes
    assert "missing_final" in codes
    assert fmt < 1.0

    traj2 = Trajectory(
        id="t2",
        system_prompt="s",
        user_query="q",
        image_paths=["a.jpg"],
        steps=[
            TrajectoryStep(
                event_type="final",
                thought="空答案",
                action=Action(tool="final_answer", params={"location": ""}),
                observation=None,
            )
        ],
    )
    gates2, _, _ = evaluate_programmatic_gates(
        traj2,
        _task(),
        [TranscriptSegment(start=0, end=1, text="旁白")],
    )
    assert any(g.code == "empty_location" for g in gates2)


def test_answer_leakage_selected_gate() -> None:
    task = _task(
        frame_assessments=[
            KeyframeAssessment(
                timestamp=1.0,
                image_path="img.jpg",
                kind="still",
                quality_score=0.9,
                answer_leakage=True,
                selected=True,
            )
        ]
    )
    gates, _, _ = evaluate_programmatic_gates(
        _traj_ok(),
        task,
        [TranscriptSegment(start=0, end=1, text="旁白")],
    )
    assert any(g.code == "answer_leakage_selected" for g in gates)


def test_run_stage4_writes_sidecar_and_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    clear_settings_cache()

    frame = tmp_path / "img.jpg"
    frame.write_bytes(b"jpg")
    traj = _traj_ok()
    traj = traj.model_copy(update={"image_paths": [str(frame)]})

    def fake_judge(**_k: Any) -> ConfidenceJudgeDraft:
        return ConfidenceJudgeDraft(
            evidence_grounding=0.8,
            final_answer_support=0.7,
            tool_param_correctness=0.75,
            logical_consistency=0.8,
            input_quality_alignment=0.7,
            hard_gates=[
                HardGateHit(code="fabricated_observation", evidence="假观测")
            ],
            notes="注入裁判",
        )

    report_path = tmp_path / "stage4_confidence.json"
    shard = tmp_path / "out.jsonl"
    original_messages = _entry().messages
    entry = _entry()
    shard.write_text(entry.model_dump_json() + "\n", encoding="utf-8")

    report = run_stage4(
        task=_task(image_paths=[str(frame)]),
        transcript=[TranscriptSegment(start=0, end=1, text="红瓦屋顶定位")],
        freeform=_freeform(),
        trajectory=traj,
        entry=entry,
        out_report_path=str(report_path),
        out_jsonl_path=str(shard),
        judge=fake_judge,
    )
    assert report_path.is_file()
    assert report.quality_score <= 0.3
    assert any(g.code == "fabricated_observation" for g in report.hard_gates)

    rewritten = DatasetEntry.model_validate_json(
        shard.read_text(encoding="utf-8").splitlines()[0]
    )
    assert rewritten.quality_score == pytest.approx(report.quality_score)
    assert rewritten.messages == original_messages


def test_judge_failure_fail_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAGE4_JUDGE_NEUTRAL_SCORE", "0.5")
    clear_settings_cache()

    def boom(**_k: Any) -> ConfidenceJudgeDraft:
        raise RuntimeError("api down")

    report = run_stage4(
        task=_task(),
        transcript=[TranscriptSegment(start=0, end=1, text="旁白")],
        freeform=_freeform(),
        trajectory=_traj_ok(),
        entry=_entry(),
        out_report_path=str(tmp_path / "r.json"),
        out_jsonl_path=str(tmp_path / "s.jsonl"),
        judge=boom,
    )
    assert report.judge_call_failed is True
    assert "judge_call_failed" in report.notes
    # 程序化格式完整时 base 约为中性分加权 + 格式 1.0
    assert 0.4 <= report.quality_score <= 0.7
    dim_map = {d.name: d.score for d in report.dimensions}
    assert dim_map["sft_format_completeness"] == pytest.approx(1.0)
    assert dim_map["evidence_grounding"] == pytest.approx(0.5)


def test_rewrite_only_quality_score() -> None:
    entry = _entry()
    updated = rewrite_entry_quality_score(entry, 0.66)
    assert updated.quality_score == pytest.approx(0.66)
    assert updated.messages == entry.messages
    assert updated.id == entry.id


def test_judge_prompt_has_score_anchors() -> None:
    """裁判 prompt 必须含分档刻度，避免「能讲通就 0.9+」。"""
    from pipeline.stage4_confidence.judge import JUDGE_HINT, build_judge_prompt

    assert "0.95–1.00" in JUDGE_HINT
    assert "默认从 0.75 起评" in JUDGE_HINT
    assert "incomplete_final_targets" in JUDGE_HINT
    prompt = build_judge_prompt(
        task=_task(),
        transcript=[TranscriptSegment(start=0, end=1, text="旁白")],
        freeform=_freeform(),
        trajectory=_traj_ok(),
    )
    assert "禁止把「能复述字幕 / 主链能讲通」直接打 0.9+" in prompt
    assert "final_answer_support" in prompt
