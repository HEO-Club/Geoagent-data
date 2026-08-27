"""Human review sidecars must remain evidence-linked and non-blocking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.schemas.audit import (
    GeoTaskSpec,
    KeyframeAssessment,
    ProcessInterval,
    ProcessRole,
    TargetKind,
    TaskStatus,
)
from pipeline.schemas.confidence import (
    ConfidenceJudgeDraft,
    ConfidenceReport,
    HardGateHit,
)
from pipeline.schemas.dataset import DatasetEntry
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage3_normalize_format.format_jsonl import format_dataset_entry
from pipeline.stage4_confidence.review_cards import (
    build_review_packet,
    render_review_markdown,
)
from pipeline.stage4_confidence.run import merge_confidence, run_stage4
from scripts import recompute_fused_stage4_from_reports as recompute


def _trajectory(images: list[str] | None = None) -> Trajectory:
    return Trajectory(id="v__t01", system_prompt="system", user_query="query", image_paths=images or [], steps=[TrajectoryStep(event_type="final", thought="提交已有证据支持的地点", action=Action(tool="final_answer", params={"location": "某地"}))])


def _task(**kwargs) -> GeoTaskSpec:
    return GeoTaskSpec(task_id="v__t01", time_start=90, time_end=120, target_kind=TargetKind.still_image, **kwargs)


def _draft() -> ConfidenceJudgeDraft:
    return ConfidenceJudgeDraft(evidence_grounding=.9, final_answer_support=.9, logical_consistency=.9, input_quality_alignment=.75, notes="基于现有材料的审核")


def _report() -> ConfidenceReport:
    return merge_confidence(task_id="v__t01", format_score=1, format_reason="通过", programmatic_gates=[], draft=_draft(), judge_call_failed=False, param_score=1, audit_coverage=.9)


def test_unselected_leaking_frame_is_not_reported_as_input_leak(tmp_path: Path) -> None:
    candidate = KeyframeAssessment(timestamp=94, image_path=str(tmp_path / "candidate.jpg"), kind="teaching_ui", answer_leakage=True, selected=False, reason="候选带答案")
    packet = build_review_packet(report=_report(), task=_task(frame_assessments=[candidate]), trajectory=_trajectory(), transcript=[])
    assert not packet["selected_frames"]
    assert packet["unselected_candidates"][0]["timestamp"] == 94
    codes = [issue["code"] for issue in packet["issues"]]
    assert "no_selected_image" in codes
    assert "selected_answer_leakage" not in codes
    assert "候选帧含答案不等于最终输入泄露" in render_review_markdown(packet)


def test_selected_timestamp_and_context_are_not_fabricated(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"test")
    task = _task(image_paths=[str(image)], frame_assessments=[KeyframeAssessment(timestamp=95.25, image_path=str(image), kind="target_photo", selected=True, clean_source=False, tutorial_overlay=True, quality_score=.6, reason="带讲解字幕")])
    full = [TranscriptSegment(start=60, end=90, text="先对比地图再判断几个镜头"), TranscriptSegment(start=90, end=120, text="本题的弯道对应"), TranscriptSegment(start=120, end=150, text="另一题")]
    packet = build_review_packet(report=_report(), task=task, trajectory=_trajectory([str(image)]), transcript=full[1:2], context_transcript=full)
    assert packet["selected_frames"][0]["timestamp"] == 95.25
    assert packet["selected_frames"][0]["timestamp_source"] == "assessment_path"
    assert [s["role"] for s in packet["transcript_context"]] == ["preceding_context", "task_overlap", "following_context"]
    assert all(issue["verification"] == "requires_human_confirmation" for issue in packet["issues"])
    assert "01:35.250" in render_review_markdown(packet)


def test_multiple_show_windows_do_not_assert_multiple_missing_images() -> None:
    task = _task(status=TaskStatus.needs_review, status_reason="多出示窗仅选一张", process_intervals=[ProcessInterval(start=90, end=95, role=ProcessRole.show_source), ProcessInterval(start=100, end=110, role=ProcessRole.show_source)])
    packet = build_review_packet(report=_report(), task=task, trajectory=_trajectory(["unknown.jpg"]), transcript=[])
    issue = next(i for i in packet["issues"] if i["code"] == "stage15_review_reason")
    assert "同一张原图重复出现" in issue["repair_guidance"]
    assert packet["selected_frames"][0]["timestamp"] is None


def test_reject_still_saves_messages_and_review_sidecar(tmp_path: Path) -> None:
    trajectory = _trajectory()
    entry = format_dataset_entry(trajectory, source_video="v")
    draft = _draft().model_copy(update={"hard_gates": [HardGateHit(code="fabricated_observation", evidence="步骤1需复核")]})
    path = tmp_path / "stage4_confidence.json"
    shard = tmp_path / "sample.jsonl"
    result = run_stage4(task=_task(), transcript=[TranscriptSegment(start=90, end=120, text="某地")], freeform=FreeFormTrajectory(source_video="v", steps=[]), trajectory=trajectory, entry=entry, parameter_audits=[], judge=lambda **_: draft, out_report_path=str(path), out_jsonl_path=str(shard))
    assert result.decision == "reject"
    assert DatasetEntry.model_validate_json(shard.read_text(encoding="utf-8")).messages == entry.messages
    assert path.with_name("stage4_confidence.review.json").is_file()
    assert path.with_name("stage4_confidence.review.md").is_file()


def test_recomputed_shard_keeps_full_conversation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source" / "v__t01"
    source.mkdir(parents=True)
    trajectory = _trajectory()
    task = _task()
    freeform = FreeFormTrajectory(source_video="v", steps=[])
    for name, model in (("stage3_trajectory.json", trajectory), ("stage15_task.json", task), ("stage2_freeform_tao.json", freeform), ("stage4_confidence.json", _report())):
        (source / name).write_text(model.model_dump_json(indent=2), encoding="utf-8")
    (source / "transcript_slice.json").write_text('[{"start":90,"end":120,"text":"某地"}]', encoding="utf-8")
    (source / "stage3_tool_mapping.json").write_text("{}", encoding="utf-8")
    audits = tmp_path / "audits.json"
    audits.write_text(json.dumps({"items": [{"task_id": task.task_id, "calls": []}]}), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["recompute", "--root", str(source), "--revalidation", str(audits), "--out", str(out), "--context-root", str(tmp_path / "none")])
    recompute.main()
    shard = DatasetEntry.model_validate_json((out / "output" / "shards" / "v__t01.jsonl").read_text(encoding="utf-8"))
    assert len(shard.messages) == 3
    assert shard.messages == format_dataset_entry(trajectory, source_video="v").messages


def test_replay_does_not_promote_failed_judge_to_success() -> None:
    report = _report().model_copy(update={"judge_call_failed": True})
    with pytest.raises(ValueError, match="原审核失败"):
        recompute._replay_judge(report)()
