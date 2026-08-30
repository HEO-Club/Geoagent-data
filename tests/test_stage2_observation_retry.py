"""Bounded Observation regeneration: private feedback, three attempts, fail open."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas.audit import GeoTaskSpec, TargetKind
from pipeline.schemas.clues import ClueExtractionResult
from pipeline.schemas.confidence import ConfidenceJudgeDraft
from pipeline.schemas.dataset import DatasetEntry
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao import run as stage2
from pipeline.stage2_freeform_tao.observation_review import (
    ObservationReviewItem,
    ObservationReviewResult,
    observation_fingerprint,
    review_observations,
)
from pipeline.stage3_normalize_format.format_jsonl import run_stage3
from pipeline.stage4_confidence.rules import evaluate_observation_audit
from pipeline.stage4_confidence.run import run_stage4


def _result(value: str = "发现原文地标", thought: str = "检索已知地标"):
    return stage2._LLMFreeFormResult.model_validate({"steps": [
        {"event_type": "tool_call", "thought": thought, "tool": "web_search",
         "params": {"operation": "keyword_search", "inputs": {"query": "地标"}},
         "observation": {"result": value}},
        {"event_type": "final", "thought": "提交位置", "tool": "final_answer",
         "params": {"location": "示例地"}, "observation": None},
    ]})


def _review(verdict: str = "supported", confidence: float = 1.0):
    return ObservationReviewResult(items=[ObservationReviewItem(
        step_index=1, verdict=verdict, confidence=confidence,
        reason="上一轮补写了42公里，原始材料没有距离数值" if verdict == "fabricated" else "检索动作和结果在材料中出现",
        correction="保留地标匹配，删除无来源距离", evidence="对应片段0–10秒",
    )])


def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("STAGE2_OBSERVATION_REVIEW", "true")
    monkeypatch.setenv("STAGE2_MAX_GENERATIONS", "3")
    clear_settings_cache()
    monkeypatch.setattr(stage2, "extract_working_scope", lambda _: ClueExtractionResult())
    dest = tmp_path / "stage2_freeform_tao.json"
    transcript = [TranscriptSegment(start=0, end=10, text="查找地标后确认是示例地")]
    return dest, transcript


def _run(dest, transcript, reviewer, **kwargs):
    return stage2.run_stage2("v.mp4", transcript, out_path=str(dest), image_paths=["frame.jpg"], observation_reviewer=reviewer, **kwargs)


def _trace(dest: Path) -> dict:
    return json.loads(dest.with_name("stage2_observation_audit.json").read_text(encoding="utf-8"))


def test_first_pass_stops_without_extra_generation(monkeypatch, tmp_path):
    dest, transcript = _setup(monkeypatch, tmp_path)
    prompts = []

    def generate(prompt, schema, **kwargs):
        prompts.append(prompt)
        assert kwargs["max_attempts"] == 1
        return _result()

    monkeypatch.setattr(stage2, "call_structured", generate)
    _run(dest, transcript, lambda **_: _review())
    assert len(prompts) == 1
    assert _trace(dest)["accepted"] is True
    assert _trace(dest)["generation_count"] == 1
    hard, soft, observed = evaluate_observation_audit(_trace(dest))
    assert observed and not hard and not soft


def test_retry_warning_contains_specific_error_but_not_training_output(monkeypatch, tmp_path):
    dest, transcript = _setup(monkeypatch, tmp_path)
    prompts = []
    reviews = iter([_review("fabricated"), _review()])

    def generate(prompt, schema, **kwargs):
        prompts.append(prompt)
        return _result("距离42公里" if len(prompts) == 1 else "发现原文地标")

    monkeypatch.setattr(stage2, "call_structured", generate)
    result = _run(dest, transcript, lambda **_: next(reviews))
    assert len(prompts) == 2
    assert "（上一轮 Observation 纠错提醒" in prompts[1]
    assert "上一轮第1步" in prompts[1] and "42公里" in prompts[1]
    assert "保留地标匹配，删除无来源距离" in prompts[1]
    assert "纠错提醒" not in result.model_dump_json()
    assert _trace(dest)["passes"][-1]["items"][0]["verdict"] == "supported"
    assert _trace(dest)["accepted"] is True


def test_three_bad_generations_still_continue_stage3_and_stage4(monkeypatch, tmp_path):
    dest, transcript = _setup(monkeypatch, tmp_path)
    generations = []
    monkeypatch.setattr(stage2, "call_structured", lambda *_a, **_k: generations.append(1) or _result("距离42公里"))
    freeform = _run(dest, transcript, lambda **_: _review("fabricated"), max_attempts=99)
    audit = _trace(dest)
    assert len(generations) == audit["generation_count"] == audit["generation_limit"] == 3
    assert audit["selected_generation"] == 3
    assert audit["accepted"] is False and audit["continued_with_issues"] is True
    assert audit["stop_reason"] == "generation_limit"
    trajectory_path = tmp_path / "stage3_trajectory.json"
    shard = tmp_path / "sample.jsonl"
    entry = run_stage3(freeform, trees_path=tmp_path / "tools.json",
        out_trajectory_path=str(trajectory_path), out_jsonl_path=str(shard),
        image_paths=["frame.jpg"], compile_params=False)
    trajectory = Trajectory.model_validate_json(trajectory_path.read_text(encoding="utf-8"))
    report = run_stage4(task=GeoTaskSpec(task_id="v", time_start=0, time_end=10, target_kind=TargetKind.still_image),
        transcript=transcript, freeform=freeform, trajectory=trajectory, entry=entry,
        parameter_audit_path=tmp_path / "stage3_parameter_audit.json",
        observation_audit_path=dest.with_name("stage2_observation_audit.json"),
        judge=lambda **_: ConfidenceJudgeDraft(evidence_grounding=.9, final_answer_support=.9, logical_consistency=.9, input_quality_alignment=.9, notes="测试裁判"),
        out_report_path=str(tmp_path / "stage4_confidence.json"), out_jsonl_path=str(shard))
    assert report.decision == "reject"
    assert shard.is_file() and any(m.role == "assistant" for m in DatasetEntry.model_validate_json(shard.read_text(encoding="utf-8")).messages)


@pytest.mark.parametrize("verdict,confidence", [("uncertain", 1.0), ("fabricated", .5)])
def test_uncertainty_does_not_spend_another_generation(monkeypatch, tmp_path, verdict, confidence):
    dest, transcript = _setup(monkeypatch, tmp_path)
    count = []
    monkeypatch.setattr(stage2, "call_structured", lambda *_a, **_k: count.append(1) or _result())
    _run(dest, transcript, lambda **_: _review(verdict, confidence))
    assert len(count) == 1
    assert _trace(dest)["accepted"] is False


def test_audit_failure_keeps_valid_generation(monkeypatch, tmp_path):
    dest, transcript = _setup(monkeypatch, tmp_path)
    count = []
    monkeypatch.setattr(stage2, "call_structured", lambda *_a, **_k: count.append(1) or _result())

    def fail(**_):
        raise RuntimeError("service unavailable")

    result = _run(dest, transcript, fail)
    assert len(count) == 1 and result.steps[-1].params["location"] == "示例地"
    assert _trace(dest)["passes"][-1]["status"] == "audit_failed"
    assert _trace(dest)["continued_with_issues"] is True


def test_generation_failure_retains_last_valid_draft(monkeypatch, tmp_path):
    dest, transcript = _setup(monkeypatch, tmp_path)
    count = []

    def generate(*_a, **_k):
        count.append(1)
        if len(count) > 1:
            raise ValueError("malformed response")
        return _result("保留的有效结构")

    monkeypatch.setattr(stage2, "call_structured", generate)
    result = _run(dest, transcript, lambda **_: _review("fabricated"))
    assert len(count) == 3
    assert result.steps[0].observation == {"result": "保留的有效结构"}
    assert _trace(dest)["selected_generation"] == 1


def test_auditor_must_cover_every_call(monkeypatch, tmp_path):
    dest, transcript = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(stage2, "call_structured", lambda *_a, **_k: _result())
    _run(dest, transcript, lambda **_: ObservationReviewResult(items=[]))
    assert _trace(dest)["passes"][-1]["status"] == "audit_failed"
    assert _trace(dest)["accepted"] is False


def test_reviewer_gets_adjacent_method_context_without_groundtruth():
    transcript = [TranscriptSegment(start=90, end=120, text="本题河道对应")]
    full = [TranscriptSegment(start=60, end=90, text="先对比地图"), *transcript]
    task = GeoTaskSpec(task_id="v__t04", time_start=90, time_end=120, target_kind=TargetKind.still_image, final_location_text="不得注入的答案")
    captured: dict[str, Any] = {}

    def reviewer(**kwargs):
        captured.update(kwargs)
        return _review()

    trajectory = stage2.FreeFormTrajectory(source_video="v", steps=_result().steps)
    review_observations(trajectory, transcript=transcript, images=[], task=task, context_transcript=full, reviewer=reviewer)
    assert "先对比地图" in captured["prompt"]
    assert "preceding_context" in captured["prompt"]
    assert "不得注入的答案" not in captured["prompt"]


def test_stale_observation_audit_is_not_reused():
    trajectory = Trajectory.model_validate({"id": "v", "system_prompt": "s", "user_query": "q", "steps": [
        {"event_type": "tool_call", "thought": "查询", "action": {"tool": "web_search", "params": {}}, "observation": {"result": "已修改"}},
        {"event_type": "final", "thought": "提交", "action": {"tool": "final_answer", "params": {"location": "某地"}}},
    ]})
    audit = {"policy": "bounded_observation_regeneration_v1", "accepted": True, "passes": [{"status": "complete", "items": [
        {"step_index": 1, "verdict": "supported", "observation_sha256": observation_fingerprint({"result": "旧值"})},
    ]}]}
    hard, soft, observed = evaluate_observation_audit(audit, trajectory=trajectory)
    assert not hard and not observed
    assert soft[0].code == "observation_audit_stale"


def test_three_structural_failures_do_not_fabricate_a_trajectory(monkeypatch, tmp_path):
    dest, transcript = _setup(monkeypatch, tmp_path)
    count = []

    def fail(*_a, **_k):
        count.append(1)
        raise ValueError("no valid structure")

    monkeypatch.setattr(stage2, "call_structured", fail)
    with pytest.raises(ValueError, match="no valid structure"):
        _run(dest, transcript, lambda **_: _review())
    assert len(count) == 3
    assert not dest.exists()


def test_reasoning_only_does_not_call_observation_reviewer(monkeypatch, tmp_path):
    dest, transcript = _setup(monkeypatch, tmp_path)
    result = _result().model_copy(update={"steps": [_result().steps[-1]]})
    monkeypatch.setattr(stage2, "call_structured", lambda *_a, **_k: result)

    def must_not_review(**_):
        raise AssertionError("no tool calls need no audit")

    _run(dest, transcript, must_not_review)
    assert _trace(dest)["accepted"] is True
    assert _trace(dest)["generation_count"] == 1
