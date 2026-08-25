from __future__ import annotations

from pathlib import Path

from pipeline.quality.scorer import score_trajectory_quality
from pipeline.schemas.audit import GeoTaskSpec, KeyframeAssessment, TargetKind
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.quality import SemanticQualityReview
from pipeline.schemas.tools import ToolForest
from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from pipeline.stage3_normalize_format.params import attach_operation_input_schemas
from pipeline.stage3_normalize_format.trees import load_forest


def _forest() -> ToolForest:
    return attach_operation_input_schemas(
        load_forest(Path("canonical_tool_catalog.json"))
    )


def _freeform() -> FreeFormTrajectory:
    return FreeFormTrajectory.model_validate(
        {
            "source_video": "demo",
            "steps": [
                {
                    "event_type": "reasoning",
                    "thought": "图中可见河流和桥梁，先形成候选区域。",
                },
                {
                    "event_type": "tool_call",
                    "thought": "查询候选区域内的桥梁。",
                    "tool": "map_lookup",
                    "params": {"区域": "上海", "关键词": "桥梁"},
                    "observation": {"result": "检索到杨浦大桥"},
                },
                {
                    "event_type": "final",
                    "thought": "证据一致，提交地点。",
                    "tool": "final_answer",
                    "params": {"location": "上海杨浦大桥"},
                    "observation": None,
                },
            ],
        }
    )


def _trajectory(*, location: str = "上海杨浦大桥") -> Trajectory:
    return Trajectory(
        id="demo__t01",
        system_prompt="system",
        user_query="query",
        image_paths=["selected.jpg"],
        steps=[
            TrajectoryStep(event_type="reasoning", thought="图中可见河流和桥梁。"),
            TrajectoryStep(
                event_type="tool_call",
                thought="查询候选区域内的桥梁。",
                action=Action(
                    tool="map_query",
                    params={
                        "operation": "browse",
                        "purpose": "查询候选区域内的桥梁。",
                        "inputs": {"area": "上海", "query": "桥梁"},
                    },
                ),
                observation={"result": "检索到杨浦大桥"},
            ),
            TrajectoryStep(
                event_type="final",
                thought="证据一致，提交地点。",
                action=Action(tool="final_answer", params={"location": location}),
                observation=None,
            ),
        ],
    )


def _task() -> GeoTaskSpec:
    return GeoTaskSpec(
        task_id="demo__t01",
        time_start=0,
        time_end=60,
        target_kind=TargetKind.still_image,
        image_paths=["selected.jpg"],
        final_location_text="上海杨浦大桥",
        frame_assessments=[
            KeyframeAssessment(
                timestamp=2,
                image_path="selected.jpg",
                kind="target_photo",
                quality_score=0.95,
                clean_source=True,
                chain_support_score=0.95,
                selected=True,
            )
        ],
    )


def _observation_audit(*, verdict: str = "supported") -> dict:
    return {
        "accepted": verdict == "supported",
        "passes": [{"items": [{"call_id": "C001", "verdict": verdict}]}],
    }


def _semantic_review() -> SemanticQualityReview:
    return SemanticQualityReview(
        evidence_grounding=0.96,
        final_answer_support=0.95,
        reasoning_consistency=0.93,
        tool_semantics=0.95,
        input_alignment=0.96,
        summary="证据、逻辑、工具和最终答案一致。",
    )


def test_fully_audited_good_trajectory_is_accepted() -> None:
    report = score_trajectory_quality(
        _freeform(),
        _trajectory(),
        _forest(),
        task=_task(),
        observation_audit=_observation_audit(),
        trajectory_consistency={
            "conflict": False,
            "confidence": 0.95,
            "reason": "图片与轨迹一致",
        },
        parameter_audits=[],
        semantic_review=_semantic_review(),
    )
    assert report.decision == "accept"
    assert report.quality_score >= 0.9
    assert report.audit_coverage >= 0.9
    assert report.hard_failures == []


def test_structurally_good_but_unaudited_trajectory_requires_review() -> None:
    report = score_trajectory_quality(_freeform(), _trajectory(), _forest())
    assert report.decision == "needs_review"
    assert 0.65 <= report.quality_score < 0.85
    assert report.audit_coverage < 0.7
    issue_codes = {issue.code for issue in report.issues}
    assert "observation_direct_evidence_not_audited" in issue_codes
    assert "operation_inputs_not_validated" in issue_codes
    assert "stage15_context_unavailable" in issue_codes


def test_failed_observation_audit_is_hard_reject() -> None:
    report = score_trajectory_quality(
        _freeform(),
        _trajectory(),
        _forest(),
        observation_audit=_observation_audit(verdict="reject"),
    )
    assert report.decision == "reject"
    assert "observation_direct_evidence" in report.hard_failures


def test_task_answer_mismatch_is_hard_reject() -> None:
    report = score_trajectory_quality(
        _freeform(),
        _trajectory(location="北京市"),
        _forest(),
        task=_task(),
    )
    assert report.decision == "reject"
    assert "answer_matches_task" in report.hard_failures


def test_equivalent_location_wording_is_not_a_hard_mismatch() -> None:
    task = _task().model_copy(
        update={"final_location_text": "兰州近水广场东侧第二列花坛旁第三个台阶"}
    )
    report = score_trajectory_quality(
        _freeform(),
        _trajectory(location="甘肃省兰州市近水广场东侧第二列花坛旁第三个台阶"),
        _forest(),
        task=task,
    )
    assert "answer_matches_task" not in report.hard_failures


def test_stage15_review_and_image_version_mismatch_route_review_not_reject() -> None:
    task = _task().model_copy(
        update={
            "status": "needs_review",
            "status_reason": "截图带讲解覆盖",
            "image_paths": ["new_selected.jpg"],
        }
    )
    report = score_trajectory_quality(
        _freeform(), _trajectory(), _forest(), task=task
    )
    assert report.decision == "needs_review"
    assert report.quality_score <= 0.75
    assert "task_gate" not in report.hard_failures
    assert "trajectory_uses_selected_images" not in report.hard_failures


def test_repairable_parameters_do_not_override_low_audit_coverage() -> None:
    report = score_trajectory_quality(
        _freeform(),
        _trajectory(),
        _forest(),
        parameter_audits=[
            {
                "step_index": 2,
                "tool": "map_query",
                "operation": "browse",
                "valid": False,
                "readiness": "repairable",
                "issues": [],
            }
        ],
    )
    assert report.audit_coverage < 0.7
    assert report.decision == "needs_review"


def test_missing_final_is_hard_reject() -> None:
    traj = _trajectory().model_copy(update={"steps": _trajectory().steps[:-1]})
    report = score_trajectory_quality(_freeform(), traj, _forest())
    assert report.decision == "reject"
    assert "final_event_count" in report.hard_failures
    assert "location_present" in report.hard_failures


def test_low_score_with_low_audit_coverage_is_review_not_false_reject() -> None:
    traj = _trajectory()
    traj.steps[1].action = Action(
        tool="unknown_custom_tool",
        params={"operation": "execute", "purpose": "x", "inputs": {}},
    )
    report = score_trajectory_quality(_freeform(), traj, _forest())
    assert report.audit_coverage < 0.7
    assert report.decision == "needs_review"
