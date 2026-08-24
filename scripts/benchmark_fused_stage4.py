"""三并发离线基准：对比原 Stage4 与融合 Stage4 的路由差异。"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.schemas.audit import GeoTaskSpec, KeyframeAssessment, TargetKind
from pipeline.schemas.confidence import ConfidenceJudgeDraft
from pipeline.schemas.dataset import ChatMessage, DatasetEntry
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage3_normalize_format.params import attach_operation_input_schemas
from pipeline.stage3_normalize_format.trees import load_forest
from pipeline.stage4_confidence.run import merge_confidence, run_stage4


def _judge(**_kwargs) -> ConfidenceJudgeDraft:
    return ConfidenceJudgeDraft(
        evidence_grounding=0.95,
        final_answer_support=0.95,
        tool_param_correctness=0.95,
        logical_consistency=0.95,
        input_quality_alignment=0.95,
    )


def _case(case_id: str, image: Path):
    task = GeoTaskSpec(
        task_id=case_id,
        time_start=0,
        time_end=10,
        target_kind=TargetKind.still_image,
        image_paths=[str(image)],
        final_location_text="某市某镇",
        frame_assessments=[
            KeyframeAssessment(
                timestamp=1,
                image_path=str(image),
                kind="target_photo",
                quality_score=0.95,
                clean_source=True,
                chain_support_score=0.95,
                selected=True,
            )
        ],
    )
    freeform = FreeFormTrajectory.model_validate(
        {
            "source_video": case_id,
            "steps": [
                {
                    "event_type": "tool_call",
                    "thought": "查询候选区域。",
                    "tool": "map_query",
                    "params": {"area": "某市", "query": "目标"},
                    "observation": {"result": "找到某市某镇"},
                },
                {
                    "event_type": "final",
                    "thought": "提交地点。",
                    "tool": "final_answer",
                    "params": {"location": "某市某镇"},
                    "observation": None,
                },
            ],
        }
    )
    trajectory = Trajectory(
        id=case_id,
        system_prompt="system",
        user_query="query",
        image_paths=[str(image)],
        steps=[
            TrajectoryStep(
                event_type="tool_call",
                thought="查询候选区域。",
                action=Action(
                    tool="map_query",
                    params={
                        "operation": "browse",
                        "purpose": "查询候选区域。",
                        "inputs": {"area": "某市", "query": "目标"},
                    },
                ),
                observation={"result": "找到某市某镇"},
            ),
            TrajectoryStep(
                event_type="final",
                thought="提交地点。",
                action=Action(
                    tool="final_answer", params={"location": "某市某镇"}
                ),
            ),
        ],
    )
    entry = DatasetEntry(
        id=case_id,
        source_video=case_id,
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="query"),
        ],
    )
    return task, freeform, trajectory, entry


def _run_case(case_id: str, out: Path, forest) -> dict:
    image = out / f"{case_id}.jpg"
    image.write_bytes(b"jpg")
    task, freeform, trajectory, entry = _case(case_id, image)
    readiness = "repairable" if case_id == "repairable_params" else "ready"
    parameter_audit = {
        "calls": [
            {
                "step_index": 1,
                "tool": "map_query",
                "operation": "browse",
                "valid": readiness == "ready",
                "readiness": readiness,
                "issues": (
                    []
                    if readiness == "ready"
                    else [
                        {
                            "code": "required_input_missing",
                            "severity": "error",
                            "field": "query",
                            "message": "缺少查询对象",
                        }
                    ]
                ),
            }
        ]
    }
    verdict = "reject" if case_id == "fabricated_observation" else "supported"
    observation_audit = {
        "accepted": verdict == "supported",
        "passes": [{"items": [{"call_id": "C001", "verdict": verdict}]}],
    }
    legacy = merge_confidence(
        task_id=case_id,
        format_score=1.0,
        format_reason="格式通过",
        programmatic_gates=[],
        draft=_judge(),
        judge_call_failed=False,
    )
    fused = run_stage4(
        task=task,
        transcript=[TranscriptSegment(start=0, end=1, text="最终地点是某市某镇")],
        freeform=freeform,
        trajectory=trajectory,
        entry=entry,
        parameter_audit=parameter_audit,
        observation_audit=observation_audit,
        trajectory_consistency={"conflict": False, "confidence": 1.0},
        forest=forest,
        out_report_path=str(out / f"{case_id}.confidence.json"),
        out_jsonl_path=str(out / f"{case_id}.jsonl"),
        judge=_judge,
    )
    return {
        "case_id": case_id,
        "legacy": {
            "quality_score": legacy.quality_score,
            "review_priority": legacy.review_priority,
        },
        "fused": {
            "quality_score": fused.quality_score,
            "audit_coverage": fused.audit_coverage,
            "decision": fused.decision,
            "review_priority": fused.review_priority,
            "hard_gates": [item.code for item in fused.hard_gates],
            "parameter_readiness_counts": fused.parameter_readiness_counts,
        },
    }


def main() -> None:
    out = Path("data/runs/fused_stage4_benchmark")
    out.mkdir(parents=True, exist_ok=True)
    forest = attach_operation_input_schemas(
        load_forest(Path("canonical_tool_catalog.json"))
    )
    case_ids = ["good", "repairable_params", "fabricated_observation"]
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_run_case, case, out, forest): case for case in case_ids}
        for future in as_completed(futures):
            rows.append(future.result())
    report = {"max_workers": 3, "cases": sorted(rows, key=lambda row: row["case_id"])}
    path = out / "benchmark.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={path}")


if __name__ == "__main__":
    main()
