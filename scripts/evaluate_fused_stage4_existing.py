"""三并发评估既有真实 Stage2/3 结果；默认不调用外部裁判。"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.schemas.audit import (
    AnswerStatus,
    AuditSplitResult,
    GeoTaskSpec,
    TargetKind,
    TaskStatus,
)
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage3_normalize_format.format_jsonl import format_dataset_entry
from pipeline.stage3_normalize_format.params import (
    attach_operation_input_schemas,
    initial_parameter_context,
    normalize_and_validate_tool_inputs,
    update_parameter_context,
)
from pipeline.stage3_normalize_format.trees import load_forest
from pipeline.stage4_confidence.run import run_stage4

STAGE15_ROOTS = [
    Path("rerun_stage15_latest_20260822_promptfix_v2"),
    Path("rerun_stage15_latest_20260822_promptfix"),
    Path("rerun_stage15_latest_20260822_v4"),
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _load_transcript(root: Path, video_id: str) -> list[TranscriptSegment]:
    raw = json.loads(
        (root / video_id / "stage1_transcript.json").read_text(encoding="utf-8")
    )
    values = raw.get("segments", []) if isinstance(raw, dict) else raw
    return [TranscriptSegment.model_validate(item) for item in values]


def _stage15_tasks(video_id: str) -> list[GeoTaskSpec]:
    for root in STAGE15_ROOTS:
        path = root / "intermediate" / video_id / "stage_audit_split.json"
        if path.is_file():
            return AuditSplitResult.model_validate_json(
                path.read_text(encoding="utf-8")
            ).tasks
    return []


def _quality_task(video_id: str, trajectory: Trajectory) -> GeoTaskSpec:
    tasks = [
        task
        for task in _stage15_tasks(video_id)
        if task.answer_status == AnswerStatus.resolved
    ]
    if len(tasks) == 1:
        return tasks[0]
    return GeoTaskSpec(
        task_id=trajectory.id,
        time_start=min((task.time_start for task in tasks), default=0.0),
        time_end=max((task.time_end for task in tasks), default=1.0),
        target_kind=(
            TargetKind.video_derived
            if any(task.target_kind == TargetKind.video_derived for task in tasks)
            else TargetKind.still_image
        ),
        image_paths=list(trajectory.image_paths),
        status=TaskStatus.needs_review,
        status_reason=(
            f"历史轨迹聚合了 {len(tasks)} 个 Stage 1.5 task，需按 task 重新运行 Stage 2/3"
        ),
        answer_status=AnswerStatus.resolved,
        final_location_text="",
    )


def _parameter_audit(trajectory: Trajectory, forest):
    context = initial_parameter_context(trajectory.image_paths)
    calls = []
    for index, step in enumerate(trajectory.steps, start=1):
        if step.event_type != "tool_call" or step.action is None:
            continue
        params = step.action.params
        audit = normalize_and_validate_tool_inputs(
            forest,
            tool=step.action.tool,
            operation=str(params.get("operation") or "execute"),
            inputs=dict(params.get("inputs") or {}),
            step_index=index,
            available_context=context,
        )
        update_parameter_context(context, audit)
        calls.append(audit)
    return calls


def _offline_judge(**_kwargs):
    raise RuntimeError("offline benchmark: semantic judge intentionally disabled")


def _run_one(
    trajectory_path: Path,
    *,
    transcript_root: Path,
    out: Path,
    forest,
    real_judge: bool,
) -> dict[str, Any]:
    base = trajectory_path.parent
    freeform = FreeFormTrajectory.model_validate_json(
        (base / "stage2_freeform_tao.json").read_text(encoding="utf-8")
    )
    trajectory = Trajectory.model_validate_json(
        trajectory_path.read_text(encoding="utf-8")
    )
    video_id = freeform.source_video
    task = _quality_task(video_id, trajectory)
    transcript = _load_transcript(transcript_root, video_id)
    parameters = _parameter_audit(trajectory, forest)
    observations = _load_json(base / "stage2_observation_audit.json")
    entry = format_dataset_entry(trajectory, source_video=video_id)
    case_out = out / trajectory.id
    case_out.mkdir(parents=True, exist_ok=True)
    report = run_stage4(
        task=task,
        transcript=transcript,
        freeform=freeform,
        trajectory=trajectory,
        entry=entry,
        parameter_audits=parameters,
        observation_audit=observations,
        out_report_path=str(case_out / "stage4_confidence.json"),
        out_jsonl_path=str(case_out / "sample.jsonl"),
        judge=None if real_judge else _offline_judge,
    )
    return {
        "trajectory_id": trajectory.id,
        "video_id": video_id,
        "quality_score": report.quality_score,
        "audit_coverage": report.audit_coverage,
        "decision": report.decision,
        "review_priority": report.review_priority,
        "judge_call_failed": report.judge_call_failed,
        "hard_gates": [item.code for item in report.hard_gates],
        "parameter_readiness": (
            report.parameter_readiness.model_dump()
            if report.parameter_readiness is not None
            else None
        ),
        "dimensions": {
            item.name: {"score": item.score, "reason": item.reason}
            for item in report.dimensions
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--transcript-root",
        type=Path,
        default=Path("rerun_qwen35omni_stage1/intermediate"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/runs/fused_stage4_existing")
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--real-judge", action="store_true")
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    forest = attach_operation_input_schemas(
        load_forest(Path("canonical_tool_catalog_v2.json"))
    )
    paths = sorted(args.root.rglob("stage3_trajectory.json"))
    if args.ids:
        wanted = set(args.ids)
        paths = [
            path
            for path in paths
            if path.parent.name in wanted
            or any(token in path.parent.name for token in wanted)
        ]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {
            pool.submit(
                _run_one,
                path,
                transcript_root=args.transcript_root,
                out=args.out,
                forest=forest,
                real_judge=args.real_judge,
            ): path
            for path in paths
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda item: item["trajectory_id"])
    summary = {
        "judge_mode": "real" if args.real_judge else "offline",
        "max_workers": args.max_workers,
        "count": len(rows),
        "mean_score": mean(item["quality_score"] for item in rows) if rows else 0,
        "mean_coverage": mean(item["audit_coverage"] for item in rows) if rows else 0,
        "decision_counts": {
            decision: sum(item["decision"] == decision for item in rows)
            for decision in (
                "accept",
                "provisional_pass",
                "parameter_repair",
                "needs_review",
                "reject",
            )
        },
        "items": rows,
    }
    path = args.out / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report={path}")


if __name__ == "__main__":
    main()
