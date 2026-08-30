"""Run Stage 2/3 and fused Stage 4 on the nine existing evaluation videos."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.schemas.audit import AuditSplitResult, GeoTaskSpec, TaskStatus
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao.run import run_stage2
from pipeline.stage3_normalize_format.format_jsonl import run_stage3
from pipeline.stage4_confidence.run import run_stage4

DEFAULT_IDS = [
    "01_BV13m61BJEQC",
    "02_BV1RC2XBdEnq",
    "03_BV1JFp7zdEzH",
    "04_BV1SbeqzoE5y",
    "05_BV1ibbWzQEfN",
    "06_BV1CDtizwEhE",
    "08_BV1jjN2zdEQH",
    "09_BV15zyUY2EKs",
    "10_BV1ze2JY5EMV",
]

STAGE15_ROOTS = [
    Path("rerun_stage15_latest_20260822_promptfix_v2"),
    Path("rerun_stage15_latest_20260822_promptfix"),
    Path("rerun_stage15_latest_20260822_v4"),
    Path("rerun_stage15_latest_20260822_v3"),
]


def _video_id(case_id: str) -> str:
    return case_id.split("_", 1)[-1]


def _load_transcript(video_id: str) -> list[TranscriptSegment]:
    path = Path("data/transcripts") / f"{video_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("segments", [])
    return [TranscriptSegment.model_validate(item) for item in value]


def _find_audit(video_id: str) -> tuple[AuditSplitResult, Path]:
    for root in STAGE15_ROOTS:
        candidates = list(root.rglob(f"{video_id}/stage_audit_split.json"))
        if not candidates:
            candidates = [
                path
                for path in root.rglob("stage_audit_split.json")
                if path.parent.name == video_id
            ]
        for path in candidates:
            audit = AuditSplitResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if audit.tasks:
                return audit, path
    raise FileNotFoundError(f"未找到 {video_id} 的可用 Stage 1.5 tasks")


def _task_transcript(
    task: GeoTaskSpec, transcript: list[TranscriptSegment]
) -> list[TranscriptSegment]:
    if task.segment_start_idx is not None and task.segment_end_idx is not None:
        start = max(0, task.segment_start_idx)
        end = min(len(transcript) - 1, task.segment_end_idx)
        if start <= end:
            return transcript[start : end + 1]
    selected = [
        segment
        for segment in transcript
        if segment.end > task.time_start + 1e-6
        and segment.start < task.time_end - 1e-6
    ]
    return selected or transcript


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _run_task(
    *,
    case_id: str,
    task: GeoTaskSpec,
    transcript: list[TranscriptSegment],
    out_root: Path,
    trees_path: Path,
) -> dict[str, Any]:
    video_id = _video_id(case_id)
    task_dir = out_root / "intermediate" / case_id / "tasks" / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    task_path = task_dir / "stage15_task.json"
    task_path.write_text(task.model_dump_json(indent=2), encoding="utf-8")
    task_segments = _task_transcript(task, transcript)
    (task_dir / "transcript_slice.json").write_text(
        json.dumps([item.model_dump() for item in task_segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    freeform_path = task_dir / "stage2_freeform_tao.json"
    freeform = run_stage2(
        str(Path("data/raw_videos") / f"{video_id}.mp4"),
        task_segments,
        out_path=str(freeform_path),
        image_paths=task.image_paths,
        source_video=task.task_id,
        max_attempts=3,
        task=task,
        observation_context=transcript,
    )

    trajectory_path = task_dir / "stage3_trajectory.json"
    shard_path = out_root / "output" / "shards" / f"{task.task_id}.jsonl"
    entry = run_stage3(
        freeform,
        trees_path=trees_path,
        out_trajectory_path=str(trajectory_path),
        out_jsonl_path=str(shard_path),
        image_paths=task.image_paths,
        shard_id=task.task_id,
    )
    trajectory = Trajectory.model_validate_json(
        trajectory_path.read_text(encoding="utf-8")
    )
    mapping_path = task_dir / "stage3_tool_mapping.json"
    parameter_path = task_dir / "stage3_parameter_audit.json"
    report_path = task_dir / "stage4_confidence.json"
    report = run_stage4(
        task=task,
        transcript=task_segments,
        freeform=freeform,
        trajectory=trajectory,
        entry=entry,
        tool_mapping_path=mapping_path,
        parameter_audit_path=parameter_path,
        observation_audit_path=freeform_path.with_name("stage2_observation_audit.json"),
        review_context_transcript=transcript,
        source_video_path=str(Path("data/raw_videos") / f"{video_id}.mp4"),
        out_report_path=str(report_path),
        out_jsonl_path=str(shard_path),
    )

    final = trajectory.steps[-1]
    location = final.action.params.get("location") if final.action else None
    mapping = _read_json(mapping_path)
    parameter = _read_json(parameter_path)
    calls = parameter.get("calls", []) if isinstance(parameter.get("calls"), list) else []
    return {
        "case_id": case_id,
        "task_id": task.task_id,
        "stage15_status": task.status.value,
        "images": len(task.image_paths),
        "location": location,
        "stage2_steps": len(freeform.steps),
        "stage3_tools": sorted(
            {
                step.action.tool
                for step in trajectory.steps
                if step.event_type == "tool_call" and step.action is not None
            }
        ),
        "tool_calls_before": mapping.get("tool_calls_before_stage3"),
        "tool_calls_after": mapping.get("tool_calls_after_stage3"),
        "parameter_readiness": {
            level: sum(call.get("readiness") == level for call in calls)
            for level in ("ready", "context_resolvable", "repairable", "invalid")
        },
        "quality_score": report.quality_score,
        "audit_coverage": report.audit_coverage,
        "decision": report.decision,
        "hard_gates": [gate.code for gate in report.hard_gates],
        "soft_flags": [flag.code for flag in report.soft_flags],
        "judge_call_failed": report.judge_call_failed,
        "task_dir": str(task_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--task-ids", nargs="*", default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    trees_path = args.out / "tool_trees_runtime_v2.json"
    jobs: list[tuple[str, GeoTaskSpec, list[TranscriptSegment], str]] = []
    stage15_sources: dict[str, str] = {}
    for case_id in args.ids:
        video_id = _video_id(case_id)
        transcript = _load_transcript(video_id)
        audit, audit_path = _find_audit(video_id)
        stage15_sources[case_id] = str(audit_path)
        for task in audit.tasks:
            if task.status == TaskStatus.rejected:
                continue
            if args.task_ids and task.task_id not in set(args.task_ids):
                continue
            jobs.append((case_id, task, transcript, str(audit_path)))

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        future_map = {
            pool.submit(
                _run_task,
                case_id=case_id,
                task=task,
                transcript=transcript,
                out_root=args.out,
                trees_path=trees_path,
            ): (case_id, task.task_id)
            for case_id, task, transcript, _ in jobs
        }
        for future in as_completed(future_map):
            case_id, task_id = future_map[future]
            try:
                row = future.result()
                rows.append(row)
                print(
                    f"done {case_id}/{task_id} score={row['quality_score']:.3f} "
                    f"decision={row['decision']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "case_id": case_id,
                        "task_id": task_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"failed {case_id}/{task_id}: {type(exc).__name__}: {exc}", flush=True)

    rows.sort(key=lambda row: (row["case_id"], row["task_id"]))
    summary = {
        "catalog": "canonical_tool_catalog_v2.json",
        "model": "claude-sonnet-5",
        "case_count": len({row["case_id"] for row in rows}),
        "task_count": len(rows),
        "error_count": len(errors),
        "mean_score": mean(row["quality_score"] for row in rows) if rows else 0.0,
        "mean_coverage": mean(row["audit_coverage"] for row in rows) if rows else 0.0,
        "decision_counts": {
            decision: sum(row["decision"] == decision for row in rows)
            for decision in (
                "accept",
                "provisional_pass",
                "parameter_repair",
                "needs_review",
                "reject",
            )
        },
        "stage15_sources": stage15_sources,
        "items": rows,
        "errors": errors,
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"items", "errors"}}, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
