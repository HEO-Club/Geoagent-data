"""Re-run Stage 3/4 for selected existing Tool v2 Stage 2 trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao.run import load_freeform
from pipeline.stage3_normalize_format.format_jsonl import run_stage3
from pipeline.stage4_confidence.run import run_stage4


def _find_task_dir(root: Path, task_id: str) -> Path:
    candidates = [
        path.parent
        for path in root.rglob("stage2_freeform_tao.json")
        if path.parent.name == task_id
    ]
    if not candidates:
        raise FileNotFoundError(task_id)
    return candidates[0]


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("segments", [])
    return [TranscriptSegment.model_validate(item) for item in value]


def _run(source_dir: Path, out: Path, trees_path: Path) -> dict:
    task = GeoTaskSpec.model_validate_json(
        (source_dir / "stage15_task.json").read_text(encoding="utf-8")
    )
    transcript = _load_transcript(source_dir / "transcript_slice.json")
    freeform = load_freeform(source_dir / "stage2_freeform_tao.json")
    task_dir = out / "intermediate" / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = task_dir / "stage3_trajectory.json"
    shard_path = out / "output" / "shards" / f"{task.task_id}.jsonl"
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
    report = run_stage4(
        task=task,
        transcript=transcript,
        freeform=freeform,
        trajectory=trajectory,
        entry=entry,
        tool_mapping_path=task_dir / "stage3_tool_mapping.json",
        parameter_audit_path=task_dir / "stage3_parameter_audit.json",
        out_report_path=str(task_dir / "stage4_confidence.json"),
        out_jsonl_path=str(shard_path),
    )
    return {
        "task_id": task.task_id,
        "quality_score": report.quality_score,
        "audit_coverage": report.audit_coverage,
        "decision": report.decision,
        "hard_gates": [gate.code for gate in report.hard_gates],
        "soft_flags": [flag.code for flag in report.soft_flags],
        "parameter_readiness": (
            report.parameter_readiness.model_dump()
            if report.parameter_readiness is not None
            else None
        ),
        "judge_call_failed": report.judge_call_failed,
        "task_dir": str(task_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trees_path = args.out / "tool_trees_runtime_v2.json"
    sources = [_find_task_dir(args.source, task_id) for task_id in args.task_ids]
    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {
            pool.submit(_run, source, args.out, trees_path): source.name
            for source in sources
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(
                    f"done {task_id} score={row['quality_score']:.3f} "
                    f"decision={row['decision']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"failed {task_id}: {type(exc).__name__}: {exc}", flush=True)
    rows.sort(key=lambda item: item["task_id"])
    summary = {
        "task_count": len(rows),
        "error_count": len(errors),
        "mean_score": mean(item["quality_score"] for item in rows) if rows else 0.0,
        "items": rows,
        "errors": errors,
    }
    path = args.out / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"items", "errors"}}, ensure_ascii=False, indent=2))
    print(f"summary={path}")


if __name__ == "__main__":
    main()
