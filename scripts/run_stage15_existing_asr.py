"""复用既有 Stage 1 字幕，在隔离目录批量运行最新 Stage 1.5。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import clear_settings_cache
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage_audit_split.run import run_audit_split


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("segments", []) if isinstance(raw, dict) else raw
    return [TranscriptSegment.model_validate(item) for item in items]


def _run_one(
    video: Path,
    transcript_root: Path,
    intermediate: Path,
    *,
    resume: bool,
) -> dict:
    video_id = video.stem
    transcript_path = transcript_root / video_id / "stage1_transcript.json"
    if not transcript_path.is_file():
        raise FileNotFoundError(transcript_path)
    out_path = intermediate / video_id / "stage_audit_split.json"
    result = run_audit_split(
        str(video),
        _load_transcript(transcript_path),
        out_path=str(out_path),
        resume_tasks=resume,
    )
    tasks = []
    for task in result.tasks:
        selected = [item for item in task.frame_assessments if item.selected]
        tasks.append(
            {
                "task_id": task.task_id,
                "status": task.status.value,
                "answer_status": task.answer_status.value,
                "time_start": task.time_start,
                "time_end": task.time_end,
                "final_location_text": task.final_location_text,
                "image_count": len(task.image_paths),
                "image_paths": task.image_paths,
                "selected_quality": [item.quality_score for item in selected],
                "selected_chain_support": [
                    item.chain_support_score for item in selected
                ],
                "selected_clean": [item.clean_source for item in selected],
                "selected_leakage": [item.answer_leakage for item in selected],
                "status_reason": task.status_reason,
            }
        )
    return {
        "video_id": video_id,
        "ok": True,
        "decision": result.decision.value,
        "reason": result.reason,
        "task_count": len(tasks),
        "accepted_tasks": sum(item["status"] == "accepted" for item in tasks),
        "review_tasks": sum(item["status"] == "needs_review" for item in tasks),
        "rejected_tasks": sum(item["status"] == "rejected" for item in tasks),
        "selected_images": sum(item["image_count"] for item in tasks),
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--transcript-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.out_root.resolve()
    intermediate = root / "intermediate"
    os.environ["INTERMEDIATE_DIR"] = str(intermediate)
    os.environ["SELECTED_DIR"] = str(root / "selected")
    os.environ["RUNS_DIR"] = str(root / "runs")
    os.environ["CACHE_DIR"] = str(root / "cache")
    clear_settings_cache()

    videos = sorted(args.video_dir.glob("*.mp4"))
    results = []
    report_path = root / "runs" / "stage15_batch_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {
            pool.submit(
                _run_one,
                video,
                args.transcript_root,
                intermediate,
                resume=args.resume,
            ): video
            for video in videos
        }
        for future in as_completed(futures):
            video = futures[future]
            try:
                item = future.result()
            except Exception as exc:  # noqa: BLE001
                item = {
                    "video_id": video.stem,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            results.append(item)
            report = {
                "video_count": len(videos),
                "completed": len(results),
                "ok": sum(row.get("ok", False) for row in results),
                "failed": sum(not row.get("ok", False) for row in results),
                "results": sorted(results, key=lambda row: str(row["video_id"])),
            }
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {key: value for key, value in item.items() if key not in {"tasks", "traceback"}},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
