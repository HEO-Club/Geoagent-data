"""批量跑 stage1（可复用）+ stage1.5 选图；不跑 stage2/3。"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import clear_settings_cache, get_settings
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage1_transcript.run import run_stage1
from pipeline.stage_audit_split.run import run_audit_split

EXCLUDE = {"BV1o94y1H7GK", "BV1Q94y1x7rY", "BV1ww411Q7Fo"}


def _ok_ids_from_reports(runs_dir: Path) -> set[str]:
    """合并所有 batch 报告里已成功的 video_id，避免重复重跑。"""
    done: set[str] = set()
    for report in runs_dir.glob("batch_audit_select_*.json"):
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for item in data.get("results", []):
            if item.get("ok") and item.get("video_id"):
                done.add(str(item["video_id"]))
    return done


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["segments"] if isinstance(raw, dict) and "segments" in raw else raw
    return [TranscriptSegment.model_validate(x) for x in items]


def _summarize(audit: object) -> dict[str, object]:
    tasks = []
    for t in getattr(audit, "tasks", []) or []:
        tasks.append(
            {
                "task_id": t.task_id,
                "status": t.status.value,
                "status_reason": t.status_reason,
                "target_kind": t.target_kind.value,
                "time_start": t.time_start,
                "time_end": t.time_end,
                "expected_image_count": t.expected_image_count,
                "keyframe_timestamps": t.keyframe_timestamps,
                "image_paths": [str(Path(p).as_posix()) for p in t.image_paths],
                "final_location_text": t.final_location_text,
                "answer_status": t.answer_status.value,
                "n_assessments": len(t.frame_assessments),
            }
        )
    return {
        "video_id": audit.video_id,
        "decision": audit.decision.value,
        "reason": audit.reason,
        "tasks": tasks,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    clear_settings_cache()
    settings = get_settings()
    assert settings.ALLOW_REAL_API, "ALLOW_REAL_API must be true"

    root = Path("data/raw_videos")
    inter = Path(settings.INTERMEDIATE_DIR)
    runs = Path(settings.RUNS_DIR)
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = runs / f"batch_audit_select_{stamp}.json"
    log_path = runs / f"batch_audit_select_{stamp}.log"

    skip_ok = _ok_ids_from_reports(runs)
    exclude = set(EXCLUDE) | skip_ok
    videos = sorted(p for p in root.glob("*.mp4") if p.stem not in exclude)
    results: list[dict[str, object]] = []
    print(
        f"total={len(videos)} exclude={sorted(exclude)} "
        f"(base={sorted(EXCLUDE)}, skip_ok={sorted(skip_ok)})",
        flush=True,
    )

    for i, video in enumerate(videos, start=1):
        vid = video.stem
        print(f"==== [{i}/{len(videos)}] {vid} ====", flush=True)
        item: dict[str, object] = {"video_id": vid, "ok": False}
        try:
            tpath = inter / vid / "stage1_transcript.json"
            if tpath.is_file():
                print(f"skip stage1, use {tpath}", flush=True)
                transcript = _load_transcript(tpath)
            else:
                print("run stage1...", flush=True)
                transcript = run_stage1(str(video))
            for name in (
                "stage_audit_split.json",
                "stage_audit_split_draft.json",
                "stage_audit_split.partial.json",
            ):
                p = inter / vid / name
                if p.is_file():
                    p.unlink()
            audit = run_audit_split(
                str(video),
                transcript,
                out_path=str(inter / vid / "stage_audit_split.json"),
                resume_tasks=False,
            )
            summary = _summarize(audit)
            item.update(summary)
            item["ok"] = True
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001
            item["error"] = f"{type(exc).__name__}: {exc}"
            item["traceback"] = traceback.format_exc()
            print(f"FAILED {vid}: {item['error']}", flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n==== {vid} ====\n{item['traceback']}\n")
        results.append(item)
        out_path.write_text(
            json.dumps(
                {
                    "started": stamp,
                    "exclude": sorted(EXCLUDE),
                    "done": i,
                    "total": len(videos),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    ok_n = sum(1 for r in results if r.get("ok"))
    print(f"ALL_DONE ok={ok_n}/{len(results)} report={out_path}", flush=True)


if __name__ == "__main__":
    main()
