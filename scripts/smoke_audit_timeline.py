"""Smoke: 4 videos through stage1.5 selection with process timeline."""

from __future__ import annotations

import json
import logging
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import clear_settings_cache, get_settings
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage_audit_split.run import run_audit_split

VIDEOS = [
    "BV1yu4y1z7JF",
    "BV1rPxEeVEEV",
    "BV1ze2JY5EMV",
    "BV1ny411v71H",
]


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["segments"] if isinstance(raw, dict) and "segments" in raw else raw
    return [TranscriptSegment.model_validate(x) for x in items]


def _clear_audit(vid: str, inter: Path, selected: Path) -> None:
    root = inter / vid
    for name in (
        "stage_audit_split.json",
        "stage_audit_split_draft.json",
        "stage_audit_split.partial.json",
    ):
        p = root / name
        if p.is_file():
            p.unlink()
    tasks = root / "tasks"
    if tasks.is_dir():
        shutil.rmtree(tasks)
    sel = selected / vid
    if sel.is_dir():
        shutil.rmtree(sel)
    cache = Path(get_settings().CACHE_DIR) / "audit_candidates" / vid
    if cache.is_dir():
        shutil.rmtree(cache)


def _summarize(audit: object) -> dict[str, object]:
    tasks = []
    for t in getattr(audit, "tasks", []) or []:
        intervals = []
        for iv in getattr(t, "process_intervals", []) or []:
            intervals.append(
                {
                    "start": float(iv.start),
                    "end": float(iv.end),
                    "role": iv.role.value if hasattr(iv.role, "value") else str(iv.role),
                    "confidence": float(iv.confidence),
                }
            )
        selected = [a for a in t.frame_assessments if a.selected]
        tasks.append(
            {
                "task_id": t.task_id,
                "status": t.status.value,
                "status_reason": t.status_reason,
                "target_kind": t.target_kind.value,
                "time_start": t.time_start,
                "time_end": t.time_end,
                "expected_image_count": t.expected_image_count,
                "multi_target_images": t.multi_target_images,
                "keyframe_timestamps": t.keyframe_timestamps,
                "image_names": [Path(p).name for p in t.image_paths],
                "final_location_text": t.final_location_text,
                "visual_evidence_brief": (t.visual_evidence_brief or "")[:240],
                "process_intervals": intervals,
                "n_assessments": len(t.frame_assessments),
                "selected_kinds": [a.kind for a in selected],
                "selected_roles": [a.evidence_role for a in selected],
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

    inter = Path(settings.INTERMEDIATE_DIR)
    selected = Path(settings.SELECTED_DIR)
    runs = Path(settings.RUNS_DIR)
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = runs / f"smoke_audit_timeline_{stamp}.json"

    results: list[dict[str, object]] = []
    for i, vid in enumerate(VIDEOS, start=1):
        print(f"==== [{i}/{len(VIDEOS)}] {vid} ====", flush=True)
        item: dict[str, object] = {"video_id": vid, "ok": False}
        try:
            video = Path("data/raw_videos") / f"{vid}.mp4"
            tpath = inter / vid / "stage1_transcript.json"
            if not video.is_file():
                raise FileNotFoundError(video)
            if not tpath.is_file():
                raise FileNotFoundError(tpath)
            _clear_audit(vid, inter, selected)
            transcript = _load_transcript(tpath)
            print(f"reuse transcript segments={len(transcript)}", flush=True)
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
            print(item["traceback"], flush=True)
        results.append(item)
        out_path.write_text(
            json.dumps(
                {"started": stamp, "videos": VIDEOS, "results": results},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    ok_n = sum(1 for r in results if r.get("ok"))
    print(f"ALL_DONE ok={ok_n}/{len(results)} report={out_path}", flush=True)


if __name__ == "__main__":
    main()
