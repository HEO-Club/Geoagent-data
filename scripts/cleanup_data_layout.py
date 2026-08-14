"""清理杂乱中间产物，并把已有选中图迁到 data/selected/。

保留：
- data/raw_videos、data/transcripts、data/groundtruth
- 各视频 stage1_transcript.json / stage_audit_split.json / task_audit.json
- accepted 的 stage2/stage3 JSON
- 最终选中图（迁入 data/selected/）

清理：
- intermediate 根目录 _rerun_/_eval_/_batch_ 等测试摘要（归档到 data/runs/archive/）
- tasks/*/candidates/ 探测帧目录
- video 根目录散落的 *_t*.jpg 副本
- *.partial.json、*before_*.json
- .cache/audit_candidates、过大的 .cache/keyframes（可选）
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTER = ROOT / "data" / "intermediate"
SELECTED = ROOT / "data" / "selected"
RUNS = ROOT / "data" / "runs"
CACHE = ROOT / ".cache"


def _copy_selected(src: Path, video_id: str, task_id: str) -> Path | None:
    if not src.is_file():
        return None
    dest_dir = SELECTED / video_id / task_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    return dest


def migrate_video(video_dir: Path) -> dict[str, object]:
    video_id = video_dir.name
    stats: dict[str, object] = {
        "video_id": video_id,
        "selected": 0,
        "removed_candidates": 0,
        "removed_loose_jpg": 0,
        "removed_partials": 0,
        "updated_json": False,
    }
    audit_path = video_dir / "stage_audit_split.json"
    if audit_path.is_file():
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        changed = False
        for task in data.get("tasks", []):
            task_id = str(task.get("task_id") or "")
            new_paths: list[str] = []
            for raw in task.get("image_paths") or []:
                src = Path(raw)
                if not src.is_file():
                    # 尝试在 candidates / 根目录找同名
                    alt = video_dir / "tasks" / task_id / "candidates" / src.name
                    if not alt.is_file():
                        alt = video_dir / src.name
                    src = alt if alt.is_file() else src
                dest = _copy_selected(src, video_id, task_id)
                if dest is not None:
                    new_paths.append(str(dest.resolve()))
                    stats["selected"] = int(stats["selected"]) + 1
                    changed = True
                elif raw:
                    new_paths.append(str(raw))
            task["image_paths"] = new_paths
            # 同步 task_audit.json
            task_audit = video_dir / "tasks" / task_id / "task_audit.json"
            if task_audit.is_file() and new_paths:
                try:
                    ta = json.loads(task_audit.read_text(encoding="utf-8"))
                    ta["image_paths"] = new_paths
                    task_audit.write_text(
                        json.dumps(ta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:  # noqa: BLE001
                    pass
        if changed:
            audit_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            stats["updated_json"] = True

    # 删 candidates
    for cand in video_dir.glob("tasks/*/candidates"):
        n = sum(1 for _ in cand.rglob("*") if _.is_file())
        shutil.rmtree(cand, ignore_errors=True)
        stats["removed_candidates"] = int(stats["removed_candidates"]) + n

    # 删根目录与 task 目录下散落 jpg 副本（已迁到 selected）
    for jpg in list(video_dir.glob("*_t*.jpg")) + list(
        video_dir.glob("tasks/*/*.jpg")
    ):
        jpg.unlink(missing_ok=True)
        stats["removed_loose_jpg"] = int(stats["removed_loose_jpg"]) + 1

    # partial / before
    for pattern in (
        "*.partial.json",
        "*before*.json",
        "stage_audit_split.partial.json",
    ):
        for f in video_dir.rglob(pattern):
            if f.name == "stage_audit_split.json":
                continue
            f.unlink(missing_ok=True)
            stats["removed_partials"] = int(stats["removed_partials"]) + 1

    return stats


def archive_root_test_artifacts() -> int:
    archive = RUNS / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    moved = 0
    prefixes = (
        "_rerun_",
        "_eval_",
        "_batch_",
        "_verify_",
        "_temp",
        "_restore",
        "_stage2_",
    )
    for f in INTER.iterdir():
        if not f.is_file():
            continue
        if f.name == ".gitkeep":
            continue
        if f.name.startswith(prefixes) or f.name.startswith("_"):
            dest = archive / f.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(f), str(dest))
            moved += 1
    return moved


def clear_cache_dirs() -> dict[str, int]:
    removed: dict[str, int] = {}
    for name in ("audit_candidates", "keyframes", "audit_sparse", "cropped"):
        path = CACHE / name
        if not path.exists():
            removed[name] = 0
            continue
        n = sum(1 for _ in path.rglob("*") if _.is_file())
        shutil.rmtree(path, ignore_errors=True)
        removed[name] = n
    return removed


def main() -> None:
    SELECTED.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (SELECTED / ".gitkeep").write_text("", encoding="utf-8")
    (RUNS / ".gitkeep").write_text("", encoding="utf-8")

    moved = archive_root_test_artifacts()
    video_stats = []
    if INTER.is_dir():
        for video_dir in sorted(p for p in INTER.iterdir() if p.is_dir()):
            video_stats.append(migrate_video(video_dir))
    cache_stats = clear_cache_dirs()
    report = {
        "archived_root_files": moved,
        "videos": video_stats,
        "cache_removed_files": cache_stats,
    }
    out = RUNS / "cleanup_data_layout_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
