"""可选串联阶段1 → 审核切分 → 按 task 跑阶段2–3；manifest 断点续跑。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import get_settings
from pipeline.schemas.audit import AuditDecision, TaskStatus
from pipeline.schemas.dataset import DatasetEntry, ManifestV2
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage1_transcript.run import run_stage1
from pipeline.stage2_freeform_tao.run import load_freeform, run_stage2
from pipeline.stage3_normalize_format.format_jsonl import run_stage3
from pipeline.stage_audit_split.run import (
    load_audit_split,
    run_audit_split,
    slice_transcript_for_task,
)
from pipeline.stage_audit_split.trajectory_image_check import (
    check_trajectory_image_consistency,
)

logger = logging.getLogger(__name__)

STAGE_ORDER = ("stage1", "stage_audit_split", "stage2", "stage3")


def _utcnow() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _manifest_path(video_id: str) -> Path:
    settings = get_settings()
    return Path(settings.INTERMEDIATE_DIR) / video_id / "manifest_v2.json"


def load_manifest(video_id: str) -> ManifestV2:
    path = _manifest_path(video_id)
    if not path.is_file():
        return ManifestV2(video_id=video_id, stages={}, updated_at=_utcnow())
    return ManifestV2.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(manifest: ManifestV2) -> None:
    path = _manifest_path(manifest.video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = _utcnow()
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _load_transcript_from_intermediate(video_id: str) -> list[TranscriptSegment]:
    settings = get_settings()
    path = Path(settings.INTERMEDIATE_DIR) / video_id / "stage1_transcript.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "segments" in data:
        return [TranscriptSegment.model_validate(x) for x in data["segments"]]
    if isinstance(data, list):
        return [TranscriptSegment.model_validate(x) for x in data]
    raise ValueError(f"无法解析 stage1 产物: {path}")


def _audit_path(video_id: str) -> Path:
    settings = get_settings()
    return Path(settings.INTERMEDIATE_DIR) / video_id / "stage_audit_split.json"


def _task_stage_key(task_id: str, stage: str) -> str:
    return f"task:{task_id}:{stage}"


def run_one_video(
    video_path: str,
    *,
    video_id: str | None = None,
    anchor_transcript_path: str | None = None,
    image_path: str = "",
    image_paths: list[str] | None = None,
    skip_completed: bool = True,
    stage3_matcher=None,
) -> list[DatasetEntry]:
    """串联阶段1 → 审核切分 → 按 task 跑阶段2–3。

    Args:
        video_path: 视频路径。
        video_id: 覆盖默认 stem。
        anchor_transcript_path: 可选 ASR 锚点。
        image_path: 兼容旧单图；无审核任务时可作为回退视觉输入。
        image_paths: 兼容外部多图；无审核任务时回退。
        skip_completed: True 时跳过 manifest 已标记 done 的阶段。
        stage3_matcher: 注入阶段3 匹配器（测试用）。

    Returns:
        每个 accept 的 task 对应一条 DatasetEntry；reject 时返回空列表。
    """
    settings = get_settings()
    vid = video_id or Path(video_path).stem
    manifest = load_manifest(vid)

    if not (skip_completed and manifest.stages.get("stage1") == "done"):
        run_stage1(video_path, anchor_transcript_path=anchor_transcript_path)
        manifest.stages["stage1"] = "done"
        save_manifest(manifest)
    else:
        logger.info("skip stage1 (done) for %s", vid)

    transcript = _load_transcript_from_intermediate(vid)
    audit_file = _audit_path(vid)

    if not (
        skip_completed
        and manifest.stages.get("stage_audit_split") in {"done", "rejected"}
        and audit_file.is_file()
    ):
        audit = run_audit_split(video_path, transcript, out_path=str(audit_file))
        manifest.stages["stage_audit_split"] = (
            "rejected" if audit.decision == AuditDecision.reject else "done"
        )
        save_manifest(manifest)
    else:
        audit = load_audit_split(audit_file)
        logger.info(
            "skip stage_audit_split (%s) for %s",
            manifest.stages.get("stage_audit_split"),
            vid,
        )

    if audit.decision == AuditDecision.reject:
        logger.info("audit rejected %s: %s", vid, audit.reason)
        return []

    entries: list[DatasetEntry] = []
    fallback_images = [p for p in (image_paths or []) if str(p).strip()] or (
        [image_path] if image_path.strip() else []
    )

    for task in audit.tasks:
        video_dir = Path(settings.INTERMEDIATE_DIR) / vid
        task_dir = video_dir / "tasks" / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        freeform_path = task_dir / "stage2_freeform_tao.json"
        traj_path = task_dir / "stage3_trajectory.json"
        shard_path = Path(settings.OUTPUT_DIR) / "shards" / f"{task.task_id}.jsonl"
        stage2_key = _task_stage_key(task.task_id, "stage2")
        stage3_key = _task_stage_key(task.task_id, "stage3")

        if task.status != TaskStatus.accepted:
            skipped = task.status.value
            manifest.stages[stage2_key] = skipped
            manifest.stages[stage3_key] = skipped
            save_manifest(manifest)
            logger.info(
                "skip downstream for %s status=%s reason=%s",
                task.task_id,
                task.status.value,
                task.status_reason,
            )
            continue

        task_images = list(task.image_paths) or list(fallback_images)
        task_transcript = slice_transcript_for_task(transcript, task)

        if not (
            skip_completed
            and manifest.stages.get(stage2_key) == "done"
            and freeform_path.is_file()
        ):
            run_stage2(
                video_path,
                task_transcript,
                out_path=str(freeform_path),
                image_paths=task_images or None,
                source_video=vid,
            )
            manifest.stages[stage2_key] = "done"
            save_manifest(manifest)
        else:
            logger.info("skip stage2 (done) for %s", task.task_id)

        freeform = load_freeform(freeform_path)
        if freeform.source_video != vid:
            freeform.source_video = vid

        consistency_path = task_dir / "image_trajectory_consistency.json"
        if settings.AUDIT_TRAJECTORY_IMAGE_CHECK:
            consistency = check_trajectory_image_consistency(
                image_paths=task_images,
                visual_evidence_brief=str(
                    getattr(task, "visual_evidence_brief", "") or ""
                ),
                trajectory=freeform,
            )
            consistency_path.write_text(
                consistency.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if consistency.conflict:
                manifest.stages[stage3_key] = "needs_review"
                save_manifest(manifest)
                logger.info(
                    "skip stage3 for %s: trajectory-image conflict (%s)",
                    task.task_id,
                    consistency.reason,
                )
                continue

        if not (
            skip_completed
            and manifest.stages.get(stage3_key) == "done"
            and shard_path.is_file()
        ):
            entry = run_stage3(
                freeform,
                out_trajectory_path=str(traj_path),
                out_jsonl_path=str(shard_path),
                image_paths=task_images or None,
                shard_id=task.task_id,
                matcher=stage3_matcher,
            )
            manifest.stages[stage3_key] = "done"
            save_manifest(manifest)
        else:
            logger.info("skip stage3 (done) for %s", task.task_id)
            entry = DatasetEntry.model_validate_json(
                shard_path.read_text(encoding="utf-8").splitlines()[0]
            )
        entries.append(entry)

    return entries


def merge_jsonl_shards(output_dir: str | Path | None = None) -> int:
    """单 writer：合并 shards/*.jsonl → geolocate_agent.jsonl。返回行数。"""
    settings = get_settings()
    out = Path(output_dir) if output_dir else Path(settings.OUTPUT_DIR)
    shard_dir = out / "shards"
    lines: list[str] = []
    if shard_dir.is_dir():
        for path in sorted(shard_dir.glob("*.jsonl")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                lines.extend(text.splitlines())
    dest = out / "geolocate_agent.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)
