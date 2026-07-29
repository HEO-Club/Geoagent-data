"""可选串联阶段1–3；manifest 断点续跑。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import get_settings
from pipeline.schemas.dataset import DatasetEntry, ManifestV2
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage1_transcript.run import run_stage1
from pipeline.stage2_freeform_tao.run import load_freeform, run_stage2
from pipeline.stage3_normalize_format.format_jsonl import run_stage3

logger = logging.getLogger(__name__)

STAGE_ORDER = ("stage1", "stage2", "stage3")


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


def run_one_video(
    video_path: str,
    *,
    video_id: str | None = None,
    anchor_transcript_path: str | None = None,
    image_path: str = "",
    skip_completed: bool = True,
    stage3_matcher=None,
) -> DatasetEntry:
    """串联阶段1–3；manifest 断点续跑。

    Args:
        video_path: 视频路径。
        video_id: 覆盖默认 stem。
        anchor_transcript_path: 可选 ASR 锚点。
        image_path: 训练样本代表图路径。
        skip_completed: True 时跳过 manifest 已标记 done 的阶段。
        stage3_matcher: 注入阶段3 匹配器（测试用）。
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

    freeform_path = Path(settings.INTERMEDIATE_DIR) / vid / "stage2_freeform_tao.json"
    if not (
        skip_completed
        and manifest.stages.get("stage2") == "done"
        and freeform_path.is_file()
    ):
        run_stage2(video_path, transcript, image_path=image_path or None)
        manifest.stages["stage2"] = "done"
        save_manifest(manifest)
    else:
        logger.info("skip stage2 (done) for %s", vid)

    freeform = load_freeform(freeform_path)

    entry = run_stage3(
        freeform,
        image_path=image_path,
        matcher=stage3_matcher,
    )
    manifest.stages["stage3"] = "done"
    save_manifest(manifest)
    return entry


def merge_jsonl_shards(output_dir: str | Path | None = None) -> int:
    """单 writer：合并 shards/*.jsonl → geolocate_agent.jsonl。返回行数。"""
    settings = get_settings()
    out = Path(output_dir) if output_dir else Path(settings.OUTPUT_DIR)
    shards_dir = out / "shards"
    final_path = out / "geolocate_agent.jsonl"
    lines: list[str] = []
    if shards_dir.is_dir():
        for shard in sorted(shards_dir.glob("*.jsonl")):
            text = shard.read_text(encoding="utf-8").strip()
            if text:
                lines.extend(text.splitlines())
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return len(lines)
