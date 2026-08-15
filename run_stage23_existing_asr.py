"""复用既有 Stage 1 字幕与关键帧，跳过 Stage 1.5 批量运行 Stage 2/3。"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao.run import load_freeform, run_stage2
from pipeline.stage3_normalize_format.format_jsonl import run_stage3

logger = logging.getLogger(__name__)


def _load_segments(path: Path) -> list[TranscriptSegment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw.get("segments") if isinstance(raw, dict) else raw
    if not isinstance(payload, list):
        raise TypeError(f"无法解析字幕 segments: {path}")
    return [TranscriptSegment.model_validate(item) for item in payload]


def _timestamp_key(path: Path) -> float:
    match = re.search(r"t([0-9]+(?:\.[0-9]+)?)", path.stem)
    return float(match.group(1)) if match else float("inf")


def _discover_samples(
    transcript_root: Path, order_root: Path | None
) -> list[tuple[str, str]]:
    if order_root and order_root.is_dir():
        ordered: list[tuple[str, str]] = []
        for folder in sorted(p for p in order_root.iterdir() if p.is_dir()):
            match = re.match(r"^(\d+)_(.+)$", folder.name)
            if not match:
                continue
            video_id = match.group(2)
            if (transcript_root / video_id / "stage1_transcript.json").is_file():
                ordered.append((folder.name, video_id))
        if ordered:
            return ordered
    return [
        (folder.name, folder.name)
        for folder in sorted(p for p in transcript_root.iterdir() if p.is_dir())
        if (folder / "stage1_transcript.json").is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="复用既有 ASR 和关键帧，跳过 Stage 1.5 跑 Stage 2/3"
    )
    parser.add_argument("--transcript-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--keyframes-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--order-root", default="")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rerun-stage3",
        action="store_true",
        help="复用已有 Stage 2，但重新执行 Stage 3 映射与序列化",
    )
    parser.add_argument(
        "--only",
        default="",
        help="可选，逗号分隔的序号目录名或 video_id；用于探针或局部续跑",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    transcript_root = Path(args.transcript_root)
    video_root = Path(args.video_root)
    keyframes_root = Path(args.keyframes_root)
    output_root = Path(args.out)
    order_root = Path(args.order_root) if args.order_root else None
    runtime_trees = output_root / "tool_trees.runtime.json"
    shard_dir = output_root / "output" / "shards"
    status_path = output_root / "run_status.json"
    status_by_label: dict[str, dict[str, object]] = {}
    if status_path.is_file():
        existing_status = json.loads(status_path.read_text(encoding="utf-8"))
        if isinstance(existing_status, list):
            status_by_label = {
                str(item["label"]): item
                for item in existing_status
                if isinstance(item, dict) and item.get("label")
            }

    all_samples = _discover_samples(transcript_root, order_root)
    samples = all_samples
    if args.only:
        selected = {item.strip() for item in args.only.split(",") if item.strip()}
        samples = [
            (label, video_id)
            for label, video_id in samples
            if label in selected or video_id in selected
        ]

    for label, video_id in samples:
        sample_dir = output_root / "intermediate" / label
        stage2_path = sample_dir / "stage2_freeform_tao.json"
        stage3_path = sample_dir / "stage3_trajectory.json"
        shard_path = shard_dir / f"{label}.jsonl"
        transcript_path = transcript_root / video_id / "stage1_transcript.json"
        video_path = video_root / f"{video_id}.mp4"
        frame_dir = keyframes_root / video_id
        image_paths = [
            str(path.resolve())
            for path in sorted(frame_dir.glob("*.jpg"), key=_timestamp_key)
        ]
        status: dict[str, object] = {
            "label": label,
            "video_id": video_id,
            "images": len(image_paths),
        }
        try:
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            if not image_paths:
                raise FileNotFoundError(f"没有既有关键帧: {frame_dir}")
            transcript = _load_segments(transcript_path)
            sample_dir.mkdir(parents=True, exist_ok=True)
            if args.resume and stage2_path.is_file():
                freeform = load_freeform(stage2_path)
                logger.info("resume Stage 2: %s", label)
            else:
                freeform = run_stage2(
                    str(video_path),
                    transcript,
                    out_path=str(stage2_path),
                    image_paths=image_paths,
                    source_video=video_id,
                    max_attempts=max(1, args.max_attempts),
                )
            if args.rerun_stage3 or not (
                args.resume and stage3_path.is_file() and shard_path.is_file()
            ):
                run_stage3(
                    freeform,
                    trees_path=runtime_trees,
                    out_trajectory_path=str(stage3_path),
                    out_jsonl_path=str(shard_path),
                    image_paths=image_paths,
                    shard_id=label,
                )
            else:
                logger.info("resume Stage 3: %s", label)
            status["ok"] = True
            status["events"] = len(freeform.steps)
        except Exception as exc:
            logger.exception("sample failed: %s", label)
            status["ok"] = False
            status["error"] = f"{type(exc).__name__}: {exc}"
        status_by_label[label] = status
        results = [
            status_by_label[label]
            for label, _video_id in all_samples
            if label in status_by_label
        ]
        output_root.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    merged: list[str] = []
    if shard_dir.is_dir():
        for shard in sorted(shard_dir.glob("*.jsonl")):
            text = shard.read_text(encoding="utf-8").strip()
            if text:
                merged.extend(text.splitlines())
    (output_root / "output" / "geolocate_agent.jsonl").parent.mkdir(
        parents=True, exist_ok=True
    )
    (output_root / "output" / "geolocate_agent.jsonl").write_text(
        "\n".join(merged) + ("\n" if merged else ""), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "samples": len(status_by_label),
                "ok": sum(
                    bool(result.get("ok")) for result in status_by_label.values()
                ),
                "jsonl": len(merged),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
