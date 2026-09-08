"""阶段2 CLI：视频 + 字幕 → 自由 TAO。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pipeline.config import get_settings
from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao.run import run_stage2


def _load_transcript(path: str) -> list[TranscriptSegment]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "segments" in raw:
        items = raw["segments"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"无法解析字幕: {path}")
    return [TranscriptSegment.model_validate(x) for x in items]


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段2：视频+字幕 → 自由 TAO")
    parser.add_argument("--video", required=True)
    parser.add_argument("--transcript", required=True, help="阶段1 字幕 JSON")
    parser.add_argument("--out", default=None)
    parser.add_argument("--image", default=None, help="可选单图（兼容）")
    parser.add_argument("--task-json", default=None, help="可选 Stage 1.5 GeoTaskSpec")
    parser.add_argument("--context-transcript", default=None, help="仅供 Observation 审核补充相邻语境的全字幕")
    parser.add_argument("--max-generations", type=int, choices=(1, 2, 3), default=None, help="含首次的总生成次数上限，默认3")
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="任务关键帧（可多图）",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    get_settings()
    task = (
        GeoTaskSpec.model_validate_json(Path(args.task_json).read_text(encoding="utf-8"))
        if args.task_json else None
    )
    traj = run_stage2(
        args.video,
        _load_transcript(args.transcript),
        out_path=args.out,
        image_path=args.image,
        image_paths=(args.images if args.images is not None else task.image_paths if task and not args.image else None),
        task=task,
        max_attempts=args.max_generations,
        observation_context=_load_transcript(args.context_transcript) if args.context_transcript else None,
    )
    print(
        json.dumps(
            {"source_video": traj.source_video, "steps": len(traj.steps)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
