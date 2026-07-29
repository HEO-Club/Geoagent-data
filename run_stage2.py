"""阶段2 CLI：视频 + 字幕 → 自由 TAO。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pipeline.config import get_settings
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
    parser.add_argument("--image", default=None, help="可选代表图")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    get_settings()
    traj = run_stage2(
        args.video,
        _load_transcript(args.transcript),
        out_path=args.out,
        image_path=args.image,
    )
    print(
        json.dumps(
            {"source_video": traj.source_video, "steps": len(traj.steps)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
