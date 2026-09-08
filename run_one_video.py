"""单视频全链路 CLI。"""

from __future__ import annotations

import argparse
import json
import logging

from pipeline.config import get_settings
from pipeline.orchestrator import run_one_video


def main() -> None:
    parser = argparse.ArgumentParser(description="单视频流水线：阶段1 → 审核切分 → 阶段2–4")
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--anchor-transcript", default=None)
    parser.add_argument("--image", default="", help="可选回退单图（无审核帧时）")
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="可选回退多图（无审核帧时）",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="忽略 manifest，强制重跑各阶段",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    get_settings()
    entries = run_one_video(
        args.video,
        video_id=args.video_id,
        anchor_transcript_path=args.anchor_transcript,
        image_path=args.image,
        image_paths=args.images,
        skip_completed=not args.no_skip,
        stage3_matcher=lambda _n, _f: None,
    )
    print(
        json.dumps(
            {
                "count": len(entries),
                "entries": [
                    {
                        "id": e.id,
                        "source_video": e.source_video,
                        "messages": len(e.messages),
                        "quality_score": e.quality_score,
                    }
                    for e in entries
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
