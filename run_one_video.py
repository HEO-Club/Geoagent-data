"""单视频全链路 CLI。"""

from __future__ import annotations

import argparse
import json
import logging

from pipeline.config import get_settings
from pipeline.orchestrator import run_one_video


def main() -> None:
    parser = argparse.ArgumentParser(description="单视频流水线：阶段1–3")
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--anchor-transcript", default=None)
    parser.add_argument("--image", default="")
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="忽略 manifest，强制重跑各阶段",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    get_settings()
    entry = run_one_video(
        args.video,
        video_id=args.video_id,
        anchor_transcript_path=args.anchor_transcript,
        image_path=args.image,
        skip_completed=not args.no_skip,
        stage3_matcher=lambda _n, _f: None,
    )
    print(
        json.dumps(
            {
                "id": entry.id,
                "source_video": entry.source_video,
                "messages": len(entry.messages),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
