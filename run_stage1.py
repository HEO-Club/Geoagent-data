"""阶段1 CLI：视频 → 字幕。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pipeline.config import get_settings
from pipeline.stage1_transcript.run import run_stage1


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段1：视频 → 带时间戳字幕")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--out", default=None, help="字幕 JSON 输出路径")
    parser.add_argument(
        "--anchor-transcript",
        default=None,
        help="可选旧字幕（仅时间锚）",
    )
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    get_settings()
    segs = run_stage1(
        args.video,
        anchor_transcript_path=args.anchor_transcript,
        out_path=args.out,
        window_sec=args.window_sec,
        max_frames=args.max_frames,
    )
    print(
        json.dumps(
            {
                "video": args.video,
                "segments": len(segs),
                "stem": Path(args.video).stem,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
