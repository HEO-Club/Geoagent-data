"""CLI：从字幕准备 groundtruth 坐标（map_query / Nominatim）。

示例::

    python prep_groundtruth.py --transcript data/transcripts/BV13m61BJEQC.json
    python prep_groundtruth.py --transcript ... --query 郑州黄河文化公园
"""

from __future__ import annotations

import argparse
import json
import sys

from pipeline.prep_groundtruth import lookup_groundtruth_from_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从带时间戳字幕推断答案地名，并用地图解析经纬度"
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="TranscriptSegment 列表 JSON，或含 segments 的 whisper 风格 JSON",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="手动指定地名（跳过/优先于自动抽取）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出完整 JSON（默认打印可读摘要 + --gt 字符串）",
    )
    args = parser.parse_args(argv)

    suggestion = lookup_groundtruth_from_file(args.transcript, query=args.query)
    if args.json:
        print(json.dumps(suggestion.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0 if suggestion.status == "success" else 1

    print(f"answer_timestamp: {suggestion.answer_timestamp}")
    if suggestion.candidates:
        print("candidates:")
        for c in suggestion.candidates[:8]:
            print(f"  - {c.query!r} @ {c.source_start:.1f}s")
    print(f"query_used: {suggestion.query}")
    print(f"status: {suggestion.status}")
    if suggestion.status == "success":
        print(f"address: {suggestion.formatted_address}")
        print(f"place_type: {suggestion.place_type}")
        print(f"gt: {suggestion.gt_cli()}")
        print()
        print("下一步示例:")
        print(
            "python run_one_video.py "
            f"--video <视频> --transcript {args.transcript} "
            f"--gt {suggestion.gt_cli()} --platform bilibili"
        )
        return 0

    print("未能解析到坐标；可加 --query 手动指定地名后重试。", file=sys.stderr)
    if suggestion.observation:
        print(
            json.dumps(suggestion.observation, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
