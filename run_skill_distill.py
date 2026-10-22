"""技能蒸馏 CLI。阶段3/4 跑完后，把产物目录传给 --input 即可。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline.config import get_settings
from pipeline.skill_distill import run_skill_distill


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 stage3_trajectory.json 蒸馏技能库。不读取 groundtruth，不改训练 JSONL。"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="含 stage3_trajectory.json 的目录，通常是 data/intermediate 或一次跑批的输出根",
    )
    parser.add_argument(
        "--output",
        default="data/skill_bank",
        help="写出 skill_bank.json、patches.jsonl、merge_report.json",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="正式工具目录，缺省为 TOOL_CATALOG_PATH",
    )
    parser.add_argument(
        "--bank",
        default=None,
        help="已有 skill_bank.json。提供时在冻结副本上加深，而不是从空库开始",
    )
    parser.add_argument("--min-quality", type=float, default=0.85)
    parser.add_argument(
        "--allow-unscored",
        action="store_true",
        help="没有 stage4_confidence.json 的轨迹也进入蒸馏",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    report = run_skill_distill(
        Path(args.input),
        Path(args.output),
        catalog_path=Path(args.catalog or settings.TOOL_CATALOG_PATH),
        min_quality=args.min_quality,
        allow_unscored=args.allow_unscored,
        bank_path=Path(args.bank) if args.bank else None,
    )
    print(
        f"admitted={report.admitted} skipped={len(report.skipped)} "
        f"skills={report.skills} examples={report.examples} "
        f"review={report.review_queue}"
    )


if __name__ == "__main__":
    main()
