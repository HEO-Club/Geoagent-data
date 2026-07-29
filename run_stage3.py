"""阶段3 CLI：自由 TAO → tool 树归一化 JSONL。"""

from __future__ import annotations

import argparse
import json
import logging

from pipeline.config import get_settings
from pipeline.stage2_freeform_tao.run import load_freeform
from pipeline.stage3_normalize_format.format_jsonl import run_stage3


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段3：自由 TAO → 标准 JSONL")
    parser.add_argument("--freeform", required=True, help="stage2_freeform_tao.json")
    parser.add_argument("--trees", default=None, help="tool_trees.json 路径")
    parser.add_argument("--out-jsonl", default=None)
    parser.add_argument("--image", default="")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--user-query", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    get_settings()
    freeform = load_freeform(args.freeform)
    entry = run_stage3(
        freeform,
        trees_path=args.trees,
        out_jsonl_path=args.out_jsonl,
        image_path=args.image,
        system_prompt=args.system_prompt,
        user_query=args.user_query,
        matcher=lambda _name, _forest: None,
    )
    print(
        json.dumps(
            {"id": entry.id, "messages": len(entry.messages)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
