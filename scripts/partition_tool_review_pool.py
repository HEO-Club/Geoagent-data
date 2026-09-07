"""Partition Stage 3 outputs into canonical-only and temporary-review pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    canonical_only: list[dict[str, Any]] = []
    temporary_review: list[dict[str, Any]] = []
    for path in sorted(args.root.rglob("stage3_tool_mapping.json")):
        mapping = _read(path)
        routing = mapping.get("tool_routing")
        has_temporary = bool(
            mapping.get("temporary_tools") or mapping.get("temporary_operations")
        )
        if isinstance(routing, dict):
            has_temporary = bool(routing.get("requires_tool_review", has_temporary))
        item = {
            "mapping_path": str(path.resolve()),
            "task_dir": str(path.parent.resolve()),
            "pool": "temporary_review" if has_temporary else "canonical_only",
            "temporary_tools": mapping.get("temporary_tools") or [],
            "temporary_operations": mapping.get("temporary_operations") or [],
            "candidates": mapping.get("tool_review_candidates") or [],
        }
        (temporary_review if has_temporary else canonical_only).append(item)

    payload = {
        "schema_version": "tool_review_partition_v1",
        "policy": "仅分流、不改轨迹、不自动新增Tool",
        "counts": {
            "total": len(canonical_only) + len(temporary_review),
            "canonical_only": len(canonical_only),
            "temporary_review": len(temporary_review),
        },
        "canonical_only": canonical_only,
        "temporary_review": temporary_review,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
