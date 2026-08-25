"""合并融合 Stage4 基线与定向重跑结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = _load(args.base)
    by_id = {item["trajectory_id"]: item for item in base.get("items", [])}
    for path in args.override:
        for item in _load(path).get("items", []):
            by_id[item["trajectory_id"]] = item
    rows = sorted(by_id.values(), key=lambda item: item["trajectory_id"])
    result = {
        "judge_mode": "real",
        "count": len(rows),
        "mean_score": mean(item["quality_score"] for item in rows) if rows else 0,
        "mean_coverage": mean(item["audit_coverage"] for item in rows) if rows else 0,
        "decision_counts": {
            decision: sum(item["decision"] == decision for item in rows)
            for decision in (
                "accept",
                "provisional_pass",
                "parameter_repair",
                "needs_review",
                "reject",
            )
        },
        "items": rows,
        "sources": [str(args.base), *[str(path) for path in args.override]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "items"}, ensure_ascii=False, indent=2))
    print(f"report={args.out}")


if __name__ == "__main__":
    main()
