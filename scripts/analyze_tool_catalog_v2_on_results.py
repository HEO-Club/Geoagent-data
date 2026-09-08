"""把既有 Stage3 调用离线映射到 Tool v2 提案并统计拆分效果。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.schemas.trajectory import Trajectory
from pipeline.stage3_normalize_format.params import (
    attach_operation_input_schemas,
    resolve_operation_alias,
)
from pipeline.stage3_normalize_format.trees import find_tree_for_name, load_forest
from pipeline.tool_catalog_v2_proposal import build_tool_catalog_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("data/runs/tool_catalog_v2_analysis.json")
    )
    args = parser.parse_args()

    proposal = build_tool_catalog_v2()
    migration = {
        (item["from_tool"], item["from_operation"]): (
            item["to_tool"],
            item["to_operation"],
        )
        for item in proposal["migration"]
    }
    forest = attach_operation_input_schemas(
        load_forest(Path("canonical_tool_catalog.json"))
    )
    old_counts: Counter[str] = Counter()
    new_counts: Counter[str] = Counter()
    split_matrix: Counter[str] = Counter()
    unresolved = []
    calls = []
    for path in sorted(args.root.rglob("stage3_trajectory.json")):
        trajectory = Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
        for index, step in enumerate(trajectory.steps, start=1):
            if step.event_type != "tool_call" or step.action is None:
                continue
            old_tool = step.action.tool
            raw_operation = str(step.action.params.get("operation") or "execute")
            tree = find_tree_for_name(forest, old_tool)
            if tree is None:
                unresolved.append(
                    {
                        "trajectory_id": trajectory.id,
                        "step": index,
                        "tool": old_tool,
                        "operation": raw_operation,
                        "reason": "unknown_source_tool",
                    }
                )
                continue
            old_canonical = tree.canonical.name
            operation, _ = resolve_operation_alias(tree, raw_operation)
            target = migration.get((old_canonical, operation))
            if target is None:
                unresolved.append(
                    {
                        "trajectory_id": trajectory.id,
                        "step": index,
                        "tool": old_canonical,
                        "operation": operation,
                        "reason": "missing_migration",
                    }
                )
                continue
            new_tool, new_operation = target
            old_counts[old_canonical] += 1
            new_counts[new_tool] += 1
            split_matrix[f"{old_canonical} -> {new_tool}"] += 1
            calls.append(
                {
                    "trajectory_id": trajectory.id,
                    "step": index,
                    "from": f"{old_canonical}.{operation}",
                    "to": f"{new_tool}.{new_operation}",
                }
            )

    report = {
        "root": str(args.root.resolve()),
        "trajectory_count": len(list(args.root.rglob("stage3_trajectory.json"))),
        "mapped_calls": len(calls),
        "unresolved_calls": len(unresolved),
        "old_unique_tools": len(old_counts),
        "new_unique_tools": len(new_counts),
        "old_tool_counts": dict(old_counts.most_common()),
        "new_tool_counts": dict(new_counts.most_common()),
        "split_matrix": dict(split_matrix.most_common()),
        "unresolved": unresolved,
        "calls": calls,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "calls"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"report={args.out}")


if __name__ == "__main__":
    main()
