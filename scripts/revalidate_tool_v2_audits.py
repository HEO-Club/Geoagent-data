"""Revalidate existing Stage 3 calls against the latest Tool v2 contracts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.schemas.trajectory import Trajectory
from pipeline.stage3_normalize_format.params import (
    initial_parameter_context,
    normalize_and_validate_tool_inputs,
    update_parameter_context,
)
from pipeline.stage3_normalize_format.trees import load_forest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--catalog", type=Path, default=Path("canonical_tool_catalog_v2.json")
    )
    args = parser.parse_args()

    by_task: dict[str, Path] = {}
    for root in args.root:
        for path in root.rglob("stage3_trajectory.json"):
            by_task[path.parent.name] = path

    forest = load_forest(args.catalog)
    before = Counter()
    after = Counter()
    issue_codes = Counter()
    rows = []
    for task_id, trajectory_path in sorted(by_task.items()):
        trajectory = Trajectory.model_validate_json(
            trajectory_path.read_text(encoding="utf-8")
        )
        old_path = trajectory_path.with_name("stage3_parameter_audit.json")
        if old_path.is_file():
            old_value = json.loads(old_path.read_text(encoding="utf-8"))
            for call in old_value.get("calls", []):
                before[str(call.get("readiness") or "unknown")] += 1

        context = initial_parameter_context(trajectory.image_paths)
        calls = []
        for index, step in enumerate(trajectory.steps, start=1):
            if step.event_type != "tool_call" or step.action is None:
                continue
            params = step.action.params or {}
            audit = normalize_and_validate_tool_inputs(
                forest,
                tool=step.action.tool,
                operation=str(params.get("operation") or "execute"),
                inputs=(
                    dict(params.get("inputs") or {})
                    if isinstance(params.get("inputs"), dict)
                    else {}
                ),
                step_index=index,
                available_context=context,
            )
            update_parameter_context(context, audit)
            after[audit.readiness] += 1
            issue_codes.update(issue.code for issue in audit.issues)
            calls.append(audit.model_dump())
        rows.append({"task_id": task_id, "trajectory": str(trajectory_path), "calls": calls})

    result = {
        "task_count": len(rows),
        "before": dict(before),
        "after": dict(after),
        "issue_codes": dict(issue_codes),
        "items": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "items"}, ensure_ascii=False, indent=2))
    print(f"report={args.out}")


if __name__ == "__main__":
    main()
