"""离线审计既有 Stage 3 轨迹的 operation/inputs，不调用外部 API。"""

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
    initial_parameter_context,
    normalize_and_validate_tool_inputs,
    update_parameter_context,
)
from pipeline.stage3_normalize_format.trees import load_forest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--catalog", type=Path, default=Path("canonical_tool_catalog.json"))
    parser.add_argument("--out", type=Path, default=Path("data/runs/parameter_audit.json"))
    args = parser.parse_args()

    forest = attach_operation_input_schemas(load_forest(args.catalog))
    trajectories = []
    issue_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    readiness_counts: Counter[str] = Counter()
    invalid_calls = 0
    total_calls = 0
    for path in sorted(args.root.rglob("stage3_trajectory.json")):
        trajectory = Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
        calls = []
        context = initial_parameter_context(trajectory.image_paths)
        for index, step in enumerate(trajectory.steps, start=1):
            if step.event_type != "tool_call" or step.action is None:
                continue
            params = step.action.params
            audit = normalize_and_validate_tool_inputs(
                forest,
                tool=step.action.tool,
                operation=str(params.get("operation") or "execute"),
                inputs=dict(params.get("inputs") or {}),
                step_index=index,
                available_context=context,
            )
            update_parameter_context(context, audit)
            dumped = audit.model_dump()
            calls.append(dumped)
            total_calls += 1
            invalid_calls += not audit.valid
            tool_counts[audit.tool] += 1
            readiness_counts[audit.readiness] += 1
            issue_counts.update(issue.code for issue in audit.issues)
        trajectories.append(
            {
                "trajectory_id": trajectory.id,
                "path": str(path),
                "valid_calls": sum(item["valid"] for item in calls),
                "total_calls": len(calls),
                "calls": calls,
            }
        )

    result = {
        "root": str(args.root.resolve()),
        "trajectory_count": len(trajectories),
        "total_calls": total_calls,
        "valid_calls": total_calls - invalid_calls,
        "invalid_calls": invalid_calls,
        "valid_rate": round((total_calls - invalid_calls) / total_calls, 4)
        if total_calls
        else 0.0,
        "issue_counts": dict(issue_counts.most_common()),
        "tool_counts": dict(tool_counts.most_common()),
        "readiness_counts": dict(readiness_counts),
        "trajectories": trajectories,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "trajectories"}, ensure_ascii=False, indent=2))
    print(f"report={args.out}")


if __name__ == "__main__":
    main()
