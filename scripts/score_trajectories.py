"""离线批量计算 Stage 3 轨迹质量置信度，不调用外部 API。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.quality.scorer import score_trajectory_quality
from pipeline.schemas.audit import AuditSplitResult, GeoTaskSpec
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.quality import SemanticQualityReview
from pipeline.schemas.trajectory import Trajectory
from pipeline.stage3_normalize_format.params import (
    attach_operation_input_schemas,
    initial_parameter_context,
    normalize_and_validate_tool_inputs,
    update_parameter_context,
)
from pipeline.stage3_normalize_format.trees import load_forest


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _find_task(path: Path, trajectory_id: str) -> GeoTaskSpec | None:
    for parent in [path.parent, *path.parents]:
        candidate = parent / "stage_audit_split.json"
        if not candidate.is_file():
            continue
        audit = AuditSplitResult.model_validate_json(candidate.read_text(encoding="utf-8"))
        for task in audit.tasks:
            if task.task_id == trajectory_id or path.parent.name == task.task_id:
                return task
        return None
    return None


def _load_parameter_audits(path: Path) -> list[dict[str, Any]] | None:
    raw = _read_json(path)
    if raw is None:
        return None
    items = raw.get("calls")
    return [item for item in items or [] if isinstance(item, dict)]


def _audit_existing_parameters(trajectory: Trajectory, forest) -> list[dict[str, Any]]:
    audits = []
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
        audits.append(audit.model_dump())
    return audits


def _report_row(path: Path, report: Any) -> dict[str, Any]:
    return {
        "path": str(path),
        "source_video": report.source_video,
        "trajectory_id": report.trajectory_id,
        "quality_score": report.quality_score,
        "audit_coverage": report.audit_coverage,
        "decision": report.decision,
        "hard_failures": report.hard_failures,
        "dimensions": {
            item.name: {
                "score": item.score,
                "coverage": item.audit_coverage,
            }
            for item in report.dimensions
        },
        "issues": [item.model_dump() for item in report.issues],
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# 轨迹质量置信度报告",
        "",
        (
            f"共评估 {summary['count']} 条；平均分 {summary['mean_score']:.3f}，"
            f"中位数 {summary['median_score']:.3f}，"
            f"平均审核覆盖率 {summary['mean_coverage']:.3f}。"
        ),
        "",
        "| 轨迹 | 分数 | 覆盖率 | 决策 | 硬错误 |",
        "|---|---:|---:|---|---|",
    ]
    for item in summary["items"]:
        lines.append(
            f"| `{item['trajectory_id']}` | {item['quality_score']:.3f} | "
            f"{item['audit_coverage']:.3f} | {item['decision']} | "
            f"{', '.join(item['hard_failures']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="包含 stage3_trajectory.json 的结果目录")
    parser.add_argument("--catalog", type=Path, default=Path("canonical_tool_catalog.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--write-sidecars", action="store_true")
    args = parser.parse_args()

    forest = attach_operation_input_schemas(load_forest(args.catalog))
    rows: list[dict[str, Any]] = []
    for trajectory_path in sorted(args.root.rglob("stage3_trajectory.json")):
        freeform_path = trajectory_path.with_name("stage2_freeform_tao.json")
        if not freeform_path.is_file():
            continue
        freeform = FreeFormTrajectory.model_validate_json(
            freeform_path.read_text(encoding="utf-8")
        )
        trajectory = Trajectory.model_validate_json(
            trajectory_path.read_text(encoding="utf-8")
        )
        task = _find_task(trajectory_path, trajectory.id)
        observation_audit = _read_json(
            trajectory_path.with_name("stage2_observation_audit.json")
        )
        consistency = _read_json(
            trajectory_path.with_name("image_trajectory_consistency.json")
        )
        parameter_audits = _load_parameter_audits(
            trajectory_path.with_name("stage3_parameter_audit.json")
        )
        if parameter_audits is None:
            parameter_audits = _audit_existing_parameters(trajectory, forest)
        semantic_raw = _read_json(
            trajectory_path.with_name("stage3_semantic_quality_review.json")
        )
        semantic = (
            SemanticQualityReview.model_validate(semantic_raw)
            if semantic_raw is not None
            else None
        )
        report = score_trajectory_quality(
            freeform,
            trajectory,
            forest,
            task=task,
            observation_audit=observation_audit,
            trajectory_consistency=consistency,
            parameter_audits=parameter_audits,
            semantic_review=semantic,
        )
        rows.append(_report_row(trajectory_path, report))
        if args.write_sidecars:
            trajectory_path.with_name("stage3_quality_report.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8"
            )

    scores = [item["quality_score"] for item in rows]
    coverages = [item["audit_coverage"] for item in rows]
    summary = {
        "root": str(args.root.resolve()),
        "count": len(rows),
        "mean_score": mean(scores) if scores else 0.0,
        "median_score": median(scores) if scores else 0.0,
        "mean_coverage": mean(coverages) if coverages else 0.0,
        "decision_counts": {
            value: sum(item["decision"] == value for item in rows)
            for value in (
                "accept",
                "provisional_pass",
                "parameter_repair",
                "needs_review",
                "reject",
            )
        },
        "items": rows,
    }
    out = args.out or args.root / "trajectory_quality_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, ensure_ascii=False, indent=2))
    print(f"report={out}")


if __name__ == "__main__":
    main()
