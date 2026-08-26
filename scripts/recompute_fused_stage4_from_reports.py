"""Recompute fused Stage 4 locally from real VLM reports and latest param audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.confidence import ConfidenceJudgeDraft, ConfidenceReport
from pipeline.schemas.dataset import DatasetEntry
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.tools import ToolParameterAudit
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage4_confidence.run import run_stage4


def _component_maps(roots: list[Path]) -> dict[str, dict[str, Path]]:
    names = {
        "task": "stage15_task.json",
        "transcript": "transcript_slice.json",
        "freeform": "stage2_freeform_tao.json",
        "trajectory": "stage3_trajectory.json",
        "mapping": "stage3_tool_mapping.json",
        "report": "stage4_confidence.json",
        "report_retry": "stage4_confidence.retry.json",
    }
    result = {key: {} for key in names}
    for root in roots:
        for key, filename in names.items():
            for path in root.rglob(filename):
                task_id = path.parent.name
                result[key][task_id] = path
                if key == "report":
                    result["report_retry"].pop(task_id, None)
    return result


def _draft_from_report(report: ConfidenceReport) -> ConfidenceJudgeDraft:
    dimensions = {item.name: item for item in report.dimensions}
    model_flags = [
        *report.hard_gates,
        *[
            flag
            for flag in report.soft_flags
            if flag.code not in {"task_needs_review", "parameter_inputs_invalid"}
        ],
    ]
    return ConfidenceJudgeDraft(
        evidence_grounding=dimensions["evidence_grounding"].score,
        final_answer_support=dimensions["final_answer_support"].score,
        tool_param_correctness=dimensions["tool_param_correctness"].score,
        logical_consistency=dimensions["logical_consistency"].score,
        input_quality_alignment=dimensions["input_quality_alignment"].score,
        sft_format_completeness=dimensions["sft_format_completeness"].score,
        dimension_reasons={
            name: item.reason for name, item in dimensions.items()
        },
        hard_gates=model_flags,
        notes="复用已完成的真实 VLM 审核；仅以最新参数合同重新融合分数和路由。",
    )


def _transcript(path: Path) -> list[TranscriptSegment]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("segments", [])
    return [TranscriptSegment.model_validate(item) for item in value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--revalidation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    components = _component_maps(args.root)
    revalidation = json.loads(args.revalidation.read_text(encoding="utf-8"))
    audits_by_task = {
        item["task_id"]: [ToolParameterAudit.model_validate(call) for call in item["calls"]]
        for item in revalidation.get("items", [])
    }
    rows = []
    for task_id, trajectory_path in sorted(components["trajectory"].items()):
        required = ("task", "transcript", "freeform", "mapping", "report")
        if any(task_id not in components[key] for key in required):
            continue
        report_path = components["report_retry"].get(
            task_id, components["report"][task_id]
        )
        previous = ConfidenceReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        draft = _draft_from_report(previous)
        task = GeoTaskSpec.model_validate_json(
            components["task"][task_id].read_text(encoding="utf-8")
        )
        freeform = FreeFormTrajectory.model_validate_json(
            components["freeform"][task_id].read_text(encoding="utf-8")
        )
        trajectory = Trajectory.model_validate_json(
            trajectory_path.read_text(encoding="utf-8")
        )
        entry = DatasetEntry(
            id=trajectory.id,
            source_video=freeform.source_video,
            messages=[],
        )
        task_dir = args.out / "intermediate" / task_id
        report = run_stage4(
            task=task,
            transcript=_transcript(components["transcript"][task_id]),
            freeform=freeform,
            trajectory=trajectory,
            entry=entry,
            tool_mapping_path=components["mapping"][task_id],
            parameter_audits=audits_by_task.get(task_id, []),
            out_report_path=str(task_dir / "stage4_confidence.json"),
            out_jsonl_path=str(args.out / "output" / "shards" / f"{task_id}.jsonl"),
            judge=lambda _draft=draft, **_kwargs: _draft,
        )
        final = trajectory.steps[-1]
        rows.append(
            {
                "task_id": task_id,
                "location": final.action.params.get("location") if final.action else None,
                "quality_score": report.quality_score,
                "audit_coverage": report.audit_coverage,
                "decision": report.decision,
                "hard_gates": [gate.code for gate in report.hard_gates],
                "soft_flags": [flag.code for flag in report.soft_flags],
                "parameter_readiness": (
                    report.parameter_readiness.model_dump()
                    if report.parameter_readiness is not None
                    else None
                ),
                "trajectory": str(trajectory_path),
                "source_report": str(report_path),
            }
        )

    summary = {
        "task_count": len(rows),
        "mean_score": mean(item["quality_score"] for item in rows) if rows else 0.0,
        "mean_coverage": mean(item["audit_coverage"] for item in rows) if rows else 0.0,
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
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
