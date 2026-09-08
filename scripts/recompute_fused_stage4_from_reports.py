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
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.tools import ToolParameterAudit
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage3_normalize_format.format_jsonl import format_dataset_entry
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
    if report.judge_call_failed:
        raise ValueError("原审核失败，不能把中性分包装成成功 VLM 审核")
    dimensions = {item.name: item for item in report.dimensions}
    model_flags = [
        *report.hard_gates,
        *[
            flag
            for flag in report.soft_flags
            if flag.code not in {"parameter_inputs_invalid"}
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


def _replay_judge(report: ConfidenceReport):
    def judge(**_kwargs):
        return _draft_from_report(report)
    return judge


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
    parser.add_argument("--context-root", type=Path, default=Path("data/transcripts"))
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
        task = GeoTaskSpec.model_validate_json(
            components["task"][task_id].read_text(encoding="utf-8")
        )
        freeform = FreeFormTrajectory.model_validate_json(
            components["freeform"][task_id].read_text(encoding="utf-8")
        )
        trajectory = Trajectory.model_validate_json(
            trajectory_path.read_text(encoding="utf-8")
        )
        # Rebuild the exact conversation represented by the saved trajectory instead
        # of writing an empty messages=[] entry during score recomputation.
        entry = format_dataset_entry(trajectory, source_video=freeform.source_video)
        video_id = task_id.split("__t", 1)[0]
        context_path = args.context_root / f"{video_id}.json"
        context = _transcript(context_path) if context_path.is_file() else None
        video_path = Path("data/raw_videos") / f"{video_id}.mp4"
        task_dir = args.out / "intermediate" / task_id
        report = run_stage4(
            task=task,
            transcript=_transcript(components["transcript"][task_id]),
            freeform=freeform,
            trajectory=trajectory,
            entry=entry,
            tool_mapping_path=components["mapping"][task_id],
            parameter_audits=audits_by_task.get(task_id),
            review_context_transcript=context,
            source_video_path=str(video_path) if video_path.is_file() else None,
            out_report_path=str(task_dir / "stage4_confidence.json"),
            out_jsonl_path=str(args.out / "output" / "shards" / f"{task_id}.jsonl"),
            judge=_replay_judge(previous),
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
                "task_window": {"start": task.time_start, "end": task.time_end},
                "judge_call_failed": report.judge_call_failed,
                "review_markdown": str(task_dir / "stage4_confidence.review.md"),
                "review_json": str(task_dir / "stage4_confidence.review.json"),
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
    index = [
        "# 逐题人工审核入口", "",
        "本包复用已有 VLM 审核，仅本地重算参数评分并生成问题清单；没有新调用模型或真实地理工具。",
        "评分建议不阻止样本保存；报告的模型判断需要人工核验。新 JSONL 保留完整 messages，旧输出目录不变。", "",
        "| Task | 题目秒数 | 分数 | 质量建议 | 逐项问题、选帧与修改建议 |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        window = row["task_window"]
        link = str(Path(row["review_markdown"]).resolve()).replace("\\", "/")
        index.append(f"| {row['task_id']} | {window['start']:.3f}–{window['end']:.3f} | {row['quality_score']:.4f} | {row['decision']} | [打开审核清单](<{link}>) |")
    (args.out / "REVIEW_INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
