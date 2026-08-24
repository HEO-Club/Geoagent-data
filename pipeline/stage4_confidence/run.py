"""阶段4：样本置信度评分（人工检查用，不拦入库）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pipeline.config import get_settings
from pipeline.quality.scorer import score_trajectory_quality
from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.confidence import (
    DIMENSION_NAMES,
    ConfidenceJudgeDraft,
    ConfidenceReport,
    DimensionScore,
    HardGateHit,
    ReviewPriority,
)
from pipeline.schemas.dataset import DatasetEntry
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.quality import SemanticQualityReview, TrajectoryQualityReport
from pipeline.schemas.tools import ToolForest
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage3_normalize_format.params import attach_operation_input_schemas
from pipeline.stage3_normalize_format.trees import load_forest
from pipeline.stage4_confidence.judge import JudgeFn, call_confidence_judge
from pipeline.stage4_confidence.rules import evaluate_programmatic_gates

logger = logging.getLogger(__name__)

_QUALITY_TO_STAGE4_DIMENSION = {
    "evidence_grounding": "evidence_grounding",
    "final_answer_support": "final_answer_support",
    "tool_parameter_validity": "tool_param_correctness",
    "reasoning_consistency": "logical_consistency",
    "input_alignment": "input_quality_alignment",
    "sft_format": "sft_format_completeness",
}


def _dimension_weights() -> dict[str, float]:
    settings = get_settings()
    raw = {
        "evidence_grounding": float(settings.STAGE4_W_EVIDENCE_GROUNDING),
        "final_answer_support": float(settings.STAGE4_W_FINAL_ANSWER_SUPPORT),
        "tool_param_correctness": float(settings.STAGE4_W_TOOL_PARAM_CORRECTNESS),
        "logical_consistency": float(settings.STAGE4_W_LOGICAL_CONSISTENCY),
        "input_quality_alignment": float(settings.STAGE4_W_INPUT_QUALITY_ALIGNMENT),
        "sft_format_completeness": float(settings.STAGE4_W_SFT_FORMAT_COMPLETENESS),
    }
    total = sum(max(0.0, v) for v in raw.values())
    if total <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: max(0.0, v) / total for k, v in raw.items()}


def _review_priority(score: float) -> ReviewPriority:
    settings = get_settings()
    if score < float(settings.STAGE4_PRIORITY_HIGH_BELOW):
        return "high"
    if score < float(settings.STAGE4_PRIORITY_MEDIUM_BELOW):
        return "medium"
    return "low"


def _route_priority(score: float, decision: str) -> ReviewPriority:
    priority = _review_priority(score)
    if decision == "reject":
        return "high"
    if decision in {"parameter_repair", "needs_review", "provisional_pass"}:
        return "medium" if priority == "low" else priority
    return priority


def _draft_score(draft: ConfidenceJudgeDraft | None, name: str, neutral: float) -> float:
    if draft is None:
        return neutral
    value = getattr(draft, name, None)
    if value is None:
        return neutral
    return float(max(0.0, min(1.0, value)))


def _draft_reason(draft: ConfidenceJudgeDraft | None, name: str, fallback: str) -> str:
    if draft is None:
        return fallback
    reasons = draft.dimension_reasons or {}
    text = str(reasons.get(name) or "").strip()
    return text or fallback


def merge_confidence(
    *,
    task_id: str,
    format_score: float,
    format_reason: str,
    programmatic_gates: list[HardGateHit],
    draft: ConfidenceJudgeDraft | None,
    judge_call_failed: bool,
) -> ConfidenceReport:
    """加权合成 base_score；硬门槛命中则压到 cap。"""
    settings = get_settings()
    weights = _dimension_weights()
    neutral = float(settings.STAGE4_JUDGE_NEUTRAL_SCORE)
    cap = float(settings.STAGE4_HARD_GATE_CAP)

    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for name in DIMENSION_NAMES:
        if name == "sft_format_completeness":
            scores[name] = float(max(0.0, min(1.0, format_score)))
            reasons[name] = format_reason
        else:
            scores[name] = _draft_score(draft, name, neutral)
            reasons[name] = _draft_reason(
                draft,
                name,
                "裁判调用失败，记中性分" if judge_call_failed else "",
            )

    dimensions = [
        DimensionScore(
            name=name,
            score=scores[name],
            weight=weights[name],
            reason=reasons[name],
        )
        for name in DIMENSION_NAMES
    ]
    base = sum(d.score * d.weight for d in dimensions)
    base = float(max(0.0, min(1.0, base)))

    model_gates = list(draft.hard_gates) if draft else []
    # 去重 code
    gates: list[HardGateHit] = []
    seen: set[str] = set()
    for g in [*programmatic_gates, *model_gates]:
        code = str(g.code).strip()
        if not code or code in seen:
            continue
        seen.add(code)
        gates.append(HardGateHit(code=code, evidence=str(g.evidence or "").strip()))

    quality = min(base, cap) if gates else base
    quality = float(max(0.0, min(1.0, quality)))

    notes_parts: list[str] = []
    if judge_call_failed:
        notes_parts.append("judge_call_failed")
    if draft and draft.notes.strip():
        notes_parts.append(draft.notes.strip())
    if gates:
        notes_parts.append(f"hard_gates={len(gates)}")

    return ConfidenceReport(
        task_id=task_id,
        base_score=base,
        quality_score=quality,
        dimensions=dimensions,
        hard_gates=gates,
        review_priority=_review_priority(quality),
        notes="；".join(notes_parts),
        judge_call_failed=judge_call_failed,
    )


def rewrite_entry_quality_score(
    entry: DatasetEntry,
    quality_score: float,
    *,
    out_jsonl_path: str | Path | None = None,
) -> DatasetEntry:
    """只回写 quality_score，不改 messages。"""
    updated = entry.model_copy(update={"quality_score": float(quality_score)})
    if out_jsonl_path:
        path = Path(out_jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated.model_dump_json() + "\n", encoding="utf-8")
    return updated


def load_confidence_report(path: str | Path) -> ConfidenceReport:
    """从 intermediate 加载阶段4报告。"""
    return ConfidenceReport.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def _load_tool_mapping(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("无法读取 tool mapping %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _load_optional_artifact(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _load_tool_mapping(Path(path))


def _load_quality_forest() -> ToolForest:
    settings = get_settings()
    runtime_path = Path(settings.TOOL_TREES_PATH)
    catalog_path = Path(settings.TOOL_CATALOG_PATH)
    path = runtime_path if runtime_path.is_file() else catalog_path
    return attach_operation_input_schemas(load_forest(path))


def _semantic_review_from_draft(
    draft: ConfidenceJudgeDraft | None,
) -> SemanticQualityReview | None:
    if draft is None:
        return None
    return SemanticQualityReview(
        evidence_grounding=draft.evidence_grounding,
        final_answer_support=draft.final_answer_support,
        reasoning_consistency=draft.logical_consistency,
        tool_semantics=draft.tool_param_correctness,
        input_alignment=draft.input_quality_alignment,
        summary=draft.notes,
    )


def _quality_dimensions(report: TrajectoryQualityReport) -> list[DimensionScore]:
    result: list[DimensionScore] = []
    for item in report.dimensions:
        actionable = [
            check.message
            for check in item.checks
            if not check.observed or check.score < 0.999
        ]
        reason = "；".join(actionable[:3]) or "该维度检查通过"
        result.append(
            DimensionScore(
                name=_QUALITY_TO_STAGE4_DIMENSION[item.name],
                score=item.score,
                weight=item.weight,
                reason=reason,
            )
        )
    return result


def merge_fused_confidence(
    *,
    quality_report: TrajectoryQualityReport,
    programmatic_gates: list[HardGateHit],
    draft: ConfidenceJudgeDraft | None,
    judge_call_failed: bool,
    evidence_sources: list[str],
) -> ConfidenceReport:
    """融合确定性审计、参数合同、严格证据和 VLM 裁判。"""

    settings = get_settings()
    cap = float(settings.STAGE4_HARD_GATE_CAP)
    gates: list[HardGateHit] = []
    seen: set[str] = set()
    issue_by_code = {item.code: item.message for item in quality_report.issues}
    model_gates = list(draft.hard_gates) if draft else []
    quality_gates = [
        HardGateHit(code=code, evidence=issue_by_code.get(code, "质量审计硬门槛"))
        for code in quality_report.hard_failures
    ]
    for gate in [*programmatic_gates, *model_gates, *quality_gates]:
        code = str(gate.code or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        gates.append(HardGateHit(code=code, evidence=str(gate.evidence or "").strip()))

    base = quality_report.quality_score
    quality = min(base, cap) if gates else base
    decision = "reject" if gates else quality_report.decision
    notes = []
    if judge_call_failed:
        notes.append("judge_call_failed: semantic dimensions remain uncovered")
    if draft and draft.notes.strip():
        notes.append(draft.notes.strip())
    if gates:
        notes.append(f"hard_gates={len(gates)}")

    return ConfidenceReport(
        task_id=quality_report.trajectory_id,
        base_score=base,
        quality_score=quality,
        audit_coverage=quality_report.audit_coverage,
        decision=decision,
        dimensions=_quality_dimensions(quality_report),
        hard_gates=gates,
        review_priority=_route_priority(quality, decision),
        notes="；".join(notes),
        judge_call_failed=judge_call_failed,
        parameter_readiness_counts=dict(
            quality_report.metadata.get("parameter_readiness_counts") or {}
        ),
        evidence_sources=evidence_sources,
    )


def run_stage4(
    *,
    task: GeoTaskSpec,
    transcript: list[TranscriptSegment],
    freeform: FreeFormTrajectory,
    trajectory: Trajectory,
    entry: DatasetEntry,
    tool_mapping: dict[str, Any] | None = None,
    tool_mapping_path: str | Path | None = None,
    parameter_audit: dict[str, Any] | None = None,
    parameter_audit_path: str | Path | None = None,
    observation_audit: dict[str, Any] | None = None,
    observation_audit_path: str | Path | None = None,
    trajectory_consistency: dict[str, Any] | None = None,
    trajectory_consistency_path: str | Path | None = None,
    forest: ToolForest | None = None,
    out_report_path: str | None = None,
    out_jsonl_path: str | None = None,
    judge: JudgeFn | None = None,
) -> ConfidenceReport:
    """阶段4：多维置信度评分 → 报告落盘 + 回写 JSONL.quality_score。

    硬门槛只压分，不拦入库、不改轨迹。
    """
    mapping = tool_mapping
    if mapping is None and tool_mapping_path is not None:
        mapping = _load_tool_mapping(Path(tool_mapping_path))
    parameters = parameter_audit or _load_optional_artifact(parameter_audit_path)
    observations = observation_audit or _load_optional_artifact(observation_audit_path)
    consistency = trajectory_consistency or _load_optional_artifact(
        trajectory_consistency_path
    )

    programmatic_gates, _, _ = evaluate_programmatic_gates(
        trajectory, task, transcript
    )

    draft: ConfidenceJudgeDraft | None = None
    judge_failed = False
    try:
        draft = call_confidence_judge(
            task=task,
            transcript=transcript,
            freeform=freeform,
            trajectory=trajectory,
            tool_mapping=mapping,
            parameter_audit=parameters,
            observation_audit=observations,
            trajectory_consistency=consistency,
            judge=judge,
        )
    except Exception as exc:  # noqa: BLE001 — 失败开放
        judge_failed = True
        logger.warning(
            "stage4 judge failed for %s: %s",
            task.task_id,
            type(exc).__name__,
        )

    semantic_review = _semantic_review_from_draft(draft)
    parameter_calls = [
        item for item in (parameters or {}).get("calls", []) if isinstance(item, dict)
    ]
    quality_report = score_trajectory_quality(
        freeform,
        trajectory,
        forest or _load_quality_forest(),
        task=task,
        observation_audit=observations,
        trajectory_consistency=consistency,
        parameter_audits=parameter_calls if parameters is not None else None,
        semantic_review=semantic_review,
    )
    evidence_sources = ["programmatic_gates", "stage1.5_task"]
    if draft is not None:
        evidence_sources.append("vlm_judge")
    if parameters is not None:
        evidence_sources.append("parameter_audit")
    if observations is not None:
        evidence_sources.append("strict_observation_audit")
    if consistency is not None:
        evidence_sources.append("trajectory_image_consistency")
    report = merge_fused_confidence(
        quality_report=quality_report,
        programmatic_gates=programmatic_gates,
        draft=draft,
        judge_call_failed=judge_failed,
        evidence_sources=evidence_sources,
    )

    settings = get_settings()
    report_path = (
        Path(out_report_path)
        if out_report_path
        else (
            Path(settings.INTERMEDIATE_DIR)
            / freeform.source_video
            / "tasks"
            / task.task_id
            / "stage4_confidence.json"
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    shard = (
        Path(out_jsonl_path)
        if out_jsonl_path
        else Path(settings.OUTPUT_DIR) / "shards" / f"{task.task_id}.jsonl"
    )
    rewrite_entry_quality_score(entry, report.quality_score, out_jsonl_path=shard)

    return report
