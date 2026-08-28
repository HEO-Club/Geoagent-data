"""阶段4：样本置信度评分（人工检查用，不拦入库）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pipeline.config import get_settings
from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.confidence import (
    DIMENSION_NAMES,
    VLM_DIMENSION_NAMES,
    ConfidenceJudgeDraft,
    ConfidenceReport,
    DimensionScore,
    HardGateHit,
    ParameterReadinessSummary,
    ReviewPriority,
)
from pipeline.schemas.dataset import DatasetEntry
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.tools import ToolParameterAudit
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage4_confidence.judge import JudgeFn, call_confidence_judge
from pipeline.stage4_confidence.review_cards import (
    build_review_packet,
    write_review_packet,
)
from pipeline.stage4_confidence.rules import (
    evaluate_observation_audit,
    evaluate_parameter_readiness,
    evaluate_programmatic_gates,
)

logger = logging.getLogger(__name__)

SOFT_GATE_CAPS: dict[str, float] = {
    "observation_needs_repair": 0.82,
    "parameter_inputs_invalid": 0.75,
}
# 选图质量只作标注/维度分，不进软审查或硬门槛（与 SPEC v3.4.11 一致）。
SELECTION_QUALITY_GATE_CODES: frozenset[str] = frozenset(
    {
        "image_trajectory_mismatch",
        "task_needs_review",
    }
)
ACCEPT_SCORE = 0.90
PROVISIONAL_SCORE = 0.78
ACCEPT_COVERAGE = 0.85


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


def _enrich_weak_dimension_reasons(
    dimensions: list[DimensionScore],
    *,
    draft: ConfidenceJudgeDraft | None,
    programmatic_gates: list[HardGateHit],
    model_gates: list[HardGateHit],
    judge_call_failed: bool,
    weak_below: float,
) -> list[DimensionScore]:
    """弱维度补全可核对 reason（不重试 API）。"""
    gate_evidence = [
        f"{g.code}:{g.evidence}" if g.evidence else g.code
        for g in [*programmatic_gates, *model_gates]
        if g.code
    ]
    enriched: list[DimensionScore] = []
    for dim in dimensions:
        reason = (dim.reason or "").strip()
        if dim.score >= weak_below and reason:
            enriched.append(dim)
            continue
        if dim.score >= weak_below:
            enriched.append(dim)
            continue
        # 弱维：reason 过短则补证据
        if len(reason) >= 12:
            enriched.append(dim)
            continue
        extras: list[str] = []
        if judge_call_failed and dim.name in VLM_DIMENSION_NAMES:
            extras.append("裁判调用失败，本维中性分")
        if gate_evidence:
            extras.append("相关硬门槛: " + "; ".join(gate_evidence[:4]))
        draft_note = (draft.notes.strip() if draft and draft.notes else "")
        if draft_note and dim.name in VLM_DIMENSION_NAMES:
            extras.append(f"裁判备注摘录: {draft_note[:120]}")
        if not extras:
            extras.append(f"本维得分 {dim.score:.2f}，低于弱维阈值 {weak_below:.2f}")
        new_reason = reason + ("；" if reason else "") + "；".join(extras)
        enriched.append(dim.model_copy(update={"reason": new_reason}))
    return enriched


def compose_evaluation_notes(
    *,
    quality_score: float,
    base_score: float,
    review_priority: ReviewPriority,
    hard_gates: list[HardGateHit],
    dimensions: list[DimensionScore],
    parameter_summary: ParameterReadinessSummary | None,
    draft: ConfidenceJudgeDraft | None,
    judge_call_failed: bool,
    weak_below: float,
    image_selection_note: str = "",
    audit_coverage: float = 0.0,
    decision: str = "needs_review",
    soft_flags: list[HardGateHit] | None = None,
    applied_soft_caps: dict[str, float] | None = None,
) -> str:
    """组装每条样本必填的评价 notes（含弱维明细）。"""
    parts: list[str] = [
        (
            f"quality_score={quality_score:.3f} base_score={base_score:.3f} "
            f"coverage={audit_coverage:.3f} decision={decision} "
            f"priority={review_priority}"
        )
    ]
    if judge_call_failed:
        parts.append("judge_call_failed")
    if hard_gates:
        gate_bits = [
            f"{g.code}({g.evidence})" if g.evidence else g.code for g in hard_gates
        ]
        parts.append("硬门槛: " + "; ".join(gate_bits))
    else:
        parts.append("硬门槛: 无")
    if soft_flags:
        parts.append(
            "软审查项: "
            + "; ".join(
                f"{item.code}({item.evidence})" if item.evidence else item.code
                for item in soft_flags
            )
        )
    if applied_soft_caps:
        parts.append(
            "软上限: "
            + "; ".join(f"{key}<={value:.2f}" for key, value in applied_soft_caps.items())
        )

    if parameter_summary is not None:
        if parameter_summary.audit_missing:
            parts.append("参数质检: audit 缺失或损坏，参数分记中性")
        else:
            parts.append(
                "参数质检: "
                f"total={parameter_summary.total_calls} "
                f"ready={parameter_summary.ready} "
                f"context_resolvable={parameter_summary.context_resolvable} "
                f"repairable={parameter_summary.repairable} "
                f"invalid={parameter_summary.invalid}"
                + (
                    f" worst={parameter_summary.worst}"
                    if parameter_summary.worst
                    else ""
                )
            )
            for line in parameter_summary.detail_lines[:8]:
                parts.append(f"参数问题: {line}")
            if len(parameter_summary.detail_lines) > 8:
                parts.append(
                    f"参数问题: …另有 {len(parameter_summary.detail_lines) - 8} 次"
                )

    weak_dims = [d for d in dimensions if d.score < weak_below]
    if weak_dims:
        parts.append("弱维度明细:")
        for d in weak_dims:
            reason = (d.reason or "").strip() or "（无明细）"
            parts.append(f"- {d.name}={d.score:.2f}: {reason}")
    else:
        parts.append("弱维度: 无")

    selection_note = (image_selection_note or "").strip()
    if selection_note:
        parts.append(f"选图评价: {selection_note}")

    if draft and draft.notes.strip():
        parts.append(f"裁判notes: {draft.notes.strip()}")

    text = "\n".join(parts).strip()
    return text or "评价说明已生成（无额外瑕疵）"


def merge_confidence(
    *,
    task_id: str,
    format_score: float,
    format_reason: str,
    programmatic_gates: list[HardGateHit],
    draft: ConfidenceJudgeDraft | None,
    judge_call_failed: bool,
    param_score: float | None = None,
    param_reason: str = "",
    parameter_summary: ParameterReadinessSummary | None = None,
    image_selection_note: str = "",
    audit_coverage: float = 0.0,
    evidence_sources: list[str] | None = None,
    soft_flags: list[HardGateHit] | None = None,
) -> ConfidenceReport:
    """Fuse VLM scores, programmatic gates, coverage and repair routing."""
    settings = get_settings()
    weights = _dimension_weights()
    neutral = float(settings.STAGE4_JUDGE_NEUTRAL_SCORE)
    cap = float(settings.STAGE4_HARD_GATE_CAP)
    weak_below = float(settings.STAGE4_WEAK_DIMENSION_BELOW)

    resolved_param_score = (
        float(max(0.0, min(1.0, param_score)))
        if param_score is not None
        else float(max(0.0, min(1.0, neutral)))
    )
    resolved_param_reason = (
        param_reason.strip()
        or (
            "参数审计未提供，记中性分"
            if param_score is None
            else "参数质检通过"
        )
    )

    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for name in DIMENSION_NAMES:
        if name == "sft_format_completeness":
            scores[name] = float(max(0.0, min(1.0, format_score)))
            reasons[name] = format_reason
        elif name == "tool_param_correctness":
            scores[name] = resolved_param_score
            reasons[name] = resolved_param_reason
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
    all_gates: list[HardGateHit] = []
    seen: set[str] = set()
    for g in [*programmatic_gates, *model_gates]:
        code = str(g.code).strip()
        if not code or code in seen:
            continue
        seen.add(code)
        all_gates.append(
            HardGateHit(code=code, evidence=str(g.evidence or "").strip())
        )

    resolved_soft_flags = list(soft_flags or [])
    blocking_gates: list[HardGateHit] = []
    for gate in all_gates:
        if gate.code in SELECTION_QUALITY_GATE_CODES:
            continue
        if gate.code in SOFT_GATE_CAPS:
            resolved_soft_flags.append(gate)
        else:
            blocking_gates.append(gate)
    deduped_soft: list[HardGateHit] = []
    soft_seen: set[str] = set()
    for flag in resolved_soft_flags:
        if (
            not flag.code
            or flag.code in soft_seen
            or flag.code in SELECTION_QUALITY_GATE_CODES
        ):
            continue
        soft_seen.add(flag.code)
        deduped_soft.append(flag)

    applied_soft_caps = {
        flag.code: SOFT_GATE_CAPS[flag.code]
        for flag in deduped_soft
        if flag.code in SOFT_GATE_CAPS
    }
    if parameter_summary and parameter_summary.invalid:
        applied_soft_caps["parameter_inputs_invalid"] = SOFT_GATE_CAPS[
            "parameter_inputs_invalid"
        ]

    quality = min(base, cap) if blocking_gates else base
    if applied_soft_caps:
        quality = min(quality, *applied_soft_caps.values())
    quality = float(max(0.0, min(1.0, quality)))

    worst = parameter_summary.worst if parameter_summary else None
    if blocking_gates:
        decision = "reject"
    elif deduped_soft or audit_coverage < ACCEPT_COVERAGE:
        decision = "needs_review"
    elif worst in {"repairable", "invalid"}:
        decision = "parameter_repair"
    elif quality >= ACCEPT_SCORE:
        decision = "accept"
    elif quality >= PROVISIONAL_SCORE:
        decision = "provisional_pass"
    else:
        decision = "needs_review"
    priority = "high" if decision == "reject" else _review_priority(quality)

    dimensions = _enrich_weak_dimension_reasons(
        dimensions,
        draft=draft,
        programmatic_gates=blocking_gates,
        model_gates=model_gates,
        judge_call_failed=judge_call_failed,
        weak_below=weak_below,
    )

    notes = compose_evaluation_notes(
        quality_score=quality,
        base_score=base,
        review_priority=priority,
        hard_gates=blocking_gates,
        dimensions=dimensions,
        parameter_summary=parameter_summary,
        draft=draft,
        judge_call_failed=judge_call_failed,
        weak_below=weak_below,
        image_selection_note=image_selection_note,
        audit_coverage=audit_coverage,
        decision=decision,
        soft_flags=deduped_soft,
        applied_soft_caps=applied_soft_caps,
    )

    return ConfidenceReport(
        task_id=task_id,
        base_score=base,
        quality_score=quality,
        audit_coverage=audit_coverage,
        decision=decision,
        dimensions=dimensions,
        hard_gates=blocking_gates,
        review_priority=priority,
        notes=notes,
        judge_call_failed=judge_call_failed,
        parameter_readiness=parameter_summary,
        evidence_sources=list(evidence_sources or []),
        soft_flags=deduped_soft,
        applied_soft_caps=applied_soft_caps,
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


def _load_optional_audit(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("无法读取辅助审计 %s: %s", file_path, exc)
        return None
    return value if isinstance(value, dict) else None


def _calculate_audit_coverage(
    *,
    transcript: list[TranscriptSegment],
    trajectory: Trajectory,
    tool_mapping: dict[str, Any] | None,
    parameter_audit_present: bool,
    observation_audit_present: bool,
    judge_succeeded: bool,
) -> tuple[float, list[str]]:
    checks = [
        ("timestamped_transcript", 0.15, bool(transcript and any(s.text.strip() for s in transcript))),
        (
            "selected_images",
            0.15,
            bool(trajectory.image_paths)
            and all(Path(path).is_file() for path in trajectory.image_paths),
        ),
        ("tool_mapping", 0.10, tool_mapping is not None),
        ("parameter_contract_audit", 0.20, parameter_audit_present),
        ("strict_observation_audit", 0.10, observation_audit_present),
        ("vlm_semantic_judge", 0.30, judge_succeeded),
    ]
    sources = [name for name, _, present in checks if present]
    coverage = sum(weight for _, weight, present in checks if present)
    return float(max(0.0, min(1.0, coverage))), sources


def load_parameter_audits(
    path: str | Path | None,
) -> tuple[list[ToolParameterAudit] | None, bool]:
    """加载 stage3_parameter_audit.json。

    Returns:
        (audits, audit_missing)。文件不存在或损坏时 audits=None、audit_missing=True。
        path 为 None 时视为缺失。
    """
    if path is None:
        return None, True
    file_path = Path(path)
    if not file_path.is_file():
        return None, True
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("无法读取 parameter audit %s: %s", file_path, exc)
        return None, True
    if not isinstance(data, dict):
        return None, True
    raw_calls = data.get("calls")
    if raw_calls is None:
        return [], False
    if not isinstance(raw_calls, list):
        return None, True
    audits: list[ToolParameterAudit] = []
    try:
        for item in raw_calls:
            audits.append(ToolParameterAudit.model_validate(item))
    except Exception as exc:  # noqa: BLE001
        logger.warning("parameter audit 校验失败 %s: %s", file_path, exc)
        return None, True
    return audits, False


def run_stage4(
    *,
    task: GeoTaskSpec,
    transcript: list[TranscriptSegment],
    freeform: FreeFormTrajectory,
    trajectory: Trajectory,
    entry: DatasetEntry,
    tool_mapping: dict[str, Any] | None = None,
    tool_mapping_path: str | Path | None = None,
    parameter_audits: list[ToolParameterAudit] | None = None,
    parameter_audit_path: str | Path | None = None,
    observation_audit: dict[str, Any] | None = None,
    observation_audit_path: str | Path | None = None,
    trajectory_consistency: dict[str, Any] | None = None,
    review_context_transcript: list[TranscriptSegment] | None = None,
    source_video_path: str | None = None,
    out_report_path: str | None = None,
    out_jsonl_path: str | None = None,
    judge: JudgeFn | None = None,
) -> ConfidenceReport:
    """阶段4：多维置信度评分 → 报告落盘 + 回写 JSONL.quality_score。

    硬门槛只压分，不拦入库、不改轨迹。``tool_param_correctness`` 由参数审计程序化覆盖。
    ``trajectory_consistency`` 保留兼容参数，选图质量不再进入软审查。
    """
    _ = trajectory_consistency
    mapping = tool_mapping
    if mapping is None and tool_mapping_path is not None:
        mapping = _load_tool_mapping(Path(tool_mapping_path))

    audits = parameter_audits
    audit_missing = False
    if audits is None:
        audits, audit_missing = load_parameter_audits(parameter_audit_path)
    # 显式传入空列表视为「有审计、无 Tool」；仅 path/解析失败才 missing

    programmatic_gates, format_score, format_reason = evaluate_programmatic_gates(
        trajectory, task, transcript
    )
    param_score, param_reason, param_gates, param_summary = (
        evaluate_parameter_readiness(audits, audit_missing=audit_missing)
    )
    resolved_observation_audit = observation_audit
    if resolved_observation_audit is None:
        resolved_observation_audit = _load_optional_audit(observation_audit_path)
    observation_gates, observation_soft, observation_observed = (
        evaluate_observation_audit(resolved_observation_audit, trajectory=trajectory)
    )
    programmatic_gates = [
        *programmatic_gates,
        *param_gates,
        *observation_gates,
    ]
    soft_flags = list(observation_soft)

    draft: ConfidenceJudgeDraft | None = None
    judge_failed = False
    try:
        draft = call_confidence_judge(
            task=task,
            transcript=transcript,
            freeform=freeform,
            trajectory=trajectory,
            tool_mapping=mapping,
            parameter_audits=audits if not audit_missing else None,
            parameter_summary=param_summary,
            judge=judge,
        )
    except Exception as exc:  # noqa: BLE001 — 失败开放
        judge_failed = True
        logger.warning(
            "stage4 judge failed for %s: %s",
            task.task_id,
            type(exc).__name__,
        )

    audit_coverage, evidence_sources = _calculate_audit_coverage(
        transcript=transcript,
        trajectory=trajectory,
        tool_mapping=mapping,
        parameter_audit_present=not audit_missing,
        observation_audit_present=observation_observed,
        judge_succeeded=not judge_failed,
    )

    report = merge_confidence(
        task_id=task.task_id,
        format_score=format_score,
        format_reason=format_reason,
        programmatic_gates=programmatic_gates,
        draft=draft,
        judge_call_failed=judge_failed,
        param_score=param_score,
        param_reason=param_reason,
        parameter_summary=param_summary,
        image_selection_note=str(
            getattr(task, "image_selection_note", "") or ""
        ),
        audit_coverage=audit_coverage,
        evidence_sources=evidence_sources,
        soft_flags=soft_flags,
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

    # Human-review material is advisory only: no model call, no retry loop and
    # no quality gate. A sidecar failure must not discard the completed sample.
    try:
        packet = build_review_packet(
            report=report,
            task=task,
            trajectory=trajectory,
            transcript=transcript,
            parameter_audits=audits,
            context_transcript=review_context_transcript,
            source_video_path=source_video_path,
        )
        write_review_packet(packet, report_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("stage4 review sidecar failed for %s: %s", task.task_id, type(exc).__name__)

    return report
