"""阶段4：样本置信度评分（人工检查用，不拦入库）。"""

from __future__ import annotations

import json
import logging
import re
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


_DECISION_HELP: dict[str, tuple[str, str]] = {
    "accept": ("建议通过", "可以入库"),
    "provisional_pass": ("暂可入库", "建议抽查后再当高质量样本用"),
    "parameter_repair": ("参数待修", "样本仍保留，工具参数需要补"),
    "needs_review": ("需要人工看一眼", "有软审查项、覆盖率不够，或分数未到通过线"),
    "reject": ("质量建议不通过", "文件仍保留，不删除轨迹"),
}

_PRIORITY_HELP: dict[str, str] = {
    "high": "优先看",
    "medium": "抽查",
    "low": "低优先",
}

_DIMENSION_HELP: dict[str, str] = {
    "evidence_grounding": "证据是否对得上",
    "final_answer_support": "最终地点是否推得出来",
    "tool_param_correctness": "工具参数能不能执行",
    "logical_consistency": "推理有没有跳步",
    "input_quality_alignment": "选图和输入质量",
    "sft_format_completeness": "训练格式",
}

_GATE_HELP: dict[str, str] = {
    "fabricated_observation": "伪造回执",
    "tool_params_unexecutable": "工具参数无法执行或和 purpose 矛盾",
    "hallucinated_precision": "无源高精度数字",
    "answer_leaking_image": "选中图含答案泄露",
    "answer_leakage_selected": "选中帧被标了答案泄露",
    "incomplete_final_targets": "最终地点少答了独立目标",
    "empty_transcript": "字幕切片是空的",
    "missing_final": "没有最终答案步",
    "last_not_final": "最后一步不是 final",
    "final_has_observation": "final 步不该有 Observation",
    "final_bad_tool": "final 步工具名不对",
    "empty_location": "最终地点是空的",
    "reasoning_has_action": "reasoning 步不该带工具动作",
    "observation_review_unavailable": "Observation 审核没做完",
    "observation_audit_stale": "Observation 和审核版本对不上",
    "observation_needs_repair": "Observation 需要人工修",
    "parameter_inputs_invalid": "参数合同过不去",
}

_READINESS_HELP: dict[str, str] = {
    "ready": "齐，可以直接执行",
    "context_resolvable": "缺的能从当前图或上一步结果补",
    "repairable": "能修，但现在不完整",
    "invalid": "合同过不去",
}

_SELECTION_FRAME_RE = re.compile(
    r"-\s*\[(?P<idx>\d+)\]\s*"
    r"t=(?P<t>[-+]?\d+(?:\.\d+)?)s?\s+"
    r"quality=(?P<quality>[-+]?\d+(?:\.\d+)?)\s+"
    r"overlay=(?P<overlay>\w+)\s+"
    r"clean=(?P<clean>\w+)\s+"
    r"support=(?P<support>[-+]?\d+(?:\.\d+)?)\s+"
    r"reason=(?P<reason>.*)$"
)


def _truthy_token(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y"}


def _seconds_clock(seconds: float) -> str:
    total = max(0.0, float(seconds))
    minutes, rest = divmod(int(round(total * 1000)), 60000)
    secs, millis = divmod(rest, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _describe_decision(decision: str) -> tuple[str, str]:
    return _DECISION_HELP.get(decision, (decision, "对照 JSON 里的 decision"))


def _describe_priority(priority: ReviewPriority) -> str:
    return _PRIORITY_HELP.get(priority, str(priority))


def _describe_gate(item: HardGateHit) -> str:
    meaning = _GATE_HELP.get(item.code, "见码名")
    evidence = (item.evidence or "").strip()
    if evidence:
        return f"{meaning}（{item.code}）：{evidence}"
    return f"{meaning}（{item.code}）"


def _describe_readiness(code: str | None) -> str:
    if not code:
        return ""
    return _READINESS_HELP.get(code, code)


def _programmatic_overview(
    *,
    hard_gates: list[HardGateHit],
    weak_dims: list[DimensionScore],
    image_selection_note: str,
    judge_call_failed: bool,
) -> str:
    if judge_call_failed:
        return "语义裁判没跑成，下面分数里的语义维是中性分，不要当成已经证实的低质量。"
    if hard_gates:
        return "有硬门槛，先看下面点名的问题。"
    if weak_dims:
        return "主链能看，但有低于阈值的维度，对照下面明细核对。"
    if "needs_review" in image_selection_note:
        return "主链未见明显硬伤；选图标了待复核，对照选图段看。"
    return "主链未见明显硬伤。"


def _format_selection_note(raw: str) -> str:
    """把 Stage 1.5 程序化选图串转成口语句；解析失败则原样附上。"""
    text = raw.strip()
    if not text:
        return ""
    grade = ""
    count: str | None = None
    reason = ""
    frames: list[str] = []
    recognized = False
    leftover: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("选图质量等级="):
            grade = stripped.split("=", 1)[1].strip()
            recognized = True
            continue
        if stripped.startswith("选中张数="):
            count = stripped.split("=", 1)[1].strip()
            recognized = True
            continue
        if stripped.startswith("选图原因:") or stripped.startswith("选图原因："):
            reason = stripped.split(":", 1)[1].strip()
            recognized = True
            continue
        if stripped in {"选中帧: 无", "选中帧：无", "选中帧明细:"}:
            recognized = True
            continue
        match = _SELECTION_FRAME_RE.match(stripped)
        if match:
            recognized = True
            overlay = "有讲解覆盖" if _truthy_token(match.group("overlay")) else "没有讲解覆盖"
            clean = (
                "是干净原图"
                if _truthy_token(match.group("clean"))
                else "不是干净原图"
            )
            frame_reason = match.group("reason").strip() or "（无逐帧理由）"
            frames.append(
                f"- 第 {int(match.group('idx'))} 张，约 {_seconds_clock(float(match.group('t')))}，"
                f"质量 {float(match.group('quality')):.2f}，{overlay}，{clean}，"
                f"支撑分 {float(match.group('support')):.2f}。理由：{frame_reason}"
            )
            continue
        leftover.append(stripped)

    if not recognized:
        return "选图：" + text

    bits = ["选图："]
    if grade:
        bits[0] += f"质量等级 {grade}"
    if count is not None:
        joiner = "，" if grade else ""
        bits[0] += f"{joiner}选中 {count} 张"
    if reason:
        bits[0] += f"。原因：{reason}"
    elif bits[0] == "选图：":
        bits[0] += "见下方明细"
    if not frames and count == "0":
        frames.append("- 没有选中帧。")
    parts = [bits[0]]
    parts.extend(frames)
    if leftover:
        parts.append("原始选图记录：" + "；".join(leftover))
    return "\n".join(parts)


def _collect_temporary_labels(
    temporary_tools: list[dict[str, Any]] | None,
) -> tuple[list[str], list[str]]:
    tool_bits: list[str] = []
    operation_bits: list[str] = []
    for item in temporary_tools or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "new_executor").strip() or "new_executor"
        reason = str(item.get("reason") or "").strip()
        if kind == "new_operation":
            parent = str(item.get("canonical_name") or "").strip()
            op_name = str(
                item.get("temporary_operation") or item.get("temporary_name") or ""
            ).strip()
            label = f"{parent}.{op_name}" if parent and op_name else (parent or op_name)
            if not label:
                continue
            if reason:
                operation_bits.append(f"{label}。原因：{reason}")
            else:
                operation_bits.append(label)
            continue
        raw_name = str(item.get("raw_tool") or "").strip()
        temp_name = str(item.get("temporary_name") or "").strip()
        if not raw_name and not temp_name:
            continue
        if raw_name and temp_name:
            arrow = f"{raw_name} → {temp_name}"
        else:
            arrow = temp_name or raw_name
        if reason:
            tool_bits.append(f"{arrow}。原因：{reason}")
        else:
            tool_bits.append(arrow)
    return tool_bits, operation_bits


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
    temporary_tools: list[dict[str, Any]] | None = None,
) -> str:
    """组装给人看的评价 notes（中文核对清单，弱维必有明细）。"""
    selection_note = (image_selection_note or "").strip()
    weak_dims = [item for item in dimensions if item.score < weak_below]
    draft_note = draft.notes.strip() if draft and draft.notes.strip() else ""

    decision_label, decision_hint = _describe_decision(decision)
    parts: list[str] = [
        (
            f"先看结论：{decision_label}（{decision}）。"
            f"综合分 {quality_score:.2f}（未压上限前 {base_score:.2f}），"
            f"覆盖率 {audit_coverage:.2f}，复核优先级{_describe_priority(review_priority)}。"
            f"{decision_hint}。"
        )
    ]
    parts.append("")
    parts.append(draft_note or _programmatic_overview(
        hard_gates=hard_gates,
        weak_dims=weak_dims,
        image_selection_note=selection_note,
        judge_call_failed=judge_call_failed,
    ))

    if judge_call_failed:
        parts.extend(
            [
                "",
                "语义裁判没跑成（judge_call_failed）。"
                "分数里的语义维是中性分，不要当成已经证实的低质量。",
            ]
        )

    parts.append("")
    if hard_gates:
        parts.append("硬门槛：" + "；".join(_describe_gate(item) for item in hard_gates))
    else:
        parts.append(
            "硬门槛：没有触发。没有发现伪造回执、无源高精度数字、答案图泄露这类硬伤。"
        )
    if soft_flags:
        parts.append(
            "软审查：" + "；".join(_describe_gate(item) for item in soft_flags)
        )
    if applied_soft_caps:
        cap_bits = [
            f"{_GATE_HELP.get(code, code)}（{code}）≤{value:.2f}"
            for code, value in applied_soft_caps.items()
        ]
        parts.append("软上限把分数压到了：" + "；".join(cap_bits))

    if parameter_summary is not None:
        parts.append("")
        if parameter_summary.audit_missing:
            parts.append("参数：审计文件缺失或损坏，参数分记中性，不要当低质量证据。")
        else:
            worst = parameter_summary.worst
            worst_text = (
                f"最差一次是{_describe_readiness(worst)}（{worst}）"
                if worst
                else "没有最差档记录"
            )
            parts.append(
                f"参数：共 {parameter_summary.total_calls} 次工具调用，"
                f"{parameter_summary.ready} 次齐（ready），"
                f"{parameter_summary.context_resolvable} 次能从上下文补"
                f"（context_resolvable），"
                f"{parameter_summary.repairable} 次能修（repairable），"
                f"{parameter_summary.invalid} 次合同过不去（invalid）。"
                f"{worst_text}。"
            )
            for line in parameter_summary.detail_lines[:8]:
                parts.append(f"- {line}")
            extra = len(parameter_summary.detail_lines) - 8
            if extra > 0:
                parts.append(f"- …另有 {extra} 条参数问题，对照 parameter_readiness")

    parts.append("")
    parts.append(f"各维度（低于 {weak_below:.2f} 的要重点看）：")
    if dimensions:
        for item in dimensions:
            label = _DIMENSION_HELP.get(item.name, item.name)
            reason = (item.reason or "").strip()
            if not reason:
                if item.score < weak_below:
                    reason = "（没有写出可核对细节，请对照 dimensions）"
                else:
                    reason = "未见需要单独说明的问题"
            parts.append(
                f"- {label}（{item.name}）{item.score:.2f}：{reason}"
            )
    else:
        parts.append("- （没有维度分，请对照 JSON）")

    if selection_note:
        parts.append("")
        parts.append(_format_selection_note(selection_note))

    tool_bits, operation_bits = _collect_temporary_labels(temporary_tools)
    if tool_bits or operation_bits:
        parts.append("")
        parts.append("下面只是汇报，不加软上限，也不单独把 decision 压成 needs_review。")
    if tool_bits:
        parts.append("临时工具：" + "；".join(tool_bits))
    if operation_bits:
        parts.append("临时操作：" + "；".join(operation_bits))

    if draft_note:
        parts.append("")
        parts.append("裁判补充：" + draft_note)

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
    temporary_tools: list[dict[str, Any]] | None = None,
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
        temporary_tools=temporary_tools,
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


def _temporary_tools_from_mapping(
    mapping: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """从 mapping 读取临时 tool 与临时 operation，供 notes 按 kind 分行。"""
    if not mapping:
        return []
    merged: list[dict[str, Any]] = []
    tools = mapping.get("temporary_tools")
    if isinstance(tools, list):
        for item in tools:
            if isinstance(item, dict):
                payload = dict(item)
                payload.setdefault("kind", "new_executor")
                merged.append(payload)
    operations = mapping.get("temporary_operations")
    if isinstance(operations, list):
        for item in operations:
            if isinstance(item, dict):
                payload = dict(item)
                payload["kind"] = "new_operation"
                merged.append(payload)
    return merged


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
        temporary_tools=_temporary_tools_from_mapping(mapping),
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
