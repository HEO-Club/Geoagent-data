"""阶段4：程序化硬门槛、SFT 格式分与参数 readiness 分。"""

from __future__ import annotations

from typing import Any, Literal

from pipeline.config import get_settings
from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.confidence import HardGateHit, ParameterReadinessSummary
from pipeline.schemas.tools import ToolParameterAudit
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment

Readiness = Literal["ready", "context_resolvable", "repairable", "invalid"]

_READINESS_RANK: dict[str, int] = {
    "ready": 0,
    "context_resolvable": 1,
    "repairable": 2,
    "invalid": 3,
}


def _location_nonempty(params: dict[str, Any] | None) -> bool:
    if not params:
        return False
    loc = params.get("location")
    if loc is None:
        return False
    if isinstance(loc, str):
        return bool(loc.strip())
    if isinstance(loc, list):
        return any(str(x).strip() for x in loc)
    return bool(str(loc).strip())


def evaluate_programmatic_gates(
    trajectory: Trajectory,
    task: GeoTaskSpec,
    transcript: list[TranscriptSegment],
) -> tuple[list[HardGateHit], float, str]:
    """程序化硬门槛 + SFT 格式完整性分。

    Returns:
        (gates, format_score∈[0,1], format_reason)
    """
    gates: list[HardGateHit] = []
    deductions: list[str] = []
    score = 1.0

    if not transcript or not any(seg.text.strip() for seg in transcript):
        gates.append(
            HardGateHit(code="empty_transcript", evidence="字幕切片为空或无有效文本")
        )
        score -= 0.4
        deductions.append("字幕切片为空")

    selected_leaks = [
        a
        for a in task.frame_assessments
        if a.selected and a.answer_leakage
    ]
    if selected_leaks:
        stamps = ", ".join(f"{a.timestamp:.1f}s" for a in selected_leaks[:3])
        gates.append(
            HardGateHit(
                code="answer_leakage_selected",
                evidence=f"选中帧标记答案泄露: {stamps}",
            )
        )
        score -= 0.5
        deductions.append("选中帧答案泄露")

    if not trajectory.steps:
        gates.append(HardGateHit(code="missing_final", evidence="轨迹无步骤"))
        score -= 0.5
        deductions.append("无步骤")
        format_score = max(0.0, min(1.0, score))
        return gates, format_score, "；".join(deductions) or "格式完整"

    last = trajectory.steps[-1]
    if last.event_type != "final":
        gates.append(
            HardGateHit(
                code="last_not_final",
                evidence=f"末步 event_type={last.event_type!r}，期望 final",
            )
        )
        score -= 0.35
        deductions.append("末步非 final")

    finals = [s for s in trajectory.steps if s.event_type == "final"]
    if not finals:
        gates.append(HardGateHit(code="missing_final", evidence="无 final 事件"))
        score -= 0.5
        deductions.append("缺 final")
    else:
        for idx, step in enumerate(finals):
            if step.observation is not None:
                gates.append(
                    HardGateHit(
                        code="final_has_observation",
                        evidence=f"final[{idx}] observation 非 null",
                    )
                )
                score -= 0.25
                deductions.append("final 带 observation")
            action = step.action
            if action is None or action.tool != "final_answer":
                gates.append(
                    HardGateHit(
                        code="final_bad_tool",
                        evidence=f"final[{idx}] tool 非 final_answer",
                    )
                )
                score -= 0.3
                deductions.append("final tool 错误")
            elif not _location_nonempty(action.params):
                gates.append(
                    HardGateHit(
                        code="empty_location",
                        evidence=f"final[{idx}] params.location 缺失或为空",
                    )
                )
                score -= 0.4
                deductions.append("location 为空")

    for idx, step in enumerate(trajectory.steps):
        if step.event_type == "reasoning" and (
            step.action is not None or step.observation is not None
        ):
            gates.append(
                HardGateHit(
                    code="reasoning_has_action",
                    evidence=f"reasoning[{idx}] 含 action/observation",
                )
            )
            score -= 0.2
            deductions.append("reasoning 含 action")
            break

    # 去重同 code（保留首次证据）
    seen: set[str] = set()
    unique_gates: list[HardGateHit] = []
    for g in gates:
        if g.code in seen:
            continue
        seen.add(g.code)
        unique_gates.append(g)

    format_score = max(0.0, min(1.0, score))
    reason = "；".join(dict.fromkeys(deductions)) if deductions else "SFT 格式契约通过"
    return unique_gates, format_score, reason


def evaluate_observation_audit(
    audit: dict[str, Any] | None,
) -> tuple[list[HardGateHit], list[HardGateHit], bool]:
    """Convert a strict direct-evidence audit into hard/soft Stage 4 flags."""

    if not isinstance(audit, dict):
        return [], [], False
    passes = audit.get("passes")
    items: list[dict[str, Any]] = []
    if isinstance(passes, list) and passes:
        final_pass = passes[-1]
        if isinstance(final_pass, dict) and isinstance(final_pass.get("items"), list):
            items = [item for item in final_pass["items"] if isinstance(item, dict)]
    elif isinstance(audit.get("items"), list):
        items = [item for item in audit["items"] if isinstance(item, dict)]
    if not items:
        return [], [], False

    unsupported = []
    repair = []
    for item in items:
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict in {"unsupported", "fabricated", "reject", "hallucinated"}:
            unsupported.append(item)
        elif verdict not in {"supported", "pass", "accepted"}:
            repair.append(item)
    hard: list[HardGateHit] = []
    soft: list[HardGateHit] = []
    if unsupported:
        evidence = "；".join(
            str(item.get("reason") or item.get("call_id") or "无直接证据")
            for item in unsupported[:3]
        )
        hard.append(HardGateHit(code="fabricated_observation", evidence=evidence))
    if repair or audit.get("accepted") is False:
        evidence = "；".join(
            str(item.get("reason") or item.get("call_id") or "需修复")
            for item in repair[:3]
        ) or "Observation 严格审计尚未完全通过"
        soft.append(HardGateHit(code="observation_needs_repair", evidence=evidence))
    return hard, soft, True


def _call_detail_line(audit: ToolParameterAudit) -> str:
    """单次非 ready / 需说明的调用明细。"""
    issue_bits: list[str] = []
    for issue in audit.issues[:6]:
        field = f"{issue.field}=" if issue.field else ""
        issue_bits.append(f"{field}{issue.code}:{issue.message}")
    if len(audit.issues) > 6:
        issue_bits.append(f"…另有 {len(audit.issues) - 6} 条 issue")
    repair_bits: list[str] = []
    for action in audit.repair_actions[:4]:
        repair_bits.append(
            f"{action.field}/{action.strategy}:{action.guidance[:80]}"
        )
    if len(audit.repair_actions) > 4:
        repair_bits.append(f"…另有 {len(audit.repair_actions) - 4} 条 repair")
    parts = [
        f"step={audit.step_index}",
        f"tool={audit.tool}",
        f"operation={audit.operation}",
        f"readiness={audit.readiness}",
    ]
    if issue_bits:
        parts.append("issues=[" + "; ".join(issue_bits) + "]")
    if repair_bits:
        parts.append("repairs=[" + "; ".join(repair_bits) + "]")
    return " | ".join(parts)


def evaluate_parameter_readiness(
    audits: list[ToolParameterAudit] | None,
    *,
    audit_missing: bool = False,
) -> tuple[float, str, list[HardGateHit], ParameterReadinessSummary]:
    """将 stage3 参数审计四级 readiness 映射为 tool_param_correctness。

    按最差一次 Tool 调用取分；``invalid`` 另记硬门槛 ``tool_params_invalid``。
    审计缺失时记中性分，不加硬门槛（失败开放）。

    Returns:
        (score, reason, gates, summary)
    """
    settings = get_settings()
    score_map: dict[str, float] = {
        "ready": float(settings.STAGE4_PARAM_SCORE_READY),
        "context_resolvable": float(settings.STAGE4_PARAM_SCORE_CONTEXT_RESOLVABLE),
        "repairable": float(settings.STAGE4_PARAM_SCORE_REPAIRABLE),
        "invalid": float(settings.STAGE4_PARAM_SCORE_INVALID),
    }
    missing_score = float(settings.STAGE4_PARAM_SCORE_AUDIT_MISSING)

    if audit_missing:
        summary = ParameterReadinessSummary(audit_missing=True)
        return (
            missing_score,
            "stage3_parameter_audit 缺失或损坏，参数分记中性",
            [],
            summary,
        )

    calls = list(audits or [])
    counts = {
        "ready": 0,
        "context_resolvable": 0,
        "repairable": 0,
        "invalid": 0,
    }
    worst: str | None = None
    detail_lines: list[str] = []

    for audit in calls:
        readiness = str(audit.readiness or "ready").strip()
        if readiness not in counts:
            readiness = "invalid"
        counts[readiness] += 1
        if worst is None or _READINESS_RANK[readiness] > _READINESS_RANK[worst]:
            worst = readiness
        if readiness != "ready":
            detail_lines.append(_call_detail_line(audit))

    summary = ParameterReadinessSummary(
        total_calls=len(calls),
        ready=counts["ready"],
        context_resolvable=counts["context_resolvable"],
        repairable=counts["repairable"],
        invalid=counts["invalid"],
        worst=worst,  # type: ignore[arg-type]
        audit_missing=False,
        detail_lines=detail_lines,
    )

    if not calls:
        return (
            float(settings.STAGE4_PARAM_SCORE_READY),
            "无 Tool 调用，参数合同不适用，记满分",
            [],
            summary,
        )

    assert worst is not None
    score = float(max(0.0, min(1.0, score_map[worst])))
    count_bits = (
        f"total={summary.total_calls} ready={summary.ready} "
        f"context_resolvable={summary.context_resolvable} "
        f"repairable={summary.repairable} invalid={summary.invalid}"
    )
    reason_parts = [f"最差 readiness={worst}", count_bits]
    if detail_lines:
        reason_parts.append("非 ready 调用: " + " || ".join(detail_lines[:8]))
        if len(detail_lines) > 8:
            reason_parts.append(f"…另有 {len(detail_lines) - 8} 次")
    else:
        reason_parts.append("全部 Tool 调用 ready")

    gates: list[HardGateHit] = []
    unexecutable = [
        audit
        for audit in calls
        if audit.readiness == "invalid"
        and any(
            issue.code in {"unknown_canonical_tool", "unknown_operation"}
            for issue in audit.issues
        )
    ]
    if unexecutable:
        evidence = " || ".join(_call_detail_line(audit) for audit in unexecutable[:3])
        gates.append(HardGateHit(code="tool_params_unexecutable", evidence=evidence))

    return score, "；".join(reason_parts), gates, summary
