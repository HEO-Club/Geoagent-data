"""阶段4：程序化硬门槛与 SFT 格式分。"""

from __future__ import annotations

from typing import Any

from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.confidence import HardGateHit
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment


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
