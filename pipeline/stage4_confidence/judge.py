"""阶段4：VLM 结构化裁判。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from pipeline.llm import call_structured
from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.confidence import (
    ConfidenceJudgeDraft,
    ParameterReadinessSummary,
)
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.tools import ToolParameterAudit
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

JudgeFn = Callable[..., ConfidenceJudgeDraft]

JUDGE_HINT = (
    "你是地理定位 SFT 样本的质量裁判。根据字幕切片、视觉证据简报、选中图、"
    "自由轨迹与规范化轨迹，给出各维度 0~1 分，并列出模型侧硬门槛命中项。\n"
    "禁止使用 groundtruth；不要建议改写轨迹或换图。\n"
    "刻度（各维独立对照，禁止把「能复述字幕 / 主链能讲通」直接打 0.9+）：\n"
    "- 0.95–1.00：近乎无可指摘。证据可逐条落到字幕/选中图/工具回执；"
    "location 覆盖题面要求的全部独立目标且粒度不粗于字幕最终结论；"
    "Observation 像外部回执而非讲解总结；选中图是干净题面原图。\n"
    "- 0.80–0.94：主链正确，但有明确可指出的少数瑕疵，例如答案比字幕结论粗一档、"
    "个别 Observation 是总结性复述、选中图带平台水印或视频字幕但仍是同一场景。\n"
    "- 0.65–0.79：能讲通但有实质性缺口，例如多目标漏答其一、逻辑有跳跃但仍收敛、"
    "输入图明显是讲解包装帧。\n"
    "- 0.40–0.64：核心任务未完成或证据链明显不闭合。\n"
    "- 0.00–0.39：严重缺陷；高精度时应同时填写 hard_gates。\n"
    "默认从 0.75 起评，按瑕疵下修、仅当几乎无瑕才上修到 0.95+。"
    "同一瑕疵不要在所有维度重复打到 0.4 以下。\n"
    "维度：\n"
    "- evidence_grounding：Thought/Observation 能否在字幕、选中图或工具回执中找到依据；"
    "Observation 若只是旁白总结而非材料中的工具回执，本维应落在 0.80–0.90 而非 0.95+\n"
    "- final_answer_support：location 是否完整、是否由前置证据链推出、有无无源精细坐标；"
    "题面/字幕要求的独立定位目标未全部出现在 location 时，本维应 ≤0.65\n"
    "- tool_param_correctness：可选参考分；程序化参数审计（ready/"
    "context_resolvable/repairable/invalid）会覆盖本维\n"
    "- logical_consistency：候选生成→排除→收敛是否自洽，有无矛盾或跳跃\n"
    "- input_quality_alignment：ASR 是否缺关键条件；选中图是否像题面原图；"
    "带讲解水印/字幕条但仍是同一场景时本维应 0.70–0.85，不得因「内容能对上」打 0.95+\n"
    "- sft_format_completeness：可选参考分；程序化规则会覆盖\n"
    "硬门槛 hard_gates（仅高精度命中时填写，code 用英文蛇形）：\n"
    "- fabricated_observation：明显伪造 Observation\n"
    "- tool_params_unexecutable：工具参数无法执行或与 purpose 矛盾"
    "（schema 级 invalid 由程序化门槛覆盖，此处只记语义矛盾）\n"
    "- hallucinated_precision：无源精细坐标/距离/检索命中/库结果\n"
    "- image_trajectory_mismatch：轨迹所依原图与实选图明显不一致\n"
    "- answer_leaking_image：选中图含答案泄露（评估记录未覆盖时）\n"
    "- incomplete_final_targets：题面要求多个独立最终地点，location 明显少答\n"
    "低置信问题不要记硬门槛。\n"
    "输出要求：\n"
    "- notes 必填：概括样本质量与主要问题，禁止空字符串。\n"
    "- 任一维度得分 < 0.80 时，dimension_reasons 必须写可核对证据"
    "（引用字幕/轨迹/选中图/参数审计中的具体事实），禁止只写「较差」。"
)


def _format_transcript(transcript: list[TranscriptSegment], *, max_chars: int = 4000) -> str:
    lines = [f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}" for seg in transcript]
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n…(截断)"
    return text or "（空）"


def _truncate_json(value: Any, *, max_chars: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def _format_trajectory_brief(traj: Trajectory, *, max_steps: int = 24) -> str:
    parts: list[str] = []
    for i, step in enumerate(traj.steps[:max_steps], start=1):
        tool = step.action.tool if step.action else None
        params = step.action.params if step.action else None
        loc = None
        if step.action and step.event_type == "final":
            loc = (step.action.params or {}).get("location")
        obs = step.observation
        obs_s = json.dumps(obs, ensure_ascii=False) if obs is not None else "null"
        if len(obs_s) > 200:
            obs_s = obs_s[:200] + "…"
        thought = step.thought[:300] + ("…" if len(step.thought) > 300 else "")
        line = f"{i}. [{step.event_type}] thought={thought}"
        if tool:
            line += f" tool={tool}"
        if loc is not None:
            line += f" location={loc!r}"
        if step.event_type == "tool_call" and isinstance(params, dict):
            op = params.get("operation")
            purpose = params.get("purpose")
            inputs = params.get("inputs")
            if op is not None:
                line += f" operation={op!r}"
            if purpose is not None:
                purpose_s = str(purpose)
                if len(purpose_s) > 80:
                    purpose_s = purpose_s[:80] + "…"
                line += f" purpose={purpose_s!r}"
            if inputs is not None:
                line += f" inputs={_truncate_json(inputs)}"
            line += f" obs={obs_s}"
        parts.append(line)
    if len(traj.steps) > max_steps:
        parts.append(f"…共 {len(traj.steps)} 步，已截断")
    return "\n".join(parts) or "（空轨迹）"


def _format_freeform_brief(freeform: FreeFormTrajectory, *, max_steps: int = 24) -> str:
    parts: list[str] = []
    for i, step in enumerate(freeform.steps[:max_steps], start=1):
        thought = step.thought[:280] + ("…" if len(step.thought) > 280 else "")
        line = f"{i}. [{step.event_type}] {thought}"
        if step.tool:
            line += f" | tool={step.tool}"
        if step.event_type == "tool_call" and step.params:
            line += f" | params={_truncate_json(step.params, max_chars=160)}"
        parts.append(line)
    return "\n".join(parts) or "（空）"


def _format_parameter_audit_brief(
    audits: list[ToolParameterAudit] | None,
    summary: ParameterReadinessSummary | None,
    *,
    max_calls: int = 16,
) -> str:
    if summary is not None and summary.audit_missing:
        return "（parameter audit 缺失或损坏）"
    if not audits:
        if summary is not None and summary.total_calls == 0:
            return "（无 Tool 调用）"
        return "（无）"
    lines: list[str] = []
    if summary is not None:
        lines.append(
            f"汇总 total={summary.total_calls} ready={summary.ready} "
            f"context_resolvable={summary.context_resolvable} "
            f"repairable={summary.repairable} invalid={summary.invalid}"
            + (f" worst={summary.worst}" if summary.worst else "")
        )
    for audit in audits[:max_calls]:
        issue_codes = ",".join(i.code for i in audit.issues[:5]) or "-"
        line = (
            f"step={audit.step_index} tool={audit.tool} "
            f"op={audit.operation} readiness={audit.readiness} "
            f"issues={issue_codes}"
        )
        if audit.repair_actions:
            strategies = ",".join(
                f"{a.field}:{a.strategy}" for a in audit.repair_actions[:3]
            )
            line += f" repairs={strategies}"
        lines.append(line)
    if len(audits) > max_calls:
        lines.append(f"…共 {len(audits)} 次调用，已截断")
    return "\n".join(lines)


def build_judge_prompt(
    *,
    task: GeoTaskSpec,
    transcript: list[TranscriptSegment],
    freeform: FreeFormTrajectory,
    trajectory: Trajectory,
    tool_mapping: dict[str, Any] | None = None,
    parameter_audits: list[ToolParameterAudit] | None = None,
    parameter_summary: ParameterReadinessSummary | None = None,
) -> str:
    """组装裁判 prompt（不含 groundtruth）。"""
    mapping_summary = ""
    if tool_mapping:
        mapping_summary = json.dumps(
            {
                "tool_calls_before_stage3": tool_mapping.get("tool_calls_before_stage3"),
                "tool_calls_after_stage3": tool_mapping.get("tool_calls_after_stage3"),
                "pseudo_tools_demoted": tool_mapping.get("pseudo_tools_demoted"),
                "unique_canonical_tools": tool_mapping.get("unique_canonical_tools"),
            },
            ensure_ascii=False,
        )
    brief = (task.visual_evidence_brief or "").strip() or "（空）"
    return (
        f"{JUDGE_HINT}\n\n"
        f"task_id={task.task_id}\n"
        f"visual_evidence_brief:\n{brief}\n\n"
        f"字幕切片:\n{_format_transcript(transcript)}\n\n"
        f"自由轨迹摘要:\n{_format_freeform_brief(freeform)}\n\n"
        f"规范化轨迹摘要:\n{_format_trajectory_brief(trajectory)}\n\n"
        f"tool mapping 摘要: {mapping_summary or '（无）'}\n\n"
        "参数审计摘要:\n"
        f"{_format_parameter_audit_brief(parameter_audits, parameter_summary)}\n\n"
        f"已附上 {len(trajectory.image_paths)} 张选中图。"
        "请输出各维度分数、dimension_reasons（弱维必详述）、hard_gates、非空 notes。"
    )


def call_confidence_judge(
    *,
    task: GeoTaskSpec,
    transcript: list[TranscriptSegment],
    freeform: FreeFormTrajectory,
    trajectory: Trajectory,
    tool_mapping: dict[str, Any] | None = None,
    parameter_audits: list[ToolParameterAudit] | None = None,
    parameter_summary: ParameterReadinessSummary | None = None,
    judge: Optional[JudgeFn] = None,
) -> ConfidenceJudgeDraft:
    """调用 VLM 裁判；可注入 judge 回调（测试用）。"""
    images = [
        p
        for p in trajectory.image_paths
        if str(p).strip() and Path(p).is_file()
    ]
    prompt = build_judge_prompt(
        task=task,
        transcript=transcript,
        freeform=freeform,
        trajectory=trajectory,
        tool_mapping=tool_mapping,
        parameter_audits=parameter_audits,
        parameter_summary=parameter_summary,
    )
    if judge is not None:
        return judge(
            prompt=prompt,
            images=images,
            task=task,
            transcript=transcript,
            freeform=freeform,
            trajectory=trajectory,
            tool_mapping=tool_mapping,
            parameter_audits=parameter_audits,
            parameter_summary=parameter_summary,
        )
    return call_structured(
        prompt,
        ConfidenceJudgeDraft,
        images=images or None,
        lane="vlm",
        max_attempts=1,
    )
