"""Stage 2 action-coverage review: recover missed map/satellite/streetview calls."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from pipeline.llm import call_structured
from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao.observation_review import _adjacent_context

REGENERATE_CONFIDENCE = 0.85


class ActionCoverageItem(BaseModel):
    """One external action attested by the transcript."""

    action_summary: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    covered_by_trajectory: bool
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_tool: str = ""
    correction: str = ""
    kind: Literal[
        "web_search",
        "map_or_satellite",
        "streetview",
        "measure",
        "media_search",
        "image_edit",
        "other_external",
    ] = "other_external"


class ActionCoverageResult(BaseModel):
    items: list[ActionCoverageItem] = Field(default_factory=list)


ActionCoverageReviewer = Callable[..., ActionCoverageResult]


def missed_actions(
    result: ActionCoverageResult,
    *,
    min_confidence: float = REGENERATE_CONFIDENCE,
) -> list[ActionCoverageItem]:
    """Return high-confidence external actions missing from the trajectory."""

    return [
        item
        for item in result.items
        if (not item.covered_by_trajectory) and item.confidence >= min_confidence
    ]


def review_action_coverage(
    trajectory: FreeFormTrajectory,
    *,
    transcript: list[TranscriptSegment],
    task: GeoTaskSpec | None = None,
    context_transcript: list[TranscriptSegment] | None = None,
    reviewer: ActionCoverageReviewer | None = None,
) -> ActionCoverageResult:
    """Check whether transcript-attested external actions became tool_call steps."""

    steps = [
        {
            "step_index": index,
            "event_type": step.event_type,
            "thought": step.thought,
            "tool": step.tool,
            "params": step.params,
            "observation": step.observation,
        }
        for index, step in enumerate(trajectory.steps, start=1)
    ]
    context = _adjacent_context(task, transcript, context_transcript)
    task_info = (
        {
            "task_id": task.task_id,
            "time_start": task.time_start,
            "time_end": task.time_end,
            "visual_evidence_brief": task.visual_evidence_brief,
        }
        if task
        else None
    )
    prompt = (
        "审核下列蒸馏轨迹是否漏写了字幕已明确执行的外部动作。"
        "只报告材料明确展示/报告过的外部动作；不得要求补做未执行的核验。\n"
        "属于外部动作：打开/平移/调时相地图或卫星、在底图或街景上测量、打开街景、"
        "网页/媒体检索、图像裁剪增强、OCR、反向搜图、GIS/计算程序。\n"
        "不属于外部动作：直接看待定位图、镜头切换、基于已有结果的比较筛选排除排名总结。\n"
        "对每个已确认的外部动作返回一项：action_summary、evidence、covered_by_trajectory、"
        "confidence（0到1）、可选 suggested_tool 与 correction。\n"
        "covered_by_trajectory=true 表示轨迹中已有对应 tool_call（工具名不必字面相同，语义覆盖即可）。\n"
        "若字幕没有任何外部动作，返回空 items。\n"
        "相邻字幕只用于确认方法归属，不能把相邻题结果当成本题证据。不得提供或猜测 groundtruth。\n"
        f"本题范围与视觉简报：{json.dumps(task_info, ensure_ascii=False)}\n"
        f"本题带时间戳字幕：{json.dumps([s.model_dump() for s in transcript], ensure_ascii=False)}\n"
        f"相邻语境（不是本题答案）：{json.dumps(context, ensure_ascii=False)}\n"
        f"当前轨迹步骤：{json.dumps(steps, ensure_ascii=False)}"
    )
    if reviewer is None:
        result = call_structured(
            prompt, ActionCoverageResult, lane="llm", max_attempts=1
        )
    else:
        result = reviewer(
            prompt=prompt,
            trajectory=trajectory,
            transcript=transcript,
            task=task,
        )
    if not isinstance(result, ActionCoverageResult):
        result = ActionCoverageResult.model_validate(result)
    return result


def action_coverage_retry_warning(items: list[ActionCoverageItem]) -> str:
    """Parenthesized private feedback for missed external actions."""

    if not items:
        return ""
    details: list[str] = []
    for item in items:
        fix = item.correction[:200] or "补写对应 tool_call，Observation 仅用旁白已报告结果"
        if item.suggested_tool:
            fix = f"{fix}；建议工具 {item.suggested_tool}"
        details.append(
            f"漏动作：{item.action_summary[:200]}；证据：{item.evidence[:200]}；修正：{fix}"
        )
    return (
        "\n（上一轮动作覆盖提醒，仅用于本轮生成，禁止复制到thought/params/observation/notes："
        + json.dumps(details, ensure_ascii=False)
        + "。以上是字幕已明确的外部动作，不是新事实；保持本题最终答案粒度，"
        "不要把镜头切换或直接看图伪造成工具。）\n"
    )
