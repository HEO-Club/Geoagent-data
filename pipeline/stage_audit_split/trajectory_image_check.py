"""Stage2 后：轨迹 / 选图 / visual_evidence_brief 三方一致性门禁。

只拦高精度冲突；不改轨迹、不换图、不重蒸。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.llm import call_structured
from pipeline.schemas.freeform import FreeFormTrajectory

logger = logging.getLogger(__name__)

CONSISTENCY_HINT = (
    "判断已选定位输入图是否与视觉证据简报、轨迹开篇 reasoning 明显冲突。\n"
    "只抓高精度冲突：\n"
    "- brief/thought 依赖的核心视觉事实在选中图中明显不存在；或\n"
    "- 选中图主体与 brief 明显不是同一场景；或\n"
    "- 开篇 reasoning 明确依赖第二份独立输入（如图1与图2、另一张原图），"
    "但只附上 1 张选中图。\n"
    "低置信、细节遗漏、表述风格差异一律 conflict=false。"
    "单图题仅写「图中/待定位图」不算多输入依赖。\n"
    "禁止使用 groundtruth；不要建议换图或重写轨迹。"
)

# 通用「明确依赖第二份独立输入」表述；非样本特判词表
_MULTI_INPUT_RE = re.compile(
    r"(?:"
    r"图\s*[1１一]\s*.{0,24}图\s*[2２二]"
    r"|图\s*[2２二]"
    r"|两张(?:原图|图|照片|输入)"
    r"|另一张(?:原图|图|照片|输入)"
    r"|第二张(?:原图|图|照片|输入)"
    r"|image\s*1.{0,24}image\s*2"
    r"|second\s+(?:image|photo|input)"
    r"|another\s+(?:image|photo|input)"
    r")",
    re.IGNORECASE | re.DOTALL,
)


class TrajectoryImageConsistencyResult(BaseModel):
    """轨迹–选图一致性结果。"""

    conflict: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


def _collect_early_reasoning(trajectory: FreeFormTrajectory, *, max_steps: int = 4) -> str:
    """取前若干条 reasoning thought，供一致性检查。"""
    thoughts: list[str] = []
    for step in trajectory.steps:
        if step.event_type != "reasoning":
            continue
        text = step.thought.strip()
        if text:
            thoughts.append(text)
        if len(thoughts) >= max_steps:
            break
    return "\n".join(f"- {t}" for t in thoughts)


def _early_reasoning_implies_multi_input(early: str) -> bool:
    """开篇 reasoning 是否明确依赖第二份独立输入。"""
    return bool(_MULTI_INPUT_RE.search(early or ""))


def check_trajectory_image_consistency(
    *,
    image_paths: list[str],
    visual_evidence_brief: str,
    trajectory: FreeFormTrajectory,
    min_confidence: float = 0.8,
) -> TrajectoryImageConsistencyResult:
    """选中图 + brief + 轨迹前段 reasoning 的高精度冲突检测。

    Args:
        image_paths: 已选关键帧路径。
        visual_evidence_brief: 题级视觉证据简报；可为空。
        trajectory: Stage2 自由轨迹。
        min_confidence: 低于该置信度的 conflict 视为无效。

    Returns:
        结构化判定；调用失败时默认无冲突。
    """
    images = [p for p in image_paths if str(p).strip() and Path(p).is_file()]
    brief = visual_evidence_brief.strip()
    early = _collect_early_reasoning(trajectory)
    if not images:
        return TrajectoryImageConsistencyResult(
            conflict=False,
            confidence=0.0,
            reason="无可用选中图，跳过一致性检查",
        )
    if not brief and not early:
        return TrajectoryImageConsistencyResult(
            conflict=False,
            confidence=0.0,
            reason="无 brief 与 early reasoning，跳过一致性检查",
        )

    if len(images) == 1 and _early_reasoning_implies_multi_input(early):
        return TrajectoryImageConsistencyResult(
            conflict=True,
            confidence=1.0,
            reason="开篇 reasoning 明确依赖第二份独立输入，但只选出 1 张图",
        )

    prompt = (
        f"{CONSISTENCY_HINT}\n\n"
        f"视觉证据简报：\n{brief or '（空）'}\n\n"
        f"轨迹开篇 reasoning：\n{early or '（空）'}\n\n"
        f"已附上 {len(images)} 张选中图（按顺序）。"
        "请输出 conflict、confidence、reason。"
    )
    try:
        result = call_structured(
            prompt,
            TrajectoryImageConsistencyResult,
            images=images,
            lane="vlm",
            max_attempts=1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("trajectory-image consistency check failed: %s", exc)
        return TrajectoryImageConsistencyResult(
            conflict=False,
            confidence=0.0,
            reason=f"一致性检查调用失败：{type(exc).__name__}",
        )

    conflict = bool(result.conflict) and float(result.confidence) >= float(min_confidence)
    return TrajectoryImageConsistencyResult(
        conflict=conflict,
        confidence=float(result.confidence),
        reason=str(result.reason or "").strip(),
    )
