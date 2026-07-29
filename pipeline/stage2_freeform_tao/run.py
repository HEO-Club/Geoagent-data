"""阶段2：视频 + 字幕 → 自由 TAO 逻辑链。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from pipeline.config import get_settings
from pipeline.llm import call_structured
from pipeline.media.keyframes import extract_keyframes, video_duration_sec
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_HINT = (
    "你是地理定位推理蒸馏器。根据视频关键帧与字幕，复现博主的地理定位推理逻辑链。"
    "删除社交开场、纯 UI、无关科普等无增益内容。"
    "每步包含 thought、自定义 tool 名、params、observation。"
    "tool 名由你发明，无需匹配任何预置工具池。"
    "禁止使用或猜测 groundtruth / 官方真值坐标。"
)


class _LLMFreeFormStep(BaseModel):
    """LLM 输出步（与软信封对齐）。"""

    thought: str
    tool: str
    params: dict = Field(default_factory=dict)
    observation: Optional[dict] = None


class _LLMFreeFormResult(BaseModel):
    """LLM 整条自由链。"""

    steps: list[_LLMFreeFormStep] = Field(default_factory=list)
    notes: Optional[str] = None


def _format_transcript(transcript: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for seg in transcript:
        lines.append(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}")
    return "\n".join(lines)


def _pick_overview_timestamps(duration: float, count: int = 6) -> list[float]:
    if duration <= 0:
        return [0.0]
    if count <= 1:
        return [duration * 0.5]
    return [duration * i / (count - 1) for i in range(count)]


def run_stage2(
    video_path: str,
    transcript: list[TranscriptSegment],
    *,
    out_path: str | None = None,
    image_path: str | None = None,
) -> FreeFormTrajectory:
    """视频+字幕 → 自由 TAO 逻辑链（内容优先，无统一 tool schema）。

    Args:
        video_path: 视频路径。
        transcript: 阶段1 字幕。
        out_path: 可选落盘路径；默认 intermediate/{id}/stage2_freeform_tao.json。
        image_path: 可选代表图；缺省时从视频抽若干概览帧。

    Returns:
        FreeFormTrajectory 软信封。
    """
    settings = get_settings()
    video_id = Path(video_path).stem
    images: list[str] = []
    if image_path:
        images = [image_path]
    else:
        try:
            duration = video_duration_sec(video_path)
            stamps = _pick_overview_timestamps(duration)
            images = extract_keyframes(video_path, stamps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("stage2 keyframe extract failed: %s", exc)

    prompt = (
        f"{DEFAULT_SYSTEM_HINT}\n\n"
        f"视频 ID: {video_id}\n"
        "字幕（带时间戳）：\n"
        f"{_format_transcript(transcript)}\n\n"
        "请输出 steps：每步 thought / tool / params / observation；"
        "notes 可简述删除了哪些无用部分。\n"
        "observation 用 JSON 对象；终端步可 observation=null。"
    )
    result = call_structured(
        prompt,
        _LLMFreeFormResult,
        images=images or None,
        lane="llm",
    )
    traj = FreeFormTrajectory(
        source_video=video_id,
        steps=[
            FreeFormStep(
                thought=s.thought,
                tool=s.tool,
                params=dict(s.params or {}),
                observation=s.observation,
            )
            for s in result.steps
        ],
        notes=result.notes,
    )

    dest = Path(out_path) if out_path else (
        Path(settings.INTERMEDIATE_DIR) / video_id / "stage2_freeform_tao.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    return traj


def load_freeform(path: str | Path) -> FreeFormTrajectory:
    """从落盘 JSON 加载自由链。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FreeFormTrajectory.model_validate(data)
