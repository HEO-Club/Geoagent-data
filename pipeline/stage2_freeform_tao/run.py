"""阶段2：视频 + 字幕 → 地理图片定位 agent 自由 TAO 链。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from pipeline.config import get_settings
from pipeline.llm import call_structured
from pipeline.media.keyframes import extract_keyframes, video_duration_sec
from pipeline.schemas.clues import WorkingScope
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao.extract_scope import extract_working_scope

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_HINT = (
    "你从地理定位讲解视频中，蒸馏一条可供 SFT 训练的地理图片定位 agent 的 ReAct/TAO 轨迹。"
    "Agent 面对的是待定位图片与场景证据，不是读字幕的观众。"
    "讲解内容参考（含时间戳旁白）仅为你的内部蒸馏材料；"
    "产物 thought / params / observation / notes 中禁止出现「字幕」「旁白」「博主说」「视频里提到」等元话语；"
    "线索应写成 agent 已观察到的视觉/地理证据或工作假设。"
    "若提供「Agent 已知工作范围」，它来自问题设置的外部给定先验（非地理推理结论、非博主演绎候选）；"
    "须当作已知先验使用，禁止在 thought 中解释来源，禁止把博主候选升格为已知范围。"
    "社交开场、纯 UI、无关感慨等无增益内容静默跳过：不要生成对应步骤，也不要在 notes 里罗列删了什么；"
    "notes 默认 null（或极短质量备注，禁止去噪清单）。"
    "每步 thought 必须体现：当前假设/已确认状态 + 仍缺什么信息 → 因此调用本步 tool；"
    "禁止空转复述；禁止预知本步 observation。"
    "每步包含 thought、自定义 tool 名、params、observation；tool 名由你发明，无需匹配任何预置工具池。"
    "禁止使用或猜测 groundtruth / 官方真值坐标。"
    "Observation 只能复现讲解材料中明确展示、报告的工具结果，或直接可见的画面事实；"
    "不得把常识推测包装成已执行工具的返回，不得自行补写材料中没有的坐标、距离、角度、像素差、"
    "日期时间、编号、候选数量、置信度百分比等精细数据。"
    "材料只支持定性判断时必须保持定性；材料明确表示尚待核验时，Observation 也必须保留未核验状态。"
    "最后一步必须且只能使用 tool=final_answer，params 必须且只能包含 location；"
    "单地点写字符串，多道题写地点字符串数组，地点名称须忠实保留讲解最终结论，不得换成 result/site/answer 等字段；"
    "最后一步 observation 必须为 null。"
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

    @model_validator(mode="after")
    def validate_final_answer(self) -> "_LLMFreeFormResult":
        """硬校验统一终端答案，避免模型只“准备总结”却不落答案。"""
        if not self.steps:
            raise ValueError("steps 不能为空，且末步必须输出 final_answer")
        if any(step.tool == "final_answer" for step in self.steps[:-1]):
            raise ValueError("final_answer 只能出现在最后一步")

        final = self.steps[-1]
        if final.tool != "final_answer":
            raise ValueError("最后一步 tool 必须严格等于 final_answer")
        if set(final.params) != {"location"}:
            raise ValueError("final_answer.params 必须且只能包含 location")
        location = final.params["location"]
        if isinstance(location, str):
            valid_location = bool(location.strip())
        elif isinstance(location, list):
            valid_location = bool(location) and all(
                isinstance(item, str) and bool(item.strip()) for item in location
            )
        else:
            valid_location = False
        if not valid_location:
            raise ValueError("final_answer.params.location 必须是非空地点或地点数组")
        if final.observation is not None:
            raise ValueError("final_answer.observation 必须为 null")
        return self


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


def _format_scope_block(working_scope: WorkingScope | None) -> str:
    """蒸馏 prompt 中的已知工作范围块（仅展示短语）。"""
    if working_scope is None:
        return "Agent 已知工作范围：无外部工作范围。\n"
    return (
        "Agent 已知工作范围（外部给定先验，禁止解释来源）：\n"
        f"{working_scope.region}\n"
    )


def run_stage2(
    video_path: str,
    transcript: list[TranscriptSegment],
    *,
    out_path: str | None = None,
    image_path: str | None = None,
) -> FreeFormTrajectory:
    """蒸馏为地理图片定位 agent 自由 TAO（内容优先，无统一 tool schema）。

    Args:
        video_path: 视频路径。
        transcript: 阶段1 字幕（仅作内部蒸馏材料，不得写入产物元话语）。
        out_path: 可选落盘路径；默认 intermediate/{id}/stage2_freeform_tao.json。
        image_path: 可选代表图；缺省时从视频抽若干概览帧。

    Returns:
        FreeFormTrajectory 软信封（含可选 working_scope）。
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

    extraction = extract_working_scope(transcript)
    working_scope = extraction.working_scope

    prompt = (
        f"{DEFAULT_SYSTEM_HINT}\n\n"
        f"视频 ID: {video_id}\n"
        f"{_format_scope_block(working_scope)}"
        "讲解内容参考（仅供蒸馏，禁止写入产物）：\n"
        f"{_format_transcript(transcript)}\n\n"
        "请输出 steps：每步 thought / tool / params / observation；"
        "每步 thought 写清当前假设缺口与为何调用本步 tool；"
        "notes 默认 null。\n"
        "普通步骤 observation 用 JSON 对象。无论前面有多少步，最后一步必须严格写成："
        '{"thought":"基于已有证据提交最终地点","tool":"final_answer",'
        '"params":{"location":"最终地点"},"observation":null}。'
        "若视频包含多道独立定位题，location 使用字符串数组并按讲解顺序列出全部最终地点。"
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
        working_scope=working_scope,
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
