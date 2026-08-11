"""阶段2：视频 + 字幕 → 地理图片定位 agent 自由 TAO 链。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

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
    "你从地理定位讲解视频中，蒸馏一条可供 SFT 训练的地理图片定位 agent 事件轨迹。"
    "Agent 面对的是待定位图片与场景证据，不是读字幕的观众，也不是视频讲解员。"
    "讲解内容参考（含时间戳旁白）仅为你的内部蒸馏材料；"
    "产物 thought / params / observation / notes 中禁止出现渠道与媒介元话语，包括但不限于："
    "「字幕」「旁白」「博主说」「视频里提到」「求助者」「网友」「评论区」「私信」"
    "「观众」「UP主」「本期视频」「求助图」；"
    "待定位图一律称为「图中 / 待定位图 / 图1 / 图2」。"
    "线索应写成 agent 已观察到的视觉/地理证据或工作假设。"
    "若提供「Agent 已知工作范围」，它来自问题设置的外部给定先验（非地理推理结论、非博主演绎候选）；"
    "须当作已知先验使用，禁止在 thought 中解释来源，禁止把博主候选升格为已知范围。"
    "社交开场、纯 UI、无关感慨等无增益内容静默跳过：不要生成对应步骤，也不要在 notes 里罗列删了什么；"
    "notes 默认 null（或极短质量备注，禁止去噪清单）。"
    "输出事件分为 reasoning、tool_call、final 三类。"
    "reasoning 只包含 thought，用于直接观察输入图、合并已有证据、更新假设、排除候选、解释矛盾或规划下一步；"
    "reasoning 的 tool 必须为 null、params 必须为空对象、observation 必须为 null。"
    "允许 Thought→Thought→Action：当一段内部推理较长时，可以拆成连续 reasoning，"
    "但每条必须完成一次实质性的认知更新，不得把同一句话切碎，不得空转复述；通常连续不超过三条。"
    "如果一段思考只是解释为何马上调用某个工具，应直接写入该 tool_call 的 thought，不要额外拆 reasoning。"
    "tool_call 只用于真实外部动作：访问搜索引擎、数据库、地图/街景/卫星/天气服务，"
    "执行图像增强/OCR/反向搜图、GIS/计算程序，或检索外部档案和媒体。"
    "直接看图得出的天空、植被、朝向、建筑形态，基于已有结果进行比较、筛选、排名、总结、"
    "形成目标签名或时间一致性判断，都属于 reasoning，不得发明 inspect/filter/build/assess 类伪工具。"
    "只有 action 确实带来上下文中原本没有的新信息时才建立 tool_call；"
    "其 thought 写清当前信息缺口与调用原因，tool 使用忠实描述真实执行器的自由名称，params 写实际输入，"
    "observation 只写该外部动作实际返回或明确报告的新结果。"
    "禁止使用或猜测 groundtruth / 官方真值坐标。"
    "Observation 只能复现讲解材料中明确展示、报告的工具结果，或直接可见的画面事实；"
    "不得把常识推测包装成已执行工具的返回，不得自行补写材料中没有的坐标、距离、角度、像素差、"
    "日期时间、编号、候选数量、置信度百分比等精细数据。"
    "材料只支持定性判断时必须保持定性；材料明确表示尚待核验时，Observation 也必须保留未核验状态。"
    "最后一步必须是 event_type=final 且只能使用 tool=final_answer，params 必须且只能包含 location；"
    "单地点写字符串；若本任务内讲解给出多个并列最终地点，可用地点字符串数组；"
    "地点名称须忠实保留讲解最终结论，不得换成 result/site/answer 等字段；"
    "最后一步 observation 必须为 null。"
)

# 通用渠道元话语（非样本特判）；命中则触发一次口吻重写
META_LEAK_TERMS = (
    "求助者",
    "网友",
    "评论区",
    "私信",
    "观众",
    "博主",
    "UP主",
    "本期视频",
    "求助图",
    "字幕",
    "旁白",
)


class _LLMFreeFormStep(FreeFormStep):
    """LLM 输出事件（与自由事件软信封对齐）。"""


class _LLMFreeFormResult(BaseModel):
    """LLM 整条自由链。"""

    steps: list[_LLMFreeFormStep] = Field(default_factory=list)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_final_answer(self) -> "_LLMFreeFormResult":
        """硬校验统一终端答案，避免模型只“准备总结”却不落答案。"""
        if not self.steps:
            raise ValueError("steps 不能为空，且末步必须输出 final_answer")
        if any(step.event_type == "final" for step in self.steps[:-1]):
            raise ValueError("final_answer 只能出现在最后一步")

        final = self.steps[-1]
        if final.event_type != "final" or final.tool != "final_answer":
            raise ValueError("最后一步必须是 event_type=final 且 tool=final_answer")
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
        for step in self.steps[:-1]:
            if step.event_type == "tool_call" and step.observation is None:
                raise ValueError(
                    "真实 tool_call 必须提供 observation；无结果也要写结构化失败状态"
                )
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
        f"Agent 已知工作范围（外部给定先验，禁止解释来源）：\n{working_scope.region}\n"
    )


def _steps_blob(steps: list[_LLMFreeFormStep] | list[FreeFormStep] | list[Any]) -> str:
    payload: list[dict] = []
    for step in steps:
        if hasattr(step, "model_dump"):
            payload.append(step.model_dump())  # type: ignore[union-attr]
        else:
            tool = getattr(step, "tool", None)
            event_type = getattr(step, "event_type", None) or (
                "final"
                if tool == "final_answer"
                else "tool_call"
                if tool
                else "reasoning"
            )
            payload.append(
                {
                    "event_type": event_type,
                    "thought": step.thought,
                    "tool": tool,
                    "params": dict(step.params or {}),
                    "observation": step.observation,
                }
            )
    return json.dumps(payload, ensure_ascii=False)


def trajectory_has_meta_leak(
    steps: list[_LLMFreeFormStep] | list[FreeFormStep],
) -> bool:
    """检测产物是否含渠道元话语。"""
    parts: list[str] = []
    for step in steps:
        parts.append(step.thought)
        parts.append(json.dumps(step.params or {}, ensure_ascii=False))
        if step.observation is not None:
            parts.append(json.dumps(step.observation, ensure_ascii=False))
    blob = "\n".join(parts)
    return any(term in blob for term in META_LEAK_TERMS)


def _rewrite_agent_voice(
    result: _LLMFreeFormResult,
    *,
    images: list[str] | None,
    max_attempts: int | None = None,
) -> _LLMFreeFormResult:
    """命中元话语时做一次口吻重写。"""
    prompt = (
        f"{DEFAULT_SYSTEM_HINT}\n\n"
        "下列轨迹含渠道元话语，请原样保留推理结构与 final_answer.location，"
        "仅把 thought/params/observation 改写成地理定位 agent 口吻；"
        "严格保留每条 event_type，禁止把 reasoning 改成 tool_call"
        "（用「图中/待定位图/图1/图2」，删除求助者/网友等词）。\n"
        f"{_steps_blob(result.steps)}\n"
    )
    return call_structured(
        prompt,
        _LLMFreeFormResult,
        images=images or None,
        lane="llm",
        max_attempts=max_attempts,
    )


def run_stage2(
    video_path: str,
    transcript: list[TranscriptSegment],
    *,
    out_path: str | None = None,
    image_path: str | None = None,
    image_paths: list[str] | None = None,
    source_video: str | None = None,
    max_attempts: int | None = None,
) -> FreeFormTrajectory:
    """蒸馏为地理图片定位 agent 自由 TAO（内容优先，无统一 tool schema）。

    Args:
        video_path: 视频路径。
        transcript: 字幕（仅作内部蒸馏材料，不得写入产物元话语）。
        out_path: 可选落盘路径；默认 intermediate/{id}/stage2_freeform_tao.json。
        image_path: 兼容单图；与 ``image_paths`` 二选一优先后者。
        image_paths: 任务关键帧（可多图）；编排器应传入审核切分结果。
        source_video: 写入产物的来源 id；默认视频 stem。
        max_attempts: 可选单次运行重试上限；默认沿用全局配置。

    Returns:
        FreeFormTrajectory 软信封（含可选 working_scope）。
    """
    settings = get_settings()
    video_id = (source_video or Path(video_path).stem).strip() or Path(video_path).stem
    images: list[str] = []
    if image_paths:
        images = [p for p in image_paths if str(p).strip()]
    elif image_path:
        images = [image_path]
    else:
        # 独立 CLI 兼容：无审核帧时回退均匀概览帧
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
        "请输出 steps，每条显式包含 event_type。reasoning 事件只写 thought；"
        "tool_call 事件才写真实 tool、params 与 observation。notes 默认 null。\n"
        "三种事件的结构示例："
        '{"event_type":"reasoning","thought":"根据已有日落时间可排除较早天黑的候选",'
        '"tool":null,"params":{},"observation":null}；'
        '{"event_type":"tool_call","thought":"仍缺少候选区的连续云量证据，因此查询历史天气档案",'
        '"tool":"historical_weather_query","params":{"region":"候选区"},'
        '"observation":{"result":"明确查询结果"}}；'
        "无论前面有多少步，最后一步必须严格写成："
        '{"event_type":"final","thought":"基于已有证据提交最终地点","tool":"final_answer",'
        '"params":{"location":"最终地点"},"observation":null}。'
        "若本任务内讲解给出多个并列最终地点，location 使用字符串数组。"
    )
    result = call_structured(
        prompt,
        _LLMFreeFormResult,
        images=images or None,
        lane="llm",
        max_attempts=max_attempts,
    )
    if trajectory_has_meta_leak(result.steps):
        logger.info("stage2 meta leak detected; rewriting agent voice once")
        result = _rewrite_agent_voice(
            result,
            images=images or None,
            max_attempts=max_attempts,
        )
    traj = FreeFormTrajectory(
        source_video=video_id,
        steps=[
            FreeFormStep(
                event_type=(
                    getattr(s, "event_type", None)
                    or (
                        "final"
                        if getattr(s, "tool", None) == "final_answer"
                        else "tool_call"
                        if getattr(s, "tool", None)
                        else "reasoning"
                    )
                ),
                thought=s.thought,
                tool=getattr(s, "tool", None),
                params=dict(s.params or {}),
                observation=s.observation,
            )
            for s in result.steps
        ],
        notes=result.notes,
        working_scope=working_scope,
    )

    dest = (
        Path(out_path)
        if out_path
        else (Path(settings.INTERMEDIATE_DIR) / video_id / "stage2_freeform_tao.json")
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    return traj


def load_freeform(path: str | Path) -> FreeFormTrajectory:
    """从落盘 JSON 加载自由链。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FreeFormTrajectory.model_validate(data)
