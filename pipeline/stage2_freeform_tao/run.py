"""阶段2：视频 + 字幕 → 地理图片定位 agent 自由 TAO 链。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from pipeline.config import get_settings
from pipeline.llm import call_structured
from pipeline.media.keyframes import extract_keyframes, video_duration_sec
from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.clues import WorkingScope
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao.action_review import (
    ActionCoverageReviewer,
    action_coverage_retry_warning,
    missed_actions,
    review_action_coverage,
)
from pipeline.stage2_freeform_tao.extract_scope import extract_working_scope
from pipeline.stage2_freeform_tao.observation_review import (
    ObservationReviewer,
    observation_fingerprint,
    retry_warning,
    review_observations,
)
from pipeline.stage3_normalize_format.trees import load_forest
from pipeline.tool_catalog_v2 import render_tool_contract_guidance

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
    "旁白明确打开、平移或调时相地图/卫星，在底图或街景上测量，或打开街景会话时，必须写 tool_call；"
    "Observation 可以是旁白已报告的定性结果（如「许昌几乎都是平原」「河宽约80m」），"
    "不必伪造 URL、API 载荷或标准搜索结果页；禁止因「回执不够像搜索结果」把上述动作改成 reasoning。"
    "执行器选择原则（非词表特判）：网页关键词检索→web_search；调历史底图/遥感时相→satellite_imagery_query；"
    "在图上量宽/距→distance_bearing_calculator 或卫星/地图查询对应测量结果；打开街景→streetview_query；"
    "禁止把后三类默认写成 web_search。"
    "直接看图得出的天空、植被、朝向、建筑形态，基于已有结果进行比较、筛选、排名、总结、"
    "形成目标签名或时间一致性判断，都属于 reasoning，不得发明 inspect/filter/build/assess 类伪工具。"
    "只有 action 确实带来上下文中原本没有的新信息时才建立 tool_call；"
    "其 thought 写清当前信息缺口与调用原因，tool 优先使用提供的 Canonical Tool 名称，"
    "params 按 operation/purpose/inputs 固定合同填写，"
    "observation 只写该外部动作实际返回或明确报告的新结果。"
    "禁止使用或猜测 groundtruth / 官方真值坐标。"
    "Observation 只能复现讲解材料中明确展示、报告的真实工具执行结果；"
    "直接看输入图得到的画面事实应写入 reasoning，不得为了组成 tool_call 而包装成外部工具回执。"
    "只有材料明确说明某个外部动作已经执行并直接报告其返回内容时，才能生成对应 Observation；"
    "必须忠实还原讲解者实际做过的动作，不能为了让推理链显得完整而补做反向搜图、Web检索、"
    "卫星核验或数据库查询，也不能在已经得到结论后自行增加一轮验证工具。"
    "材料中的「视频换了一个视角」「接着画面出现」「同一段视频继续拍到」只是既有材料的镜头切换，"
    "不代表执行了 reverse_image_search、video_frame_extract 或外部媒体检索；"
    "除非材料明确说执行了检索/打开/提取并报告结果，否则应写为 reasoning，或在输入确实含多图时直接综合观察。"
    "检索 query 与 inputs 只能使用该步骤之前已经掌握的条件，禁止把尚未推出的最终地名塞进查询词再反向确认。"
    "计划查询、查询条件、待验证假设、候选值和常识推断都不是工具结果。"
    "必须保持证据的原始边界：不得把不同题目、不同镜头、不同时间段、不同辅助线或不同轮次搜索的"
    "动作与结果拼接成同一个 Observation；材料后续纠正前文时，以最终纠正结果为准。"
    "Observation 中的每个原子结论都必须有明确依据，不得从已报告属性扩展出未报告属性，"
    "也不得由某一天或少数样本外推出长期频率、稳定规律或更强结论。"
    "不得把常识推测包装成已执行工具的返回，不得自行补写材料中没有的坐标、距离、角度、像素差、"
    "日期时间、编号、候选数量、置信度百分比等精细数据。"
    "材料只支持定性判断时必须保持定性；材料明确表示尚待核验时，Observation 也必须保留未核验状态。"
    "最后一步必须是 event_type=final 且只能使用 tool=final_answer，params 必须且只能包含 location；"
    "单地点写字符串；若视频包含多道独立定位题，location 使用字符串数组并按讲解顺序列出全部最终地点；"
    "地点名称须忠实保留讲解最终结论，不得换成 result/site/answer 等字段；"
    "最后一步 observation 必须为 null。"
    "输出前请在内部静默复查每个 tool_call：若无法指出材料中明确执行的动作和直接返回结果，"
    "就把该内容改为 reasoning 或删除，不要输出额外的审查说明。"
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
    "上一轮 Observation 纠错提醒",
)


class _LLMFreeFormStep(FreeFormStep):
    """LLM 输出事件（与自由事件软信封对齐）。"""


class _LLMFreeFormResult(BaseModel):
    """LLM 整条自由链。"""

    steps: list[_LLMFreeFormStep] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_final_answer(self) -> _LLMFreeFormResult:
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


def _format_tool_contract() -> str:
    """加载生产 Tool v2；目录不可用时失败开放为最小合同。"""

    path = Path(get_settings().TOOL_CATALOG_PATH)
    try:
        forest = load_forest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("stage2 tool catalog unavailable %s: %s", path, exc)
        return (
            "Tool 合同：params.operation=具体操作，params.purpose=证据缺口，"
            "params.inputs=真实执行输入；缺少输入时不得编造。"
        )
    return render_tool_contract_guidance(forest)


def _format_task_block(task: GeoTaskSpec | None) -> str:
    """Provide split-task identity without exposing Stage 1.5 final_location_text."""

    if task is None:
        return ""
    brief = task.visual_evidence_brief.strip() or "（暂无视觉简报，以所附图片为准）"
    return (
        "当前只蒸馏这一道已拆分定位题；即使粗粒度字幕片段同时提到其他镜头/题目，"
        "也必须忽略与下列视觉目标不一致的内容，final_answer 只能回答本题：\n"
        f"task_id={task.task_id}\n"
        f"task_time=[{task.time_start:.1f}, {task.time_end:.1f})\n"
        f"target_kind={task.target_kind.value}\n"
        f"visual_evidence_brief={brief}\n"
    )


def _format_tool_interval_hint(task: GeoTaskSpec | None) -> str:
    """Compact soft prior: only role=tool windows from Stage 1.5 process_intervals."""

    if task is None:
        return ""
    windows = [
        (float(item.start), float(item.end))
        for item in (task.process_intervals or [])
        if getattr(item.role, "value", item.role) == "tool"
    ]
    if not windows:
        return ""
    lines = [
        f"[{start:.1f}, {end:.1f})"
        for start, end in sorted(windows, key=lambda pair: pair[0])
    ]
    return (
        "过程工具时段软先验（仅对照字幕判断是否写 tool_call；非配额、不附工具画面、"
        "不含 show_source/reveal，禁止写入产物）：\n"
        + "；".join(lines)
        + "\n"
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


def run_stage2(
    video_path: str,
    transcript: list[TranscriptSegment],
    *,
    out_path: str | None = None,
    image_path: str | None = None,
    image_paths: list[str] | None = None,
    source_video: str | None = None,
    task: GeoTaskSpec | None = None,
    max_attempts: int | None = None,
    observation_reviewer: ObservationReviewer | None = None,
    action_coverage_reviewer: ActionCoverageReviewer | None = None,
    observation_context: list[TranscriptSegment] | None = None,
) -> FreeFormTrajectory:
    """蒸馏为地理图片定位 agent 自由 TAO（内容优先，无统一 tool schema）。

    Args:
        video_path: 视频路径。
        transcript: 字幕（仅作内部蒸馏材料，不得写入产物元话语）。
        out_path: 可选落盘路径；默认 intermediate/{id}/stage2_freeform_tao.json。
        image_path: 兼容单图；与 ``image_paths`` 二选一优先后者。
        image_paths: 任务关键帧（可多图）；编排器应传入审核切分结果。
        source_video: 写入产物的来源 id；默认视频 stem。
        max_attempts: 可选 Stage 2 总生成尝试上限（含首次，最多3次）；每轮结构化调用不再内部重试。
        task: 可选 Stage 1.5 单题边界；只注入不含最终答案的安全字段。
        observation_reviewer: 可注入批量 Observation 审核回调（测试用）。
        action_coverage_reviewer: 可注入动作覆盖审核回调（测试用）。
        observation_context: 全字幕；仅取本题相邻段帮助审核，不注入相邻题答案到生成提示。

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
        f"{_format_task_block(task)}"
        f"{_format_tool_interval_hint(task)}"
        f"{_format_tool_contract()}\n\n"
        "讲解内容参考（仅供蒸馏，禁止写入产物）：\n"
        f"{_format_transcript(transcript)}\n\n"
        "请输出 steps，每条显式包含 event_type。reasoning 事件只写 thought；"
        "tool_call 事件才写真实 tool、params 与 observation。notes 默认 null。\n"
        "三种事件的结构示例："
        '{"event_type":"reasoning","thought":"根据已有日落时间可排除较早天黑的候选",'
        '"tool":null,"params":{},"observation":null}；'
        '{"event_type":"tool_call","thought":"仍缺少候选区的连续云量证据，因此查询历史天气档案",'
        '"tool":"weather_archive_query","params":{"operation":"cloud_cover",'
        '"purpose":"核验候选区在目标时段的连续云量",'
        '"inputs":{"area":"候选区","time_range":"材料明确给出的目标时段"}},'
        '"observation":{"result":"明确查询结果"}}；'
        "无论前面有多少步，最后一步必须严格写成："
        '{"event_type":"final","thought":"基于已有证据提交最终地点","tool":"final_answer",'
        '"params":{"location":"最终地点"},"observation":null}。'
        "若视频包含多道独立定位题，location 使用字符串数组并按讲解顺序列出全部最终地点。"
    )
    dest = (
        Path(out_path)
        if out_path
        else (Path(settings.INTERMEDIATE_DIR) / video_id / "stage2_freeform_tao.json")
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    limit = min(3, max(1, int(max_attempts if max_attempts is not None else settings.STAGE2_MAX_GENERATIONS)))
    trace: dict[str, Any] = {
        "policy": "bounded_observation_regeneration_v1",
        "source_video": video_id,
        "generation_limit": limit,
        "generation_count": 0,
        "generation_image_paths": images,
        "passes": [],
        "attempts": [],
        "accepted": False,
        "continue_downstream": True,
    }
    warning = ""
    traj: FreeFormTrajectory | None = None
    selected_pass: dict[str, Any] | None = None
    last_error: Exception | None = None
    for generation in range(1, limit + 1):
        trace["generation_count"] = generation
        try:
            raw = call_structured(
                prompt + warning,
                _LLMFreeFormResult,
                images=images or None,
                lane="llm",
                max_attempts=1,
            )
            result = _LLMFreeFormResult.model_validate(
                {
                    "steps": json.loads(_steps_blob(raw.steps)),
                    "notes": raw.notes,
                }
            )
            traj = FreeFormTrajectory(
                source_video=video_id,
                steps=[
                    FreeFormStep.model_validate(step.model_dump())
                    for step in result.steps
                ],
                notes=result.notes,
                working_scope=working_scope,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            trace["attempts"].append(
                {
                    "generation": generation,
                    "status": "generation_failed",
                    "error_type": type(exc).__name__,
                }
            )
            warning += (
                "\n（上一轮结构化生成未成功，请严格返回完整事件轨迹及末步"
                "final_answer.location；不要把本提醒写入产物。）\n"
            )
            trace["stop_reason"] = "generation_failed_using_last_valid"
            continue

        # Save a valid candidate before review; neither review errors nor a third
        # unresolved attempt may discard the usable Stage 2 artifact.
        dest.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
        trace["attempts"].append(
            {
                "generation": generation,
                "status": "generated",
                "trajectory": traj.model_dump(mode="json"),
            }
        )
        style_warning = trajectory_has_meta_leak(traj.steps)
        selected_pass = {
            "generation": generation,
            "items": [],
            "action_coverage": {"status": "not_run", "items": [], "missed": []},
            "status": "not_run",
            "style_warning": style_warning,
        }
        trace["selected_generation"] = generation
        failures = []
        review_items = []
        missed = []
        has_calls = any(step.event_type == "tool_call" for step in traj.steps)
        observation_status = "complete"
        if not has_calls:
            observation_status = "complete"
        elif not settings.STAGE2_OBSERVATION_REVIEW:
            observation_status = "disabled"
        elif not settings.ALLOW_REAL_API and observation_reviewer is None:
            observation_status = "api_not_allowed"
        else:
            try:
                review = review_observations(
                    traj,
                    transcript=transcript,
                    images=images,
                    task=task,
                    context_transcript=observation_context,
                    reviewer=observation_reviewer,
                )
                review_items = review.items
                selected_pass["items"] = [
                    {
                        **item.model_dump(),
                        "call_id": f"step_{item.step_index}",
                        "observation_sha256": observation_fingerprint(
                            traj.steps[item.step_index - 1].observation
                        ),
                    }
                    for item in review.items
                ]
                observation_status = "complete"
                failures = [
                    item for item in review.items if item.verdict == "fabricated"
                ]
            except Exception as exc:  # noqa: BLE001
                observation_status = "audit_failed"
                selected_pass["error_type"] = type(exc).__name__
                logger.warning(
                    "stage2 observation review unavailable: %s", type(exc).__name__
                )

        action_status = "not_run"
        action_payload: dict[str, Any] = {
            "status": "not_run",
            "items": [],
            "missed": [],
        }
        if observation_status == "audit_failed":
            action_status = "skipped_after_observation_failure"
        elif not settings.STAGE2_ACTION_COVERAGE_REVIEW:
            action_status = "disabled"
        elif not settings.ALLOW_REAL_API and action_coverage_reviewer is None:
            action_status = "api_not_allowed"
        else:
            try:
                coverage = review_action_coverage(
                    traj,
                    transcript=transcript,
                    task=task,
                    context_transcript=observation_context,
                    reviewer=action_coverage_reviewer,
                )
                missed = missed_actions(coverage)
                action_payload = {
                    "status": "complete",
                    "items": [item.model_dump() for item in coverage.items],
                    "missed": [item.model_dump() for item in missed],
                }
                action_status = "complete"
            except Exception as exc:  # noqa: BLE001
                action_status = "audit_failed"
                action_payload = {
                    "status": "audit_failed",
                    "items": [],
                    "missed": [],
                    "error_type": type(exc).__name__,
                }
                logger.warning(
                    "stage2 action coverage review unavailable: %s",
                    type(exc).__name__,
                )

        selected_pass["action_coverage"] = action_payload
        if observation_status == "audit_failed" or action_status == "audit_failed":
            selected_pass["status"] = "audit_failed"
        elif observation_status in {"disabled", "api_not_allowed"} and action_status in {
            "disabled",
            "api_not_allowed",
            "not_run",
            "skipped_after_observation_failure",
        }:
            selected_pass["status"] = observation_status
        else:
            selected_pass["status"] = "complete"

        trace["passes"].append(selected_pass)
        observation_ok = (
            observation_status in {"complete", "disabled", "api_not_allowed"}
            and all(item.verdict == "supported" for item in review_items)
        )
        action_ok = action_status in {
            "complete",
            "disabled",
            "api_not_allowed",
            "not_run",
            "skipped_after_observation_failure",
        } and not missed
        trace["accepted"] = (
            selected_pass["status"] == "complete"
            and observation_ok
            and action_ok
            and not style_warning
        )
        trace["stop_reason"] = (
            "audit_failed"
            if selected_pass["status"] == "audit_failed"
            else "passed"
            if trace["accepted"]
            else "uncertain_or_unreviewed"
        )
        if selected_pass["status"] == "audit_failed":
            break
        if failures or missed or style_warning:
            trace["stop_reason"] = (
                "generation_limit" if generation == limit else "retry_requested"
            )
            warning = retry_warning(failures, style_warning)
            warning += action_coverage_retry_warning(missed)
            continue
        break

    if traj is None:
        assert last_error is not None
        raise last_error
    last_missed = bool(
        selected_pass
        and selected_pass.get("action_coverage", {}).get("missed")
    )
    trace["continued_with_issues"] = (
        not trace["accepted"]
        or bool(selected_pass and selected_pass["style_warning"])
        or last_missed
    )
    dest.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    dest.with_name("stage2_observation_audit.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return traj


def load_freeform(path: str | Path) -> FreeFormTrajectory:
    """从落盘 JSON 加载自由链。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FreeFormTrajectory.model_validate(data)
