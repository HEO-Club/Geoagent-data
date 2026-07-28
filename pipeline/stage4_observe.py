"""stage4：LLM 合成 NormalizedStep 中 Action 的 Observation。

依赖 stage3 产出的 NormalizedStep 与 tools.base.execute_action；
不访问 groundtruth，不生成 Thought（Thought 属于 stage5）。
"""

from __future__ import annotations

from typing import Optional, Sequence

from pipeline.evidence_routing import (
    ContentRegion,
    ContentType,
    EvidenceIntent,
    heuristic_content_region,
    is_meta_setup_narration,
    parse_evidence_intent,
)
from pipeline.image_utils import (
    candidate_keyframes_near_move,
    parse_keyframe_time,
    resolve_keyframe_for_time,
)
from pipeline.schemas import (
    Action,
    AgentRole,
    NormalizedStep,
    ObservationExecutionResult,
    ObservationSource,
)
from pipeline.tools.base import execute_action


class ObservationSynthesisExhausted(RuntimeError):
    """保留兼容；generate_observations 不再抛出（耗尽改降 empty）。"""


def _is_synthesis_exhausted(result: ObservationExecutionResult) -> bool:
    """识别 LLM 合成耗尽（区别于权限/未知 tool 等预检错误）。"""
    return (
        result.status == "error"
        and result.source is ObservationSource.LLM_SYNTHESIZED
        and result.observation is None
    )


def resolve_image_for_step(
    step: NormalizedStep,
    *,
    image_path: str,
    keyframes: Optional[Sequence[str]] = None,
) -> str:
    """按 Move 时间窗 + EvidenceIntent 目标感知选帧；无 keyframes 时回退。"""
    if not keyframes:
        return image_path
    intent = parse_evidence_intent(step.thought_draft)
    candidates = candidate_keyframes_near_move(
        list(keyframes),
        float(step.move.start_time),
        float(step.move.end_time),
    )
    if not candidates:
        chosen = resolve_keyframe_for_time(
            list(keyframes),
            float(step.move.start_time),
            float(step.move.end_time),
            fallback=image_path,
        )
        return chosen or image_path

    # candidates 已按与 Move 时间窗距离排序；优先最近帧。
    # Move 已在中后段时，丢弃过早开场帧，避免「中位候选」回退到 t≈0 UI。
    move_mid = (
        float(step.move.start_time) + float(step.move.end_time)
    ) / 2.0
    if move_mid >= 12.0:
        non_early = [
            path
            for path in candidates
            if (parse_keyframe_time(path) or 0.0) >= 10.0
        ]
        if non_early:
            candidates = non_early
    # 在候选中再选最靠近 Move 中点的一帧（提升主场景命中率）
    best = min(
        candidates,
        key=lambda p: abs((parse_keyframe_time(p) or move_mid) - move_mid),
    )
    return best


def _candidate_images_for_step(
    step: NormalizedStep,
    *,
    image_path: str,
    keyframes: Optional[Sequence[str]] = None,
    max_frames: int = 3,
) -> list[str]:
    """为画面类 Tool 准备最多 max_frames 个候选帧（中点优先，再近邻）。"""
    primary = resolve_image_for_step(
        step, image_path=image_path, keyframes=keyframes
    )
    out: list[str] = [primary]
    if not keyframes:
        return out
    near = candidate_keyframes_near_move(
        list(keyframes),
        float(step.move.start_time),
        float(step.move.end_time),
        max_candidates=max(max_frames, 5),
    )
    move_mid = (
        float(step.move.start_time) + float(step.move.end_time)
    ) / 2.0
    if move_mid >= 12.0:
        near = [
            p for p in near if (parse_keyframe_time(p) or 0.0) >= 10.0
        ] or near
    for path in near:
        if path not in out:
            out.append(path)
        if len(out) >= max_frames:
            break
    return out


def _pair_steps_with_observations(
    steps: list[NormalizedStep],
    observations: list[ObservationExecutionResult],
) -> list[tuple[NormalizedStep, Optional[ObservationExecutionResult]]]:
    """按 Action 展开顺序对齐 step 与 Observation。"""
    paired: list[tuple[NormalizedStep, Optional[ObservationExecutionResult]]] = []
    obs_i = 0
    for step in steps:
        if not step.actions:
            paired.append((step, None))
            continue
        for _action in step.actions:
            obs = observations[obs_i] if obs_i < len(observations) else None
            paired.append((step, obs))
            obs_i += 1
    return paired


def _is_usable_geo_obs(obs: Optional[ObservationExecutionResult]) -> bool:
    """success 且非空描述，才算通过内容区门禁的可用地理 Obs。"""
    if obs is None or obs.status != "success" or not obs.observation:
        return False
    blob = str(obs.observation)
    if "no in-scene geography" in blob.lower():
        return False
    return True


def _step_is_early_meta(step: NormalizedStep) -> bool:
    """开场元叙事步：不应决定 Agent1 代表帧。"""
    if float(step.move.start_time) >= 15.0:
        return False
    narr = step.move.narration or ""
    if is_meta_setup_narration(narr):
        return True
    intent = parse_evidence_intent(step.thought_draft)
    if intent is None:
        return float(step.move.start_time) < 8.0 and not (step.move.visible_clues)
    if intent.content_type is ContentType.INTERFACE_ONLY:
        return True
    # 无地理特征且时间很早 → 视为元叙事/片头
    if float(step.move.start_time) < 8.0 and not intent.target_features:
        return True
    return False


def pick_agent1_representative_image(
    observations: list[ObservationExecutionResult],
    steps: list[NormalizedStep],
    *,
    keyframes: Sequence[str],
    fallback: str,
) -> str:
    """优先选择已通过内容区门禁的 primary_scene 代表图，供 stage5 轨迹生成。

    优先级：
    1) 非开场元叙事步 + success 非空 Observation 对应选帧；
    2) 非开场的 primary_scene / supporting_geo_visual 步选帧；
    3) keyframes 中时间靠后的中后段帧（避开 t≈0 开场 UI）。
    """
    frames = list(keyframes)
    paired = _pair_steps_with_observations(steps, observations)

    # 1) 有可用地理 Obs 的步
    for step, obs in paired:
        if _step_is_early_meta(step) or not _is_usable_geo_obs(obs):
            continue
        intent = parse_evidence_intent(step.thought_draft)
        if intent is not None and intent.content_type is ContentType.INTERFACE_ONLY:
            continue
        path = resolve_image_for_step(
            step, image_path=fallback, keyframes=frames
        )
        if path:
            return path

    # 2) 意图指向主场景/地图，且非开场元叙事
    for step, _obs in paired:
        if _step_is_early_meta(step):
            continue
        intent = parse_evidence_intent(step.thought_draft)
        if intent is None:
            continue
        if intent.content_type is ContentType.INTERFACE_ONLY:
            continue
        if intent.content_type in (
            ContentType.PRIMARY_SCENE,
            ContentType.SUPPORTING_GEO_VISUAL,
        ):
            path = resolve_image_for_step(
                step, image_path=fallback, keyframes=frames
            )
            if path:
                return path

    # 3) 中后段 keyframe：优先 t>=15s，否则中间帧
    if frames:
        timed: list[tuple[float, str]] = []
        for path in frames:
            t = parse_keyframe_time(path)
            if t is not None:
                timed.append((t, path))
        late = [p for t, p in timed if t >= 15.0]
        if late:
            return late[min(len(late) // 2, len(late) - 1)]
        if timed:
            timed.sort(key=lambda x: x[0])
            return timed[min(len(timed) - 1, max(1, len(timed) * 2 // 3))][1]
        return frames[min(len(frames) - 1, max(1, len(frames) // 2))]
    return fallback


_VISUAL_TOOLS: frozenset[str] = frozenset(
    {"zoom_inspect", "ocr", "sun_position_calc"}
)


def _execute_visual_with_frame_fallback(
    action: Action,
    *,
    frame_paths: list[str],
    agent_role: AgentRole,
    narration: str,
    registry_path: Optional[str],
    use_cache: bool,
    evidence_intent: Optional[EvidenceIntent],
    content_region: ContentRegion,
) -> ObservationExecutionResult:
    """画面类 Tool：首帧 empty 时换近邻关键帧再合成，提升可用 Obs 召回。"""
    last: Optional[ObservationExecutionResult] = None
    for frame in frame_paths:
        result = execute_action(
            action,
            frame,
            agent_role,
            narration=narration,
            registry_path=registry_path,
            use_cache=use_cache,
            evidence_intent=evidence_intent,
            content_region=content_region,
        )
        last = result
        if result.status == "success":
            return result
        # error（合成耗尽等）也试下一帧；skipped 不应出现在非 terminal
        if result.status not in ("empty", "error"):
            return result
    assert last is not None
    return last


def generate_observations(
    normalized_steps: list[NormalizedStep],
    image_path: str,
    agent_role: AgentRole,
    *,
    keyframes: Optional[Sequence[str]] = None,
    registry_path: Optional[str] = None,
    use_cache: bool = True,
) -> list[ObservationExecutionResult]:
    """展开 normalized_steps 中的全部 Action，逐个 LLM 合成 Observation。

    thought_only 步（actions=[]）不产生 execution result。
    composed 步可产生多个 ObservationExecutionResult。
    每步目标感知选帧 + EvidenceIntent；内容区裁剪在 execute_action 内完成。
    VisualObs 在首帧 empty 时自动换近邻关键帧重试（不加硬门禁，只提升召回）。

    任一非 terminal 合成/schema 耗尽 → status=error（诚实失败载荷），全角色继续（不抛异常）。
    不得伪装成「无场景地理」的 empty。
    """
    results: list[ObservationExecutionResult] = []
    for step in normalized_steps:
        narration = step.move.narration or ""
        intent = parse_evidence_intent(step.thought_draft)
        frame_paths = _candidate_images_for_step(
            step, image_path=image_path, keyframes=keyframes, max_frames=3
        )
        # 内容区门禁：interface_only 且无目标 → 直接 empty（不合成 UI）
        region = heuristic_content_region(
            intent=intent, screen_action=step.move.screen_action
        )
        for action in step.actions:
            if (
                region.content_type is ContentType.INTERFACE_ONLY
                and not region.target_visible
                and action.tool in ("zoom_inspect", "ocr", "sun_position_calc")
            ):
                results.append(
                    ObservationExecutionResult(
                        action=action,
                        observation=_empty_obs_for_tool(action.tool),
                        source=ObservationSource.LLM_SYNTHESIZED,
                        status="empty",
                        error_message=None,
                        cache_hit=False,
                    )
                )
                continue
            if action.tool in _VISUAL_TOOLS and len(frame_paths) > 1:
                results.append(
                    _execute_visual_with_frame_fallback(
                        action,
                        frame_paths=frame_paths,
                        agent_role=agent_role,
                        narration=narration,
                        registry_path=registry_path,
                        use_cache=use_cache,
                        evidence_intent=intent,
                        content_region=region,
                    )
                )
            else:
                results.append(
                    execute_action(
                        action,
                        frame_paths[0],
                        agent_role,
                        narration=narration,
                        registry_path=registry_path,
                        use_cache=use_cache,
                        evidence_intent=intent,
                        content_region=region,
                    )
                )

    # 全角色：合成耗尽标 error（诚实失败），继续流水线
    fixed: list[ObservationExecutionResult] = []
    for r in results:
        if _is_synthesis_exhausted(r):
            err_msg = (
                r.error_message
                or "observation synthesis exhausted"
            )
            fixed.append(
                ObservationExecutionResult(
                    action=r.action,
                    observation=_error_obs_for_tool(r.action.tool, err_msg),
                    source=ObservationSource.LLM_SYNTHESIZED,
                    status="error",
                    error_message=err_msg,
                    cache_hit=False,
                )
            )
        else:
            fixed.append(r)
    return fixed


def _empty_obs_for_tool(tool_name: str) -> dict:
    """interface_only / 真·目标不可见时返回 schema 友好的 empty Observation。"""
    if tool_name == "ocr":
        return {"status": "empty", "error_message": None, "texts": []}
    if tool_name == "sun_position_calc":
        return {
            "status": "empty",
            "error_message": None,
            "possible_latitude_range": None,
            "note": None,
        }
    if tool_name == "web_search":
        return {"status": "empty", "error_message": None, "results": None}
    if tool_name == "map_query":
        return {
            "status": "empty",
            "error_message": None,
            "formatted_address": None,
            "resolved_latlng": None,
            "place_type": None,
        }
    if tool_name == "reverse_image_search":
        return {"status": "empty", "error_message": None, "matches": None}
    if tool_name == "find_specific_features_in_satellite_map":
        return {
            "status": "empty",
            "error_message": None,
            "matched_features": [],
            "overall_match_assessment": "未找到匹配",
        }
    if tool_name == "detect_terrain_features":
        return {
            "status": "empty",
            "error_message": None,
            "detected_features": [],
            "summary": "no in-scene geography visible in content region",
        }
    return {
        "status": "empty",
        "error_message": None,
        "description": "no in-scene geography visible in content region",
    }


def _error_obs_for_tool(tool_name: str, error_message: str) -> dict:
    """合成/schema 耗尽时的最小合法 error Observation（非「无地理」empty）。"""
    msg = (error_message or "").strip() or "observation synthesis failed"
    if tool_name == "ocr":
        return {"status": "error", "error_message": msg, "texts": []}
    if tool_name == "sun_position_calc":
        return {
            "status": "error",
            "error_message": msg,
            "possible_latitude_range": None,
            "note": None,
        }
    if tool_name == "web_search":
        return {"status": "error", "error_message": msg, "results": None}
    if tool_name == "map_query":
        return {
            "status": "error",
            "error_message": msg,
            "formatted_address": None,
            "resolved_latlng": None,
            "place_type": None,
        }
    if tool_name == "reverse_image_search":
        return {"status": "error", "error_message": msg, "matches": None}
    if tool_name == "find_specific_features_in_satellite_map":
        return {
            "status": "error",
            "error_message": msg,
            "matched_features": [],
            "overall_match_assessment": "observation synthesis failed",
        }
    if tool_name == "detect_terrain_features":
        return {
            "status": "error",
            "error_message": msg,
            "detected_features": [],
            "summary": "observation synthesis failed",
        }
    if tool_name == "lookup_historical_satellite_map":
        return {
            "status": "error",
            "error_message": msg,
            "image_url": None,
            "layout_description": "observation synthesis failed",
            "matched_features": [],
        }
    if tool_name == "compare_images_for_geolocation":
        return {
            "status": "error",
            "error_message": msg,
            "visual_similarity_score": None,
            "matched_features": [],
            "mismatched_features": [],
            "geolocation_hints": [],
        }
    return {
        "status": "error",
        "error_message": msg,
        "description": "observation synthesis failed",
    }
