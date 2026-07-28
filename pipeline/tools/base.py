"""Tool Observation 合成分发器：权限、校验、LLM 合成与 diskcache。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import diskcache
from pydantic import BaseModel, Field, ValidationError

from pipeline.config import get_settings
from pipeline.evidence_routing import (
    ContentRegion,
    ContentType,
    EvidenceIntent,
    heuristic_content_region,
)
from pipeline.image_utils import crop_image_by_bbox, expand_bbox_xywh
from pipeline.llm import call_structured
from pipeline.schemas import (
    Action,
    AgentRole,
    ObservationExecutionResult,
    ObservationSource,
    ToolDefinition,
)
from pipeline.tools.registry import load_registry
from pipeline.tools.validation import validate_action_params, validate_observation

PROMPT_VERSION = "obs_synth_v10_video_step_result"

# VisualObs：依赖关键帧；bbox 为注意力提示（含单图地形/标注类）
_VISUAL_OBS_TOOLS: frozenset[str] = frozenset(
    {
        "zoom_inspect",
        "ocr",
        "sun_position_calc",
        "annotate_geographic_environment_on_image",
        "detect_terrain_features",
        "analyze_terrain_ambiguity",
        "analyze_terrain_visual_illusion",
    }
)
# RetrievalObs：纯文本按本步旁白+Action 填 schema，不传图
# （含 COARSE 开放的卫星/历史地图/双图比对结果摘要，避免无效图 URL）
_RETRIEVAL_OBS_TOOLS: frozenset[str] = frozenset(
    {
        "web_search",
        "map_query",
        "reverse_image_search",
        "find_specific_features_in_satellite_map",
        "lookup_historical_satellite_map",
        "lookup_historical_map_layout",
        "compare_images_for_geolocation",
    }
)
_BBOX_EXPAND_MARGIN = 0.08
_TEXT_ONLY_IMAGE_MARKER = "text_only"


class _ObservationGroundingCheck(BaseModel):
    """逐视频来源蕴含 + 相对既有声明的增量检查。"""

    fully_entailed_by_source_claims: bool
    unsupported_spans: list[str] = Field(default_factory=list)
    target_visibility_consistent: bool
    adds_incremental_information: bool = True
    reason: str = ""

# 仅对 VisualObs 做 H9 门禁（检索摘要常合法提及平台名）
_FRAME_OVERLAY_TOOLS: frozenset[str] = _VISUAL_OBS_TOOLS

# 视频 overlay / 元信息类别（通用结构识别，与地理事实判定无关）
_OVERLAY_CATEGORY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"水印|片头|标题卡|烧录字幕|进度条|频道(?:名|标识|水印|logo)|up主|创作者标签",
        re.I,
    ),
    re.compile(r"\b(?:watermark|title\s*card|burn(?:ed)?[- ]?in|channel\s*logo)\b", re.I),
    # 平台名须与 UI/水印共现，避免检索摘要误杀
    re.compile(
        r"(?:bilibili|youtube|tiktok|抖音|小红书|instagram).{0,12}"
        r"(?:水印|logo|角标|ui|界面|标题卡|片头)"
        r"|(?:水印|logo|角标|ui|界面|标题卡|片头).{0,12}"
        r"(?:bilibili|youtube|tiktok|抖音|小红书|instagram)",
        re.I,
    ),
    re.compile(r"难度\s*\d+\s*★|\d+\s*star(?:s)?\b|难度角标", re.I),
    re.compile(r"平台\s*(?:logo|ui|水印)", re.I),
)

_COORD_RE = re.compile(
    r"(?:[-+]?\d{1,3}(?:\.\d+)?\s*[,，]\s*[-+]?\d{1,3}(?:\.\d+)?)"
    r"|(?:北纬|南纬|东经|西经)\s*[-+]?\d+(?:\.\d+)?"
    r"|(?:lat(?:itude)?|lng|lon(?:gitude)?)\s*[:=]?\s*[-+]?\d+(?:\.\d+)?",
    re.IGNORECASE,
)
def sanitize_narration_for_obs(agent_role: AgentRole, narration: str) -> str:
    """按角色消毒旁白，降低地名经 Observation 反哺 Thought 的风险。

    COARSE：不传自由旁白，只由 EvidenceIntent.source_claims 提供来源事实。
    FINE / VERIFIER：仅剥离坐标表述（允许 params 驱动的地名查询上下文）。
    """
    text = (narration or "").strip()
    if not text:
        return ""
    if agent_role == AgentRole.COARSE:
        return ""
    text = _COORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" ，,。.;；")
    return text


def observation_contains_video_overlay(observation: dict[str, Any]) -> list[str]:
    """通用启发式：Observation 是否含视频 overlay/元信息类别。

    返回命中的类别说明列表；空列表表示未检出。
    """
    if not observation:
        return []
    blob = json.dumps(observation, ensure_ascii=False)
    hits: list[str] = []
    for pat in _OVERLAY_CATEGORY_RES:
        if pat.search(blob):
            hits.append(pat.pattern)
    return hits


def _resolve_synth_image(
    tool: ToolDefinition,
    params: dict[str, Any],
    image_path: str,
    *,
    evidence_intent: Optional[EvidenceIntent] = None,
    content_region: Optional[ContentRegion] = None,
) -> str:
    """内容区裁剪 + zoom_inspect/ocr 相对 bbox；interface_only 返回原图由上层处理。"""
    region = content_region or heuristic_content_region(intent=evidence_intent)
    settings = get_settings()
    cache_dir = str(Path(settings.CACHE_DIR))

    # 先裁内容区（老照片/地图画布）
    base = image_path
    if region.content_type is not ContentType.INTERFACE_ONLY:
        base = crop_image_by_bbox(
            image_path, list(region.content_bbox), cache_dir=cache_dir
        )

    if tool.name not in ("zoom_inspect", "ocr"):
        return base
    bbox = params.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return base
    # bbox=注意力提示：外扩后再裁，降低 stage3 框偏导致的机械 empty
    expanded = expand_bbox_xywh(
        [float(x) for x in bbox], margin=_BBOX_EXPAND_MARGIN
    )
    return crop_image_by_bbox(base, expanded, cache_dir=cache_dir)


def _tool_schema_hash(tool: ToolDefinition) -> str:
    payload = tool.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _params_hash(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _narration_hash(narration: str) -> str:
    return hashlib.sha256(narration.encode("utf-8")).hexdigest()[:16]


def _image_content_hash(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        return hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:16]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_key(
    *,
    tool: ToolDefinition,
    params: dict[str, Any],
    image_path: str,
    narration: str,
    model_name: str,
    prompt_version: str,
) -> str:
    parts = [
        tool.name,
        _tool_schema_hash(tool),
        _params_hash(params),
        _image_content_hash(image_path),
        _narration_hash(narration),
        model_name,
        prompt_version,
    ]
    return "|".join(parts)


def _get_cache() -> diskcache.Cache:
    settings = get_settings()
    Path(settings.CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(Path(settings.CACHE_DIR) / "tool_executions"))


def _check_agent_permission(tool: ToolDefinition, agent_role: AgentRole) -> None:
    if agent_role not in tool.allowed_agents:
        raise PermissionError(
            f"角色 {agent_role.value} 无权使用 tool {tool.name}；"
            f"允许: {[r.value for r in tool.allowed_agents]}"
        )


def _build_observation_model(tool: ToolDefinition) -> type[BaseModel]:
    """按 observation_fields 动态构造 Pydantic 模型（H1）。"""
    fields: dict[str, Any] = {}
    annotations: dict[str, Any] = {}
    for obs in tool.observation_fields:
        if obs.type == "string":
            py_type: Any = str
        elif obs.type == "float":
            py_type = float
        elif obs.type == "int":
            py_type = int
        elif obs.type == "bool":
            py_type = bool
        elif obs.type == "string_list":
            py_type = list[str]
        elif obs.type in ("latlng", "lat_range", "bbox"):
            py_type = list[float]
        elif obs.type == "result_list":
            py_type = list[dict[str, Any]]
        else:
            py_type = Any
        if obs.nullable:
            annotations[obs.name] = Optional[py_type]
            fields[obs.name] = Field(default=None)
        else:
            annotations[obs.name] = py_type
            fields[obs.name] = Field()
    namespace = {
        "__annotations__": annotations,
        **fields,
    }
    return type(f"SynthObs_{tool.name}", (BaseModel,), namespace)


def _is_retrieval_tool(tool_name: str) -> bool:
    return tool_name in _RETRIEVAL_OBS_TOOLS


def _is_visual_tool(tool_name: str) -> bool:
    return tool_name in _VISUAL_OBS_TOOLS


def _field_lines(tool: ToolDefinition) -> list[str]:
    lines: list[str] = []
    for f in tool.observation_fields:
        lines.append(
            f"- {f.name} ({f.type}, nullable={f.nullable}): {f.description}"
        )
    return lines


def _retry_feedback_block(retry_feedback: str) -> str:
    if not retry_feedback.strip():
        return ""
    return (
        "Previous Observation was rejected. Fix these issues and regenerate:\n"
        f"{retry_feedback.strip()}\n"
    )


def _synthesize_visual_observation(
    tool: ToolDefinition,
    params: dict[str, Any],
    image_path: str,
    narration: str,
    agent_role: AgentRole,
    *,
    retry_feedback: str = "",
    evidence_intent: Optional[EvidenceIntent] = None,
) -> dict[str, Any]:
    """VisualObs：关键帧 + bbox 提示 + 可见确认优先。"""
    model_cls = _build_observation_model(tool)
    role_hint = (
        "This is a COARSE (broad localization) step: prefer ranges/visual features; "
        "do not invent city/POI names absent from source_claims and the image.\n"
        if agent_role == AgentRole.COARSE
        else ""
    )
    intent_hint = ""
    if evidence_intent is not None:
        intent_hint = (
            f"- EvidenceIntent target: {evidence_intent.target_object}; "
            f"features={evidence_intent.target_features}; "
            f"relation={evidence_intent.expected_spatial_relation!r}; "
            f"subject_scope={evidence_intent.subject_scope.value}; "
            f"spatial_anchor={evidence_intent.spatial_anchor!r}.\n"
            f"- Per-video source claims={evidence_intent.source_claims}; "
            f"source concepts={evidence_intent.source_concepts}.\n"
            "- PRIMARY GOAL: usable Observation of what this video step shows. "
            "Prefer status=success with a short concrete description of visible "
            "cues about the EvidenceIntent target.\n"
            "- bbox in Params is an ATTENTION HINT (may be slightly off); "
            "the crop is already expanded. Describe nearby in-scene geography "
            "if the target or supporting scene is visible — do NOT return empty "
            "for a ~5–10% box misalignment.\n"
            "- A short visibility confirmation of a source_claims target IS "
            "valuable training signal.\n"
            "- Do not invent new place names / POIs absent from source_claims "
            "AND the image. Visible appearance of an already-claimed target "
            "and readable in-scene signage text are allowed.\n"
            "- Use status=empty ONLY when: (a) pure UI / black / heavy blur with "
            "no in-scene geography, OR (b) the stated target is clearly absent.\n"
            "- Keep description to 1–3 short visual sentences.\n"
        )
    else:
        intent_hint = (
            "- bbox in Params is an ATTENTION HINT; crop may be expanded. "
            "Prefer success when in-scene geography is visible near the hint.\n"
            "- empty ONLY for pure UI/black/blur or clearly absent targets.\n"
        )
    prompt = (
        "You synthesize a VisualObs tool Observation from the image and action "
        "params (plus optional sanitized narration).\n"
        "Goal: report what this video step's visual action reveals, in schema form.\n"
        "Rules:\n"
        "- Fill EVERY field listed; do not add extra fields.\n"
        "- If nullable and info is absent, use null; do not guess.\n"
        "- For result_list use 2~5 items matching item_fields when status=success.\n"
        "- Never use placeholder strings like 未知/不确定/N/A.\n"
        "- status must be success|empty|error; error_message only when status=error.\n"
        "- Do NOT use any groundtruth.\n"
        "- Style as a realistic API response for this tool.\n"
        "- H9: NEVER include video production overlays: intro/title cards, "
        "platform/channel watermarks or logos, difficulty/star badges, "
        "progress bars, burned-in subtitles, creator tags, chat bubbles, "
        "or non-scene UI. Describe only in-scene geography/architecture/"
        "nature/real signage. For OCR, ignore corner watermarks and top title "
        "bars; if only overlay text is visible, return empty texts / status=empty.\n"
        "- Tool modality: write as this Tool's visual result, NOT as a separate "
        "satellite-search / map-query / web API call. If the frame shows a map or "
        "satellite view, describe visible map/satellite-view features in-scene "
        "(e.g. plains, rivers, labels on the image); do NOT claim you 'opened a "
        "satellite API' or 'queried remote sensing service'. For zoom_inspect/"
        "ocr/sun_position_calc, never invent independent map-service metadata.\n"
        f"{intent_hint}"
        f"{role_hint}"
        f"{_retry_feedback_block(retry_feedback)}"
        f"AgentRole: {agent_role.value}\n"
        f"Tool: {tool.name}\nDescription: {tool.description}\n"
        f"Params: {json.dumps(params, ensure_ascii=False)}\n"
        f"Narration: {narration or '(empty)'}\n"
        f"Fields:\n" + "\n".join(_field_lines(tool))
    )
    result = call_structured(prompt, model_cls, images=[image_path])
    return result.model_dump(mode="json")


def _synthesize_retrieval_observation(
    tool: ToolDefinition,
    params: dict[str, Any],
    narration: str,
    agent_role: AgentRole,
    *,
    retry_feedback: str = "",
) -> dict[str, Any]:
    """RetrievalObs：无图；按本步旁白+Action 写 API 形结果，放行顺推中间地名。"""
    model_cls = _build_observation_model(tool)
    prompt = (
        "You synthesize a RetrievalObs tool Observation from Action params and "
        "this step's video narration only (NO image).\n"
        "Goal: report what this video step's search/map/match action found, "
        "as a realistic API response in the given schema.\n"
        "Rules:\n"
        "- Fill EVERY field listed; do not add extra fields.\n"
        "- If nullable and info is absent from narration/params, use null.\n"
        "- For result_list use 2~5 items matching item_fields when status=success.\n"
        "- Never use placeholder strings like 未知/不确定/N/A.\n"
        "- status must be success|empty|error; error_message only when status=error.\n"
        "- Do NOT use any groundtruth or answer revealed only at video end.\n"
        "- Place names that appear in THIS step's Narration and/or Action params "
        "(e.g. query) MAY be written into Observation fields when they are "
        "intermediate results of this step (顺推). "
        "Do NOT invent final answer-level place names absent from Narration "
        "and Action params.\n"
        "- Prefer status=success when the narration indicates useful hits; "
        "empty only when the step clearly found nothing.\n"
        f"{_retry_feedback_block(retry_feedback)}"
        f"AgentRole: {agent_role.value}\n"
        f"Tool: {tool.name}\nDescription: {tool.description}\n"
        f"Params: {json.dumps(params, ensure_ascii=False)}\n"
        f"Narration: {narration or '(empty)'}\n"
        f"Fields:\n" + "\n".join(_field_lines(tool))
    )
    result = call_structured(prompt, model_cls, images=None)
    return result.model_dump(mode="json")


def _synthesize_observation(
    tool: ToolDefinition,
    params: dict[str, Any],
    image_path: str,
    narration: str,
    agent_role: AgentRole,
    *,
    retry_feedback: str = "",
    evidence_intent: Optional[EvidenceIntent] = None,
) -> dict[str, Any]:
    """按 Tool 族分发 Visual / Retrieval 合成。"""
    if _is_retrieval_tool(tool.name):
        return _synthesize_retrieval_observation(
            tool,
            params,
            narration,
            agent_role,
            retry_feedback=retry_feedback,
        )
    return _synthesize_visual_observation(
        tool,
        params,
        image_path,
        narration,
        agent_role,
        retry_feedback=retry_feedback,
        evidence_intent=evidence_intent,
    )


def _check_observation_grounding(
    observation: dict[str, Any],
    evidence_intent: EvidenceIntent,
    image_path: str,
) -> _ObservationGroundingCheck:
    """检查 Obs 的每个事实是否被该视频引用声明蕴含。"""
    prompt = (
        "审查一个 COARSE Observation 是否受当前视频来源声明约束，"
        "同时允许对已声明目标做可见确认。不要使用任何预设地理词表。\n"
        "规则：\n"
        "1. 不得引入 source_claims 未提及的新地名、新 POI、新候选地点；"
        "同义改写允许。\n"
        "2. 图片用于确认 source_claims 指定目标是否可见，并可描述该目标上"
        "可见的外观/空间线索（形状、相对位置、材质色调、地貌类型等）；"
        "这些可见确认不算「发明新事实」。目标完全不可见时 status=empty 才一致。\n"
        "3. 若出现 source_claims 未提及的新地名/新候选/新设施名，"
        "fully_entailed=false，并列入 unsupported_spans。\n"
        "4. adds_incremental_information=true 当 Observation 给出了相对 "
        "source_claims 的可见确认、否定或空间线索；短确认句算增量。"
        "仅当整段 Observation 是对 source_claims 的长篇同义复述且无可视"
        "确认时才 false（此时仍可 fully_entailed=true）。\n"
        f"source_fact_ids={evidence_intent.video_fact_ids}\n"
        f"source_claims={evidence_intent.source_claims}\n"
        f"source_concepts={evidence_intent.source_concepts}\n"
        f"observation={json.dumps(observation, ensure_ascii=False)}"
    )
    return call_structured(
        prompt,
        _ObservationGroundingCheck,
        images=[image_path],
    )


def execute_action(
    action: Action,
    image_path: str,
    agent_role: AgentRole,
    *,
    narration: str = "",
    registry_path: Optional[str] = None,
    use_cache: bool = True,
    evidence_intent: Optional[EvidenceIntent] = None,
    content_region: Optional[ContentRegion] = None,
) -> ObservationExecutionResult:
    """分发器：权限 → params 校验 → terminal skip / LLM 合成 → observation 校验 → 缓存。"""
    settings = get_settings()
    registry = load_registry(registry_path)
    if action.tool not in registry:
        return ObservationExecutionResult(
            action=action,
            observation=None,
            source=None,
            status="error",
            error_message=f"未知 tool: {action.tool}",
            cache_hit=False,
        )
    tool = registry[action.tool]

    try:
        _check_agent_permission(tool, agent_role)
        params = validate_action_params(tool, action.params, agent_role=agent_role)
    except (PermissionError, ValidationError, ValueError) as exc:
        return ObservationExecutionResult(
            action=action,
            observation=None,
            source=None,
            status="error",
            error_message=str(exc),
            cache_hit=False,
        )

    normalized_action = Action(tool=action.tool, params=params)

    if tool.is_terminal:
        return ObservationExecutionResult(
            action=normalized_action,
            observation=None,
            source=None,
            status="skipped",
            error_message=None,
            cache_hit=False,
        )

    narr = sanitize_narration_for_obs(agent_role, narration or "")
    region = content_region or heuristic_content_region(intent=evidence_intent)
    retrieval = _is_retrieval_tool(tool.name)
    if retrieval:
        synth_image = _TEXT_ONLY_IMAGE_MARKER
    else:
        synth_image = _resolve_synth_image(
            tool,
            params,
            image_path,
            evidence_intent=evidence_intent,
            content_region=region,
        )
    model_name = settings.LLM_MODEL
    key = _cache_key(
        tool=tool,
        params=params,
        image_path=synth_image,
        narration=narr,
        model_name=model_name,
        prompt_version=PROMPT_VERSION,
    )

    cache: diskcache.Cache | None = _get_cache() if use_cache else None
    if cache is not None and key in cache:
        cached = cache[key]
        if isinstance(cached, dict):
            return ObservationExecutionResult.model_validate(
                {**cached, "cache_hit": True}
            )

    last_error: str | None = None
    retry_feedback = ""
    result = ObservationExecutionResult(
        action=normalized_action,
        observation=None,
        source=ObservationSource.LLM_SYNTHESIZED,
        status="error",
        error_message="observation synthesis failed",
        cache_hit=False,
    )
    for _ in range(max(1, settings.OBS_SYNTH_MAX_RETRY)):
        try:
            raw_obs = _synthesize_observation(
                tool,
                params,
                synth_image,
                narr,
                agent_role,
                retry_feedback=retry_feedback,
                evidence_intent=evidence_intent,
            )
            overlay_hits = (
                observation_contains_video_overlay(raw_obs)
                if tool.name in _FRAME_OVERLAY_TOOLS
                else []
            )
            if overlay_hits:
                retry_feedback = (
                    "H9 violation: Observation must not contain video overlays "
                    "(title cards, platform watermarks/logos, difficulty badges, "
                    "burned-in subtitles, creator tags, chat UI, non-scene UI). "
                    "Describe only in-scene geography/architecture/nature/signage."
                )
                last_error = "observation contains video overlay"
                result = ObservationExecutionResult(
                    action=normalized_action,
                    observation=None,
                    source=ObservationSource.LLM_SYNTHESIZED,
                    status="error",
                    error_message=last_error,
                    cache_hit=False,
                )
                continue
            obs = validate_observation(tool, raw_obs)
            # 逐视频来源蕴含：不使用全局黑/白词表
            if (
                agent_role == AgentRole.COARSE
                and evidence_intent is not None
                and _is_visual_tool(tool.name)
                and synth_image != _TEXT_ONLY_IMAGE_MARKER
            ):
                grounding = _check_observation_grounding(
                    obs,
                    evidence_intent,
                    synth_image,
                )
                if (
                    not grounding.fully_entailed_by_source_claims
                    or not grounding.target_visibility_consistent
                ):
                    retry_feedback = (
                        "Video-source entailment violation (ungrounded_video_fact): "
                        f"unsupported={grounding.unsupported_spans}; "
                        f"reason={grounding.reason}. "
                        "Rewrite: keep ONLY a short visual confirmation of "
                        "source_concepts / source_claims targets that are visible "
                        "in the image (appearance, relative position, terrain/"
                        "architecture cues). Remove new place names / new POIs / "
                        "new facilities not in source_claims. "
                        "If none of the claimed targets are visible, status=empty; "
                        "otherwise prefer status=success."
                    )
                    last_error = (
                        "ungrounded_video_fact: "
                        f"{grounding.unsupported_spans or [grounding.reason]}"
                    )
                    result = ObservationExecutionResult(
                        action=normalized_action,
                        observation=None,
                        source=ObservationSource.LLM_SYNTHESIZED,
                        status="error",
                        error_message=last_error,
                        cache_hit=False,
                    )
                    continue
                # adds_incremental_information=false 时仍接受：来源蕴含且目标可见的
                # 确认句是可用训练信号；禁止再降 empty（会掏空 COARSE 证据链）。
            status = str((obs or {}).get("status", "success"))
            if status not in ("success", "empty", "error"):
                status = "success"
            result = ObservationExecutionResult(
                action=normalized_action,
                observation=obs,
                source=ObservationSource.LLM_SYNTHESIZED,
                status=status,  # type: ignore[arg-type]
                error_message=(obs or {}).get("error_message") if obs else None,
                cache_hit=False,
            )
            last_error = None
            break
        except (ValidationError, Exception) as exc:  # noqa: BLE001
            last_error = str(exc)
            retry_feedback = f"Schema/validation error: {last_error}"
            result = ObservationExecutionResult(
                action=normalized_action,
                observation=None,
                source=ObservationSource.LLM_SYNTHESIZED,
                status="error",
                error_message=last_error,
                cache_hit=False,
            )

    # H9/overlay 或闭包违规耗尽：返回 schema 合法 empty
    if (
        result.status == "error"
        and last_error
        and (
            "overlay" in last_error.lower()
            or "ungrounded_video_fact" in last_error.lower()
        )
        and tool.name in _FRAME_OVERLAY_TOOLS
    ):
        empty_obs = _empty_observation_payload(tool.name)
        result = ObservationExecutionResult(
            action=normalized_action,
            observation=empty_obs,
            source=ObservationSource.LLM_SYNTHESIZED,
            status="empty",
            error_message=last_error,
            cache_hit=False,
        )

    if cache is not None and result.status in ("success", "empty", "skipped"):
        cache[key] = result.model_dump(mode="json")
    return result


def _empty_observation_payload(tool_name: str) -> dict[str, Any]:
    """overlay/不可见目标时的 schema 友好 empty Observation。"""
    if tool_name == "ocr":
        return {"status": "empty", "error_message": None, "texts": []}
    if tool_name == "sun_position_calc":
        return {
            "status": "empty",
            "error_message": None,
            "possible_latitude_range": None,
            "note": None,
        }
    return {
        "status": "empty",
        "error_message": None,
        "description": "no in-scene geography visible in content region",
    }
