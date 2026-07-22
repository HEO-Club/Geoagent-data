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
from pipeline.image_utils import crop_image_by_bbox
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

PROMPT_VERSION = "obs_synth_v3"

# 仅对画面观察类 Tool 做 H9 门禁（web_search/map_query 结果常合法提及平台名）
_FRAME_OVERLAY_TOOLS: frozenset[str] = frozenset(
    {"zoom_inspect", "ocr", "sun_position_calc", "reverse_image_search"}
)

# 视频 overlay / 元信息类别（通用结构启发式，非单视频标题黑名单）
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

# 中文地名/行政区/道路等后缀短语（COARSE 旁白消毒）
_PLACE_SUFFIX_RE = re.compile(
    r"[\u4e00-\u9fff]{1,12}(?:省|市|州|盟|地区|县|区|旗|镇|乡|村|庄|"
    r"街道|路|街|巷|大道|广场|公园|景区|机场|车站|地铁站|大学|医院)"
)
# 常见英文地名词形（粗粒度）
_LATIN_PLACE_RE = re.compile(
    r"\b(?:in|at|near|around)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"
    r"(?:\s+(?:City|County|Province|Street|Road|Park|Square))?\b"
)
_COORD_RE = re.compile(
    r"(?:[-+]?\d{1,3}(?:\.\d+)?\s*[,，]\s*[-+]?\d{1,3}(?:\.\d+)?)"
    r"|(?:北纬|南纬|东经|西经)\s*[-+]?\d+(?:\.\d+)?"
    r"|(?:lat(?:itude)?|lng|lon(?:gitude)?)\s*[:=]?\s*[-+]?\d+(?:\.\d+)?",
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(
    r"(?:我(?:们)?|咱们|博主)(?:觉得|认为|来到|发现|看到|知道|猜)"
)


def sanitize_narration_for_obs(agent_role: AgentRole, narration: str) -> str:
    """按角色消毒旁白，降低地名经 Observation 反哺 Thought 的风险。

    COARSE：剥离中英地名后缀短语、坐标表述与第一人称叙事残留。
    FINE / VERIFIER：仅剥离坐标表述（允许 params 驱动的地名查询上下文）。
    """
    text = (narration or "").strip()
    if not text:
        return ""
    text = _COORD_RE.sub(" ", text)
    if agent_role == AgentRole.COARSE:
        text = _PLACE_SUFFIX_RE.sub(" ", text)
        text = _LATIN_PLACE_RE.sub(" ", text)
        text = _FIRST_PERSON_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" ，,。.;；")
    return text


def observation_contains_video_overlay(observation: dict[str, Any]) -> list[str]:
    """通用启发式：Observation 是否含视频 overlay/元信息类别。

    返回命中的类别说明列表；空列表表示未检出。不做单视频标题黑名单删除。
    """
    if not observation:
        return []
    blob = json.dumps(observation, ensure_ascii=False)
    hits: list[str] = []
    for pat in _OVERLAY_CATEGORY_RES:
        if pat.search(blob):
            hits.append(pat.pattern)
    return hits


def _resolve_synth_image(tool: ToolDefinition, params: dict[str, Any], image_path: str) -> str:
    """zoom_inspect/ocr 按 bbox 裁图后送模；其余用原图。"""
    if tool.name not in ("zoom_inspect", "ocr"):
        return image_path
    bbox = params.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return image_path
    settings = get_settings()
    return crop_image_by_bbox(
        image_path,
        [float(x) for x in bbox],
        cache_dir=str(Path(settings.CACHE_DIR)),
    )


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


def _synthesize_observation(
    tool: ToolDefinition,
    params: dict[str, Any],
    image_path: str,
    narration: str,
    agent_role: AgentRole,
    *,
    retry_feedback: str = "",
) -> dict[str, Any]:
    """H 规则：LLM 按 observation_fields 逐字段合成（关键帧/裁剪图 + 消毒旁白）。"""
    model_cls = _build_observation_model(tool)
    field_lines = []
    for f in tool.observation_fields:
        field_lines.append(
            f"- {f.name} ({f.type}, nullable={f.nullable}): {f.description}"
        )
    role_hint = (
        "This is a COARSE (broad localization) step: prefer ranges/visual features; "
        "do not invent city/POI names.\n"
        if agent_role == AgentRole.COARSE
        else ""
    )
    feedback = ""
    if retry_feedback.strip():
        feedback = (
            "Previous Observation was rejected. Fix these issues and regenerate:\n"
            f"{retry_feedback.strip()}\n"
        )
    prompt = (
        "You synthesize a tool Observation from the image, sanitized step narration, "
        "and action params.\n"
        "Rules:\n"
        "- Fill EVERY field listed; do not add extra fields.\n"
        "- If nullable and info is absent in the image/narration, use null; do not guess.\n"
        "- For result_list use 2~5 items matching item_fields when status=success.\n"
        "- Never use placeholder strings like 未知/不确定/N/A.\n"
        "- status must be success|empty|error; error_message only when status=error.\n"
        "- Do NOT use any groundtruth.\n"
        "- Style the Observation as a realistic API response for this tool.\n"
        "- Narration is auxiliary visual/process context ONLY. "
        "Do NOT copy city names, POI names, or precise locations from Narration "
        "into Observation fields.\n"
        "- Place names in Observation may only come from Action params "
        "(e.g. query) or clearly visible scene text (road signs/shop signs) "
        "when the tool semantically extracts text.\n"
        "- H9: NEVER include video production overlays: intro/title cards, "
        "platform/channel watermarks or logos, difficulty/star badges, "
        "progress bars, burned-in subtitles, creator tags, or non-scene UI. "
        "Describe only in-scene geography/architecture/nature/real signage. "
        "For OCR, ignore corner watermarks and top title bars; if only overlay "
        "text is visible, return empty texts / status=empty.\n"
        f"{role_hint}"
        f"{feedback}"
        f"AgentRole: {agent_role.value}\n"
        f"Tool: {tool.name}\nDescription: {tool.description}\n"
        f"Params: {json.dumps(params, ensure_ascii=False)}\n"
        f"Narration: {narration or '(empty)'}\n"
        f"Fields:\n" + "\n".join(field_lines)
    )
    result = call_structured(prompt, model_cls, images=[image_path])
    return result.model_dump(mode="json")


def execute_action(
    action: Action,
    image_path: str,
    agent_role: AgentRole,
    *,
    narration: str = "",
    registry_path: Optional[str] = None,
    use_cache: bool = True,
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
    synth_image = _resolve_synth_image(tool, params, image_path)
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
                    "burned-in subtitles, creator tags, non-scene UI). "
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

    if cache is not None and result.status in ("success", "empty", "skipped"):
        cache[key] = result.model_dump(mode="json")
    return result
