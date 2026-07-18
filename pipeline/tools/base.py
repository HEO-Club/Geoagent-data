"""Tool Executor 分发器：权限、校验、production/draft 执行与 diskcache。"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Optional

import diskcache
from pydantic import BaseModel, Field, ValidationError

from pipeline.config import get_settings
from pipeline.llm import call_structured
from pipeline.schemas import (
    Action,
    AgentRole,
    ObservationExecutionResult,
    ObservationSource,
    ToolDefinition,
    ToolTier,
)
from pipeline.tools.registry import load_registry
from pipeline.tools.validation import validate_action_params, validate_observation

PROMPT_VERSION = "draft_obs_v1"
EXECUTOR_VERSION = "1"


def _tool_schema_hash(tool: ToolDefinition) -> str:
    payload = tool.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _params_hash(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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
    model_name: Optional[str],
    prompt_version: Optional[str],
) -> str:
    parts = [
        tool.name,
        _tool_schema_hash(tool),
        EXECUTOR_VERSION,
        _params_hash(params),
        _image_content_hash(image_path),
    ]
    if model_name:
        parts.append(model_name)
    if prompt_version:
        parts.append(prompt_version)
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


def _import_executor(executor_ref: str) -> Any:
    module_path, _, attr = executor_ref.rpartition(".")
    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"executor_ref 不可调用: {executor_ref}")
    return fn


def _build_draft_observation_model(tool: ToolDefinition) -> type[BaseModel]:
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
    return type(f"DraftObs_{tool.name}", (BaseModel,), namespace)


def _synthesize_draft_observation(
    tool: ToolDefinition,
    params: dict[str, Any],
    image_path: str,
) -> dict[str, Any]:
    """H 规则：VLM 按 observation_fields 逐字段合成。"""
    model_cls = _build_draft_observation_model(tool)
    field_lines = []
    for f in tool.observation_fields:
        field_lines.append(
            f"- {f.name} ({f.type}, nullable={f.nullable}): {f.description}"
        )
    prompt = (
        "You synthesize a tool Observation from the image and action params.\n"
        "Rules:\n"
        "- Fill EVERY field listed; do not add extra fields.\n"
        "- If nullable and info is absent in the video/image, use null; do not guess.\n"
        "- For result_list use 2~5 items matching item_fields when status=success.\n"
        "- Never use placeholder strings like 未知/不确定/N/A.\n"
        "- status must be success|empty|error; error_message only when status=error.\n"
        "- Do NOT use any groundtruth.\n\n"
        f"Tool: {tool.name}\nDescription: {tool.description}\n"
        f"Params: {json.dumps(params, ensure_ascii=False)}\n"
        f"Fields:\n" + "\n".join(field_lines)
    )
    result = call_structured(prompt, model_cls, images=[image_path])
    return result.model_dump(mode="json")


def execute_action(
    action: Action,
    image_path: str,
    agent_role: AgentRole,
    *,
    registry_path: Optional[str] = None,
    use_cache: bool = True,
) -> ObservationExecutionResult:
    """分发器：权限 → params 校验 → terminal/production/draft → observation 校验 → 缓存。"""
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

    model_name = settings.LLM_MODEL if tool.tier == ToolTier.DRAFT else None
    prompt_version = PROMPT_VERSION if tool.tier == ToolTier.DRAFT else None
    key = _cache_key(
        tool=tool,
        params=params,
        image_path=image_path,
        model_name=model_name,
        prompt_version=prompt_version,
    )

    cache: diskcache.Cache | None = _get_cache() if use_cache else None
    if cache is not None and key in cache:
        cached = cache[key]
        if isinstance(cached, dict):
            return ObservationExecutionResult.model_validate(
                {**cached, "cache_hit": True}
            )

    if tool.tier == ToolTier.PRODUCTION:
        if not tool.executor_ref:
            return ObservationExecutionResult(
                action=normalized_action,
                observation=None,
                source=None,
                status="error",
                error_message="production tool 缺少 executor_ref",
                cache_hit=False,
            )
        try:
            execute_fn = _import_executor(tool.executor_ref)
            raw_obs = execute_fn(params, image_path)
            try:
                obs = validate_observation(tool, raw_obs)
            except ValidationError as exc:
                # 非法真实结果不得用 VLM 伪装
                result = ObservationExecutionResult(
                    action=normalized_action,
                    observation=None,
                    source=ObservationSource.REAL_EXECUTION,
                    status="error",
                    error_message=f"production observation 校验失败: {exc}",
                    cache_hit=False,
                )
                return result
            status = str((obs or {}).get("status", "success"))
            if status not in ("success", "empty", "error"):
                status = "success"
            result = ObservationExecutionResult(
                action=normalized_action,
                observation=obs,
                source=ObservationSource.REAL_EXECUTION,
                status=status,  # type: ignore[arg-type]
                error_message=(obs or {}).get("error_message") if obs else None,
                cache_hit=False,
            )
        except Exception as exc:  # noqa: BLE001
            result = ObservationExecutionResult(
                action=normalized_action,
                observation=None,
                source=ObservationSource.REAL_EXECUTION,
                status="error",
                error_message=str(exc),
                cache_hit=False,
            )
    else:
        # draft → VLM 合成，失败重试
        last_error: str | None = None
        result = ObservationExecutionResult(
            action=normalized_action,
            observation=None,
            source=ObservationSource.VLM_SYNTHESIZED,
            status="error",
            error_message="draft synthesis failed",
            cache_hit=False,
        )
        for _ in range(max(1, settings.DRAFT_TOOL_MAX_RETRY)):
            try:
                raw_obs = _synthesize_draft_observation(tool, params, image_path)
                obs = validate_observation(tool, raw_obs)
                status = str((obs or {}).get("status", "success"))
                if status not in ("success", "empty", "error"):
                    status = "success"
                result = ObservationExecutionResult(
                    action=normalized_action,
                    observation=obs,
                    source=ObservationSource.VLM_SYNTHESIZED,
                    status=status,  # type: ignore[arg-type]
                    error_message=(obs or {}).get("error_message") if obs else None,
                    cache_hit=False,
                )
                last_error = None
                break
            except (ValidationError, Exception) as exc:  # noqa: BLE001
                last_error = str(exc)
                result = ObservationExecutionResult(
                    action=normalized_action,
                    observation=None,
                    source=ObservationSource.VLM_SYNTHESIZED,
                    status="error",
                    error_message=last_error,
                    cache_hit=False,
                )
        if last_error is not None:
            # 仍失败 → rejected 由上游标记；此处返回 error
            pass

    if cache is not None and result.status in ("success", "empty", "skipped"):
        cache[key] = result.model_dump(mode="json")
    return result
