"""动态 params / observation 校验（F10 与 map_query 条件规则）。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ValidationError

from pipeline.schemas import AgentRole, ObservationField, ParamField, ToolDefinition

_STATUS_VALUES = frozenset({"success", "empty", "error"})

_WEB_SEARCH_PURPOSE_BY_ROLE: dict[AgentRole, frozenset[str]] = {
    AgentRole.COARSE: frozenset({"broad_discovery"}),
    AgentRole.FINE: frozenset({"broad_discovery", "precise_lookup"}),
    AgentRole.VERIFIER: frozenset({"verification"}),
}


def _raise_validation(message: str, *, loc: tuple[str, ...] = ()) -> None:
    """抛出 pydantic ValidationError。"""
    raise ValidationError.from_exception_data(
        "ToolValidation",
        [
            {
                "type": "value_error",
                "loc": loc,
                "input": None,
                "ctx": {"error": ValueError(message)},
            }
        ],
    )


def apply_param_defaults(tool: ToolDefinition, params: dict[str, Any]) -> dict[str, Any]:
    """为缺失的可选参数补齐 default；不修改已传入值。"""
    result = dict(params)
    for field in tool.params:
        if field.name not in result and not field.required and field.default is not None:
            result[field.name] = field.default
        elif field.name not in result and not field.required and field.default is None:
            # 显式保留可选且 default=null 的键，便于后续交叉约束判断“已提供”
            pass
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_param_type(field: ParamField, value: Any) -> None:
    t = field.type
    if t == "string":
        if not isinstance(value, str):
            _raise_validation(f"参数 {field.name} 须为 string", loc=("params", field.name))
        if field.enum_values is not None and value not in field.enum_values:
            _raise_validation(
                f"参数 {field.name} 必须为枚举值之一: {field.enum_values}",
                loc=("params", field.name),
            )
        return
    if t == "float":
        if not _is_number(value):
            _raise_validation(f"参数 {field.name} 须为 float", loc=("params", field.name))
        return
    if t == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            _raise_validation(f"参数 {field.name} 须为 int", loc=("params", field.name))
        return
    if t == "bool":
        if not isinstance(value, bool):
            _raise_validation(f"参数 {field.name} 须为 bool", loc=("params", field.name))
        return
    if t == "bbox":
        if not (
            isinstance(value, list)
            and len(value) == 4
            and all(_is_number(x) for x in value)
        ):
            _raise_validation(
                f"参数 {field.name} 须为长度为 4 的 bbox 列表",
                loc=("params", field.name),
            )
        return
    if t in ("latlng", "lat_range"):
        ok_list = isinstance(value, list) and len(value) == 2 and all(_is_number(x) for x in value)
        ok_tuple = isinstance(value, tuple) and len(value) == 2 and all(_is_number(x) for x in value)
        if not (ok_list or ok_tuple):
            _raise_validation(
                f"参数 {field.name} 须为长度为 2 的 {t}",
                loc=("params", field.name),
            )
        lat_or_a, lng_or_b = float(value[0]), float(value[1])
        if t == "latlng":
            if not (-90.0 <= lat_or_a <= 90.0 and -180.0 <= lng_or_b <= 180.0):
                _raise_validation(
                    f"参数 {field.name} 经纬度超出合法范围",
                    loc=("params", field.name),
                )
        return
    if t == "string_list":
        if not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            _raise_validation(
                f"参数 {field.name} 须为 string_list",
                loc=("params", field.name),
            )
        return
    _raise_validation(f"未知参数类型 {t}", loc=("params", field.name))


def _normalize_latlng_like(value: Any) -> list[float]:
    return [float(value[0]), float(value[1])]


def _check_web_search_purpose(
    tool: ToolDefinition,
    params: dict[str, Any],
    agent_role: Optional[AgentRole],
) -> None:
    if tool.name != "web_search" or agent_role is None:
        return
    purpose = params.get("purpose")
    allowed = _WEB_SEARCH_PURPOSE_BY_ROLE.get(agent_role)
    if allowed is None:
        return
    if purpose not in allowed:
        _raise_validation(
            f"web_search.purpose={purpose!r} 不允许用于角色 {agent_role.value}；"
            f"允许: {sorted(allowed)}",
            loc=("params", "purpose"),
        )


def _check_map_query_cross_fields(tool: ToolDefinition, params: dict[str, Any]) -> None:
    if tool.name != "map_query":
        return
    query = params.get("query")
    latlng = params.get("latlng")
    query_ok = isinstance(query, str) and query.strip() != ""
    latlng_ok = latlng is not None
    if not query_ok and not latlng_ok:
        _raise_validation(
            "map_query 要求 query 与 latlng 至少提供一个",
            loc=("params",),
        )


def validate_action_params(
    tool: ToolDefinition,
    params: dict[str, Any],
    agent_role: Optional[AgentRole] = None,
) -> dict[str, Any]:
    """先 apply_param_defaults，再做类型/必填/交叉约束校验，返回规范化 params。"""
    normalized = apply_param_defaults(tool, params)
    allowed_names = {f.name for f in tool.params}
    extra = set(normalized) - allowed_names
    if extra:
        _raise_validation(f"拒绝额外参数: {sorted(extra)}", loc=("params",))

    for field in tool.params:
        if field.name not in normalized:
            if field.required:
                _raise_validation(f"缺少必填参数: {field.name}", loc=("params", field.name))
            continue
        value = normalized[field.name]
        if value is None:
            if field.required:
                _raise_validation(f"必填参数不得为 null: {field.name}", loc=("params", field.name))
            continue
        _check_param_type(field, value)
        if field.type in ("latlng", "lat_range") and value is not None:
            normalized[field.name] = _normalize_latlng_like(value)
        if field.type == "bbox" and value is not None:
            normalized[field.name] = [float(x) for x in value]
        if field.type == "float" and value is not None:
            normalized[field.name] = float(value)

    _check_web_search_purpose(tool, normalized, agent_role)
    _check_map_query_cross_fields(tool, normalized)
    return normalized


def _check_obs_type(field: ObservationField, value: Any, *, loc: tuple[str, ...]) -> None:
    t = field.type
    if t == "string":
        if not isinstance(value, str):
            _raise_validation(f"observation.{field.name} 须为 string", loc=loc)
        return
    if t == "float":
        if not _is_number(value):
            _raise_validation(f"observation.{field.name} 须为 float", loc=loc)
        return
    if t == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            _raise_validation(f"observation.{field.name} 须为 int", loc=loc)
        return
    if t == "bool":
        if not isinstance(value, bool):
            _raise_validation(f"observation.{field.name} 须为 bool", loc=loc)
        return
    if t == "string_list":
        if not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            _raise_validation(f"observation.{field.name} 须为 string_list", loc=loc)
        return
    if t == "bbox":
        if not (
            isinstance(value, list)
            and len(value) == 4
            and all(_is_number(x) for x in value)
        ):
            _raise_validation(f"observation.{field.name} 须为 bbox", loc=loc)
        return
    if t in ("latlng", "lat_range"):
        ok = (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(_is_number(x) for x in value)
        )
        if not ok:
            _raise_validation(f"observation.{field.name} 须为 {t}", loc=loc)
        if t == "latlng":
            lat, lng = float(value[0]), float(value[1])
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
                _raise_validation(f"observation.{field.name} 经纬度超出范围", loc=loc)
        return
    if t == "result_list":
        if not isinstance(value, list):
            _raise_validation(f"observation.{field.name} 须为 result_list", loc=loc)
        if field.item_fields is None:
            _raise_validation(f"observation.{field.name} 缺少 item_fields 定义", loc=loc)
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                _raise_validation(
                    f"observation.{field.name}[{i}] 须为 object",
                    loc=(*loc, str(i)),
                )
            item_names = {f.name for f in field.item_fields}
            extra = set(item) - item_names
            if extra:
                _raise_validation(
                    f"observation.{field.name}[{i}] 含额外字段: {sorted(extra)}",
                    loc=(*loc, str(i)),
                )
            missing = item_names - set(item)
            if missing:
                _raise_validation(
                    f"observation.{field.name}[{i}] 缺少字段: {sorted(missing)}",
                    loc=(*loc, str(i)),
                )
            for sub in field.item_fields:
                sub_val = item[sub.name]
                if sub_val is None:
                    if not sub.nullable:
                        _raise_validation(
                            f"observation.{field.name}[{i}].{sub.name} 不可为 null",
                            loc=(*loc, str(i), sub.name),
                        )
                    continue
                _check_obs_type(sub, sub_val, loc=(*loc, str(i), sub.name))
        return
    _raise_validation(f"未知 observation 类型 {t}", loc=loc)


def _check_map_query_observation(observation: dict[str, Any]) -> None:
    status = observation.get("status")
    err = observation.get("error_message")
    resolved = observation.get("resolved_latlng")
    if "latlng" in observation and "resolved_latlng" not in observation:
        _raise_validation(
            "map_query Observation 禁止使用 latlng 键，必须使用 resolved_latlng",
            loc=("observation", "latlng"),
        )
    if "latlng" in observation:
        _raise_validation(
            "map_query Observation 不得包含旧键 latlng",
            loc=("observation", "latlng"),
        )

    if status == "success":
        if resolved is None:
            _raise_validation(
                "map_query status=success 时 resolved_latlng 必须非空",
                loc=("observation", "resolved_latlng"),
            )
    elif status == "empty":
        if err is not None:
            _raise_validation(
                "map_query status=empty 时 error_message 必须为 null",
                loc=("observation", "error_message"),
            )
    elif status == "error":
        if not (isinstance(err, str) and err.strip()):
            _raise_validation(
                "map_query status=error 时 error_message 必须非空",
                loc=("observation", "error_message"),
            )


def validate_observation(
    tool: ToolDefinition,
    observation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """按 observation_fields 与 Tool 级条件规则校验 Observation。"""
    if tool.is_terminal:
        if observation is not None:
            _raise_validation("terminal Tool 的 observation 必须为 None", loc=("observation",))
        return None

    if observation is None:
        _raise_validation("非 terminal Tool 的 observation 不得为 None", loc=("observation",))

    assert observation is not None
    allowed = {f.name for f in tool.observation_fields}
    extra = set(observation) - allowed
    if extra:
        # 特别拒绝 map_query 旧键（即使不在 schema 中）
        if tool.name == "map_query" and "latlng" in extra:
            _raise_validation(
                "map_query Observation 不得使用 latlng，须使用 resolved_latlng",
                loc=("observation", "latlng"),
            )
        _raise_validation(f"observation 含额外字段: {sorted(extra)}", loc=("observation",))

    for field in tool.observation_fields:
        if field.name not in observation:
            _raise_validation(
                f"缺少 observation 字段: {field.name}",
                loc=("observation", field.name),
            )
        value = observation[field.name]
        if value is None:
            if not field.nullable:
                _raise_validation(
                    f"observation.{field.name} 不可为 null",
                    loc=("observation", field.name),
                )
            continue
        _check_obs_type(field, value, loc=("observation", field.name))

    status = observation.get("status")
    if status not in _STATUS_VALUES:
        _raise_validation(
            f"observation.status 必须为 success|empty|error，收到 {status!r}",
            loc=("observation", "status"),
        )
    err = observation.get("error_message")
    if status == "error":
        if not (isinstance(err, str) and err.strip()):
            _raise_validation(
                "status=error 时 error_message 必须非空",
                loc=("observation", "error_message"),
            )
    else:
        if err is not None:
            _raise_validation(
                f"status={status} 时 error_message 必须为 null",
                loc=("observation", "error_message"),
            )

    if tool.name == "map_query":
        _check_map_query_observation(observation)

    # 规范化 latlng / lat_range 为 list[float]
    out = dict(observation)
    for field in tool.observation_fields:
        val = out.get(field.name)
        if val is not None and field.type in ("latlng", "lat_range"):
            out[field.name] = _normalize_latlng_like(val)
        if val is not None and field.type == "bbox":
            out[field.name] = [float(x) for x in val]
    return out
