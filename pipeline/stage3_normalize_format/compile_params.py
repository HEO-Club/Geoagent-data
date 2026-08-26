"""Stage 3：对照 schema，从 Thought/口语 params 编译工具输入（方案 A）。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from pipeline.llm import call_structured
from pipeline.schemas.tools import (
    InputFieldSpec,
    ParameterAuditIssue,
    ToolForest,
    ToolInputSchema,
    ToolParameterAudit,
)
from pipeline.stage3_normalize_format.params import (
    _norm_name,
    normalize_and_validate_tool_inputs,
)
from pipeline.stage3_normalize_format.trees import find_tree_for_name

logger = logging.getLogger(__name__)

# 需要来源文本中出现字面量/代码才可填的字段（防编造）
_LITERAL_SOURCE_FIELDS = frozenset(
    {
        "overpass_ql",
        "image",
        "images",
        "coordinates",
    }
)

_DISTANCE_FIELD_HINTS = frozenset({"distance_m", "radius_m"})


@dataclass
class CompileRequest:
    """单次 tool_call 的参数编译请求（对照 schema 从 Thought 填表）。"""

    step_index: int
    tool: str
    operation: str
    thought: str
    raw_inputs: dict[str, Any]
    normalized_inputs: dict[str, Any]
    missing_fields: list[str]
    available_context: dict[str, Any]
    schema: ToolInputSchema
    first_audit: ToolParameterAudit


@dataclass
class CompileFill:
    """LLM 对单步返回的编译结果（未过滤）。"""

    step_index: int
    filled_inputs: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


ParamCompilerFn = Callable[[list[CompileRequest]], dict[int, CompileFill]]


class _CompileStepResult(BaseModel):
    step_index: int = Field(ge=1)
    filled_inputs: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class _BatchCompileResult(BaseModel):
    results: list[_CompileStepResult] = Field(default_factory=list)


def missing_fields_from_audit(audit: ToolParameterAudit) -> list[str]:
    """从审计 issues 提取仍缺失的字段名。"""
    names: list[str] = []
    seen: set[str] = set()
    for issue in audit.issues:
        if issue.code not in {
            "required_input_missing",
            "required_input_group_missing",
        }:
            continue
        if not issue.field:
            continue
        for part in str(issue.field).split("|"):
            name = part.strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def empty_schema_fields(audit: ToolParameterAudit, schema: ToolInputSchema) -> list[str]:
    """尚未出现在 normalized_inputs 中的 schema 字段（不含 extensions）。"""
    present = {key for key in audit.normalized_inputs if key != "extensions"}
    return [item.name for item in schema.fields if item.name not in present]


def _schema_payload(schema: ToolInputSchema) -> list[dict[str, Any]]:
    """把完整 operation schema 交给 LLM，由其对照 Thought 填表。"""
    rows: list[dict[str, Any]] = []
    for item in schema.fields:
        rows.append(
            {
                "name": item.name,
                "type": item.type,
                "required": item.required,
                "description": item.description,
                "acquisition_hint": item.acquisition_hint,
                "allowed_values": list(item.allowed_values),
                "aliases": list(item.aliases),
                "example": item.example,
            }
        )
    return rows


def _request_payload(req: CompileRequest) -> dict[str, Any]:
    return {
        "step_index": req.step_index,
        "tool": req.tool,
        "operation": req.operation,
        "thought": req.thought,
        "raw_inputs": req.raw_inputs,
        "normalized_inputs": req.normalized_inputs,
        "missing_fields": req.missing_fields,
        "available_context_keys": sorted(req.available_context.keys()),
        "available_context": dict(req.available_context),
        "schema_fields": _schema_payload(req.schema),
        "required_any": list(req.schema.required_any),
    }


def llm_compile_params(
    requests: list[CompileRequest],
) -> dict[int, CompileFill]:
    """一次结构化调用，为轨迹内所有 tool_call 对照 schema 编译 params。"""
    if not requests:
        return {}
    prompt = (
        "你在把地理定位 Agent 自由轨迹编译为执行器级 input_schema 字段。\n"
        "对照 schema_fields，根据 Thought、raw_inputs、normalized_inputs "
        "（含 extensions）和 available_context 填写 inputs。\n"
        "这是写参数，不是只补缺：Thought 里写明的约束都应对到 schema 字段"
        "（例如「5公里内有风电」→ relation=near、geometry=风力发电机、distance_m=5000；"
        "「打开历史地图」→ layers 含 historical_map）。\n"
        "已有 normalized_inputs 中的键不要覆盖、不要改掉。"
        "只填 schema 里有的字段；不确定就不要填。\n"
        "不得编造坐标、文件路径、Overpass QL 或来源中不存在的几何/数值。\n"
        "execution 字段不得填写“步骤3检索到的结果/上一步提取帧”这类自然语言占位；"
        "只能使用 available_context 中已有的 $step_…/$current_… 引用或来源里明确存在的真实 ID。\n"
        "枚举字段必须使用 schema 的 allowed_values。"
        "上下文引用只能使用 available_context 中已有的 $… 值。\n"
        "每个 step_index 返回一条 result。\n"
        f"待编译步骤：{json.dumps([_request_payload(r) for r in requests], ensure_ascii=False)}"
    )
    result = call_structured(prompt, _BatchCompileResult, lane="llm")
    out: dict[int, CompileFill] = {}
    for item in result.results:
        out[item.step_index] = CompileFill(
            step_index=item.step_index,
            filled_inputs=dict(item.filled_inputs or {}),
            reason=item.reason or "",
        )
    return out


def _source_blob(req: CompileRequest) -> str:
    parts = [
        req.thought or "",
        json.dumps(req.raw_inputs, ensure_ascii=False),
        json.dumps(req.normalized_inputs.get("extensions") or {}, ensure_ascii=False),
        json.dumps(req.normalized_inputs, ensure_ascii=False),
        json.dumps(req.available_context, ensure_ascii=False),
    ]
    return "\n".join(parts)


def _extract_numbers_from_text(text: str) -> list[float]:
    """从文本提取数字；公里/km 转为米。"""
    found: list[float] = []
    for match in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(公里|千米|km|KM|米|m|M)?", text
    ):
        value = float(match.group(1))
        unit = match.group(2) or ""
        if unit in {"公里", "千米", "km", "KM"}:
            value *= 1000.0
        found.append(value)
    return found


def _value_in_source_text(value: Any, source: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numbers = _extract_numbers_from_text(source)
        target = float(value)
        return any(abs(n - target) < 1e-6 for n in numbers)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        if text.startswith("$"):
            return text in source
        # 子串或去掉空白后的包含
        compact_source = re.sub(r"\s+", "", source.lower())
        compact_value = re.sub(r"\s+", "", text.lower())
        if compact_value and compact_value in compact_source:
            return True
        # 分词：每个长度>=2 的片段至少有一个命中（用于「风力发电机」→「风电」）
        tokens = [t for t in re.split(r"[\s,，、/;|]+", text) if len(t) >= 2]
        return bool(tokens) and any(t.lower() in source.lower() for t in tokens)
    if isinstance(value, list):
        return all(_value_in_source_text(item, source) for item in value) if value else False
    if isinstance(value, dict):
        return all(
            _value_in_source_text(k, source) or _value_in_source_text(v, source)
            for k, v in value.items()
        )
    return str(value) in source


def _enum_grounded_in_source(value: str, field: InputFieldSpec, source: str) -> bool:
    """枚举值可用中文同义/来源线索证实，不要求英文 token 字面出现。"""
    if _value_in_source_text(value, source):
        return True
    lower = source.lower()
    hints: dict[str, tuple[str, ...]] = {
        "near": ("附近", "靠近", "公里内", "千米内", "米内", "km内", "within"),
        "within": ("之内", "内部", "范围内", "within"),
        "intersects": ("相交", "交叉", "穿过", "intersects"),
        "contains": ("包含", "含有", "contains"),
        "crosses": ("穿越", "跨过", "crosses"),
    }
    for hint in hints.get(_norm_name(value), ()):
        if hint.lower() in lower:
            return True
    # 枚举合法且来源含空间关系线索时接受 Thought 改写
    return bool(field.allowed_values) and _norm_name(value) in {
        _norm_name(v) for v in field.allowed_values
    } and any(
        token in lower
        for token in ("附近", "内", "相交", "包含", "穿越", "距离", "公里", "米")
    )


def _normalize_allowed(value: Any, field: InputFieldSpec) -> Any | None:
    if not field.allowed_values:
        return value
    if not isinstance(value, str):
        return None
    raw = value.strip()
    by_norm = {_norm_name(v): v for v in field.allowed_values}
    hit = by_norm.get(_norm_name(raw))
    if hit is not None:
        return hit
    aliases = {
        "附近": "near",
        "靠近": "near",
        "within": "within",
        "内部": "within",
        "相交": "intersects",
        "交叉": "crosses",
        "包含": "contains",
    }
    mapped = aliases.get(raw) or aliases.get(_norm_name(raw))
    if mapped and mapped in field.allowed_values:
        return mapped
    return None


def _field_by_name(schema: ToolInputSchema, name: str) -> InputFieldSpec | None:
    for item in schema.fields:
        if item.name == name:
            return item
    return None


def filter_grounded_fills(
    req: CompileRequest, filled: dict[str, Any]
) -> dict[str, Any]:
    """程序化过滤：只保留可证实、且尚未填过的 schema 字段。"""
    source = _source_blob(req)
    already = {
        k: v
        for k, v in req.normalized_inputs.items()
        if k != "extensions" and v is not None
    }
    allowed_names = {f.name for f in req.schema.fields}
    fillable = {f.name for f in req.schema.fields if f.name not in already}

    accepted: dict[str, Any] = {}
    for raw_key, raw_value in (filled or {}).items():
        name = _norm_name(raw_key)
        if name not in allowed_names:
            continue
        if name in already:
            continue  # 不覆盖已有
        if name not in fillable:
            continue
        field = _field_by_name(req.schema, name)
        if field is None:
            continue

        value: Any = raw_value
        if field.allowed_values:
            value = _normalize_allowed(value, field)
            if value is None:
                continue
            if isinstance(value, str) and not _enum_grounded_in_source(
                value, field, source
            ):
                continue

        if name in _LITERAL_SOURCE_FIELDS:
            if not _value_in_source_text(value, source):
                continue
        elif name in _DISTANCE_FIELD_HINTS or field.type in {
            "number",
            "integer",
            "number_or_string",
        }:
            if isinstance(value, str):
                nums = _extract_numbers_from_text(value)
                if not nums:
                    continue
                value = nums[0] if field.type != "integer" else int(nums[0])
            if not _value_in_source_text(value, source):
                continue
        elif isinstance(value, str) and value.startswith("$"):
            ctx_values = {str(v) for v in req.available_context.values()}
            if value not in ctx_values and value not in source:
                continue
        elif field.type in {"string", "string_or_array", "string_or_object", "location"}:
            if isinstance(value, str) and not _value_in_source_text(value, source):
                # layers 枚举式图层名：允许从 thought 语义推断的常见图层词
                if name == "layers":
                    layer_tokens = {
                        "historical_map",
                        "hydrology",
                        "terrain",
                        "roads",
                        "admin",
                        "satellite",
                    }
                    items = value if isinstance(value, list) else [value]
                    ok_items = []
                    for item in items if isinstance(items, list) else [items]:
                        token = _norm_name(item)
                        if token in layer_tokens and (
                            any(
                                hint in source.lower()
                                for hint in (
                                    "历史",
                                    "historical",
                                    "水系",
                                    "hydrology",
                                    "地形",
                                    "terrain",
                                    "图层",
                                    "layer",
                                    "地图",
                                )
                            )
                        ):
                            ok_items.append(token if name == "layers" else item)
                        elif _value_in_source_text(item, source):
                            ok_items.append(item)
                    if not ok_items:
                        continue
                    value = ok_items if len(ok_items) != 1 or isinstance(raw_value, list) else ok_items[0]
                else:
                    continue
        elif isinstance(value, list) and name == "layers":
            grounded_layers: list[Any] = []
            for item in value:
                token = _norm_name(item) if isinstance(item, str) else item
                if isinstance(token, str) and token in {
                    "historical_map",
                    "hydrology",
                    "terrain",
                    "roads",
                    "admin",
                    "satellite",
                }:
                    if any(
                        hint in source.lower()
                        for hint in (
                            "历史",
                            "historical",
                            "水系",
                            "hydrology",
                            "地形",
                            "terrain",
                            "图层",
                            "layer",
                            "地图",
                        )
                    ):
                        grounded_layers.append(token)
                elif _value_in_source_text(item, source):
                    grounded_layers.append(item)
            if not grounded_layers:
                continue
            value = grounded_layers

        accepted[name] = value
    return accepted


def prune_resolved_extensions(
    normalized: dict[str, Any], filled_keys: set[str]
) -> dict[str, Any]:
    """若 extensions 中某口语键的值已被编译进 schema，可删掉该键。"""
    extras = normalized.get("extensions")
    if not isinstance(extras, dict) or not extras:
        return normalized
    filled_values = {
        json.dumps(normalized[k], ensure_ascii=False, sort_keys=True)
        for k in filled_keys
        if k in normalized
    }
    kept: dict[str, Any] = {}
    for key, value in extras.items():
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        # 值已被吸收，或键名已对应 schema 字段
        if serialized in filled_values:
            continue
        if _norm_name(key) in filled_keys:
            continue
        kept[key] = value
    result = dict(normalized)
    if kept:
        result["extensions"] = kept
    else:
        result.pop("extensions", None)
    return result


def apply_compile_and_revalidate(
    forest: ToolForest,
    req: CompileRequest,
    fill: CompileFill | None,
) -> ToolParameterAudit:
    """合并 grounded 填充后再次合同校验；无有效填充则返回原审计。"""
    if fill is None or not fill.filled_inputs:
        return req.first_audit

    grounded = filter_grounded_fills(req, fill.filled_inputs)
    if not grounded:
        return req.first_audit

    merged_raw = dict(req.raw_inputs)
    merged_raw.update(grounded)
    second = normalize_and_validate_tool_inputs(
        forest,
        tool=req.tool,
        operation=req.operation,
        inputs=merged_raw,
        step_index=req.step_index,
        available_context=req.available_context,
    )
    second.normalized_inputs = prune_resolved_extensions(
        second.normalized_inputs, set(grounded.keys())
    )
    second.issues = list(second.issues) + [
        ParameterAuditIssue(
            code="input_compiled_from_thought",
            severity="info",
            field=",".join(sorted(grounded.keys())),
            message=(
                f"已从 Thought/口语参数编译字段：{sorted(grounded.keys())}"
                + (f"；{fill.reason}" if fill.reason else "")
            ),
            repairable=False,
            guidance="仅摘录来源中已有信息，禁止编造。",
        )
    ]
    return second


def build_compile_request(
    *,
    forest: ToolForest,
    audit: ToolParameterAudit,
    thought: str,
    available_context: dict[str, Any],
) -> CompileRequest | None:
    """有 input_schema 且非 invalid 的 tool_call 都构造编译请求（方案 A）。"""
    if audit.readiness == "invalid":
        return None
    tree = find_tree_for_name(forest, audit.tool)
    if tree is None:
        return None
    op = next(
        (item for item in tree.canonical.operations if item.name == audit.operation),
        None,
    )
    if op is None or op.input_schema is None:
        return None
    missing = missing_fields_from_audit(audit)
    empty = empty_schema_fields(audit, op.input_schema)
    return CompileRequest(
        step_index=audit.step_index,
        tool=audit.tool,
        operation=audit.operation,
        thought=thought,
        raw_inputs=dict(audit.raw_inputs),
        normalized_inputs=dict(audit.normalized_inputs),
        missing_fields=missing or empty,
        available_context=dict(available_context),
        schema=op.input_schema,
        first_audit=audit,
    )


def compile_params_batch(
    requests: list[CompileRequest],
    *,
    compiler: ParamCompilerFn | None = None,
) -> dict[int, CompileFill]:
    """对一批 tool_call 编译参数；失败返回空 dict（失败开放）。"""
    if not requests:
        return {}
    fn = compiler or llm_compile_params
    try:
        return fn(requests)
    except Exception:
        logger.exception(
            "Stage3 param compile failed for %s steps; keeping rule audits",
            len(requests),
        )
        return {}
