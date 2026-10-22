"""从 Canonical Tool 目录生成 MCP 工具表与 JSON Schema。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.schemas.tools import InputFieldSpec, ToolDefinition, ToolForest, ToolOperation
from pipeline.stage3_normalize_format.trees import load_forest

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = _REPO_ROOT / "canonical_tool_catalog_v2.json"
PURPOSE_FIELD = "purpose"
_JSON_SCHEMA_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array"})
_SHAPE_NOTES = {
    "region": "接受字符串、对象或 bbox 数组；由执行器校验。",
    "string_or_array": "接受字符串或数组。",
    "string_or_object": "接受字符串或对象。",
    "number_or_string": "接受数字或字符串。",
    "location": "接受地点字符串、对象或数组。",
}


@dataclass(frozen=True)
class McpToolSpec:
    """一个 MCP 工具，对应目录中的一个非终端 operation。"""

    name: str
    tool: str
    operation: str
    description: str
    input_schema: dict[str, Any]


def mcp_tool_name(tool: str, operation: str) -> str:
    """MCP 对外名：``工具名__操作名``。"""

    return f"{tool}__{operation}"


def parse_mcp_tool_name(name: str) -> tuple[str, str] | None:
    """把 MCP 工具名拆成 (tool, operation)；无法拆分时返回 None。"""

    if "__" not in name:
        return None
    tool, operation = name.split("__", 1)
    if not tool or not operation:
        return None
    return tool, operation


def load_mcp_tool_specs(catalog_path: Path | None = None) -> list[McpToolSpec]:
    """读取目录并展开可供 MCP 调用的 operation。"""

    path = catalog_path or DEFAULT_CATALOG_PATH
    return specs_from_forest(load_forest(path))


def specs_from_forest(forest: ToolForest) -> list[McpToolSpec]:
    """把 ToolForest 转成 MCP 工具表。

    终端工具和目录中 ``available=false`` 的工具不暴露。
    """

    specs: list[McpToolSpec] = []
    for tree in forest.trees:
        canonical = tree.canonical
        if canonical.is_terminal or not canonical.available:
            continue
        for operation in canonical.operations:
            specs.append(_spec_for_operation(canonical, operation))
    return specs


def _spec_for_operation(tool: ToolDefinition, operation: ToolOperation) -> McpToolSpec:
    schema = operation.input_schema
    properties: dict[str, Any] = {
        PURPOSE_FIELD: {
            "type": "string",
            "description": "本次调用原因；须说明要获取的新证据，不得编造坐标、文件或前置结果。",
        }
    }
    required: list[str] = [PURPOSE_FIELD]
    if schema is not None:
        for field in schema.fields:
            properties[field.name] = _field_schema(field)
            if field.required and field.name not in required:
                required.append(field.name)
    description_parts = [
        tool.description.strip(),
        operation.description.strip(),
    ]
    if schema is not None and schema.description.strip():
        description_parts.append(schema.description.strip())
    if tool.usage.strip():
        description_parts.append(tool.usage.strip())
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": True if schema is None else schema.allow_extra,
    }
    return McpToolSpec(
        name=mcp_tool_name(tool.name, operation.name),
        tool=tool.name,
        operation=operation.name,
        description=" ".join(part for part in description_parts if part),
        input_schema=input_schema,
    )


def _field_schema(field: InputFieldSpec) -> dict[str, Any]:
    """目录字段转 JSON Schema；非 JSON 基本类型不在 MCP 层卡死类型。"""

    parts = [field.description.strip()]
    if field.acquisition_hint.strip():
        parts.append(field.acquisition_hint.strip())
    if field.context_default is not None:
        parts.append(f"可引用 {field.context_default}。")
    shape_note = _SHAPE_NOTES.get(field.type)
    if shape_note:
        parts.append(shape_note)
    schema: dict[str, Any] = {"description": " ".join(part for part in parts if part)}
    if field.type in _JSON_SCHEMA_TYPES:
        schema["type"] = field.type
        if field.type == "array" and field.item_type in _JSON_SCHEMA_TYPES:
            schema["items"] = {"type": field.item_type}
    if field.allowed_values:
        schema["enum"] = list(field.allowed_values)
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum
    if field.example is not None:
        schema["examples"] = [field.example]
    return schema
