"""从 v2 目录读取 operation 声明字段；不依赖 pipeline。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_CATALOG_PATH = Path(__file__).resolve().parents[1] / "canonical_tool_catalog_v2.json"


@lru_cache(maxsize=1)
def catalog_input_fields() -> dict[tuple[str, str], frozenset[str]]:
    """返回 (tool, operation) -> 声明字段名。"""

    payload = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    mapping: dict[tuple[str, str], frozenset[str]] = {}
    for tree in payload.get("trees") or []:
        canonical = tree.get("canonical") or {}
        tool = str(canonical.get("name") or "")
        if not tool:
            continue
        for operation in canonical.get("operations") or []:
            name = str(operation.get("name") or "")
            if not name:
                continue
            schema = operation.get("input_schema") or {}
            fields = schema.get("fields") or []
            names = [
                str(field.get("name") or "").strip()
                for field in fields
                if isinstance(field, dict)
            ]
            mapping[(tool, name)] = frozenset(item for item in names if item)
    return mapping


def undeclared_input_extensions(
    tool: str,
    operation: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """相对目录 input_schema，收集未声明键与已有 extensions。"""

    extra: dict[str, Any] = {}
    existing = inputs.get("extensions")
    if isinstance(existing, dict):
        extra.update(existing)
    declared = catalog_input_fields().get((tool, operation))
    if declared is None:
        for key, value in inputs.items():
            if key != "extensions":
                extra[key] = value
        return extra
    for key, value in inputs.items():
        if key in declared or key == "extensions":
            continue
        extra[key] = value
    return extra
