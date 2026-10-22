"""从正式工具目录读取允许写入工具句的 (tool, operation)。"""

from __future__ import annotations

import json
from pathlib import Path


def load_operation_index(catalog_path: Path) -> set[tuple[str, str]]:
    """返回目录中的执行器与 operation 对。不含 final_answer。"""
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    trees = payload.get("trees")
    if not isinstance(trees, list):
        raise ValueError(f"工具目录缺少 trees：{catalog_path}")
    pairs: set[tuple[str, str]] = set()
    for tree in trees:
        if not isinstance(tree, dict):
            continue
        canonical = tree.get("canonical")
        if not isinstance(canonical, dict):
            continue
        tool = str(canonical.get("name") or "").strip()
        if not tool or tool == "final_answer":
            continue
        operations = canonical.get("operations") or []
        if not isinstance(operations, list):
            continue
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            name = str(operation.get("name") or "").strip()
            if name:
                pairs.add((tool, name))
    if not pairs:
        raise ValueError(f"工具目录没有可用 operation：{catalog_path}")
    return pairs


def render_operation_index(pairs: set[tuple[str, str]]) -> str:
    """把目录压成稳定的短清单，供拆句时对照。"""
    lines = [f"{tool}.{operation}" for tool, operation in sorted(pairs)]
    return "\n".join(lines)
