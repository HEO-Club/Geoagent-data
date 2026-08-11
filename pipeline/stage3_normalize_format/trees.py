"""Tool 树持久化：文件锁 + 原子写。"""

from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock

from pipeline.schemas.tools import ToolDefinition, ToolForest, ToolOperation, ToolTree


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def load_forest(path: Path) -> ToolForest:
    """读取 tool_trees.json；文件不存在时返回空森林。"""
    if not path.is_file():
        return ToolForest(trees=[])
    data = json.loads(path.read_text(encoding="utf-8"))
    return ToolForest.model_validate(data)


def save_forest(forest: ToolForest, path: Path) -> None:
    """原子写入 tool_trees.json。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = forest.model_dump_json(indent=2) + "\n"
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        import os

        os.fsync(fh.fileno())
    tmp.replace(path)


def find_tree_for_name(forest: ToolForest, tool_name: str) -> ToolTree | None:
    """精确（大小写不敏感）匹配 canonical 或 variants。"""
    key = tool_name.strip().lower()
    for tree in forest.trees:
        if tree.canonical.name.lower() == key:
            return tree
        for v in tree.variants:
            if v.lower() == key:
                return tree
    return None


def resolve_canonical_name(forest: ToolForest, tool_name: str) -> str | None:
    """若已映射则返回 canonical.name，否则 None。"""
    tree = find_tree_for_name(forest, tool_name)
    return tree.canonical.name if tree else None


def add_variant(
    forest: ToolForest,
    canonical_name: str,
    variant: str,
    *,
    operation: str | None = None,
) -> ToolForest:
    """向已有树追加 variant，并记录该写法对应的 canonical operation。"""
    trees: list[ToolTree] = []
    for tree in forest.trees:
        if tree.canonical.name == canonical_name:
            variants = list(tree.variants)
            if (
                variant not in variants
                and variant.lower() != tree.canonical.name.lower()
            ):
                variants.append(variant)
            variant_operations = dict(tree.variant_operations)
            if operation:
                variant_operations[variant.lower()] = operation.strip().lower()
            trees.append(
                ToolTree(
                    canonical=tree.canonical,
                    variants=variants,
                    variant_operations=variant_operations,
                )
            )
        else:
            trees.append(tree)
    return ToolForest(trees=trees)


def add_operation(
    forest: ToolForest,
    canonical_name: str,
    operation: str,
    description: str,
) -> ToolForest:
    """给同一执行器补充一种受解释约束的操作，不创建新 tool。"""
    op = operation.strip().lower().replace("-", "_").replace(" ", "_")
    trees: list[ToolTree] = []
    for tree in forest.trees:
        if tree.canonical.name != canonical_name:
            trees.append(tree)
            continue
        known = {item.name for item in tree.canonical.operations}
        canonical = tree.canonical
        if op and op not in known:
            canonical = canonical.model_copy(
                update={
                    "operations": list(canonical.operations)
                    + [
                        ToolOperation(
                            name=op,
                            description=description.strip()
                            or f"使用 {canonical_name} 执行 {op}",
                        )
                    ]
                }
            )
        trees.append(
            ToolTree(
                canonical=canonical,
                variants=list(tree.variants),
                variant_operations=dict(tree.variant_operations),
            )
        )
    return ToolForest(trees=trees)


def resolve_operation(forest: ToolForest, tool_name: str) -> str | None:
    """解析自由 tool 写法在 canonical executor 下对应的 operation。"""
    tree = find_tree_for_name(forest, tool_name)
    if tree is None:
        return None
    key = tool_name.strip().lower()
    mapped = tree.variant_operations.get(key)
    if mapped:
        return mapped
    if key == tree.canonical.name.lower() and tree.canonical.operations:
        return tree.canonical.operations[0].name
    return None


def create_tree(
    forest: ToolForest,
    canonical: ToolDefinition,
    *,
    initial_variant: str | None = None,
    initial_operation: str | None = None,
) -> ToolForest:
    """新建一棵树；canonical.name 不得重复。"""
    if find_tree_for_name(forest, canonical.name) is not None:
        raise ValueError(f"tool 树已存在: {canonical.name}")
    variants: list[str] = []
    if initial_variant and initial_variant.lower() != canonical.name.lower():
        variants.append(initial_variant)
    variant_operations: dict[str, str] = {}
    if initial_variant and initial_operation:
        variant_operations[initial_variant.lower()] = initial_operation
    trees = list(forest.trees) + [
        ToolTree(
            canonical=canonical,
            variants=variants,
            variant_operations=variant_operations,
        )
    ]
    return ToolForest(trees=trees)


def with_file_lock(path: Path):
    """返回 FileLock 上下文管理器。"""
    return FileLock(str(_lock_path(path)))
