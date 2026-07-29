"""Tool 树持久化：文件锁 + 原子写。"""

from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock

from pipeline.schemas.tools import ToolDefinition, ToolForest, ToolTree


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


def add_variant(forest: ToolForest, canonical_name: str, variant: str) -> ToolForest:
    """向已有树追加 variant（幂等）。"""
    trees: list[ToolTree] = []
    for tree in forest.trees:
        if tree.canonical.name == canonical_name:
            variants = list(tree.variants)
            if (
                variant not in variants
                and variant.lower() != tree.canonical.name.lower()
            ):
                variants.append(variant)
            trees.append(ToolTree(canonical=tree.canonical, variants=variants))
        else:
            trees.append(tree)
    return ToolForest(trees=trees)


def create_tree(
    forest: ToolForest,
    canonical: ToolDefinition,
    *,
    initial_variant: str | None = None,
) -> ToolForest:
    """新建一棵树；canonical.name 不得重复。"""
    if find_tree_for_name(forest, canonical.name) is not None:
        raise ValueError(f"tool 树已存在: {canonical.name}")
    variants: list[str] = []
    if initial_variant and initial_variant.lower() != canonical.name.lower():
        variants.append(initial_variant)
    trees = list(forest.trees) + [
        ToolTree(canonical=canonical, variants=variants)
    ]
    return ToolForest(trees=trees)


def with_file_lock(path: Path):
    """返回 FileLock 上下文管理器。"""
    return FileLock(str(_lock_path(path)))
