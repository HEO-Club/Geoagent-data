"""自由 tool → tool 树匹配 / 新建。"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from pipeline.llm import call_structured
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.tools import (
    MatchDecision,
    ObservationField,
    ParamSpec,
    ToolDefinition,
    ToolForest,
)
from pipeline.stage3_normalize_format.trees import (
    add_variant,
    create_tree,
    find_tree_for_name,
    load_forest,
    save_forest,
    with_file_lock,
)

MatcherFn = Callable[[str, ToolForest], Optional[str]]


def _slug_tool_name(name: str) -> str:
    """将自由 tool 名规范为小写下划线。"""
    s = name.strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed_tool"


def _default_definition_from_free(tool_name: str) -> ToolDefinition:
    slug = _slug_tool_name(tool_name)
    return ToolDefinition(
        name=slug,
        description=f"Auto-created from freeform tool {tool_name!r}",
        params=[
            ParamSpec(
                name="query",
                type="string",
                required=False,
                description="自由参数占位",
            )
        ],
        observation_fields=[
            ObservationField(
                name="result",
                type="string",
                nullable=True,
                description="自由 observation 汇总",
            )
        ],
        is_terminal=slug in {"submit_answer", "final_answer", "done"},
    )


def llm_semantic_matcher(tool_name: str, forest: ToolForest) -> Optional[str]:
    """用 LLM 判断应映射到哪棵树的 canonical；无法判断返回 None。"""
    if not forest.trees:
        return None
    catalog = [
        {
            "canonical": t.canonical.name,
            "description": t.canonical.description,
            "variants": t.variants,
        }
        for t in forest.trees
    ]
    prompt = (
        "你在维护地理定位 agent 的 tool 树。\n"
        f"自由 tool 名: {tool_name}\n"
        f"现有树目录: {catalog}\n"
        "若职能与某棵树基本一致，action=map 且 canonical_name=该树名；"
        "否则 action=create，canonical_name 可为建议的新规范名（小写下划线）。\n"
        "只输出结构化字段。"
    )
    decision = call_structured(prompt, MatchDecision, lane="llm")
    if decision.action == "map" and decision.canonical_name:
        if find_tree_for_name(forest, decision.canonical_name) is not None:
            tree = find_tree_for_name(forest, decision.canonical_name)
            assert tree is not None
            return tree.canonical.name
        return None
    return None


def ensure_tool_trees(
    freeform: FreeFormTrajectory,
    trees_path: Path,
    *,
    matcher: MatcherFn | None = None,
) -> ToolForest:
    """匹配或新建 tool 树；持久化 tool_trees.json。

    Args:
        freeform: 阶段2 自由链。
        trees_path: tool_trees.json 路径。
        matcher: 可选语义匹配器；返回 canonical 名或 None。
            默认在精确未命中时调用 ``llm_semantic_matcher``。
    """
    match_fn = matcher if matcher is not None else llm_semantic_matcher
    path = Path(trees_path)
    with with_file_lock(path):
        forest = load_forest(path)
        seen: set[str] = set()
        for step in freeform.steps:
            raw = (step.tool or "").strip()
            if not raw:
                continue
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)

            existing = find_tree_for_name(forest, raw)
            if existing is not None:
                if key != existing.canonical.name.lower():
                    forest = add_variant(forest, existing.canonical.name, raw)
                continue

            mapped = match_fn(raw, forest)
            if mapped:
                forest = add_variant(forest, mapped, raw)
                continue

            slug = _slug_tool_name(raw)
            forest = create_tree(
                forest,
                _default_definition_from_free(raw),
                initial_variant=raw if raw != slug else None,
            )

        save_forest(forest, path)
        return forest
