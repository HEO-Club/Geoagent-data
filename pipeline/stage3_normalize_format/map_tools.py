"""自由 tool → 执行器级 canonical tool 归并 / 受控新建。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.config import get_settings
from pipeline.llm import call_structured
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.tools import (
    MatchDecision,
    ObservationField,
    ParamSpec,
    ToolDefinition,
    ToolForest,
    ToolOperation,
    ToolTree,
)
from pipeline.stage3_normalize_format.params import attach_operation_input_schemas
from pipeline.stage3_normalize_format.trees import (
    add_operation,
    add_variant,
    create_tree,
    find_tree_for_name,
    load_forest,
    save_forest,
    with_file_lock,
)

MatcherFn = Callable[[str, ToolForest], str | None | MatchDecision]
RESERVED_TERMINAL_TOOLS = {"final_answer", "submit_answer", "done"}
REASONING_DEMOTION_CONFIDENCE = 0.85


class _BatchMatchResult(BaseModel):
    decisions: list[MatchDecision] = Field(default_factory=list)


def _merge_catalog(runtime: ToolForest, catalog: ToolForest) -> ToolForest:
    """以人工定义目录约束语义，以 runtime variants/new tools 保留增量。"""
    by_name = {tree.canonical.name.lower(): tree for tree in runtime.trees}
    ordered: list[ToolTree] = []
    catalog_names: set[str] = set()
    for seed in catalog.trees:
        key = seed.canonical.name.lower()
        catalog_names.add(key)
        existing = by_name.get(key)
        canonical = seed.canonical
        if not canonical.is_terminal:
            canonical = canonical.model_copy(
                update={"params": _canonical_param_specs(canonical.operations)}
            )
        variants = list(seed.variants)
        variant_operations = dict(seed.variant_operations)
        if existing is not None:
            for variant in existing.variants:
                if variant not in variants:
                    variants.append(variant)
            variant_operations.update(existing.variant_operations)
        ordered.append(
            ToolTree(
                canonical=canonical,
                variants=variants,
                variant_operations=variant_operations,
            )
        )
    ordered.extend(
        tree
        for tree in runtime.trees
        if tree.canonical.name.lower() not in catalog_names
    )
    return ToolForest(trees=ordered)


def _load_runtime_with_catalog(path: Path) -> ToolForest:
    runtime = load_forest(path)
    catalog_path = Path(get_settings().TOOL_CATALOG_PATH)
    if not catalog_path.is_file() or catalog_path.resolve() == path.resolve():
        return attach_operation_input_schemas(runtime)
    return attach_operation_input_schemas(
        _merge_catalog(runtime, load_forest(catalog_path))
    )


def _slug_tool_name(name: str) -> str:
    """将自由 tool 名规范为小写下划线。"""
    s = name.strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s:
        return s
    digest = hashlib.sha256(name.strip().encode("utf-8")).hexdigest()[:10]
    return f"custom_tool_{digest}"


def _canonical_param_specs(operations: list[ToolOperation]) -> list[ParamSpec]:
    """严格外层参数 + 宽容 inputs，避免原始参数略有差异就整步失败。"""
    return [
        ParamSpec(
            name="operation",
            type="string",
            required=True,
            description="本次调用在该执行器中执行的具体操作",
            allowed_values=[item.name for item in operations],
        ),
        ParamSpec(
            name="purpose",
            type="string",
            required=True,
            description="本次调用要补齐的证据缺口或任务目的",
        ),
        ParamSpec(
            name="inputs",
            type="object",
            required=True,
            description="保留原始调用输入；字段可随 operation 变化，不因额外字段拒绝",
        ),
    ]


def _final_answer_definition() -> ToolDefinition:
    return ToolDefinition(
        name="final_answer",
        description="证据链闭合后提交最终地理位置。",
        executor="agent_runtime",
        usage="仅能作为轨迹最后一步，params 只能包含 location，且无 Observation。",
        operations=[
            ToolOperation(name="submit", description="提交最终地点字符串或地点数组")
        ],
        params=[
            ParamSpec(
                name="location",
                type="string_or_string_array",
                required=True,
                description="最终地点；多地点任务按顺序使用字符串数组",
            )
        ],
        observation_fields=[],
        is_terminal=True,
    )


def _default_definition_from_free(
    tool_name: str,
    *,
    operation: str = "execute",
    operation_description: str = "执行该自由工具描述的外部操作",
) -> ToolDefinition:
    """LLM 提案缺字段时仍生成定义完整、参数稳定的新执行器。"""
    slug = _slug_tool_name(tool_name)
    op = ToolOperation(name=operation or "execute", description=operation_description)
    return ToolDefinition(
        name=slug,
        description=f"执行自由工具 {tool_name!r} 所代表的外部能力。",
        executor=slug,
        usage=(
            "仅当该动作确实访问外部数据、程序或服务并产生新证据时调用；"
            "纯观察、比较、筛选或总结应保留为 reasoning。"
        ),
        operations=[op],
        params=_canonical_param_specs([op]),
        observation_fields=[
            ObservationField(
                name="result",
                type="object",
                nullable=True,
                description="外部执行器实际返回的结构化结果",
            )
        ],
        is_terminal=False,
    )


def _strict_definition(
    proposal: ToolDefinition | None,
    *,
    raw_tool: str,
    operation: str,
    operation_description: str,
) -> ToolDefinition:
    """把自动新建提案收紧为可供后续任务理解和调用的定义。"""
    fallback = _default_definition_from_free(
        raw_tool,
        operation=operation,
        operation_description=operation_description,
    )
    if proposal is None:
        return fallback

    name = _slug_tool_name(proposal.name or raw_tool)
    operations = list(proposal.operations)
    known = {item.name for item in operations}
    normalized_operation = (
        operation.strip().lower().replace("-", "_").replace(" ", "_") or "execute"
    )
    if normalized_operation not in known:
        operations.append(
            ToolOperation(
                name=normalized_operation,
                description=operation_description
                or f"使用 {name} 执行 {normalized_operation}",
            )
        )
    if not operations:
        operations = list(fallback.operations)
    return ToolDefinition(
        name=name,
        description=(proposal.description.strip() or fallback.description),
        executor=(proposal.executor.strip() or name),
        usage=(proposal.usage.strip() or fallback.usage),
        operations=operations,
        params=_canonical_param_specs(operations),
        observation_fields=(
            list(proposal.observation_fields)
            if proposal.observation_fields
            else list(fallback.observation_fields)
        ),
        is_terminal=False,
    )


def _catalog_payload(forest: ToolForest) -> list[dict]:
    return [
        {
            "canonical": tree.canonical.name,
            "executor": tree.canonical.executor,
            "description": tree.canonical.description,
            "usage": tree.canonical.usage,
            "operations": [
                {
                    "name": op.name,
                    "description": op.description,
                    "aliases": op.aliases,
                    "input_schema": (
                        op.input_schema.model_dump() if op.input_schema else None
                    ),
                }
                for op in tree.canonical.operations
            ],
            "variants": tree.variants,
        }
        for tree in forest.trees
        if not tree.canonical.is_terminal
    ]


def llm_semantic_match_batch(
    steps: list[FreeFormStep], forest: ToolForest
) -> dict[str, MatchDecision]:
    """一次调用按完整上下文归并一条轨迹内的未知自由工具。"""
    if not steps:
        return {}
    raw_steps = [
        {
            "raw_tool": step.tool,
            "thought": step.thought,
            "params": step.params,
            "observation": step.observation,
        }
        for step in steps
    ]
    prompt = (
        "你在维护地理定位 Agent 的执行器级 Canonical Tool 目录。\n"
        "工具边界按真实执行器/API/数据库划分，不按自然语言动词或推理意图区分。\n"
        "同一 OSM/Overpass、街景、卫星、天气、图像处理等执行器的不同用途应 map 到同一 canonical，"
        "通过 operation 区分 query/filter/export/compare 等操作，并为 operation 写清用途。\n"
        "如果某一步没有访问外部数据或程序，只是直接看图、合并已有证据、比较、筛选、排除、排名、"
        "形成目标签名或总结，则 action=reasoning；只有置信度>=0.85 时才这样判断。\n"
        "仅当现有目录确实没有相同执行器时 action=create，并提供 proposed_definition："
        "name 必须小写下划线，description 说明能力，executor 说明底层执行器，usage 说明何时调用，"
        "operations 至少包含本次操作及解释。不要因输入字段不同就创建新工具。\n"
        "现有 operation 的 input_schema 已说明每个输入字段的含义、别名、类型和必填关系；"
        "归并时应选择能容纳原始参数语义的 operation，不得为了省事一律选择第一个 operation。\n"
        "每个 raw_tool 必须恰好返回一个 decision，并原样填写 raw_tool；confidence 范围0到1。\n"
        f"现有目录：{json.dumps(_catalog_payload(forest), ensure_ascii=False)}\n"
        f"待归并步骤：{json.dumps(raw_steps, ensure_ascii=False)}"
    )
    result = call_structured(prompt, _BatchMatchResult, lane="llm")
    return {
        item.raw_tool.strip().lower(): item
        for item in result.decisions
        if item.raw_tool.strip()
    }


def _legacy_decision(
    matcher: MatcherFn,
    raw: str,
    forest: ToolForest,
) -> MatchDecision:
    result = matcher(raw, forest)
    if isinstance(result, MatchDecision):
        return result
    if isinstance(result, str) and result.strip():
        return MatchDecision(
            raw_tool=raw,
            action="map",
            canonical_name=result.strip(),
            operation="execute",
            operation_description="执行该 canonical tool 的默认操作",
            confidence=1.0,
            reason="legacy matcher",
        )
    return MatchDecision(
        raw_tool=raw,
        action="create",
        canonical_name=_slug_tool_name(raw),
        operation="execute",
        operation_description="执行该自由工具描述的外部操作",
        confidence=0.0,
        reason="legacy matcher 未命中，使用严格 fallback 定义",
    )


def _demote_to_reasoning(step: FreeFormStep) -> None:
    """Stage 3 安全网：高置信伪工具降回 Thought，不丢掉推理结论。"""
    if step.observation:
        conclusion = json.dumps(step.observation, ensure_ascii=False)
        if conclusion not in step.thought:
            step.thought = f"{step.thought} 推理结论：{conclusion}"
    step.event_type = "reasoning"
    step.tool = None
    step.params = {}
    step.observation = None


def ensure_tool_trees(
    freeform: FreeFormTrajectory,
    trees_path: Path,
    *,
    matcher: MatcherFn | None = None,
) -> ToolForest:
    """按执行器语义匹配或严格新建 tool 树，并持久化目录。"""
    path = Path(trees_path)
    with with_file_lock(path):
        forest = _load_runtime_with_catalog(path)
        final_tree = find_tree_for_name(forest, "final_answer")
        if final_tree is None:
            forest = create_tree(forest, _final_answer_definition())

        unknown: list[FreeFormStep] = []
        seen: set[str] = set()
        for step in freeform.steps:
            if step.event_type != "tool_call" or not step.tool:
                continue
            raw = step.tool.strip()
            key = raw.lower()
            if key in seen or find_tree_for_name(forest, raw) is not None:
                continue
            seen.add(key)
            unknown.append(step)

        if not unknown:
            decisions: dict[str, MatchDecision] = {}
        elif matcher is None:
            decisions = llm_semantic_match_batch(unknown, forest)
        else:
            decisions = {
                step.tool.strip().lower(): _legacy_decision(
                    matcher, step.tool.strip(), forest
                )
                for step in unknown
                if step.tool
            }

        for step in unknown:
            raw = (step.tool or "").strip()
            key = raw.lower()
            decision = decisions.get(key)
            if decision is None:
                decision = MatchDecision(
                    raw_tool=raw,
                    action="create",
                    canonical_name=_slug_tool_name(raw),
                    operation="execute",
                    operation_description="执行该自由工具描述的外部操作",
                    confidence=0.0,
                    reason="批量 matcher 未返回该项，使用严格 fallback 定义",
                )

            if (
                decision.action == "reasoning"
                and decision.confidence >= REASONING_DEMOTION_CONFIDENCE
            ):
                _demote_to_reasoning(step)
                continue

            operation = (
                decision.operation.strip().lower().replace("-", "_").replace(" ", "_")
                or "execute"
            )
            mapped = None
            if decision.action == "map" and decision.canonical_name:
                existing = find_tree_for_name(forest, decision.canonical_name)
                if existing is not None:
                    mapped = existing.canonical.name

            if mapped:
                forest = add_operation(
                    forest,
                    mapped,
                    operation,
                    decision.operation_description,
                )
                forest = add_variant(
                    forest,
                    mapped,
                    raw,
                    operation=operation,
                )
                continue

            definition = _strict_definition(
                decision.proposed_definition,
                raw_tool=raw,
                operation=operation,
                operation_description=decision.operation_description,
            )
            existing = find_tree_for_name(forest, definition.name)
            if existing is not None:
                forest = add_operation(
                    forest,
                    existing.canonical.name,
                    operation,
                    decision.operation_description,
                )
                forest = add_variant(
                    forest,
                    existing.canonical.name,
                    raw,
                    operation=operation,
                )
                continue
            forest = create_tree(
                forest,
                definition,
                initial_variant=(
                    raw if raw.lower() != definition.name.lower() else None
                ),
                initial_operation=operation,
            )

        save_forest(forest, path)
        return forest
