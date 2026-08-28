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
)

MatcherFn = Callable[[str, ToolForest], str | None | MatchDecision]
RESERVED_TERMINAL_TOOLS = {"final_answer", "submit_answer", "done"}
REASONING_DEMOTION_CONFIDENCE = 0.85


class _BatchMatchResult(BaseModel):
    decisions: list[MatchDecision] = Field(default_factory=list)


def _load_catalog(trees_path: Path) -> ToolForest:
    """Load official catalog, or an explicit test/CLI catalog override.

    Never merges a deprecated cross-video ``tool_trees.json`` dump into the
    in-memory forest.
    """
    settings = get_settings()
    path = Path(trees_path)
    catalog_path = Path(settings.TOOL_CATALOG_PATH)
    deprecated_dump = Path(settings.TOOL_TREES_PATH).resolve()
    # Explicit override (tests/CLI) when the path is a real catalog file and not
    # the abandoned runtime dump location.
    if path.is_file() and path.resolve() != deprecated_dump:
        return attach_operation_input_schemas(load_forest(path))
    if catalog_path.is_file():
        return attach_operation_input_schemas(load_forest(catalog_path))
    if path.is_file():
        return attach_operation_input_schemas(load_forest(path))
    return attach_operation_input_schemas(ToolForest(trees=[]))


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
        "工具边界严格服从现有目录：同一执行器内用 operation 区分；目录已明确拆开的执行阶段不得重新合并。"
        "例如 OSM/Overpass 查询使用 osm_query，而对已有 OSM 结果做本地过滤/导出使用 osm_result_process；"
        "卫星影像获取使用 satellite_imagery_query，跨时相或跨候选比较使用 satellite_imagery_compare；"
        "搜索结果页检索使用 web_search，打开并读取网页使用 web_page_read。\n"
        "如果某一步没有访问外部数据或程序，只是直接看图、合并已有证据、比较、筛选、排除、排名、"
        "形成目标签名或总结，则 action=reasoning；只有置信度>=0.85 时才这样判断。\n"
        "仅当现有目录确实没有相同执行器时 action=create，并提供 proposed_definition："
        "name 必须小写下划线，description 说明能力，executor 说明底层执行器，usage 说明何时调用，"
        "operations 至少包含本次操作及解释。不要因输入字段不同就创建新工具。\n"
        "现有 operation 的 input_schema 已说明每个输入字段的含义、别名、类型、必填关系和获取方式；"
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


RECLASSIFY_CONFIDENCE = 0.85


def llm_reclassify_tool_calls(
    steps: list[FreeFormStep], forest: ToolForest
) -> list[MatchDecision]:
    """按 Thought/Params/Observation 为已有 tool_call 重选目录内执行器。

    失败开放：调用方在置信度不足或 canonical 不存在时保留原名。不得把步骤升格自 reasoning。
    """
    if not steps:
        return []
    indexed = [
        {
            "step_index": index,
            "raw_tool": step.tool,
            "thought": step.thought,
            "params": step.params,
            "observation": step.observation,
        }
        for index, step in enumerate(steps, start=1)
    ]
    prompt = (
        "你在校正地理定位 Agent 轨迹中已有 tool_call 的执行器归属。\n"
        "只允许 action=map 到现有目录中的执行器，或在当前工具名已正确时仍 action=map 并返回同一 canonical_name。\n"
        "禁止 action=create；禁止把步骤降为 reasoning；禁止发明新工具名。\n"
        "边界示例：网页关键词检索→web_search；打开并阅读网页→web_page_read；"
        "调历史底图/遥感时相或在底图上查看水系→satellite_imagery_query 或 map_layer_query；"
        "打开街景→streetview_query；在图上量宽/距→distance_bearing_calculator；"
        "勿把后几类默认写成 web_search。\n"
        "每个待审步骤必须恰好返回一个 decision：raw_tool 填原工具名，canonical_name 与 operation 必填，"
        "confidence 为0到1；reason 简述依据。\n"
        f"现有目录：{json.dumps(_catalog_payload(forest), ensure_ascii=False)}\n"
        f"待重分类步骤：{json.dumps(indexed, ensure_ascii=False)}"
    )
    result = call_structured(prompt, _BatchMatchResult, lane="llm")
    return list(result.decisions)


def _apply_reclassification(step: FreeFormStep, decision: MatchDecision) -> bool:
    """Rewrite step.tool/operation when reclassification is confident. Returns True if changed."""

    if decision.action != "map" or decision.confidence < RECLASSIFY_CONFIDENCE:
        return False
    target = (decision.canonical_name or "").strip()
    if not target:
        return False
    current = (step.tool or "").strip()
    operation = (
        decision.operation.strip().lower().replace("-", "_").replace(" ", "_")
        or "execute"
    )
    changed = current.lower() != target.lower()
    step.tool = target
    params = dict(step.params or {})
    if "inputs" in params and isinstance(params.get("inputs"), dict):
        params["operation"] = operation
        if decision.operation_description and not str(
            params.get("purpose") or ""
        ).strip():
            params["purpose"] = decision.operation_description
    else:
        params["operation"] = operation
    step.params = params
    return changed


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
    """按执行器语义匹配或严格新建 tool 树（仅本 task 内存，不落盘）。

    ``trees_path`` 保留签名兼容：可指向测试/CLI 用的目录 JSON 覆盖；
    默认流水线应传入官方 ``TOOL_CATALOG_PATH``，不再读写跨视频 dump。
    """
    forest = _load_catalog(Path(trees_path))
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

    # Second pass: reclassify surviving non-terminal tool_calls by semantics.
    # Never invent calls from reasoning; failure-open keeps the original name.
    reclassify_steps = [
        step
        for step in freeform.steps
        if step.event_type == "tool_call"
        and step.tool
        and step.tool.strip().lower() not in RESERVED_TERMINAL_TOOLS
    ]
    if reclassify_steps and matcher is None:
        try:
            reclass_decisions = llm_reclassify_tool_calls(
                reclassify_steps, forest
            )
        except Exception:  # noqa: BLE001 — fail open
            reclass_decisions = []
        by_raw: dict[str, MatchDecision] = {}
        for decision in reclass_decisions:
            key = decision.raw_tool.strip().lower()
            if key and key not in by_raw:
                by_raw[key] = decision
        for step in reclassify_steps:
            raw = (step.tool or "").strip()
            decision = by_raw.get(raw.lower())
            if decision is None:
                continue
            target = (decision.canonical_name or "").strip()
            tree = find_tree_for_name(forest, target) if target else None
            if tree is None:
                continue
            operation = (
                decision.operation.strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
                or "execute"
            )
            mapped_name = tree.canonical.name
            forest = add_operation(
                forest,
                mapped_name,
                operation,
                decision.operation_description or "语义重分类后的 operation",
            )
            if raw.lower() != mapped_name.lower():
                forest = add_variant(
                    forest, mapped_name, raw, operation=operation
                )
            _apply_reclassification(step, decision)
    elif reclassify_steps and matcher is not None:
        for step in reclassify_steps:
            raw = (step.tool or "").strip()
            result = matcher(raw, forest)
            if isinstance(result, MatchDecision):
                decision = result
            elif isinstance(result, str) and result.strip():
                # String matcher only answers unknown-name mapping; skip known tools.
                if find_tree_for_name(forest, raw) is not None:
                    continue
                decision = MatchDecision(
                    raw_tool=raw,
                    action="map",
                    canonical_name=result.strip(),
                    operation="execute",
                    confidence=1.0,
                    reason="legacy matcher reclassify",
                )
            else:
                continue
            target = (decision.canonical_name or "").strip()
            tree = find_tree_for_name(forest, target) if target else None
            if (
                decision.action != "map"
                or decision.confidence < RECLASSIFY_CONFIDENCE
                or tree is None
            ):
                continue
            operation = (
                decision.operation.strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
                or "execute"
            )
            mapped_name = tree.canonical.name
            forest = add_operation(
                forest,
                mapped_name,
                operation,
                decision.operation_description or "语义重分类后的 operation",
            )
            if raw.lower() != mapped_name.lower():
                forest = add_variant(
                    forest, mapped_name, raw, operation=operation
                )
            _apply_reclassification(step, decision)

    return forest
