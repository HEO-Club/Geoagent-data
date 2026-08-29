"""自由 tool → 执行器级 canonical tool 归并 / 受控新建。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

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
    ToolInputSchema,
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
CREATE_CONFIDENCE = 0.85
_SNAKE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ToolResolutionRecord(BaseModel):
    """本 task 未知名判定的落盘记录（不进官方目录）。"""

    raw_tool: str
    disposition: Literal["mapped", "created_proposal", "temporary", "demoted"]
    canonical_name: str | None = None
    temporary_name: str | None = None
    reason: str = ""
    proposed_definition: ToolDefinition | None = None
    create_kind: Literal["new_executor", "new_operation"] | None = None


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


def _normalize_operation_name(operation: str) -> str:
    return operation.strip().lower().replace("-", "_").replace(" ", "_") or "execute"


def _occupied_catalog_keys(forest: ToolForest) -> set[str]:
    occupied: set[str] = set()
    for tree in forest.trees:
        occupied.add(tree.canonical.name.strip().lower())
        executor = tree.canonical.executor.strip().lower()
        if executor:
            occupied.add(executor)
    return occupied


def _input_schema_gate_reason(schema: ToolInputSchema | None) -> str | None:
    """完整 v2 input_schema 才过门；失败时返回原因。"""
    if schema is None:
        return "未过 create schema 门：缺 input_schema"
    if not schema.fields:
        return "未过 create schema 门：input_schema.fields 为空"
    for field in schema.fields:
        if not (field.type or "").strip():
            return f"未过 create schema 门：字段 {field.name} 缺 type"
        if not (field.requirement_level or "").strip():
            return f"未过 create schema 门：字段 {field.name} 缺 requirement_level"
        if not (field.description or "").strip():
            return f"未过 create schema 门：字段 {field.name} 缺 description"
        if not (field.acquisition_hint or "").strip():
            return f"未过 create schema 门：字段 {field.name} 缺 acquisition_hint"
    return None


def _validate_new_executor_proposal(
    definition: ToolDefinition, forest: ToolForest
) -> str | None:
    """新执行器提案程序门；通过返回 None。"""
    name = (definition.name or "").strip()
    if not name or _SNAKE_NAME.fullmatch(name) is None:
        return "未过 create schema 门：name 必须为非空小写下划线"
    occupied = _occupied_catalog_keys(forest)
    if name.lower() in occupied:
        return f"未过 create schema 门：name 与现有 canonical/executor 撞车（{name}）"
    executor = (definition.executor or "").strip()
    if executor and executor.lower() in occupied and executor.lower() != name.lower():
        return (
            "未过 create schema 门：executor 与现有 canonical/executor 撞车"
            f"（{executor}）"
        )
    if not definition.operations:
        return "未过 create schema 门：至少需要一个 operation"
    for op in definition.operations:
        reason = _input_schema_gate_reason(op.input_schema)
        if reason is not None:
            return reason
    return None


def _pick_proposal_operation(
    decision: MatchDecision, operation: str
) -> ToolOperation | None:
    proposal = decision.proposed_definition
    if proposal is None or not proposal.operations:
        return None
    target = _normalize_operation_name(operation)
    for item in proposal.operations:
        if item.name == target:
            return item
    if len(proposal.operations) == 1:
        return proposal.operations[0]
    return None


def _temporary_slug(raw_tool: str, forest: ToolForest) -> str:
    slug = _slug_tool_name(raw_tool)
    occupied = _occupied_catalog_keys(forest)
    if slug.lower() not in occupied and find_tree_for_name(forest, slug) is None:
        return slug
    digest = hashlib.sha256(raw_tool.strip().encode("utf-8")).hexdigest()[:10]
    return f"custom_tool_{digest}"


def _apply_temporary_tree(
    forest: ToolForest,
    *,
    raw_tool: str,
    operation: str,
    operation_description: str,
) -> tuple[ToolForest, str]:
    """失败开放：内存挂临时名，不进提案。"""
    temp_name = _temporary_slug(raw_tool, forest)
    existing = find_tree_for_name(forest, temp_name)
    if existing is not None:
        forest = add_operation(
            forest,
            existing.canonical.name,
            operation,
            operation_description,
        )
        if raw_tool.lower() != existing.canonical.name.lower():
            forest = add_variant(
                forest,
                existing.canonical.name,
                raw_tool,
                operation=operation,
            )
        return forest, existing.canonical.name
    definition = _default_definition_from_free(
        raw_tool,
        operation=operation,
        operation_description=operation_description,
    )
    definition = definition.model_copy(update={"name": temp_name, "executor": temp_name})
    forest = create_tree(
        forest,
        definition,
        initial_variant=(
            raw_tool if raw_tool.lower() != temp_name.lower() else None
        ),
        initial_operation=operation,
    )
    return forest, temp_name


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
        "每个 raw_tool 必须先排除「只是 reasoning」：如果没有访问外部数据或程序，只是直接看图、合并已有证据、"
        "比较、筛选、排除、排名、形成目标签名或总结，则 action=reasoning；只有置信度>=0.85 时才这样判断。"
        "禁止把这类动词型伪工具写成 create。\n"
        "其余情况四选一，并填写 reason：\n"
        "1) action=map：必须是同一执行器，canonical_name 必须是目录内已有名；不得把不同执行器硬塞进已有名。"
        "不要因输入字段不同就拒绝 map。\n"
        "2) action=create 且 create_kind=new_operation：已有执行器需要新的 operation；canonical_name 必须已在目录中；"
        "proposed_definition.operations 给出该 operation 的完整 v2 input_schema"
        "（fields 均含 type、requirement_level、description、acquisition_hint）。\n"
        "3) action=create 且 create_kind=new_executor：目录中没有相同执行器。必须填写 not_catalog_reason，"
        "点名为何不是目录中哪几个近邻执行器。proposed_definition 必须含小写下划线 name、description、"
        "executor、usage、至少一个带完整 v2 input_schema 的 operation。外层调用仍是 operation/purpose/inputs。"
        "禁止因字段不同就新建。\n"
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
        "禁止 action=create；禁止 create_kind=new_operation；禁止把步骤降为 reasoning；禁止发明新工具名。\n"
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


def _apply_unknown_decision(
    forest: ToolForest,
    step: FreeFormStep,
    decision: MatchDecision,
    records: list[ToolResolutionRecord],
) -> ToolForest:
    """对目录外 tool 名执行 map / 合格提案 / 临时名 / 降级。"""
    raw = (step.tool or "").strip()
    if (
        decision.action == "reasoning"
        and decision.confidence >= REASONING_DEMOTION_CONFIDENCE
    ):
        _demote_to_reasoning(step)
        records.append(
            ToolResolutionRecord(
                raw_tool=raw,
                disposition="demoted",
                reason=decision.reason or "高置信伪工具降为 reasoning",
            )
        )
        return forest

    operation = _normalize_operation_name(decision.operation)
    op_desc = decision.operation_description

    if decision.action == "map":
        target = (decision.canonical_name or "").strip()
        existing = find_tree_for_name(forest, target) if target else None
        if existing is not None:
            forest = add_operation(
                forest, existing.canonical.name, operation, op_desc
            )
            forest = add_variant(
                forest, existing.canonical.name, raw, operation=operation
            )
            records.append(
                ToolResolutionRecord(
                    raw_tool=raw,
                    disposition="mapped",
                    canonical_name=existing.canonical.name,
                    reason=decision.reason or "归并到目录内同一执行器",
                )
            )
            return forest
        forest, temp_name = _apply_temporary_tree(
            forest,
            raw_tool=raw,
            operation=operation,
            operation_description=op_desc,
        )
        records.append(
            ToolResolutionRecord(
                raw_tool=raw,
                disposition="temporary",
                canonical_name=temp_name,
                temporary_name=temp_name,
                reason=decision.reason or "map 目标不在目录，记为临时名",
            )
        )
        return forest

    create_kind = decision.create_kind or "new_executor"
    if decision.action == "create" and create_kind == "new_operation":
        parent_name = (decision.canonical_name or "").strip()
        parent = find_tree_for_name(forest, parent_name) if parent_name else None
        proposed_op = _pick_proposal_operation(decision, operation)
        fail: str | None = None
        if parent is None:
            fail = "未过 create schema 门：new_operation 的 canonical 不在目录"
        elif proposed_op is None:
            fail = "未过 create schema 门：缺 input_schema"
        else:
            fail = _input_schema_gate_reason(proposed_op.input_schema)
        if (
            fail is None
            and parent is not None
            and proposed_op is not None
            and proposed_op.name
            in {item.name for item in parent.canonical.operations}
        ):
            forest = add_variant(
                forest, parent.canonical.name, raw, operation=proposed_op.name
            )
            records.append(
                ToolResolutionRecord(
                    raw_tool=raw,
                    disposition="mapped",
                    canonical_name=parent.canonical.name,
                    reason=decision.reason or "operation 已存在，按 map 处理",
                )
            )
            return forest
        if fail is None and decision.confidence < CREATE_CONFIDENCE:
            fail = (
                "未过 create schema 门：create 置信度不足"
                f"（{decision.confidence:.2f}）"
            )
        if fail is None and parent is not None and proposed_op is not None:
            op_name = proposed_op.name or operation
            forest = add_operation(
                forest,
                parent.canonical.name,
                op_name,
                proposed_op.description or op_desc,
                input_schema=proposed_op.input_schema,
            )
            forest = add_variant(
                forest, parent.canonical.name, raw, operation=op_name
            )
            stub = ToolDefinition(
                name=parent.canonical.name,
                description=parent.canonical.description,
                executor=parent.canonical.executor,
                usage=parent.canonical.usage,
                operations=[proposed_op],
                params=list(parent.canonical.params),
                observation_fields=list(parent.canonical.observation_fields),
                is_terminal=False,
            )
            records.append(
                ToolResolutionRecord(
                    raw_tool=raw,
                    disposition="created_proposal",
                    canonical_name=parent.canonical.name,
                    reason=decision.reason or "已有执行器新增 operation 提案",
                    proposed_definition=stub,
                    create_kind="new_operation",
                )
            )
            return forest
        forest, temp_name = _apply_temporary_tree(
            forest,
            raw_tool=raw,
            operation=operation,
            operation_description=op_desc,
        )
        records.append(
            ToolResolutionRecord(
                raw_tool=raw,
                disposition="temporary",
                canonical_name=temp_name,
                temporary_name=temp_name,
                reason=fail or "未过 create schema 门：缺 input_schema",
            )
        )
        return forest

    if decision.action == "create":
        definition = _strict_definition(
            decision.proposed_definition,
            raw_tool=raw,
            operation=operation,
            operation_description=op_desc,
        )
        fail = _validate_new_executor_proposal(definition, forest)
        if fail is None and decision.confidence < CREATE_CONFIDENCE:
            fail = (
                "未过 create schema 门：create 置信度不足"
                f"（{decision.confidence:.2f}）"
            )
        if fail is None:
            forest = create_tree(
                forest,
                definition,
                initial_variant=(
                    raw if raw.lower() != definition.name.lower() else None
                ),
                initial_operation=operation,
            )
            records.append(
                ToolResolutionRecord(
                    raw_tool=raw,
                    disposition="created_proposal",
                    canonical_name=definition.name,
                    reason=decision.reason or "合格新建执行器提案",
                    proposed_definition=definition,
                    create_kind="new_executor",
                )
            )
            return forest
        forest, temp_name = _apply_temporary_tree(
            forest,
            raw_tool=raw,
            operation=operation,
            operation_description=op_desc,
        )
        records.append(
            ToolResolutionRecord(
                raw_tool=raw,
                disposition="temporary",
                canonical_name=temp_name,
                temporary_name=temp_name,
                reason=fail,
            )
        )
        return forest

    forest, temp_name = _apply_temporary_tree(
        forest,
        raw_tool=raw,
        operation=operation,
        operation_description=op_desc,
    )
    records.append(
        ToolResolutionRecord(
            raw_tool=raw,
            disposition="temporary",
            canonical_name=temp_name,
            temporary_name=temp_name,
            reason=decision.reason or "未达 reasoning 降级阈值，使用临时名继续",
        )
    )
    return forest


def ensure_tool_trees(
    freeform: FreeFormTrajectory,
    trees_path: Path,
    *,
    matcher: MatcherFn | None = None,
    resolution_records: list[ToolResolutionRecord] | None = None,
) -> ToolForest:
    """按执行器语义匹配、合格提案或临时名（仅本 task 内存，不改官方目录）。

    ``trees_path`` 保留签名兼容：可指向测试/CLI 用的目录 JSON 覆盖；
    默认流水线应传入官方 ``TOOL_CATALOG_PATH``，不再读写跨视频 dump。
    ``resolution_records`` 若传入则追加本 task 判定记录，供 mapping/提案落盘。
    """
    forest = _load_catalog(Path(trees_path))
    final_tree = find_tree_for_name(forest, "final_answer")
    if final_tree is None:
        forest = create_tree(forest, _final_answer_definition())

    records: list[ToolResolutionRecord] = []
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
        forest = _apply_unknown_decision(forest, step, decision, records)

    if resolution_records is not None:
        resolution_records.extend(records)

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
            if (
                decision.action != "map"
                or decision.confidence < RECLASSIFY_CONFIDENCE
            ):
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
