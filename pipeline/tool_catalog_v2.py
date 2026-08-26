"""Production helpers for the executor-level Canonical Tool v2 catalog."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pipeline.schemas.tools import (
    ObservationField,
    ParamSpec,
    ToolDefinition,
    ToolForest,
    ToolOperation,
    ToolTree,
)
from pipeline.tool_catalog_v2_proposal import TOOL_SPLITS


def _outer_params(operations: list[ToolOperation]) -> list[ParamSpec]:
    return [
        ParamSpec(
            name="operation",
            required=True,
            description="本次调用执行的 operation",
            allowed_values=[item.name for item in operations],
        ),
        ParamSpec(
            name="purpose",
            required=True,
            description="本次调用要补齐的证据缺口，不得伪装成工具返回",
        ),
        ParamSpec(
            name="inputs",
            type="object",
            required=True,
            description="严格按 operation.input_schema 传给执行器的真实输入",
        ),
    ]


def build_tool_forest_v2(
    source_path: str | Path = "canonical_tool_catalog.json",
) -> ToolForest:
    """Build a runtime ToolForest while preserving v1 operation contracts."""

    from pipeline.stage3_normalize_format.params import (
        attach_operation_input_schemas,
    )
    from pipeline.stage3_normalize_format.trees import load_forest

    source = attach_operation_input_schemas(load_forest(Path(source_path)))
    by_name = {tree.canonical.name: tree for tree in source.trees}

    variant_targets: dict[str, set[str]] = defaultdict(set)
    variant_ops: dict[tuple[str, str], str] = {}
    for new_name, definition in TOOL_SPLITS.items():
        for old_name, operation_names in definition["sources"].items():
            tree = by_name[old_name]
            selected = set(operation_names)
            for variant in tree.variants:
                operation = tree.variant_operations.get(variant.lower())
                if operation in selected:
                    variant_targets[variant.lower()].add(new_name)
                    variant_ops[(new_name, variant.lower())] = operation

    trees: list[ToolTree] = []
    for new_name, definition in TOOL_SPLITS.items():
        operations: list[ToolOperation] = []
        variants: list[str] = []
        mapped_variant_ops: dict[str, str] = {}
        for old_name, operation_names in definition["sources"].items():
            tree = by_name[old_name]
            selected = set(operation_names)
            for operation in tree.canonical.operations:
                if operation.name in selected:
                    operations.append(operation.model_copy(deep=True))
            for variant in tree.variants:
                key = variant.lower()
                if variant_targets.get(key) == {new_name} and key not in {
                    item.lower() for item in variants
                }:
                    variants.append(variant)
                    mapped_variant_ops[key] = variant_ops[(new_name, key)]

        terminal = new_name == "final_answer"
        if terminal:
            params = [
                ParamSpec(
                    name="location",
                    type="string_or_string_array",
                    required=True,
                    description="最终地点；多题按讲解顺序使用字符串数组",
                )
            ]
            observation_fields: list[ObservationField] = []
            usage = "仅作为末步提交；params 必须且只能包含 location，Observation 为 null。"
        else:
            params = _outer_params(operations)
            observation_fields = [
                ObservationField(
                    name="result",
                    type="object",
                    nullable=True,
                    description="执行器实际返回；不得写入假设、计划或模型自行补全的数据",
                )
            ]
            usage = (
                "仅在需要访问该真实执行器并获得新证据时调用；先选择 operation，"
                "再按其 input_schema 填 inputs。缺少执行级参数时按 acquisition_hint 获取，"
                "不得编造坐标、文件、会话或前置结果。"
            )
        trees.append(
            ToolTree(
                canonical=ToolDefinition(
                    name=new_name,
                    description=str(definition["description"]),
                    executor=str(definition["executor"]),
                    usage=usage,
                    operations=operations,
                    params=params,
                    observation_fields=observation_fields,
                    is_terminal=terminal,
                ),
                variants=variants,
                variant_operations=mapped_variant_ops,
            )
        )
    return ToolForest(trees=trees)


def render_tool_contract_guidance(forest: ToolForest) -> str:
    """Render the same compact, deterministic contract for Stage 2 and SFT."""

    lines = [
        "Canonical Tool 调用固定合同：",
        "- tool 必须选择下列执行器级名称；不要因目标对象不同而创造新 Tool。",
        "- params.operation 表示该执行器本次做什么；params.purpose 表示要补齐的证据缺口；",
        "  params.inputs 才是真正传给执行器的字段，必须匹配对应 operation.input_schema。",
        "- semantic 必填字段应从当前 Thought/任务中提取；execution 必填字段若缺失，先按提示调用前置工具、",
        "  引用 $current_image/$current_images 或获取会话/结果，禁止编造路径、坐标和 source_result。",
        "- optional 字段只在有明确来源且执行需要时填写，不要求每次全量填充。",
        "- final_answer 不使用上述三层包装，末步只能是 params={\"location\": ...}。",
        "可用 Tool 与 operation：",
    ]
    for tree in forest.trees:
        tool = tree.canonical
        if tool.is_terminal:
            lines.append(f"- {tool.name}: {tool.description}; operations=submit(location)")
            continue
        operations: list[str] = []
        for operation in tool.operations:
            schema = operation.input_schema
            if schema is None:
                operations.append(operation.name)
                continue
            one_of_names = {
                name for group in schema.required_any for name in group
            }
            semantic = [
                field.name
                for field in schema.fields
                if field.required and field.requirement_level == "semantic"
            ]
            execution = [
                field.name
                for field in schema.fields
                if field.required and field.requirement_level == "execution"
            ]
            optional = [field.name for field in schema.fields if not field.required]
            bits = []
            if semantic:
                bits.append("semantic=" + ",".join(semantic))
            if execution:
                bits.append("execution=" + ",".join(execution))
            if schema.required_any:
                groups = ["|".join(group) for group in schema.required_any]
                bits.append("one_of=" + ";".join(groups))
            if optional:
                bits.append("optional=" + ",".join(optional))
            explained = []
            for field in schema.fields:
                if not field.required and field.name not in one_of_names:
                    continue
                detail = f"{field.name}:{field.type}={field.description}"
                if field.requirement_level == "execution" and field.acquisition_hint:
                    detail += f" 获取:{field.acquisition_hint}"
                explained.append(detail)
            if explained:
                bits.append("字段说明=" + " | ".join(explained))
            suffix = ";".join(bits) if bits else "no_inputs"
            operations.append(f"{operation.name}({suffix})")
        lines.append(
            f"- {tool.name}: {tool.description}; " + " / ".join(operations)
        )
    return "\n".join(lines)
