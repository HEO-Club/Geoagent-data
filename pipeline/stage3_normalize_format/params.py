"""Canonical Tool 的 operation 级参数归一与宽容校验。"""

from __future__ import annotations

import re
from typing import Any

from pipeline.schemas.tools import (
    InputFieldSpec,
    ParameterAuditIssue,
    ParameterRepairAction,
    ToolForest,
    ToolInputSchema,
    ToolParameterAudit,
    ToolTree,
)
from pipeline.stage3_normalize_format.trees import find_tree_for_name


def _norm_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", text)


def resolve_operation_alias(tree: ToolTree, operation: str) -> tuple[str, bool]:
    """把 compare 等常见简写解析为目录中的正式 operation。"""

    raw = _norm_name(operation)
    for item in tree.canonical.operations:
        if raw == item.name:
            return item.name, False
        if raw in {_norm_name(alias) for alias in item.aliases}:
            return item.name, True
    return raw, False


def initial_parameter_context(image_paths: list[str] | None = None) -> dict[str, Any]:
    """为参数归一提供可显式引用的运行时上下文。"""

    context: dict[str, Any] = {}
    if image_paths:
        context["current_image"] = "$current_image"
        context["current_images"] = "$current_images"
    return context


def update_parameter_context(
    context: dict[str, Any], audit: ToolParameterAudit
) -> None:
    """用一次真实 Tool 调用更新后续可引用的结果、区域和会话。"""

    index = audit.step_index
    context["previous_tool_result"] = f"$step_{index}_tool_result"
    if "area" in audit.normalized_inputs:
        context["active_area"] = audit.normalized_inputs["area"]
    if audit.tool == "streetview_query" and audit.operation == "open":
        context["active_session"] = f"$step_{index}_streetview_session"
    if audit.operation in {
        "query",
        "search",
        "browse",
        "poi_search",
        "keyword_search",
        "site_search",
        "enumerate",
    }:
        context["previous_candidates"] = f"$step_{index}_tool_result.candidates"
        context["previous_coordinates"] = f"$step_{index}_tool_result.coordinates"


def _type_matches(value: Any, expected: str, item_type: str | None) -> bool:
    expected = expected.lower()
    if expected == "any":
        return True
    if expected == "string":
        return isinstance(value, str) and bool(value.strip())
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        if not isinstance(value, list):
            return False
        if not item_type:
            return True
        return all(_type_matches(item, item_type, None) for item in value)
    if expected == "string_or_array":
        return _type_matches(value, "string", None) or _type_matches(
            value, "array", item_type or "string"
        )
    if expected == "string_or_object":
        return _type_matches(value, "string", None) or isinstance(value, dict)
    if expected == "number_or_string":
        return _type_matches(value, "number", None) or _type_matches(
            value, "string", None
        )
    if expected == "location":
        return (
            _type_matches(value, "string", None)
            or isinstance(value, dict)
            or (
                isinstance(value, list)
                and bool(value)
                and all(isinstance(item, (str, int, float, dict)) for item in value)
            )
        )
    return True


def _field_alias_map(schema: ToolInputSchema) -> dict[str, InputFieldSpec]:
    result: dict[str, InputFieldSpec] = {}
    for field in schema.fields:
        for name in [field.name, *field.aliases]:
            result[_norm_name(name)] = field
    return result


def _default_acquisition_hint(name: str) -> str:
    hints = {
        "image": (
            "优先引用 $current_image；若只需要局部特征，先调用 image_process.crop/zoom "
            "生成截图并把返回的图片 ID 传入，禁止编造文件名。"
        ),
        "images": "从当前输入图或前置 crop/capture 返回值收集真实图片 ID，至少两张。",
        "region": "在当前图片上明确裁剪框或可复现的目标区域描述。",
        "area": "从 working_scope、当前候选或前一步地图结果提取真实区域；来源没有时不得补写。",
        "query": "从本步 purpose/Thought 中提取明确检索对象和约束，不得只写“查询一下”。",
        "source_result": (
            "引用产生该数据的前置 Tool 返回 ID；若尚未执行前置查询，先调用对应 query/retrieve。"
        ),
        "session": "引用前一步 open/retrieve 返回的活动会话 ID；没有会话时先执行打开操作。",
        "candidates": "引用前一步候选生成/搜索返回的完整候选列表，不能只传候选数量。",
        "time_range": "从字幕、task 或已有结果提取真实时间范围；来源没有时不得猜测。",
        "datetime": "组合来源中明确给出的日期、时间和时区；缺任一关键项时标记待补充。",
        "coordinates": "引用地图/OSM/地理编码前一步真实返回的坐标，不得由模型编造。",
        "direction": "从图像朝向、道路方向或用户指定的导航方向中提取可执行方向。",
        "layers": "根据目的明确要加载的图层，如 hydrology、terrain、historical_map。",
        "format": "根据下游用途选择明确输出格式；无法判断时由调用计划显式指定。",
        "measurement": "根据 Thought 明确要测量 distance、angle、ratio、area 或 color。",
        "site": "从任务指定平台/域名中提取；未限定站点时应改用 keyword_search。",
        "template": "从已提取的视觉证据中形成可复现模板，不得用最终答案反向构造。",
    }
    return hints.get(
        name,
        f"从当前 Thought、task 或前置工具真实结果中取得 {name}；来源没有时不得编造。",
    )


def _repair_strategy(field: InputFieldSpec) -> str:
    if field.name in {"image", "images", "region"}:
        return "request_or_capture_input"
    if field.name in {"source_result", "session", "coordinates", "candidates"}:
        return "call_prerequisite_tool"
    if field.requirement_level == "semantic":
        return "extract_from_thought"
    return "manual_review"


def _validate_field(
    field: InputFieldSpec,
    value: Any,
    issues: list[ParameterAuditIssue],
) -> None:
    if not _type_matches(value, field.type, field.item_type):
        issues.append(
            ParameterAuditIssue(
                code="input_type_mismatch",
                severity="error",
                field=field.name,
                message=f"{field.name} 应为 {field.type}，实际为 {type(value).__name__}",
            )
        )
        return
    if field.allowed_values and isinstance(value, str):
        allowed = {_norm_name(item) for item in field.allowed_values}
        if _norm_name(value) not in allowed:
            issues.append(
                ParameterAuditIssue(
                    code="input_value_not_allowed",
                    severity="error",
                    field=field.name,
                    message=f"{field.name}={value!r} 不在允许值 {field.allowed_values} 中",
                )
            )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if field.minimum is not None and value < field.minimum:
            issues.append(
                ParameterAuditIssue(
                    code="input_below_minimum",
                    severity="error",
                    field=field.name,
                    message=f"{field.name}={value} 小于最小值 {field.minimum}",
                )
            )
        if field.maximum is not None and value > field.maximum:
            issues.append(
                ParameterAuditIssue(
                    code="input_above_maximum",
                    severity="error",
                    field=field.name,
                    message=f"{field.name}={value} 大于最大值 {field.maximum}",
                )
            )


def normalize_and_validate_tool_inputs(
    forest: ToolForest,
    *,
    tool: str,
    operation: str,
    inputs: dict[str, Any],
    step_index: int,
    available_context: dict[str, Any] | None = None,
) -> ToolParameterAudit:
    """按 Tool/operation 合同归一输入；额外字段保存在 extensions，不轻易丢弃。"""

    tree = find_tree_for_name(forest, tool)
    issues: list[ParameterAuditIssue] = []
    if tree is None:
        return ToolParameterAudit(
            step_index=step_index,
            tool=tool,
            raw_operation=operation,
            operation=_norm_name(operation) or "execute",
            raw_inputs=dict(inputs),
            normalized_inputs=dict(inputs),
            valid=False,
            readiness="invalid",
            issues=[
                ParameterAuditIssue(
                    code="unknown_canonical_tool",
                    severity="error",
                    message=f"未找到 Canonical Tool：{tool}",
                )
            ],
        )

    canonical_operation, repaired = resolve_operation_alias(tree, operation)
    if repaired:
        issues.append(
            ParameterAuditIssue(
                code="operation_alias_normalized",
                severity="info",
                field="operation",
                message=f"operation {operation!r} 已归一为 {canonical_operation!r}",
            )
        )
    op = next(
        (item for item in tree.canonical.operations if item.name == canonical_operation),
        None,
    )
    if op is None:
        return ToolParameterAudit(
            step_index=step_index,
            tool=tree.canonical.name,
            raw_operation=operation,
            operation=canonical_operation,
            raw_inputs=dict(inputs),
            normalized_inputs=dict(inputs),
            valid=False,
            readiness="invalid",
            issues=issues
            + [
                ParameterAuditIssue(
                    code="unknown_operation",
                    severity="error",
                    field="operation",
                    message=f"{tree.canonical.name} 不支持 operation={operation!r}",
                )
            ],
        )

    schema = op.input_schema
    if schema is None:
        return ToolParameterAudit(
            step_index=step_index,
            tool=tree.canonical.name,
            raw_operation=operation,
            operation=canonical_operation,
            raw_inputs=dict(inputs),
            normalized_inputs=dict(inputs),
            valid=True,
            readiness="ready",
            issues=issues
            + [
                ParameterAuditIssue(
                    code="input_schema_unavailable",
                    severity="warning",
                    message=f"{tree.canonical.name}.{canonical_operation} 尚无 input_schema",
                )
            ],
        )

    alias_map = _field_alias_map(schema)
    normalized: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for raw_name, value in inputs.items():
        field = alias_map.get(_norm_name(raw_name))
        if field is None:
            extras[str(raw_name)] = value
            continue
        if field.name in normalized and normalized[field.name] != value:
            issues.append(
                ParameterAuditIssue(
                    code="input_alias_conflict",
                    severity="error",
                    field=field.name,
                    message=f"字段 {field.name} 的多个别名给出了冲突值",
                )
            )
            continue
        normalized[field.name] = value
        if str(raw_name) != field.name:
            issues.append(
                ParameterAuditIssue(
                    code="input_alias_normalized",
                    severity="info",
                    field=field.name,
                    message=f"输入字段 {raw_name!r} 已归一为 {field.name!r}",
                )
            )

    context = dict(available_context or {})
    fields_by_name = {field.name: field for field in schema.fields}
    repair_actions: list[ParameterRepairAction] = []
    context_injected = False

    def resolve_from_context(field: InputFieldSpec) -> bool:
        nonlocal context_injected
        source = next(
            (name for name in field.context_sources if name in context), None
        )
        if source is None:
            return False
        value = context[source]
        normalized[field.name] = value
        context_injected = True
        guidance = field.acquisition_hint or _default_acquisition_hint(field.name)
        issues.append(
            ParameterAuditIssue(
                code="required_input_from_context",
                severity="info",
                field=field.name,
                message=f"{field.name} 已由运行时上下文 {source} 补为 {value!r}",
                requirement_level=field.requirement_level,
                repairable=True,
                guidance=guidance,
            )
        )
        repair_actions.append(
            ParameterRepairAction(
                field=field.name,
                requirement_level=field.requirement_level,
                strategy="use_context",
                guidance=guidance,
                suggested_value=value,
            )
        )
        return True

    def add_missing(field: InputFieldSpec) -> None:
        guidance = field.acquisition_hint or _default_acquisition_hint(field.name)
        issues.append(
            ParameterAuditIssue(
                code="required_input_missing",
                severity="error",
                field=field.name,
                message=f"缺少最低输入 {field.name}：{field.description}",
                requirement_level=field.requirement_level,
                repairable=True,
                guidance=guidance,
            )
        )
        repair_actions.append(
            ParameterRepairAction(
                field=field.name,
                requirement_level=field.requirement_level,
                strategy=_repair_strategy(field),
                guidance=guidance,
            )
        )

    for field in schema.fields:
        if (
            field.required
            and field.name not in normalized
            and not resolve_from_context(field)
        ):
            add_missing(field)
        if field.name in normalized:
            _validate_field(field, normalized[field.name], issues)

    for group in schema.required_any:
        if any(name in normalized for name in group):
            continue
        resolved = False
        for name in group:
            field = fields_by_name.get(name)
            if field is not None and resolve_from_context(field):
                resolved = True
                break
        if resolved:
            continue
        candidates = [fields_by_name[name] for name in group if name in fields_by_name]
        guidance = "；".join(
            f"{field.name}: {field.acquisition_hint}" for field in candidates
        )
        issues.append(
            ParameterAuditIssue(
                code="required_input_group_missing",
                severity="error",
                field="|".join(group),
                message=f"至少需要提供以下字段之一：{group}",
                requirement_level=(
                    "semantic"
                    if any(field.requirement_level == "semantic" for field in candidates)
                    else "execution"
                ),
                repairable=True,
                guidance=guidance,
            )
        )
        repair_actions.append(
            ParameterRepairAction(
                field="|".join(group),
                requirement_level=(
                    "semantic"
                    if any(field.requirement_level == "semantic" for field in candidates)
                    else "execution"
                ),
                strategy=(
                    "extract_from_thought"
                    if any(field.requirement_level == "semantic" for field in candidates)
                    else "call_prerequisite_tool"
                ),
                guidance=guidance,
            )
        )
    for group in schema.mutually_exclusive:
        present = [name for name in group if name in normalized]
        if len(present) > 1:
            issues.append(
                ParameterAuditIssue(
                    code="mutually_exclusive_inputs",
                    severity="error",
                    field="|".join(group),
                    message=f"以下字段不能同时提供：{present}",
                )
            )

    if extras:
        if schema.allow_extra:
            normalized["extensions"] = extras
            issues.append(
                ParameterAuditIssue(
                    code="extra_inputs_preserved",
                    severity="warning",
                    field="extensions",
                    message=f"未识别字段已保存在 extensions：{sorted(extras)}",
                )
            )
        else:
            issues.append(
                ParameterAuditIssue(
                    code="extra_inputs_not_allowed",
                    severity="error",
                    message=f"不允许的额外字段：{sorted(extras)}",
                )
            )

    invalid_codes = {
        "unknown_canonical_tool",
        "unknown_operation",
        "input_alias_conflict",
        "input_type_mismatch",
        "input_value_not_allowed",
        "mutually_exclusive_inputs",
        "extra_inputs_not_allowed",
    }
    has_invalid = any(issue.code in invalid_codes for issue in issues)
    has_missing = any(
        issue.code in {"required_input_missing", "required_input_group_missing"}
        for issue in issues
    )
    if has_invalid:
        readiness = "invalid"
    elif has_missing:
        readiness = "repairable"
    elif context_injected:
        readiness = "context_resolvable"
    else:
        readiness = "ready"
    return ToolParameterAudit(
        step_index=step_index,
        tool=tree.canonical.name,
        raw_operation=operation,
        operation=canonical_operation,
        raw_inputs=dict(inputs),
        normalized_inputs=normalized,
        valid=readiness in {"ready", "context_resolvable"},
        readiness=readiness,
        issues=issues,
        repair_actions=repair_actions,
    )


def _field(
    name: str,
    description: str,
    *,
    type: str = "string",
    required: bool = False,
    requirement_level: str | None = None,
    acquisition_hint: str = "",
    context_sources: list[str] | None = None,
    context_default: Any = None,
    aliases: list[str] | None = None,
    allowed_values: list[str] | None = None,
    item_type: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    example: Any = None,
) -> InputFieldSpec:
    semantic_required = {
        "area",
        "query",
        "candidates",
        "template",
        "time_range",
        "datetime",
        "direction",
        "site",
        "layers",
        "measurement",
    }
    default_sources = {
        "image": ["current_image"],
        "images": ["current_images"],
        "source_result": ["previous_tool_result"],
        "session": ["active_session"],
        "candidates": ["previous_candidates"],
        "coordinates": ["previous_coordinates"],
        "area": ["active_area"],
    }
    default_values = {
        "image": "$current_image",
        "images": "$current_images",
        "source_result": "$previous_tool_result",
        "session": "$active_session",
        "candidates": "$previous_candidates",
        "coordinates": "$previous_coordinates",
        "area": "$active_area",
    }
    return InputFieldSpec(
        name=name,
        type=type,
        required=required,
        requirement_level=requirement_level
        or (
            "semantic"
            if required and name in semantic_required
            else ("execution" if required else "optional")
        ),
        description=description,
        acquisition_hint=acquisition_hint or _default_acquisition_hint(name),
        context_sources=list(
            default_sources.get(name, [])
            if context_sources is None
            else context_sources
        ),
        context_default=(
            default_values.get(name) if context_default is None else context_default
        ),
        aliases=list(aliases or []),
        allowed_values=list(allowed_values or []),
        item_type=item_type,
        minimum=minimum,
        maximum=maximum,
        example=example,
    )


def _schema(
    description: str,
    fields: list[InputFieldSpec],
    *,
    required_any: list[list[str]] | None = None,
    mutually_exclusive: list[list[str]] | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> ToolInputSchema:
    return ToolInputSchema(
        description=description,
        fields=fields,
        required_any=list(required_any or []),
        mutually_exclusive=list(mutually_exclusive or []),
        allow_extra=True,
        examples=list(examples or []),
    )


def _image(required: bool = True) -> InputFieldSpec:
    return _field(
        "image",
        "待处理图片的路径、图片 ID 或当前输入图片引用；不得填写不存在的图片。",
        required=required,
        requirement_level="execution" if required else "optional",
        context_sources=["current_image"],
        context_default="$current_image",
        aliases=["图片", "图像", "image_path", "image_id", "对象", "target_image"],
        example="input_image_1",
    )


def _region(required: bool = False) -> InputFieldSpec:
    return _field(
        "region",
        "图片中的目标区域，可用 [x1,y1,x2,y2]、命名区域或可复现的区域描述表示。",
        type="string_or_array",
        required=required,
        aliases=["区域", "裁剪区域", "目标区域", "roi", "bbox", "crop_region"],
        example="bridge_tower_top",
    )


def _area(
    required: bool = False, *, extra_aliases: list[str] | None = None
) -> InputFieldSpec:
    return _field(
        "area",
        "地理查询范围，可为标准地名、行政区、bbox 或中心点加半径；应保留原始范围精度。",
        type="location",
        required=required,
        requirement_level="semantic" if required else "optional",
        context_sources=["active_area"],
        aliases=["区域", "地区", "地点", "候选区域", "检索范围", "搜索区域", "省域范围", "region", "target_area", "search_area", "sample_area", "location", "cities", *(extra_aliases or [])],
        example="郑州附近黄河沿线",
    )


def _query(required: bool = False) -> InputFieldSpec:
    return _field(
        "query",
        "提交给该服务的检索词、自然语言查询或明确对象描述。",
        type="string_or_array",
        required=required,
        requirement_level="semantic" if required else "optional",
        aliases=["关键词", "检索词", "查询词", "检索对象", "对象", "目标", "特征组合", "筛选条件", "material", "keywords", "问题", "q", "search_query"],
        example="铁路桥 石栏 历史照片",
    )


def _time_range(required: bool = False) -> InputFieldSpec:
    return _field(
        "time_range",
        "查询时间点或起止范围；允许明确年份、日期或“最早可用”等可执行服务条件。",
        type="string_or_object",
        required=required,
        requirement_level="semantic" if required else "optional",
        aliases=["时期", "时间", "年份", "年代", "年代范围", "日期", "影像时期", "影像年份", "date_range", "year"],
        example="1980-1995",
    )


def _source_result(required: bool = True) -> InputFieldSpec:
    return _field(
        "source_result",
        "前一步真实工具返回的结果 ID、数据集引用或已加载图层；不能用尚未执行的假想结果。",
        type="string_or_object",
        required=required,
        requirement_level="execution" if required else "optional",
        context_sources=["previous_tool_result"],
        context_default="$previous_tool_result",
        aliases=["结果", "已有结果", "输入结果", "数据源", "source", "result_id", "layer"],
        example="osm_result_01",
    )


def _operation_schemas() -> dict[tuple[str, str], ToolInputSchema]:
    """17 个 Canonical Tool 的 operation 级输入合同。"""

    schemas: dict[tuple[str, str], ToolInputSchema] = {}

    # image_process
    schemas[("image_process", "enhance")] = _schema(
        "对真实输入图片执行可复现增强；直接肉眼观察不应调用本 operation。",
        [
            _image(),
            _region(),
            _field("adjustments", "增强设置，如亮度、对比度、阴影或锐度。", type="object", aliases=["调整", "增强参数", "settings"], example={"brightness": 0.2}),
            _field("output_format", "增强图输出格式。", allowed_values=["png", "jpeg", "webp"], aliases=["格式", "format"], example="png"),
        ],
    )
    schemas[("image_process", "crop")] = _schema(
        "从输入图片裁取指定区域。",
        [_image(), _region(required=True), _field("padding", "裁剪区域外扩像素或比例。", type="number", aliases=["边距", "padding_px"], minimum=0)],
    )
    schemas[("image_process", "zoom")] = _schema(
        "放大指定图片区域，产生供后续识别使用的派生图。",
        [_image(), _region(required=True), _field("scale", "放大倍数。", type="number", aliases=["倍数", "zoom", "zoom_factor"], minimum=1, example=2)],
    )
    schemas[("image_process", "measure")] = _schema(
        "测量图片中的像素距离、角度、比例或其他可计算视觉量。",
        [_image(), _region(), _field("measurement", "要计算的量。", required=True, aliases=["测量项", "检查项", "metric"], allowed_values=["distance", "angle", "ratio", "area", "color"], example="angle"), _field("reference", "尺度、参照线或已知尺寸。", type="string_or_object", aliases=["参照", "基准", "reference_line"])],
    )
    schemas[("image_process", "compare")] = _schema(
        "对两张或多张真实图片执行程序化差异或特征比较。",
        [_field("images", "参与比较的图片路径或 ID，至少两张。", type="array", item_type="string", required=True, aliases=["图片", "图像列表", "image_paths", "candidates"]), _field("method", "比较方法。", aliases=["方式", "比对方式"], allowed_values=["feature", "pixel", "histogram", "geometry", "auto"], example="feature"), _region()],
    )

    # OCR / reverse image search
    schemas[("ocr_read", "recognize")] = _schema(
        "识别图片区域中的文字或数字。",
        [_image(), _region(), _field("languages", "候选语言代码列表；未知时可省略。", type="array", item_type="string", aliases=["语言", "language"], example=["zh", "en"]), _field("text_kind", "预期文字类型。", aliases=["文字类型", "kind"], allowed_values=["natural_text", "number", "road_sign", "address", "auto"])],
    )
    schemas[("ocr_read", "decode")] = _schema(
        "解码二维码、条码或其他机器编码。",
        [_image(), _region(), _field("code_types", "允许识别的编码类型。", type="array", item_type="string", aliases=["编码类型", "types"], example=["qr", "barcode"])],
    )
    schemas[("reverse_image_search", "search")] = _schema(
        "提交完整图片到反向搜图服务。",
        [_image(), _field("engines", "可选搜索引擎列表。", type="array", item_type="string", aliases=["引擎", "providers"]), _field("top_k", "最多返回结果数。", type="integer", aliases=["结果数", "limit"], minimum=1, maximum=100, example=10)],
    )
    schemas[("reverse_image_search", "search_crop")] = _schema(
        "提交指定裁剪区域进行局部反向搜图。",
        [_image(), _region(required=True), _field("engines", "可选搜索引擎列表。", type="array", item_type="string", aliases=["引擎", "providers"]), _field("top_k", "最多返回结果数。", type="integer", aliases=["结果数", "limit"], minimum=1, maximum=100)],
    )

    # web_search
    schemas[("web_search", "keyword_search")] = _schema(
        "在开放网页中按关键词检索。",
        [_query(required=True), _field("domains", "可选域名白名单。", type="array", item_type="string", aliases=["站点", "网站", "site", "domain"]), _field("language", "结果语言代码。", aliases=["语言", "lang"]), _time_range(), _field("top_k", "最多返回结果数。", type="integer", aliases=["结果数", "limit"], minimum=1, maximum=100)],
    )
    schemas[("web_search", "site_search")] = _schema(
        "在指定网站或内容平台内部检索。",
        [_query(required=True), _field("site", "限定检索的网站、域名或平台。", required=True, aliases=["站点", "网站", "平台", "domain"]), _time_range(), _field("top_k", "最多返回结果数。", type="integer", aliases=["结果数", "limit"], minimum=1, maximum=100)],
    )
    schemas[("web_search", "open_result")] = _schema(
        "打开已返回的网页结果并读取内容。",
        [_field("url", "目标网页 URL。", aliases=["链接", "网址"]), _field("result_id", "前一步搜索结果 ID。", aliases=["结果ID", "搜索结果", "id"]), _field("extract", "需要抽取的内容范围。", type="string_or_array", aliases=["抽取项", "字段"])],
        required_any=[["url", "result_id"]],
    )

    # map_query
    schemas[("map_query", "poi_search")] = _schema(
        "在地图服务中按名称或类别搜索 POI。",
        [_area(), _query(), _field("categories", "POI 类别或类型。", type="string_or_array", aliases=["类别", "类型", "poi_type"]), _field("radius_m", "以中心点检索时的半径（米）。", type="number", aliases=["半径", "radius"], minimum=0), _field("top_k", "最多返回候选数。", type="integer", aliases=["结果数", "limit"], minimum=1, maximum=500)],
        required_any=[["area", "query", "categories"]],
    )
    schemas[("map_query", "geocode")] = _schema(
        "在地名、地址和坐标表达之间转换。",
        [_query(required=True), _area(), _field("direction", "正向或反向地理编码。", aliases=["方向", "mode"], allowed_values=["forward", "reverse", "auto"], example="auto")],
    )
    schemas[("map_query", "route")] = _schema(
        "查询两点或多点之间的道路与路线关系。",
        [_field("origin", "路线起点。", type="string_or_object", required=True, aliases=["起点", "from"]), _field("destination", "路线终点。", type="string_or_object", required=True, aliases=["终点", "to"]), _field("waypoints", "可选途经点。", type="array", aliases=["途经点", "via"]), _field("travel_mode", "出行方式。", aliases=["交通方式", "mode"], allowed_values=["drive", "walk", "bike", "transit", "any"])],
    )
    schemas[("map_query", "load_layer")] = _schema(
        "加载指定区域的水系、地形、道路、行政区或历史地图图层。",
        [_area(required=True), _field("layers", "需要加载的一个或多个图层。", type="string_or_array", required=True, aliases=["图层", "数据层", "layer", "地图类型"], example=["hydrology"]), _time_range(), _field("provider", "地图或数据服务提供方。", aliases=["数据源", "服务"] )],
    )
    schemas[("map_query", "browse")] = _schema(
        "在指定区域浏览地图并返回符合描述的候选要素。",
        [_area(required=True), _query(), _field("filters", "附加属性或空间过滤条件。", type="object", aliases=["筛选条件", "条件"]), _field("top_k", "最多返回候选数。", type="integer", aliases=["结果数", "limit"], minimum=1, maximum=500)],
    )

    # osm_query: 默认传结构化约束；overpass_ql 仅在确有原始代码时可选。
    schemas[("osm_query", "query")] = _schema(
        "查询 OSM 要素。默认传 area/tags 等结构化约束，由执行器生成 Overpass QL；只有轨迹确实编写了代码时才传 overpass_ql。",
        [_area(), _field("bbox", "WGS84 查询框 [west,south,east,north]。", type="array", item_type="number", aliases=["边界框", "bounds", "extent"]), _field("center", "中心坐标或地名，与 radius_m 配合。", type="string_or_object", aliases=["中心点", "中心"]), _field("radius_m", "中心点查询半径（米）。", type="number", aliases=["半径", "radius"], minimum=0), _field("tags", "OSM 标签过滤，如 {\"bridge\":\"yes\"}。", type="object", aliases=["标签", "osm_tags", "filters"], example={"bridge": "yes"}), _field("feature_types", "希望检索的自然语言要素类型；执行器负责映射为 OSM 标签。", type="string_or_array", aliases=["特征", "目标类型", "要素类型", "categories"], example=["桥梁"]), _field("element_types", "OSM 元素类型。", type="array", item_type="string", aliases=["元素类型", "types"], example=["node", "way", "relation"]), _field("spatial_relation", "空间关系约束，如 within/near/intersects。", aliases=["空间关系", "关系"]), _field("return_geometry", "是否返回完整几何。", type="boolean", aliases=["返回几何", "geometry"], example=True), _field("limit", "最多返回要素数。", type="integer", aliases=["结果数", "top_k"], minimum=1, maximum=10000), _field("overpass_ql", "可选的完整 Overpass QL；不是默认必填参数。", aliases=["代码", "查询代码", "query_code", "raw_query"])],
        required_any=[["area", "bbox", "center", "overpass_ql"]],
        examples=[{"area": "郑州市", "tags": {"bridge": "yes"}, "return_geometry": True}],
    )
    schemas[("osm_query", "filter")] = _schema(
        "在已返回的 OSM 结果上继续按属性或几何关系筛选。",
        [_source_result(), _field("filters", "OSM 属性过滤条件。", type="object", required=True, aliases=["筛选条件", "标签", "conditions"]), _field("spatial_filter", "空间过滤条件，如距河流多少米或与道路相交。", type="object", aliases=["空间条件", "geometry_filter"])],
    )
    schemas[("osm_query", "export")] = _schema(
        "把已有 OSM 结果导出为指定格式。",
        [_source_result(), _field("format", "导出格式。", required=True, aliases=["格式", "输出格式"], allowed_values=["geojson", "json", "csv", "kml", "osm_xml"]), _field("include_geometry", "是否保留几何。", type="boolean", aliases=["包含几何", "geometry"], example=True)],
    )
    schemas[("osm_query", "count")] = _schema(
        "统计已有结果或指定结构化查询范围内的 OSM 要素数量。",
        [_source_result(required=False), _area(), _field("bbox", "WGS84 查询框。", type="array", item_type="number", aliases=["边界框", "bounds"]), _field("tags", "待统计的 OSM 标签。", type="object", aliases=["标签", "特征", "filters"]), _field("group_by", "可选分组字段。", type="string_or_array", aliases=["分组", "统计维度"])],
        required_any=[["source_result", "area", "bbox"]],
    )

    # streetview
    schemas[("streetview_query", "open")] = _schema(
        "打开指定地点或坐标的街景。",
        [_area(extra_aliases=["target"]), _field("coordinates", "WGS84 坐标。", type="array", item_type="number", aliases=["坐标", "latlon", "point"]), _field("provider", "街景服务提供方。", aliases=["服务", "数据源"])],
        required_any=[["area", "coordinates"]],
    )
    schemas[("streetview_query", "navigate")] = _schema(
        "在已打开街景中沿道路或方向移动。",
        [_field("session", "已打开街景会话或结果 ID。", aliases=["会话", "街景结果", "result_id"]), _area(), _field("direction", "移动方向或道路方向。", required=True, aliases=["方向", "朝向", "bearing"]), _field("distance_m", "移动距离（米）。", type="number", aliases=["距离", "distance"], minimum=0)],
        required_any=[["session", "area"]],
    )
    schemas[("streetview_query", "change_time")] = _schema(
        "切换指定地点或会话的街景年份。",
        [_field("session", "街景会话或结果 ID。", aliases=["会话", "result_id"]), _area(), _time_range(required=True)],
        required_any=[["session", "area"]],
    )
    schemas[("streetview_query", "capture")] = _schema(
        "从街景会话获取指定视角画面。",
        [_field("session", "街景会话或结果 ID。", required=True, aliases=["会话", "result_id"]), _field("heading", "水平朝向角（度）。", type="number", aliases=["朝向", "方位角"], minimum=0, maximum=360), _field("pitch", "俯仰角（度）。", type="number", aliases=["俯仰"], minimum=-90, maximum=90), _field("fov", "视场角（度）。", type="number", aliases=["视角", "field_of_view"], minimum=1, maximum=180)],
    )

    # satellite imagery
    schemas[("satellite_imagery_query", "retrieve")] = _schema(
        "获取指定区域和时间的卫星或航片。",
        [_area(extra_aliases=["target"]), _field("coordinates", "待获取影像的一个或多个坐标。", type="string_or_array", aliases=["坐标", "points"]), _field("source_result", "上一步筛选结果或坐标结果引用。", type="string_or_object", aliases=["已有结果", "筛选结果", "result_id"]), _time_range(), _field("provider", "影像服务或数据集。", aliases=["数据源", "服务"]), _field("layer", "影像图层，如 satellite/aerial/terrain。", aliases=["图层", "影像类型"]), _field("cloud_cover_max", "最大允许云量百分比。", type="number", aliases=["最大云量", "cloud_threshold"], minimum=0, maximum=100), _field("resolution_m", "期望空间分辨率（米/像素）。", type="number", aliases=["分辨率", "resolution"], minimum=0)],
        required_any=[["area", "coordinates", "source_result"]],
    )
    schemas[("satellite_imagery_query", "change_time")] = _schema(
        "在同一区域切换历史年份、季节或水期影像。",
        [_area(required=True), _time_range(required=True), _field("source_result", "可选的上一影像会话或结果 ID。", aliases=["已有影像", "result_id"]), _field("provider", "影像服务提供方。", aliases=["数据源", "服务"])],
    )
    schemas[("satellite_imagery_query", "compare_candidates")] = _schema(
        "在多个明确候选区域的卫星影像中按同一视觉或空间模板进行横向比对；不能只给候选数量而不提供候选本身。",
        [_field("candidates", "待比较的候选地点、坐标或区域列表。", type="array", required=True, aliases=["候选点", "候选区域", "候选列表", "locations"]), _field("template", "所有候选共同使用的视觉/空间比对模板。", type="string_or_object", required=True, aliases=["比对模板", "结构模板", "特征模板", "criteria"]), _time_range(), _field("provider", "影像服务提供方。", aliases=["数据源", "服务"])],
    )
    schemas[("satellite_imagery_query", "compare_time")] = _schema(
        "获取并比较同一区域的两个或多个时间影像。",
        [_area(required=True), _field("times", "需要比较的至少两个时间点或时间范围。", type="array", item_type="string", required=True, aliases=["时间列表", "年份列表", "时相", "periods"]), _field("comparison", "比较目标，如水位、建设变化或植被。", aliases=["比对目标", "变化类型", "method"])],
    )
    schemas[("satellite_imagery_query", "oblique_view")] = _schema(
        "获取指定区域的倾斜、三维或地形视角。",
        [_area(required=True), _field("heading", "观察朝向角（度）。", type="number", aliases=["朝向", "方位角"], minimum=0, maximum=360), _field("tilt", "倾斜角（度）。", type="number", aliases=["倾角", "pitch"], minimum=0, maximum=90), _time_range()],
    )

    # weather
    for op, detail in [
        ("cloud_cover", "历史云量或云图"),
        ("weather", "历史温度、降水、风或能见度"),
        ("snow_cover", "历史积雪范围"),
    ]:
        schemas[("weather_archive_query", op)] = _schema(
            f"查询指定地点和时间的{detail}。",
            [_area(required=True), _time_range(required=True), _field("variables", "要返回的气象变量。", type="string_or_array", aliases=["指标", "气象项", "fields"]), _field("provider", "气象档案来源。", aliases=["数据源", "服务"])],
        )
    schemas[("weather_archive_query", "refine_range")] = _schema(
        "在已有气象结果上缩小或连续细化时间/空间范围。",
        [_source_result(), _time_range(required=True), _area(), _field("condition", "继续保留结果的条件。", type="string_or_object", aliases=["筛选条件", "条件"])],
    )

    # astronomy
    schemas[("astronomy_query", "sunset_time")] = _schema(
        "计算一个或多个地点在指定日期的日落和暮光时间。",
        [_field("locations", "一个或多个地点或坐标。", type="string_or_array", required=True, aliases=["地点", "区域", "候选地点", "location"]), _time_range(required=True), _field("timezone", "输出时区；缺省时使用地点当地时区。", aliases=["时区", "tz"])],
    )
    schemas[("astronomy_query", "sun_position")] = _schema(
        "计算指定地点和时刻的太阳高度角与方位角。",
        [_area(required=True), _field("datetime", "带日期和时间的时刻。", required=True, aliases=["时间", "日期时间", "timestamp"]), _field("timezone", "时区。", aliases=["时区", "tz"])],
    )
    schemas[("astronomy_query", "shadow_model")] = _schema(
        "根据太阳位置和物体高度计算理论阴影。",
        [_area(required=True), _field("datetime", "拍摄日期时间。", required=True, aliases=["时间", "日期时间"]), _field("object_height_m", "产生阴影的物体高度（米）。", type="number", aliases=["物体高度", "height"], minimum=0), _field("surface", "地面坡度或朝向。", type="string_or_object", aliases=["地面", "坡面"])],
    )

    # geospatial analysis
    schemas[("geospatial_analysis", "distance")] = _schema(
        "计算两个或多个地理点/要素之间的距离。",
        [_field("points", "至少两个坐标、地点或要素引用。", type="array", aliases=["点", "地点", "对象", "locations"]), _field("features", "需要测量宽度或间距的地理要素。", type="string_or_array", aliases=["目标", "要素", "target"]), _field("unit", "距离单位。", aliases=["单位"], allowed_values=["m", "km", "mile"], example="m"), _field("mode", "直线、沿路或沿几何距离。", aliases=["方式", "方法"], allowed_values=["geodesic", "route", "geometry", "width"])],
        required_any=[["points", "features"]],
    )
    schemas[("geospatial_analysis", "bearing")] = _schema(
        "计算起点到终点的方位角或方向。",
        [_field("origin", "起点或观察点。", type="string_or_object", required=True, aliases=["起点", "观察点", "from"]), _field("target", "终点或目标点。", type="string_or_object", required=True, aliases=["终点", "目标点", "to"]), _field("reference", "方位参考系。", aliases=["参考系"], allowed_values=["true_north", "magnetic_north", "image_axis"])],
    )
    schemas[("geospatial_analysis", "sightline")] = _schema(
        "计算观察点到目标的视线、可视域或射线交点。",
        [_field("observer", "观察点及可选高度。", type="string_or_object", required=True, aliases=["观察点", "起点", "参照物", "baseline", "viewpoint"]), _field("target", "视线最终核验的目标点、方向或目标要素。", type="string_or_object", required=True, aliases=["目标", "终点", "目的", "direction"]), _field("through_points", "用于定义射线方向的一个或多个中间参照点。", type="string_or_array", aliases=["参照点", "经过点", "控制点"]), _field("terrain", "使用的高程/地形数据引用。", type="string_or_object", aliases=["地形", "dem", "elevation_layer"])],
    )
    schemas[("geospatial_analysis", "terrain")] = _schema(
        "查询或计算区域高程、坡度和地形剖面。",
        [_area(), _field("path", "剖面线或路线。", type="string_or_array", aliases=["剖面线", "路线", "line"]), _field("metrics", "需要计算的地形量。", type="string_or_array", required=True, aliases=["指标", "计算项"], example=["elevation", "slope"])],
        required_any=[["area", "path"]],
    )
    schemas[("geospatial_analysis", "geometry_filter")] = _schema(
        "按几何形状和空间关系过滤已有要素。",
        [_source_result(), _field("relation", "空间关系。", required=True, aliases=["空间关系", "关系"], allowed_values=["within", "intersects", "contains", "near", "crosses"]), _field("geometry", "参照几何或区域。", type="string_or_object", required=True, aliases=["几何", "区域", "reference_geometry"]), _field("distance_m", "near 等关系使用的距离阈值。", type="number", aliases=["距离", "阈值"], minimum=0)],
    )

    # media search
    schemas[("media_search", "video_search")] = _schema(
        "在公开视频或内容平台检索地点和视角。",
        [_query(required=True), _area(), _time_range(), _field("platforms", "限定视频平台。", type="array", item_type="string", aliases=["平台", "站点"]), _field("top_k", "最多返回结果数。", type="integer", aliases=["结果数", "limit"], minimum=1, maximum=100)],
    )
    schemas[("media_search", "photo_search")] = _schema(
        "检索历史照片、航拍照片或公共图库。",
        [_query(), _area(), _time_range(), _field("sources", "限定图库或档案来源。", type="array", item_type="string", aliases=["来源", "站点", "archives"]), _field("top_k", "最多返回结果数。", type="integer", aliases=["结果数", "limit"], minimum=1, maximum=100)],
        required_any=[["query", "area"]],
    )
    schemas[("media_search", "frame_retrieve")] = _schema(
        "从已找到的视频获取指定时间或视角的帧。",
        [_field("video", "视频 URL、路径或搜索结果 ID。", required=True, aliases=["视频", "链接", "video_id", "result_id"]), _field("timestamps", "一个或多个时间点（秒或时间码）。", type="string_or_array", required=True, aliases=["时间点", "时间", "timestamp"]), _field("view", "可选视角描述。", aliases=["视角", "方向"])],
    )

    # registry
    for op, subject in [
        ("administrative", "行政归属、标准地名或水体名称"),
        ("construction", "建筑、桥梁或设施建设历史"),
        ("permit", "许可、登记或编号"),
        ("directory", "指定类别对象名录"),
    ]:
        schemas[("registry_lookup", op)] = _schema(
            f"查询结构化档案中的{subject}。",
            [_query(required=True), _area(), _time_range(), _field("registry", "指定官方或行业档案来源。", aliases=["档案", "名录", "数据源", "source"]), _field("fields", "需要返回的字段。", type="string_or_array", aliases=["字段", "返回项"])],
        )

    # external LLM
    schemas[("llm_query", "consult")] = _schema(
        "把明确问题及必要上下文提交给外部模型；当前 Agent 自身思考不得伪装成本调用。",
        [_field("prompt", "发送给外部模型的明确问题或指令。", required=True, aliases=["问题", "指令", "query"]), _field("context", "允许提供给外部模型的上下文。", type="string_or_object", aliases=["上下文", "材料"]), _field("model", "可选外部模型标识。", aliases=["模型"])],
    )
    schemas[("llm_query", "enumerate")] = _schema(
        "要求外部模型生成候选或结构化清单。",
        [_field("prompt", "枚举任务的明确指令。", required=True, aliases=["问题", "指令", "query"]), _field("item_type", "需要枚举的对象类型。", required=True, aliases=["对象类型", "候选类型", "type"]), _field("constraints", "候选必须满足的约束。", type="string_or_object", aliases=["约束", "筛选条件"]), _field("max_items", "最多返回条目数。", type="integer", aliases=["数量", "limit"], minimum=1, maximum=500)],
    )

    # flight archive / local metadata
    schemas[("flight_data_query", "search")] = _schema(
        "按日期、区域、机场或航线查询真实航班档案。",
        [_area(), _field("date", "航班日期或日期范围。", type="string_or_object", aliases=["日期", "时间", "date_range"]), _field("airports", "起降机场或候选机场。", type="string_or_array", aliases=["机场", "机场列表"]), _field("route", "航线或起终点。", type="string_or_object", aliases=["航线", "起终点"]), _field("flight_number", "航班号。", aliases=["航班", "航班号"])],
        required_any=[["area", "airports", "route", "flight_number"]],
    )
    schemas[("flight_data_query", "track")] = _schema(
        "查询指定航班或区域内的真实航迹。",
        [_area(), _field("date", "航迹日期或时间范围。", type="string_or_object", aliases=["日期", "时间", "date_range"]), _field("flight_number", "指定航班号。", aliases=["航班", "航班号"]), _field("airports", "用于限定航迹的机场。", type="string_or_array", aliases=["机场", "机场列表"])],
        required_any=[["area", "flight_number", "airports"]],
    )
    schemas[("flight_data_query", "nearby_traffic")] = _schema(
        "统计指定时空范围内的航空器活动。",
        [_area(required=True), _field("time_range", "统计时间范围。", type="string_or_object", required=True, aliases=["时间", "日期", "date_range"]), _field("radius_km", "空间统计半径（公里）。", type="number", aliases=["半径", "radius"], minimum=0)],
    )
    schemas[("metadata_read", "exif")] = _schema(
        "读取真实图片文件的 EXIF/GPS 元数据。",
        [_field("file", "可访问的图片文件路径或文件 ID。", required=True, aliases=["文件", "图片", "image", "path"]), _field("fields", "要读取的 EXIF 字段；缺省时读取全部安全字段。", type="string_or_array", aliases=["字段", "读取项"])],
    )
    schemas[("metadata_read", "file")] = _schema(
        "读取媒体文件容器和通用属性。",
        [_field("file", "可访问的媒体文件路径或文件 ID。", required=True, aliases=["文件", "媒体", "path"]), _field("fields", "要读取的属性字段。", type="string_or_array", aliases=["字段", "读取项"])],
    )

    schemas[("final_answer", "submit")] = _schema(
        "提交最终地点；单题为字符串，多题为按讲解顺序排列的字符串数组。",
        [_field("location", "最终地点字符串或地点数组，不得使用 result/site 等替代字段。", type="string_or_array", required=True, aliases=["地点", "答案"])],
        examples=[{"location": "上海市杨浦大桥"}],
    )
    return schemas


OPERATION_ALIASES: dict[tuple[str, str], list[str]] = {
    ("image_process", "enhance"): ["adjust", "exposure"],
    ("image_process", "compare"): ["diff", "match"],
    ("web_search", "keyword_search"): ["search", "query"],
    ("map_query", "browse"): ["search", "query"],
    ("osm_query", "query"): ["search", "overpass"],
    ("satellite_imagery_query", "retrieve"): ["open", "query"],
    ("satellite_imagery_query", "compare_candidates"): ["compare", "match_candidates"],
    ("satellite_imagery_query", "compare_time"): ["time_compare", "temporal_compare"],
    ("streetview_query", "open"): ["retrieve", "query"],
    ("media_search", "photo_search"): ["image_search"],
    ("flight_data_query", "search"): ["execute", "query", "schedule"],
    ("flight_data_query", "track"): ["route_track"],
    ("flight_data_query", "nearby_traffic"): ["density", "traffic_density"],
    ("metadata_read", "exif"): ["execute", "read"],
    ("final_answer", "submit"): ["answer"],
}


def attach_operation_input_schemas(forest: ToolForest) -> ToolForest:
    """把内置 operation 合同附加到人工目录；保留目录中显式定义的 schema。"""

    registry = _operation_schemas()
    trees: list[ToolTree] = []
    for tree in forest.trees:
        operations = []
        for operation in tree.canonical.operations:
            key = (tree.canonical.name, operation.name)
            aliases = list(operation.aliases)
            for alias in OPERATION_ALIASES.get(key, []):
                if alias not in aliases:
                    aliases.append(alias)
            operations.append(
                operation.model_copy(
                    update={
                        "aliases": aliases,
                        "input_schema": operation.input_schema or registry.get(key),
                    }
                )
            )
        trees.append(
            tree.model_copy(
                update={
                    "canonical": tree.canonical.model_copy(
                        update={"operations": operations}
                    )
                }
            )
        )
    return ToolForest(trees=trees)
