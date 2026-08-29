"""Tool 树 schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ParamSpec(BaseModel):
    """Tool 参数规格。"""

    name: str
    type: str = "string"
    required: bool = False
    description: str = ""
    allowed_values: list[str] = Field(default_factory=list)


class ObservationField(BaseModel):
    """Observation 字段规格。"""

    name: str
    type: str = "string"
    nullable: bool = True
    description: str = ""


class InputFieldSpec(BaseModel):
    """某个 operation 的一个实际输入字段。"""

    name: str
    type: str = "string"
    required: bool = False
    requirement_level: Literal["semantic", "execution", "optional"] = "optional"
    description: str
    acquisition_hint: str = ""
    context_sources: list[str] = Field(default_factory=list)
    context_default: Any = None
    aliases: list[str] = Field(default_factory=list)
    allowed_values: list[str] = Field(default_factory=list)
    item_type: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    example: Any = None

    @field_validator("name")
    @classmethod
    def _field_name(cls, value: str) -> str:
        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        if not cleaned:
            raise ValueError("input field name 不能为空")
        return cleaned


class ToolInputSchema(BaseModel):
    """operation 级参数合同；宽容接收别名和扩展字段。"""

    description: str = ""
    fields: list[InputFieldSpec] = Field(default_factory=list)
    required_any: list[list[str]] = Field(default_factory=list)
    mutually_exclusive: list[list[str]] = Field(default_factory=list)
    allow_extra: bool = True
    examples: list[dict[str, Any]] = Field(default_factory=list)


class ToolOperation(BaseModel):
    """同一执行器支持的一种操作及其调用语义。"""

    name: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    input_schema: ToolInputSchema | None = None

    @field_validator("name")
    @classmethod
    def _operation_name(cls, value: str) -> str:
        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        if not cleaned:
            raise ValueError("operation name 不能为空")
        return cleaned


class ToolDefinition(BaseModel):
    """规范 tool 定义（树根）。"""

    name: str
    description: str = ""
    executor: str = ""
    usage: str = ""
    operations: list[ToolOperation] = Field(default_factory=list)
    params: list[ParamSpec] = Field(default_factory=list)
    observation_fields: list[ObservationField] = Field(default_factory=list)
    is_terminal: bool = False


class ToolTree(BaseModel):
    """一棵 tool 树：canonical + 近义变体名。"""

    canonical: ToolDefinition
    variants: list[str] = Field(default_factory=list)
    variant_operations: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_variant_operations(self) -> ToolTree:
        self.variant_operations = {
            key.strip().lower(): value.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            for key, value in self.variant_operations.items()
            if key.strip() and value.strip()
        }
        return self


class ToolForest(BaseModel):
    """全部 tool 树集合。"""

    trees: list[ToolTree] = Field(default_factory=list)


class MatchDecision(BaseModel):
    """LLM 对自由 tool 的执行器级归并或严格新建决策。"""

    raw_tool: str = ""
    action: Literal["map", "create", "reasoning"]
    canonical_name: str | None = None
    operation: str = "execute"
    operation_description: str = "执行该自由工具所描述的外部操作"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    proposed_definition: ToolDefinition | None = None
    reason: str = ""
    not_catalog_reason: str = ""
    create_kind: Literal["new_executor", "new_operation"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_new_operation_action(cls, data: Any) -> Any:
        """允许 LLM 把 new_operation 写成 action；内部仍走 create + create_kind。"""
        if not isinstance(data, dict) or data.get("action") != "new_operation":
            return data
        coerced = dict(data)
        coerced["action"] = "create"
        if not coerced.get("create_kind"):
            coerced["create_kind"] = "new_operation"
        return coerced


class ParameterAuditIssue(BaseModel):
    """单次 Tool 参数归一/校验问题。"""

    code: str
    severity: Literal["info", "warning", "error", "hard_fail"]
    field: str | None = None
    message: str
    requirement_level: Literal["semantic", "execution", "optional"] | None = None
    repairable: bool = False
    guidance: str = ""


class ParameterRepairAction(BaseModel):
    """缺参时供生成 Agent 或修复 Agent 执行的下一步。"""

    field: str
    requirement_level: Literal["semantic", "execution", "optional"]
    strategy: Literal[
        "use_context",
        "extract_from_thought",
        "call_prerequisite_tool",
        "request_or_capture_input",
        "manual_review",
    ]
    guidance: str
    suggested_value: Any = None


class ToolParameterAudit(BaseModel):
    """单次 canonical Tool 调用的参数审计结果。"""

    step_index: int = Field(ge=1)
    tool: str
    raw_operation: str
    operation: str
    raw_inputs: dict[str, Any] = Field(default_factory=dict)
    normalized_inputs: dict[str, Any] = Field(default_factory=dict)
    valid: bool = True
    readiness: Literal[
        "ready", "context_resolvable", "repairable", "invalid"
    ] = "ready"
    issues: list[ParameterAuditIssue] = Field(default_factory=list)
    repair_actions: list[ParameterRepairAction] = Field(default_factory=list)
