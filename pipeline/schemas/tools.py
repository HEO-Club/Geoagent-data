"""Tool 树 schema。"""

from __future__ import annotations

from typing import Literal

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


class ToolOperation(BaseModel):
    """同一执行器支持的一种操作及其调用语义。"""

    name: str
    description: str

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
