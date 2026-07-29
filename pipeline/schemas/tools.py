"""Tool 树 schema。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ParamSpec(BaseModel):
    """Tool 参数规格。"""

    name: str
    type: str = "string"
    required: bool = False
    description: str = ""


class ObservationField(BaseModel):
    """Observation 字段规格。"""

    name: str
    type: str = "string"
    nullable: bool = True
    description: str = ""


class ToolDefinition(BaseModel):
    """规范 tool 定义（树根）。"""

    name: str
    description: str = ""
    params: list[ParamSpec] = Field(default_factory=list)
    observation_fields: list[ObservationField] = Field(default_factory=list)
    is_terminal: bool = False


class ToolTree(BaseModel):
    """一棵 tool 树：canonical + 近义变体名。"""

    canonical: ToolDefinition
    variants: list[str] = Field(default_factory=list)


class ToolForest(BaseModel):
    """全部 tool 树集合。"""

    trees: list[ToolTree] = Field(default_factory=list)


class MatchDecision(BaseModel):
    """LLM/matcher 对自由 tool 的归并决策。"""

    action: Literal["map", "create"]
    canonical_name: str | None = None
    reason: str = ""
