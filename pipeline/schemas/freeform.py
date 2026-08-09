"""阶段2 自由 TAO 软信封（不做 tool 语义校验）。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from pipeline.schemas.clues import WorkingScope


class FreeFormStep(BaseModel):
    """自由 TAO 单步：字段存在即可，tool/params 不做统一 schema。"""

    thought: str
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    observation: Optional[dict[str, Any]] = None


class FreeFormTrajectory(BaseModel):
    """阶段2 输出：内容优先的自由逻辑链。"""

    source_video: str
    steps: list[FreeFormStep] = Field(default_factory=list)
    notes: Optional[str] = None
    working_scope: Optional[WorkingScope] = None
