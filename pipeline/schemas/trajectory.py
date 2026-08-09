"""单 Agent 标准轨迹 schema。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Action(BaseModel):
    """规范化后的 Tool 调用。"""

    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    """轨迹中的单步 T→A→O。"""

    thought: str
    action: Action
    observation: Optional[dict[str, Any]] = None


class Trajectory(BaseModel):
    """单条 Agent 轨迹。"""

    id: str
    system_prompt: str
    user_query: str
    image_paths: list[str] = Field(min_length=1)
    steps: list[TrajectoryStep] = Field(default_factory=list)

    @field_validator("image_paths")
    @classmethod
    def _nonempty_paths(cls, value: list[str]) -> list[str]:
        cleaned = [p.strip() for p in value if str(p).strip()]
        if not cleaned:
            raise ValueError("image_paths 至少包含一张图")
        return cleaned
