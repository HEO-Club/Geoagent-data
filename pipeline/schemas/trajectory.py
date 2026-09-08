"""单 Agent 标准轨迹 schema。"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Action(BaseModel):
    """规范化后的 Tool 调用。"""

    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    """规范轨迹事件；reasoning 可以连续出现且不伪造 Action。"""

    event_type: Literal["reasoning", "tool_call", "final"] = "tool_call"
    thought: str
    action: Optional[Action] = None
    observation: Optional[dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_event_type(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("event_type"):
            return data
        copied = dict(data)
        action = copied.get("action") or {}
        tool = (
            action.get("tool")
            if isinstance(action, dict)
            else getattr(action, "tool", None)
        )
        if tool == "final_answer":
            copied["event_type"] = "final"
        elif tool:
            copied["event_type"] = "tool_call"
        else:
            copied["event_type"] = "reasoning"
        return copied

    @model_validator(mode="after")
    def _validate_event_shape(self) -> "TrajectoryStep":
        if self.event_type == "reasoning":
            if self.action is not None or self.observation is not None:
                raise ValueError("reasoning 事件不得包含 action/observation")
            return self
        if self.action is None:
            raise ValueError(f"{self.event_type} 事件必须包含 action")
        if self.event_type == "final":
            if self.action.tool != "final_answer" or self.observation is not None:
                raise ValueError("final 必须调用 final_answer 且 observation=null")
        elif self.action.tool == "final_answer":
            raise ValueError("final_answer 必须使用 event_type=final")
        return self


class Trajectory(BaseModel):
    """单条 Agent 轨迹。"""

    id: str
    system_prompt: str
    user_query: str
    # 允许空列表：选图失败但仍需入库的 needs_review 样本
    image_paths: list[str] = Field(default_factory=list)
    steps: list[TrajectoryStep] = Field(default_factory=list)

    @field_validator("image_paths")
    @classmethod
    def _clean_paths(cls, value: list[str]) -> list[str]:
        return [p.strip() for p in value if str(p).strip()]
