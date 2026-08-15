"""阶段2自由事件轨迹软信封。

阶段2允许连续 reasoning；只有真实外部动作才携带 tool/params/observation。
旧版没有 ``event_type`` 的 TAO JSON 会按 tool 自动推断，保持向后兼容。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from pipeline.schemas.clues import WorkingScope

EventType = Literal["reasoning", "tool_call", "final"]


class FreeFormStep(BaseModel):
    """阶段2单事件：reasoning、真实 tool_call 或 final。"""

    event_type: EventType = "tool_call"
    thought: str
    tool: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    observation: Optional[dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_event_type(cls, data: Any) -> Any:
        """旧数据没有 event_type；按 tool 推断而不破坏历史产物。"""
        if not isinstance(data, dict) or data.get("event_type"):
            return data
        copied = dict(data)
        tool = str(copied.get("tool") or "").strip()
        if tool == "final_answer":
            copied["event_type"] = "final"
        elif tool:
            copied["event_type"] = "tool_call"
        else:
            copied["event_type"] = "reasoning"
        return copied

    @model_validator(mode="after")
    def _validate_event_shape(self) -> "FreeFormStep":
        self.thought = self.thought.strip()
        if not self.thought:
            raise ValueError("thought 不能为空")

        tool = (self.tool or "").strip()
        if self.event_type == "reasoning":
            if tool or self.params or self.observation is not None:
                raise ValueError(
                    "reasoning 事件只能包含 thought，不得伪造 tool/params/observation"
                )
            self.tool = None
            return self

        if self.event_type == "final":
            if tool != "final_answer":
                raise ValueError("final 事件的 tool 必须是 final_answer")
            self.tool = "final_answer"
            return self

        if not tool:
            raise ValueError("tool_call 事件必须提供非空 tool")
        if tool == "final_answer":
            raise ValueError("final_answer 必须使用 event_type=final")
        self.tool = tool
        return self


class FreeFormTrajectory(BaseModel):
    """阶段2输出：内容优先、动作边界明确的自由事件链。"""

    source_video: str
    steps: list[FreeFormStep] = Field(default_factory=list)
    notes: Optional[str] = None
    working_scope: Optional[WorkingScope] = None
