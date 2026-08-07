"""阶段1.5：审核切分产物 schema。"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AuditDecision(str, Enum):
    """审核结论。"""

    reject = "reject"
    accept = "accept"


class TargetKind(str, Enum):
    """任务视觉目标类型。"""

    still_image = "still_image"
    video_derived = "video_derived"


class GeoTaskSpec(BaseModel):
    """单个地理定位任务（切分后的一条下游样本）。"""

    task_id: str
    time_start: float
    time_end: float
    target_kind: TargetKind
    keyframe_timestamps: list[float] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    multi_target_images: bool = False
    segment_start_idx: Optional[int] = None
    segment_end_idx: Optional[int] = None
    task_summary: str = ""

    @field_validator("task_id")
    @classmethod
    def _nonempty_task_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("task_id 不得为空")
        return cleaned

    @model_validator(mode="after")
    def _validate_time_range(self) -> "GeoTaskSpec":
        if self.time_end < self.time_start:
            raise ValueError("time_end 不得小于 time_start")
        return self


class AuditSplitResult(BaseModel):
    """阶段1.5 审核切分结果。"""

    video_id: str
    decision: AuditDecision
    reason: str = ""
    has_unresolved_target: Optional[bool] = None
    tasks: list[GeoTaskSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_decision_tasks(self) -> "AuditSplitResult":
        if self.decision == AuditDecision.accept and not self.tasks:
            raise ValueError("accept 时 tasks 不得为空")
        if self.decision == AuditDecision.reject and self.tasks:
            raise ValueError("reject 时 tasks 必须为空")
        return self
