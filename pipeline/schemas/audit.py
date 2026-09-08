"""阶段1.5：审核切分产物 schema。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class AuditDecision(str, Enum):
    """审核结论。"""

    reject = "reject"
    accept = "accept"


class TargetKind(str, Enum):
    """任务视觉目标类型。"""

    still_image = "still_image"
    video_derived = "video_derived"


class TaskStatus(str, Enum):
    """单个定位题的质量门禁状态。"""

    accepted = "accepted"
    needs_review = "needs_review"
    rejected = "rejected"


class AnswerStatus(str, Enum):
    """字幕中最终答案是否足够明确。"""

    resolved = "resolved"
    ambiguous = "ambiguous"
    unsolved = "unsolved"


class ProcessRole(str, Enum):
    """过程时间线区间角色（视频过程事实，非 agent 口吻）。"""

    show_source = "show_source"
    tool = "tool"
    reveal = "reveal"
    other = "other"


class ProcessInterval(BaseModel):
    """蒸馏窗内一段过程区间。"""

    start: float
    end: float
    role: ProcessRole
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_range(self) -> ProcessInterval:
        if self.end < self.start:
            raise ValueError("process interval end 不得小于 start")
        return self


class KeyframeAssessment(BaseModel):
    """候选帧的视觉验收记录，便于人工审计选图。"""

    timestamp: float
    image_path: str
    kind: str
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_leakage: bool = False
    tutorial_overlay: bool = False
    clean_source: bool = False
    evidence_role: str = "unknown"
    chain_support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    selected: bool = False
    reason: str = ""


class GeoTaskSpec(BaseModel):
    """单个地理定位任务（切分后的一条下游样本）。"""

    task_id: str
    time_start: float
    time_end: float
    target_kind: TargetKind
    keyframe_timestamps: list[float] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    multi_target_images: bool = False
    segment_start_idx: int | None = None
    segment_end_idx: int | None = None
    task_summary: str = ""
    visual_evidence_brief: str = ""
    process_intervals: list[ProcessInterval] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.accepted
    status_reason: str = ""
    answer_status: AnswerStatus = AnswerStatus.resolved
    final_location_text: str = ""
    expected_image_count: int = Field(default=1, ge=1)
    frame_assessments: list[KeyframeAssessment] = Field(default_factory=list)
    # 程序化选图评价：质量等级 + 选中帧明细；不拦下游
    image_selection_note: str = ""

    @field_validator("task_id")
    @classmethod
    def _nonempty_task_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("task_id 不得为空")
        return cleaned

    @model_validator(mode="after")
    def _validate_time_range(self) -> GeoTaskSpec:
        if self.time_end < self.time_start:
            raise ValueError("time_end 不得小于 time_start")
        return self


class AuditSplitResult(BaseModel):
    """阶段1.5 审核切分结果。"""

    video_id: str
    decision: AuditDecision
    reason: str = ""
    has_unresolved_target: bool | None = None
    tasks: list[GeoTaskSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_decision_tasks(self) -> AuditSplitResult:
        if self.decision == AuditDecision.accept and not self.tasks:
            raise ValueError("accept 时 tasks 不得为空")
        if self.decision == AuditDecision.reject and self.tasks:
            raise ValueError("reject 时 tasks 必须为空")
        return self
