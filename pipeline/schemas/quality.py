"""轨迹质量置信度与审核报告 schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

QualityDimensionName = Literal[
    "evidence_grounding",
    "final_answer_support",
    "tool_parameter_validity",
    "reasoning_consistency",
    "input_alignment",
    "sft_format",
]
QualityDecision = Literal[
    "accept",
    "provisional_pass",
    "parameter_repair",
    "needs_review",
    "reject",
]
IssueSeverity = Literal["info", "warning", "error", "hard_fail"]


class QualityCheck(BaseModel):
    """一个可复现的原子检查；observed=False 表示尚无足够证据。"""

    code: str
    score: float = Field(ge=0.0, le=1.0)
    observed: bool = True
    importance: float = Field(default=1.0, gt=0.0)
    message: str
    step_index: int | None = Field(default=None, ge=1)
    evidence: list[str] = Field(default_factory=list)


class QualityIssue(BaseModel):
    """供人工或修复 Agent 定位的具体问题。"""

    code: str
    dimension: QualityDimensionName
    severity: IssueSeverity
    message: str
    step_index: int | None = Field(default=None, ge=1)
    evidence: list[str] = Field(default_factory=list)


class QualityDimensionScore(BaseModel):
    """单个质量维度的证据校准分数。"""

    name: QualityDimensionName
    weight: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    audit_coverage: float = Field(ge=0.0, le=1.0)
    checks: list[QualityCheck] = Field(default_factory=list)


class SemanticQualityReview(BaseModel):
    """独立审核 Agent 的结构化语义判断，不直接改写轨迹。"""

    evidence_grounding: float = Field(ge=0.0, le=1.0)
    final_answer_support: float = Field(ge=0.0, le=1.0)
    reasoning_consistency: float = Field(ge=0.0, le=1.0)
    tool_semantics: float = Field(ge=0.0, le=1.0)
    input_alignment: float = Field(ge=0.0, le=1.0)
    issues: list[QualityIssue] = Field(default_factory=list)
    hard_failures: list[str] = Field(default_factory=list)
    summary: str = ""


class TrajectoryQualityReport(BaseModel):
    """一条轨迹的最终质量置信度报告。"""

    rubric_version: str = "trajectory_quality_v1"
    source_video: str
    trajectory_id: str
    quality_score: float = Field(ge=0.0, le=1.0)
    audit_coverage: float = Field(ge=0.0, le=1.0)
    decision: QualityDecision
    dimensions: list[QualityDimensionScore]
    hard_failures: list[str] = Field(default_factory=list)
    issues: list[QualityIssue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("quality_score", "audit_coverage")
    @classmethod
    def _round_score(cls, value: float) -> float:
        return round(float(value), 4)
