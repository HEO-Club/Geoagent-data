"""阶段4：样本置信度评分 schema。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ReviewPriority = Literal["high", "medium", "low"]
ConfidenceDecision = Literal[
    "accept",
    "provisional_pass",
    "parameter_repair",
    "needs_review",
    "reject",
]

# 六维名称（与配置权重键一致）
DIMENSION_NAMES: tuple[str, ...] = (
    "evidence_grounding",
    "final_answer_support",
    "tool_param_correctness",
    "logical_consistency",
    "input_quality_alignment",
    "sft_format_completeness",
)


class DimensionScore(BaseModel):
    """单个评分维度。"""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    reason: str = ""


class HardGateHit(BaseModel):
    """硬门槛命中记录。"""

    code: str
    evidence: str = ""


class ConfidenceJudgeDraft(BaseModel):
    """VLM/LLM 裁判软信封：维度分 + 模型侧硬门槛。"""

    evidence_grounding: float = Field(default=0.5, ge=0.0, le=1.0)
    final_answer_support: float = Field(default=0.5, ge=0.0, le=1.0)
    tool_param_correctness: float = Field(default=0.5, ge=0.0, le=1.0)
    logical_consistency: float = Field(default=0.5, ge=0.0, le=1.0)
    input_quality_alignment: float = Field(default=0.5, ge=0.0, le=1.0)
    # 格式完整性由程序化规则主导；裁判可给参考分，合并时以规则分为准
    sft_format_completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    dimension_reasons: dict[str, str] = Field(default_factory=dict)
    hard_gates: list[HardGateHit] = Field(default_factory=list)
    notes: str = ""


class ConfidenceReport(BaseModel):
    """阶段4 置信度报告（人工检查用）。"""

    task_id: str
    base_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    audit_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    decision: ConfidenceDecision = "needs_review"
    dimensions: list[DimensionScore] = Field(default_factory=list)
    hard_gates: list[HardGateHit] = Field(default_factory=list)
    review_priority: ReviewPriority = "medium"
    notes: str = ""
    judge_call_failed: bool = False
    parameter_readiness_counts: dict[str, int] = Field(default_factory=dict)
    evidence_sources: list[str] = Field(default_factory=list)
