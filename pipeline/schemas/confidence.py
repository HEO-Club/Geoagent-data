"""阶段4：样本置信度评分 schema。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ReviewPriority = Literal["high", "medium", "low"]
ParameterReadiness = Literal[
    "ready", "context_resolvable", "repairable", "invalid"
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

# VLM 主导维度（程序化覆盖的不在此列）
VLM_DIMENSION_NAMES: tuple[str, ...] = (
    "evidence_grounding",
    "final_answer_support",
    "logical_consistency",
    "input_quality_alignment",
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


class ParameterReadinessSummary(BaseModel):
    """阶段3 参数审计在阶段4 的汇总（人工检查用）。"""

    total_calls: int = 0
    ready: int = 0
    context_resolvable: int = 0
    repairable: int = 0
    invalid: int = 0
    worst: Optional[ParameterReadiness] = None
    audit_missing: bool = False
    detail_lines: list[str] = Field(default_factory=list)


class ConfidenceJudgeDraft(BaseModel):
    """VLM/LLM 裁判软信封：维度分 + 模型侧硬门槛。"""

    evidence_grounding: float = Field(default=0.5, ge=0.0, le=1.0)
    final_answer_support: float = Field(default=0.5, ge=0.0, le=1.0)
    # 参数正确性由程序化审计覆盖；裁判可给参考分，合并时以规则分为准
    tool_param_correctness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    logical_consistency: float = Field(default=0.5, ge=0.0, le=1.0)
    input_quality_alignment: float = Field(default=0.5, ge=0.0, le=1.0)
    # 格式完整性由程序化规则主导；裁判可给参考分，合并时以规则分为准
    sft_format_completeness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dimension_reasons: dict[str, str] = Field(default_factory=dict)
    hard_gates: list[HardGateHit] = Field(default_factory=list)
    # 必填评价说明；合并时若为空由程序化 compose 补全
    notes: str = Field(
        default="",
        description="样本评价说明；弱维度须写可核对证据",
    )


class ConfidenceReport(BaseModel):
    """阶段4 置信度报告（人工检查用）。"""

    task_id: str
    base_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    dimensions: list[DimensionScore] = Field(default_factory=list)
    hard_gates: list[HardGateHit] = Field(default_factory=list)
    review_priority: ReviewPriority = "medium"
    notes: str = Field(min_length=1, description="每条样本必填的评价说明")
    judge_call_failed: bool = False
    parameter_readiness: Optional[ParameterReadinessSummary] = None
