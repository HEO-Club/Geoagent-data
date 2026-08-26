"""阶段4：样本置信度评分 schema。"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ReviewPriority = Literal["high", "medium", "low"]
ConfidenceDecision = Literal[
    "accept",
    "provisional_pass",
    "parameter_repair",
    "needs_review",
    "reject",
]
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_gate(cls, value: object) -> object:
        if isinstance(value, str):
            return {"code": value}
        if not isinstance(value, dict):
            return value
        copied = dict(value)
        if "code" not in copied:
            for alias in ("type", "name", "gate", "label"):
                if alias in copied:
                    copied["code"] = copied[alias]
                    break
        if "evidence" not in copied:
            for alias in ("reason", "detail", "description"):
                if alias in copied:
                    copied["evidence"] = copied[alias]
                    break
        return copied


def _unwrap_score(value: object) -> object:
    if not isinstance(value, dict):
        return value
    for key in ("score", "value", "confidence", "rating"):
        candidate = value.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return candidate
    numeric = [
        candidate
        for candidate in value.values()
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool)
    ]
    return numeric[0] if len(numeric) == 1 else value


class ParameterReadinessSummary(BaseModel):
    """阶段3 参数审计在阶段4 的汇总（人工检查用）。"""

    total_calls: int = 0
    ready: int = 0
    context_resolvable: int = 0
    repairable: int = 0
    invalid: int = 0
    worst: ParameterReadiness | None = None
    audit_missing: bool = False
    detail_lines: list[str] = Field(default_factory=list)


class ConfidenceJudgeDraft(BaseModel):
    """VLM/LLM 裁判软信封：维度分 + 模型侧硬门槛。"""

    evidence_grounding: float = Field(ge=0.0, le=1.0)
    final_answer_support: float = Field(ge=0.0, le=1.0)
    # 参数正确性由程序化审计覆盖；裁判可给参考分，合并时以规则分为准
    tool_param_correctness: float | None = Field(default=None, ge=0.0, le=1.0)
    logical_consistency: float = Field(ge=0.0, le=1.0)
    input_quality_alignment: float = Field(ge=0.0, le=1.0)
    # 格式完整性由程序化规则主导；裁判可给参考分，合并时以规则分为准
    sft_format_completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    dimension_reasons: dict[str, str] = Field(default_factory=dict)
    hard_gates: list[HardGateHit] = Field(default_factory=list)
    # 必填评价说明；合并时若为空由程序化 compose 补全
    notes: str = Field(
        min_length=1,
        description="样本评价说明；弱维度须写可核对证据",
    )

    @field_validator(
        "evidence_grounding",
        "final_answer_support",
        "tool_param_correctness",
        "logical_consistency",
        "input_quality_alignment",
        "sft_format_completeness",
        mode="before",
    )
    @classmethod
    def _unwrap_dimension_score(cls, value: object) -> object:
        return _unwrap_score(value)

    @field_validator("dimension_reasons", mode="before")
    @classmethod
    def _normalize_dimension_reasons(cls, value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): (
                    str(reason.get("reason") or reason.get("detail") or "")
                    if isinstance(reason, dict)
                    else str(reason)
                )
                for key, reason in value.items()
            }
        if isinstance(value, list):
            result: dict[str, str] = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("dimension") or "").strip()
                reason = str(item.get("reason") or item.get("detail") or "").strip()
                if name:
                    result[name] = reason
            return result
        return value

    @field_validator("hard_gates", mode="before")
    @classmethod
    def _normalize_hard_gates(cls, value: object) -> object:
        if isinstance(value, dict):
            for key in ("items", "gates", "hard_gates"):
                candidate = value.get(key)
                if isinstance(candidate, list):
                    return candidate
            return [value] if value else []
        return value

    @field_validator("notes", mode="before")
    @classmethod
    def _normalize_notes(cls, value: object) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return json.dumps(value, ensure_ascii=False)


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
    notes: str = Field(min_length=1, description="每条样本必填的评价说明")
    judge_call_failed: bool = False
    parameter_readiness: ParameterReadinessSummary | None = None
    evidence_sources: list[str] = Field(default_factory=list)
    soft_flags: list[HardGateHit] = Field(default_factory=list)
    applied_soft_caps: dict[str, float] = Field(default_factory=dict)
