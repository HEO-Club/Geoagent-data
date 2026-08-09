"""外部给定线索 / 工作范围（沿用 v2 分层，供阶段2 抽取与双端注入）。"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ClueRole(str, Enum):
    """raw_given_clue 角色。"""

    photo_location_constraint = "photo_location_constraint"
    person_or_social_attribute = "person_or_social_attribute"
    other_non_location = "other_non_location"


class BoundKind(str, Enum):
    """工作范围边界强度。"""

    inside = "inside"
    near = "near"


class RawGivenClue(BaseModel):
    """问题设置段外部沟通原话。"""

    text: str
    role: ClueRole


class WorkingScope(BaseModel):
    """可注入 user_query / 蒸馏 prompt 的工作范围展示短语。"""

    region: str
    bound_kind: BoundKind

    @field_validator("region")
    @classmethod
    def _strip_region(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("working_scope.region 不得为空")
        return cleaned


class CandidateHypothesis(BaseModel):
    """博主演绎候选；仅审计，不得注入 user prompt。"""

    text: str


class ClueExtractionResult(BaseModel):
    """阶段2 线索抽取结构化结果。"""

    raw_given_clues: list[RawGivenClue] = Field(default_factory=list)
    working_scope: Optional[WorkingScope] = None
    candidate_hypotheses: list[CandidateHypothesis] = Field(default_factory=list)
