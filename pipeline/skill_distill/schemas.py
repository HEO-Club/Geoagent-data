"""技能蒸馏的句子、补丁与技能库。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ScopeLevel = Literal["country", "region", "place", "point"]
SentenceKind = Literal["planning", "tool"]
SentenceStatus = Literal["example", "skill"]
PatchOp = Literal["add", "narrow", "duplicate"]


class DraftSentence(BaseModel):
    """模型提出的一句技能，尚未进入技能库。"""

    kind: SentenceKind
    text: str = Field(min_length=1)
    applicability: str = Field(min_length=1)
    level: ScopeLevel | None = None
    tool: str | None = None
    operation: str | None = None

    @model_validator(mode="after")
    def _check_slot(self) -> "DraftSentence":
        self.text = " ".join(self.text.split())
        self.applicability = " ".join(self.applicability.split())
        if self.kind == "planning" and self.level is None:
            raise ValueError("规划句必须给出适用范围级别")
        if self.kind == "tool" and not (
            (self.tool or "").strip() and (self.operation or "").strip()
        ):
            raise ValueError("工具句必须给出 tool 与 operation")
        return self


class AtomizeResult(BaseModel):
    """一条轨迹拆出的句子。"""

    sentences: list[DraftSentence] = Field(default_factory=list)


class SkillPatch(BaseModel):
    """对冻结技能库的一条修改建议。分析员不得直接改库。"""

    op: PatchOp
    sentence: DraftSentence | None = None
    target_id: str | None = None
    applicability: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _check_op(self) -> "SkillPatch":
        if self.op == "add" and self.sentence is None:
            raise ValueError("add 补丁必须带句子")
        if self.op in {"narrow", "duplicate"} and not (self.target_id or "").strip():
            raise ValueError(f"{self.op} 补丁必须指向已有句子")
        if self.op == "narrow" and not (self.applicability or "").strip():
            raise ValueError("narrow 补丁必须给出收窄后的适用条件")
        return self


class AnalystResult(BaseModel):
    """一个分析员对一条轨迹给出的补丁。"""

    patches: list[SkillPatch] = Field(default_factory=list)


class SkillSentence(BaseModel):
    """技能库中的一句。"""

    id: str
    kind: SentenceKind
    text: str
    applicability: str
    level: ScopeLevel | None = None
    tool: str | None = None
    operation: str | None = None
    source_trajectory_ids: list[str] = Field(default_factory=list)
    support_count: int = 0
    status: SentenceStatus = "example"


class ReviewItem(BaseModel):
    """不能进入技能库的句子或补丁。"""

    trajectory_id: str
    reason: str
    sentence: DraftSentence | None = None
    target_id: str | None = None


class SkillBank(BaseModel):
    """规划句与工具句的合并结果。"""

    sentences: list[SkillSentence] = Field(default_factory=list)
    review_queue: list[ReviewItem] = Field(default_factory=list)


class SkipRecord(BaseModel):
    """未进入蒸馏的轨迹及原因。"""

    trajectory_id: str
    path: str
    reason: str


class MergeReport(BaseModel):
    """一次蒸馏的计数，供人工查看种子库。"""

    input_trajectories: int
    admitted: int
    skipped: list[SkipRecord] = Field(default_factory=list)
    sentences: int
    skills: int
    examples: int
    review_queue: int
