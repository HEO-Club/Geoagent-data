"""把各分析员的补丁合并进技能库。补丁在这一步之前不改库。"""

from __future__ import annotations

from pipeline.skill_distill.schemas import (
    DraftSentence,
    ReviewItem,
    SkillBank,
    SkillPatch,
    SkillSentence,
)

_SKILL_SUPPORT = 2


def apply_patches(
    bank: SkillBank,
    patches: list[tuple[str, SkillPatch]],
    *,
    operations: set[tuple[str, str]],
) -> SkillBank:
    """在副本上应用补丁。原库对象不会被修改。

    同一句正文在适用条件不同时保持为两句。至少两条不同轨迹支持的句子标为 skill，
    其余标为 example。目录外的工具句进入审核队列。
    """
    merged = bank.model_copy(deep=True)
    for trajectory_id, patch in patches:
        if patch.op == "add":
            _apply_add(merged, trajectory_id, patch, operations)
        elif patch.op == "duplicate":
            _apply_duplicate(merged, trajectory_id, patch)
        else:
            _apply_narrow(merged, trajectory_id, patch)
    _refresh_status(merged)
    return merged


def _apply_add(
    bank: SkillBank,
    trajectory_id: str,
    patch: SkillPatch,
    operations: set[tuple[str, str]],
) -> None:
    sentence = patch.sentence
    if sentence is None:
        return
    refusal = _catalog_refusal(sentence, operations)
    if refusal is not None:
        bank.review_queue.append(
            ReviewItem(
                trajectory_id=trajectory_id,
                reason=refusal,
                sentence=sentence,
            )
        )
        return
    existing = _find_same(bank, sentence)
    if existing is None:
        bank.sentences.append(_new_sentence(bank, sentence, trajectory_id))
        return
    _add_support(existing, trajectory_id)


def _apply_duplicate(bank: SkillBank, trajectory_id: str, patch: SkillPatch) -> None:
    target = _find_id(bank, patch.target_id or "")
    if target is None:
        bank.review_queue.append(
            ReviewItem(
                trajectory_id=trajectory_id,
                reason="duplicate 指向的句子不在冻结库中",
                target_id=patch.target_id,
                sentence=patch.sentence,
            )
        )
        return
    _add_support(target, trajectory_id)


def _apply_narrow(bank: SkillBank, trajectory_id: str, patch: SkillPatch) -> None:
    target = _find_id(bank, patch.target_id or "")
    applicability = " ".join((patch.applicability or "").split())
    if target is None or not applicability:
        bank.review_queue.append(
            ReviewItem(
                trajectory_id=trajectory_id,
                reason="narrow 指向的句子不在冻结库中",
                target_id=patch.target_id,
            )
        )
        return
    others = [item for item in target.source_trajectory_ids if item != trajectory_id]
    same_condition = target.applicability == applicability
    if others and not same_condition:
        copy = DraftSentence(
            kind=target.kind,
            text=target.text,
            applicability=applicability,
            level=target.level,
            tool=target.tool,
            operation=target.operation,
        )
        bank.sentences.append(_new_sentence(bank, copy, trajectory_id))
        return
    target.applicability = applicability
    _add_support(target, trajectory_id)


def _catalog_refusal(
    sentence: DraftSentence,
    operations: set[tuple[str, str]],
) -> str | None:
    if sentence.kind != "tool":
        return None
    tool = (sentence.tool or "").strip()
    operation = (sentence.operation or "").strip()
    if (tool, operation) in operations:
        return None
    return f"工具句不在正式目录中：{tool}.{operation}"


def _find_same(bank: SkillBank, sentence: DraftSentence) -> SkillSentence | None:
    for item in bank.sentences:
        if item.kind != sentence.kind:
            continue
        if item.text != sentence.text or item.applicability != sentence.applicability:
            continue
        if sentence.kind == "planning" and item.level == sentence.level:
            return item
        if sentence.kind == "tool" and item.tool == sentence.tool and item.operation == sentence.operation:
            return item
    return None


def _find_id(bank: SkillBank, sentence_id: str) -> SkillSentence | None:
    for item in bank.sentences:
        if item.id == sentence_id:
            return item
    return None


def _new_sentence(
    bank: SkillBank,
    sentence: DraftSentence,
    trajectory_id: str,
) -> SkillSentence:
    return SkillSentence(
        id=_next_id(bank),
        kind=sentence.kind,
        text=sentence.text,
        applicability=sentence.applicability,
        level=sentence.level,
        tool=(sentence.tool or "").strip() or None,
        operation=(sentence.operation or "").strip() or None,
        source_trajectory_ids=[trajectory_id],
        support_count=1,
        status="example",
    )


def _next_id(bank: SkillBank) -> str:
    highest = 0
    for sentence in bank.sentences:
        suffix = sentence.id.removeprefix("sent-")
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"sent-{highest + 1:04d}"


def _add_support(sentence: SkillSentence, trajectory_id: str) -> None:
    if trajectory_id not in sentence.source_trajectory_ids:
        sentence.source_trajectory_ids.append(trajectory_id)
    sentence.support_count = len(sentence.source_trajectory_ids)


def _refresh_status(bank: SkillBank) -> None:
    for sentence in bank.sentences:
        sentence.support_count = len(sentence.source_trajectory_ids)
        sentence.status = (
            "skill" if sentence.support_count >= _SKILL_SUPPORT else "example"
        )
