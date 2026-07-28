"""working_scope 软先验「附近」vs 硬边界「内」规范化与 user_query 格式化。"""

from __future__ import annotations

from pipeline.evidence_routing import (
    ExtractedRawClue,
    ExtractedWorkingScope,
    RawClueRole,
    ScopeBoundKind,
    VideoContextExtraction,
    WorkingScope,
    context_from_extraction,
    format_working_scope_user_query,
    normalize_working_scope_phrase,
)
from pipeline.schemas import AgentRole, Move


def test_format_does_not_force_append_nei() -> None:
    """格式化不得把「附近」强行加成「内」。"""
    soft = WorkingScope(
        region="河南许昌附近",
        bound_kind=ScopeBoundKind.NEAR,
        raw_clue_texts=["拍摄地为河南许昌附近"],
    )
    q = format_working_scope_user_query(soft)
    assert "工作范围：河南许昌附近" in q
    assert "许昌内" not in q
    assert "附近内" not in q


def test_soft_clues_never_normalize_to_inside() -> None:
    """籍贯+离家不远只能得到附近软先验，不得写成市内。"""
    phrase, kind = normalize_working_scope_phrase(
        "河南许昌内",
        clue_texts=["河南许昌人", "离家不远"],
        bound_kind=ScopeBoundKind.INSIDE,
    )
    assert kind is ScopeBoundKind.NEAR
    assert phrase == "河南许昌附近"
    assert not phrase.endswith("内") or "附近" in phrase


def test_hard_unexited_normalizes_to_inside() -> None:
    """「未出X」才可规范化为「X内」。"""
    phrase, kind = normalize_working_scope_phrase(
        "示例省",
        clue_texts=["拍摄地未出示例省"],
    )
    assert kind is ScopeBoundKind.INSIDE
    assert phrase == "示例省内"
    q = format_working_scope_user_query(
        WorkingScope(region=phrase, bound_kind=kind, raw_clue_texts=["拍摄地未出示例省"])
    )
    assert q.endswith("工作范围：示例省内")


def test_context_accepts_soft_near_from_person_distance() -> None:
    """人物属性+软距离可支撑 near working_scope。"""
    moves = [
        Move(
            start_time=0.0,
            end_time=2.0,
            narration="网友说自己是河南许昌人，拍摄地离家不远。",
            screen_action="打开聊天",
            visible_clues=["拍摄地为河南许昌附近"],
            agent_role=AgentRole.COARSE,
        ),
        Move(
            start_time=3.0,
            end_time=5.0,
            narration="画面可见高地与桥。",
            screen_action="查看照片",
            visible_clues=["高地"],
            agent_role=AgentRole.COARSE,
        ),
    ]
    extraction = VideoContextExtraction(
        raw_clues=[
            ExtractedRawClue(
                move_index=0,
                text="河南许昌人、离家不远",
                clue_role=RawClueRole.PERSON_OR_SOCIAL_ATTRIBUTE,
            ),
            ExtractedRawClue(
                move_index=0,
                text="拍摄地为河南许昌附近",
                clue_role=RawClueRole.PHOTO_LOCATION_CONSTRAINT,
            ),
        ],
        working_scope=ExtractedWorkingScope(
            region="河南许昌附近",
            supporting_move_indices=[0],
            bound_kind=ScopeBoundKind.NEAR,
            rationale="聊天给出附近软先验",
        ),
        facts=[],
    )
    ctx = context_from_extraction(moves, extraction)
    assert ctx.working_scope is not None
    assert ctx.working_scope.bound_kind is ScopeBoundKind.NEAR
    assert ctx.working_scope.region == "河南许昌附近"
    assert "内" not in ctx.working_scope.region or "附近" in ctx.working_scope.region
