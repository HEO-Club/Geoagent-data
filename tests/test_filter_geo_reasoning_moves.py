"""地理推理链去噪（替代最短蒸馏）单测。"""

from __future__ import annotations

from pipeline.evidence_routing import (
    VideoChainContext,
    VideoFactClaim,
    filter_geo_reasoning_moves,
)
from pipeline.schemas import AgentRole, Move


def _move(
    start: float,
    end: float,
    narr: str,
    screen: str | None = None,
    clues: list[str] | None = None,
) -> Move:
    return Move(
        start_time=start,
        end_time=end,
        narration=narr,
        screen_action=screen,
        visible_clues=clues or [],
        agent_role=AgentRole.COARSE,
    )


def test_filter_keeps_trial_exclude_and_later_correct() -> None:
    pre = [
        _move(0, 2, "先观察屋顶与远桥。", "查看照片", ["屋顶"]),
        _move(5, 7, "排除许昌附近平原候选。", "打开地图", ["平原"]),
        _move(10, 12, "纠正：对岸是河岸不是山。", "对比地貌", ["河岸"]),
        _move(15, 17, "置顶等待以后再想。", "置顶消息", ["消息列表"]),
    ]
    routed = pre[:3]
    ctx = VideoChainContext(
        video_facts=[
            VideoFactClaim(
                fact_id="vf0",
                start_time=0,
                end_time=2,
                quote="屋顶",
                tokens=["屋顶"],
                kind="observe",
                source_move_index=0,
            ),
            VideoFactClaim(
                fact_id="vf1",
                start_time=5,
                end_time=7,
                quote="排除许昌",
                tokens=["许昌"],
                kind="exclude",
                source_move_index=1,
                excluded_candidates=["许昌"],
            ),
            VideoFactClaim(
                fact_id="vf2",
                start_time=10,
                end_time=12,
                quote="河岸",
                tokens=["河岸"],
                kind="correct",
                source_move_index=2,
                supporting_move_indices=[0],
                corrected_from="山",
                corrected_to="河岸",
            ),
            VideoFactClaim(
                fact_id="vf3",
                start_time=15,
                end_time=17,
                quote="置顶",
                tokens=["置顶"],
                kind="stall",
                source_move_index=3,
            ),
        ]
    )
    # 即使路由误把置顶带入，也应被滤掉；试错 exclude 须保留
    kept = filter_geo_reasoning_moves(pre, pre, ctx)
    narrs = " ".join(m.narration for m in kept)
    assert "排除许昌" in narrs
    assert "河岸" in narrs
    assert "屋顶" in narrs
    assert "置顶" not in narrs


def test_filter_drops_stall_only() -> None:
    pre = [
        _move(0, 1, "静待时间的流逝。", "置顶求助信息", ["置顶按钮"]),
    ]
    ctx = VideoChainContext(
        video_facts=[
            VideoFactClaim(
                fact_id="vf0",
                start_time=0,
                end_time=1,
                quote="静待",
                tokens=["静待"],
                kind="stall",
                source_move_index=0,
            )
        ]
    )
    assert filter_geo_reasoning_moves(pre, pre, ctx) == []
