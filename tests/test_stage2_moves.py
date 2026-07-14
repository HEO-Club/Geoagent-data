"""stage2 Move 构建测试：时间重叠对齐，禁止下标配对。"""

from __future__ import annotations

import pytest

from pipeline.schemas import (
    AgentRole,
    AgentTimeSegment,
    PreprocessResult,
    TimedScreenAction,
    TranscriptSegment,
    VideoInput,
)
from pipeline.stage2_moves import build_all_agent_moves, build_moves


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def _act(
    start: float,
    end: float,
    description: str,
    clues: list[str] | None = None,
) -> TimedScreenAction:
    return TimedScreenAction(
        start_time=start,
        end_time=end,
        description=description,
        visible_clues=clues or [],
    )


class TestBuildMoves:
    def test_aligns_by_time_overlap_not_index(self) -> None:
        # 刻意让列表顺序与时间顺序相反，验证不是按下标配对
        transcript = [
            _seg(0.0, 3.0, "先观察建筑。"),
            _seg(5.0, 8.0, "然后搜索地标。"),
        ]
        screen_actions = [
            _act(5.2, 7.0, "在搜索框输入地标名", ["搜索框"]),  # 应对第二条旁白
            _act(0.5, 2.5, "放大查看屋顶", ["屋顶"]),  # 应对第一条旁白
        ]
        moves = build_moves(
            transcript_segment=transcript,
            screen_actions=screen_actions,
            agent_role=AgentRole.COARSE,
            time_range=(0.0, 10.0),
        )
        assert len(moves) >= 2
        by_narration = {m.narration: m for m in moves if m.narration}
        assert by_narration["先观察建筑。"].screen_action == "放大查看屋顶"
        assert by_narration["然后搜索地标。"].screen_action == "在搜索框输入地标名"
        assert by_narration["先观察建筑。"].visible_clues == ["屋顶"]

    def test_semantic_boundary_splits_move(self) -> None:
        transcript = [
            _seg(0.0, 10.0, "先看植被。然后打开地图确认。"),
        ]
        moves = build_moves(
            transcript_segment=transcript,
            screen_actions=[],
            agent_role=AgentRole.FINE,
            time_range=(0.0, 10.0),
        )
        narrations = [m.narration for m in moves]
        assert len(narrations) >= 2
        assert any("植被" in n for n in narrations)
        assert any("地图" in n for n in narrations)

    def test_thought_only_when_no_screen_overlap(self) -> None:
        moves = build_moves(
            transcript_segment=[_seg(1.0, 2.0, "这段只是口述推理。")],
            screen_actions=[_act(50.0, 51.0, "无关操作")],
            agent_role=AgentRole.COARSE,
            time_range=(0.0, 5.0),
        )
        assert len(moves) == 1
        assert moves[0].screen_action is None
        assert moves[0].narration == "这段只是口述推理。"
        assert moves[0].agent_role == AgentRole.COARSE

    def test_filters_outside_time_range(self) -> None:
        moves = build_moves(
            transcript_segment=[
                _seg(0.0, 1.0, "区间外"),
                _seg(10.0, 11.0, "区间内推理。"),
            ],
            screen_actions=[_act(10.2, 10.8, "缩放")],
            agent_role=AgentRole.FINE,
            time_range=(10.0, 12.0),
        )
        assert all(m.start_time >= 10.0 for m in moves)
        assert all("区间外" not in m.narration for m in moves)


class TestBuildAllAgentMoves:
    def test_builds_per_role(self) -> None:
        video = VideoInput(
            video_path="v.mp4",
            transcript=[
                _seg(0.0, 5.0, "粗看像西欧。"),
                _seg(5.0, 10.0, "打开地图搜一下。"),
                _seg(12.0, 15.0, "核对一下是否吻合。"),
            ],
            groundtruth=(48.0, 2.0),
            source_platform="yt",
        )
        preprocess_result = PreprocessResult(
            answer_timestamp=10.0,
            agent_segments=[
                AgentTimeSegment(
                    agent_role=AgentRole.COARSE, start_time=0.0, end_time=5.0
                ),
                AgentTimeSegment(
                    agent_role=AgentRole.FINE, start_time=5.0, end_time=10.0
                ),
                AgentTimeSegment(
                    agent_role=AgentRole.VERIFIER, start_time=12.0, end_time=15.0
                ),
            ],
            revision_segments=[],
        )
        screen_actions_by_role = {
            AgentRole.COARSE: [_act(1.0, 2.0, "观察天空")],
            AgentRole.FINE: [_act(6.0, 7.0, "地图查询")],
            AgentRole.VERIFIER: [_act(13.0, 14.0, "对照街景")],
        }
        all_moves = build_all_agent_moves(
            video, preprocess_result, screen_actions_by_role
        )
        assert set(all_moves.keys()) == {
            AgentRole.COARSE,
            AgentRole.FINE,
            AgentRole.VERIFIER,
        }
        assert all_moves[AgentRole.COARSE][0].agent_role == AgentRole.COARSE
        assert any(
            m.screen_action == "地图查询" for m in all_moves[AgentRole.FINE]
        )
        assert any(
            m.screen_action == "对照街景" for m in all_moves[AgentRole.VERIFIER]
        )
