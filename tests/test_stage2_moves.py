"""stage2 Move 构建测试：SA 会话合并旁白，宁粗无碎。"""

from __future__ import annotations

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
            _act(5.2, 7.0, "在搜索框输入地标名", ["搜索框"]),
            _act(0.5, 2.5, "放大查看屋顶", ["屋顶"]),
        ]
        moves = build_moves(
            transcript_segment=transcript,
            screen_actions=screen_actions,
            agent_role=AgentRole.COARSE,
            time_range=(0.0, 10.0),
        )
        assert len(moves) == 2
        by_sa = {m.screen_action: m for m in moves}
        assert by_sa["放大查看屋顶"].narration == "先观察建筑。"
        assert by_sa["在搜索框输入地标名"].narration == "然后搜索地标。"
        assert by_sa["放大查看屋顶"].visible_clues == ["屋顶"]

    def test_same_sa_merges_multiple_narrations(self) -> None:
        """同一长 SA 覆盖多句旁白 → 合并为 1 个 Move。"""
        transcript = [
            _seg(52.0, 60.0, "打开许昌地图，几乎都是平原。"),
            _seg(69.0, 75.0, "桥跨度大，应是宽河。"),
            _seg(75.0, 83.0, "许昌附近没有宽河。"),
            _seg(83.0, 98.0, "更像郑州附近的黄河。"),
            _seg(98.0, 104.0, "按两山夹一桥排查。"),
        ]
        sa = _act(
            50.0,
            102.0,
            "将老照片与卫星地图并排对比，寻找两山夹一桥",
            ["卫星", "桥"],
        )
        moves = build_moves(
            transcript_segment=transcript,
            screen_actions=[sa],
            agent_role=AgentRole.COARSE,
            time_range=(0.0, 120.0),
        )
        assert len(moves) == 1
        m = moves[0]
        assert m.screen_action == sa.description
        assert "平原" in m.narration
        assert "宽河" in m.narration
        assert "黄河" in m.narration
        assert "两山夹一桥" in m.narration
        assert m.visible_clues == ["卫星", "桥"]

    def test_no_semantic_split_without_sa(self) -> None:
        """无 SA 时不按「然后」细切，保持整段旁白。"""
        transcript = [
            _seg(0.0, 10.0, "先看植被。然后打开地图确认。"),
        ]
        moves = build_moves(
            transcript_segment=transcript,
            screen_actions=[],
            agent_role=AgentRole.FINE,
            time_range=(0.0, 10.0),
        )
        assert len(moves) == 1
        assert "植被" in moves[0].narration
        assert "地图" in moves[0].narration
        assert moves[0].screen_action is None

    def test_thought_only_when_no_screen_overlap(self) -> None:
        moves = build_moves(
            transcript_segment=[_seg(1.0, 2.0, "这段只是口述推理。")],
            screen_actions=[_act(50.0, 51.0, "无关操作")],
            agent_role=AgentRole.COARSE,
            time_range=(0.0, 5.0),
        )
        # 旁白 Move + 区间外 SA 不进入；区间内仅口述
        assert len(moves) == 1
        assert moves[0].screen_action is None
        assert moves[0].narration == "这段只是口述推理。"
        assert moves[0].agent_role == AgentRole.COARSE

    def test_sa_without_narration_kept_if_geo(self) -> None:
        moves = build_moves(
            transcript_segment=[],
            screen_actions=[_act(1.0, 2.0, "打开卫星地图排查", ["卫星"])],
            agent_role=AgentRole.COARSE,
            time_range=(0.0, 5.0),
        )
        assert len(moves) == 1
        assert moves[0].screen_action == "打开卫星地图排查"

    def test_drops_pin_and_pure_chat_ui(self) -> None:
        moves = build_moves(
            transcript_segment=[
                _seg(0.0, 2.0, "我只好把这条信息置顶起来，静待时间的流逝。"),
                _seg(5.0, 8.0, "打开地图排查黄河沿岸的桥。"),
            ],
            screen_actions=[
                _act(0.0, 2.0, "展示将求助信息置顶的操作", ["消息列表", "置顶按钮"]),
                _act(5.0, 8.0, "打开地图排查黄河", ["卫星地图"]),
            ],
            agent_role=AgentRole.COARSE,
            time_range=(0.0, 10.0),
        )
        assert len(moves) == 1
        assert moves[0].screen_action == "打开地图排查黄河"
        assert "黄河" in moves[0].narration

    def test_drops_pure_chat_without_geo(self) -> None:
        moves = build_moves(
            transcript_segment=[
                _seg(0.0, 3.0, "半年前一位粉丝向我求助，想聊聊天。"),
            ],
            screen_actions=[
                _act(0.0, 3.0, "展示与粉丝的聊天记录", ["聊天记录界面"]),
            ],
            agent_role=AgentRole.COARSE,
            time_range=(0.0, 5.0),
        )
        assert moves == []

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
