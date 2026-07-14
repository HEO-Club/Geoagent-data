"""stage0 文字稿预处理测试。"""

from __future__ import annotations

import pytest

from pipeline.schemas import AgentRole, TranscriptSegment, VideoInput
from pipeline.stage0_preprocess import (
    detect_revision_segments,
    locate_answer_timestamp,
    preprocess,
    segment_by_agent_role,
)


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def _sample_transcript() -> list[TranscriptSegment]:
    return [
        _seg(0.0, 5.0, "先看建筑风格，像是北欧一带。"),
        _seg(5.0, 12.0, "植被和路牌文字也指向斯堪的纳维亚。"),
        _seg(12.0, 20.0, "接下来打开地图搜一下候选城市。"),
        _seg(20.0, 28.0, "街景里这个雕像能帮我确认具体位置。"),
        _seg(28.0, 32.0, "答案就是奥斯陆市政厅附近。"),
        _seg(32.0, 38.0, "再验证一下坐标和图像特征是否吻合。"),
    ]


class TestLocateAnswerTimestamp:
    def test_finds_chinese_answer_phrase(self) -> None:
        ts = locate_answer_timestamp(_sample_transcript())
        assert ts == pytest.approx(28.0)

    def test_prefers_later_match(self) -> None:
        transcript = [
            _seg(1.0, 2.0, "这里是大致区域。"),
            _seg(10.0, 11.0, "答案是某个候选。"),
            _seg(20.0, 21.0, "最终答案是巴黎。"),
        ]
        assert locate_answer_timestamp(transcript) == pytest.approx(20.0)

    def test_english_answer_phrase(self) -> None:
        transcript = [
            _seg(0.0, 3.0, "Looking at the architecture."),
            _seg(8.0, 10.0, "The answer is Tokyo."),
        ]
        assert locate_answer_timestamp(transcript) == pytest.approx(8.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="为空"):
            locate_answer_timestamp([])

    def test_no_match_raises(self) -> None:
        with pytest.raises(ValueError, match="未能"):
            locate_answer_timestamp([_seg(0.0, 1.0, "今天天气不错。")])


class TestSegmentByAgentRole:
    def test_three_segments_respect_answer_boundary(self) -> None:
        transcript = _sample_transcript()
        answer_ts = locate_answer_timestamp(transcript)
        segments = segment_by_agent_role(transcript, answer_ts)
        assert len(segments) == 3
        by_role = {s.agent_role: s for s in segments}

        assert by_role[AgentRole.COARSE].end_time <= by_role[AgentRole.FINE].start_time
        assert by_role[AgentRole.FINE].end_time == pytest.approx(answer_ts)
        # VERIFIER 不得包含宣布答案句本身
        assert by_role[AgentRole.VERIFIER].start_time >= 32.0
        assert by_role[AgentRole.COARSE].end_time == pytest.approx(12.0)

    def test_midpoint_fallback_without_fine_cue(self) -> None:
        transcript = [
            _seg(0.0, 5.0, "宏观上看像南欧。"),
            _seg(5.0, 10.0, "继续观察屋顶。"),
            _seg(10.0, 12.0, "答案就是罗马。"),
        ]
        segments = segment_by_agent_role(transcript, 10.0)
        coarse = next(s for s in segments if s.agent_role == AgentRole.COARSE)
        fine = next(s for s in segments if s.agent_role == AgentRole.FINE)
        assert coarse.end_time == pytest.approx(5.0)
        assert fine.start_time == pytest.approx(5.0)
        assert fine.end_time == pytest.approx(10.0)


class TestDetectRevisionSegments:
    def test_detects_and_merges_adjacent(self) -> None:
        transcript = [
            _seg(0.0, 2.0, "先看左边。"),
            _seg(5.0, 7.0, "不对，搞错了。"),
            _seg(7.2, 9.0, "重新看右边的路牌。"),
            _seg(15.0, 16.0, "答案是柏林。"),
        ]
        ranges = detect_revision_segments(transcript)
        assert len(ranges) == 1
        assert ranges[0][0] == pytest.approx(5.0)
        assert ranges[0][1] == pytest.approx(9.0)

    def test_empty_when_no_revision(self) -> None:
        assert detect_revision_segments(_sample_transcript()) == []


class TestPreprocess:
    def test_returns_preprocess_result_not_dict(self) -> None:
        video = VideoInput(
            video_path="dummy.mp4",
            transcript=_sample_transcript(),
            groundtruth=(59.91, 10.73),
            source_platform="bilibili",
        )
        result = preprocess(video)
        assert result.answer_timestamp == pytest.approx(28.0)
        assert len(result.agent_segments) == 3
        assert result.revision_segments == []
        # 不得依赖 groundtruth（仅校验输出结构；groundtruth 未进入逻辑）
        dumped = result.model_dump()
        assert "groundtruth" not in dumped
