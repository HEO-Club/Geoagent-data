"""stage0：带时间戳文字稿预处理 — 答案时间戳、Agent 区间、可选验证证据窗、返工区间。"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from pipeline.schemas import (
    AgentRole,
    AgentTimeSegment,
    PreprocessResult,
    TranscriptSegment,
    VideoInput,
)

# 博主口头宣布答案的常见句式（中英）
_ANSWER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"答案(就是|是|锁定)",
        r"(最终|所以|因此).{0,8}(是|在|位于)",
        r"(坐标|位置|地点|这里|这儿)(就是|是|在)",
        r"锁定(在|到|为)",
        r"(我猜|我认为|我觉得).{0,12}(是|在)",
        r"the\s+answer\s+is",
        r"(so\s+)?(this|it)\s+is\s+(in|at|located)",
        r"coordinates?\s+(are|is)",
        r"located\s+(at|in)",
    )
)

# COARSE → FINE 切换线索：开始假设验证 / 查证具体地点
_FINE_TRANSITION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(搜(一下|索)|查(一下|一下地图)|打开地图|看街景|街景)",
        r"(确认|验证一下|精确|缩小到|锁定候选)",
        r"(google\s*maps|street\s*view|look\s*up|search\s*(for|it))",
        r"(reverse\s*image|以图搜图)",
    )
)

# 视频内真实纠错 / 返工
_REVISION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(不对|错了|搞错|弄错|不是这里|不是这儿)",
        r"(重新|再看|再想|修正|改一下|推翻)",
        r"(wait|actually|scratch\s+that|i\s+was\s+wrong|not\s+there)",
    )
)

# 答案后可选验证证据话术（不得用于吞掉整段片尾）
_VERIFIER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(验证|核对|对照|交叉|是否吻合|对得上)",
        r"(再确认|复查|check\s*(again|it)|verify|cross[- ]?check)",
    )
)


def _segment_matches(segment: TranscriptSegment, patterns: Sequence[re.Pattern[str]]) -> bool:
    text = segment.text.strip()
    if not text:
        return False
    return any(p.search(text) for p in patterns)


def locate_answer_timestamp(transcript: list[TranscriptSegment]) -> float:
    """从文字稿中定位博主口头宣布答案的时间点（秒）。

    优先取时间轴上靠后的匹配片段起点；找不到则抛出 ValueError。
    """
    if not transcript:
        raise ValueError("transcript 为空，无法定位 answer_timestamp")

    matches: list[TranscriptSegment] = [
        seg for seg in transcript if _segment_matches(seg, _ANSWER_PATTERNS)
    ]
    if not matches:
        raise ValueError("未能在文字稿中定位答案宣布时间戳")

    # 答案通常靠近视频后段；取最晚匹配的 start
    chosen = max(matches, key=lambda s: s.start)
    return float(chosen.start)


def _find_first_match_start(
    transcript: list[TranscriptSegment],
    patterns: Sequence[re.Pattern[str]],
    *,
    before: float,
    after: float,
) -> float | None:
    """在 (after, before) 半开区间内找最早匹配片段的 start。"""
    candidates = [
        seg
        for seg in transcript
        if seg.start >= after and seg.start < before and _segment_matches(seg, patterns)
    ]
    if not candidates:
        return None
    return float(min(candidates, key=lambda s: s.start).start)


def _answer_segment_end(
    transcript: list[TranscriptSegment],
    answer_timestamp: float,
) -> float:
    """宣布答案的那条片段结束时间；若无精确命中则退回 answer_timestamp。"""
    for seg in transcript:
        if abs(seg.start - answer_timestamp) < 1e-6:
            return float(seg.end)
        if seg.start <= answer_timestamp < seg.end:
            return float(seg.end)
    return float(answer_timestamp)


def _merge_adjacent_windows(
    windows: list[tuple[float, float]],
    *,
    gap_tol: float = 0.5,
) -> list[tuple[float, float]]:
    """合并相邻/重叠的时间窗。"""
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: (w[0], w[1]))
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap_tol:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def select_post_answer_evidence_windows(
    transcript: list[TranscriptSegment],
    answer_timestamp: float,
) -> list[tuple[float, float]]:
    """在宣布答案句结束之后筛选可选验证证据短窗。

    仅保留含验证话术的片段；合并相邻窗。排除宣布句本身；
    不得默认吞掉答案后全部字幕。
    """
    if not transcript:
        return []
    answer_end = _answer_segment_end(transcript, answer_timestamp)
    windows: list[tuple[float, float]] = []
    for seg in transcript:
        if seg.end <= answer_end + 1e-9:
            continue
        # 跨越宣布句末尾的片段跳过，避免带入宣布内容
        if seg.start < answer_end:
            continue
        if _segment_matches(seg, _VERIFIER_PATTERNS):
            windows.append((float(seg.start), float(seg.end)))
    return _merge_adjacent_windows(windows)


def segment_by_agent_role(
    transcript: list[TranscriptSegment],
    answer_timestamp: float,
    post_answer_evidence_windows: Optional[list[tuple[float, float]]] = None,
) -> list[AgentTimeSegment]:
    """划分 COARSE / FINE / VERIFIER 时间区间。

    COARSE/FINE 落在 answer_timestamp 之前。
    VERIFIER：若证据窗非空取其并集；否则为零长度占位（主链由 stage5 合成）。
    """
    if not transcript:
        raise ValueError("transcript 为空，无法划分 Agent 时间区间")

    t_min = float(min(s.start for s in transcript))
    if answer_timestamp <= t_min:
        raise ValueError("answer_timestamp 不晚于文字稿起点，无法划分 COARSE/FINE")

    fine_start = _find_first_match_start(
        transcript,
        _FINE_TRANSITION_PATTERNS,
        before=answer_timestamp,
        after=t_min,
    )
    if fine_start is None:
        fine_start = t_min + (answer_timestamp - t_min) / 2.0

    if fine_start <= t_min:
        fine_start = t_min + max((answer_timestamp - t_min) * 0.1, 0.01)
    if fine_start >= answer_timestamp:
        fine_start = answer_timestamp - max((answer_timestamp - t_min) * 0.1, 0.01)
        if fine_start <= t_min:
            fine_start = (t_min + answer_timestamp) / 2.0

    answer_end = _answer_segment_end(transcript, answer_timestamp)
    windows = (
        list(post_answer_evidence_windows)
        if post_answer_evidence_windows is not None
        else select_post_answer_evidence_windows(transcript, answer_timestamp)
    )

    if windows:
        verifier_start = max(min(w[0] for w in windows), answer_end)
        verifier_end = max(w[1] for w in windows)
        if verifier_end < verifier_start:
            verifier_end = verifier_start
    else:
        verifier_start = float(answer_end)
        verifier_end = float(answer_end)

    return [
        AgentTimeSegment(
            agent_role=AgentRole.COARSE,
            start_time=t_min,
            end_time=fine_start,
        ),
        AgentTimeSegment(
            agent_role=AgentRole.FINE,
            start_time=fine_start,
            end_time=answer_timestamp,
        ),
        AgentTimeSegment(
            agent_role=AgentRole.VERIFIER,
            start_time=verifier_start,
            end_time=verifier_end,
        ),
    ]


def detect_revision_segments(
    transcript: list[TranscriptSegment],
) -> list[tuple[float, float]]:
    """检测视频内真实纠错（video_observed）时间区间。

    连续匹配的纠错片段会合并为同一区间。
    """
    if not transcript:
        return []

    ordered = sorted(transcript, key=lambda s: s.start)
    ranges: list[tuple[float, float]] = []
    current_start: float | None = None
    current_end: float | None = None

    for seg in ordered:
        if _segment_matches(seg, _REVISION_PATTERNS):
            if current_start is None:
                current_start = float(seg.start)
                current_end = float(seg.end)
            else:
                assert current_end is not None
                if seg.start <= current_end + 1.0:
                    current_end = max(current_end, float(seg.end))
                else:
                    ranges.append((current_start, current_end))
                    current_start = float(seg.start)
                    current_end = float(seg.end)
        elif current_start is not None and current_end is not None:
            ranges.append((current_start, current_end))
            current_start = None
            current_end = None

    if current_start is not None and current_end is not None:
        ranges.append((current_start, current_end))

    return ranges


def preprocess(video_input: VideoInput) -> PreprocessResult:
    """串联 stage0 子步骤，返回强类型 PreprocessResult（禁止裸 dict）。

    注意：不得消费 groundtruth；仅使用 transcript。
    """
    transcript = video_input.transcript
    answer_timestamp = locate_answer_timestamp(transcript)
    evidence_windows = select_post_answer_evidence_windows(
        transcript, answer_timestamp
    )
    agent_segments = segment_by_agent_role(
        transcript,
        answer_timestamp,
        post_answer_evidence_windows=evidence_windows,
    )
    revision_segments = detect_revision_segments(transcript)
    return PreprocessResult(
        answer_timestamp=answer_timestamp,
        agent_segments=agent_segments,
        revision_segments=revision_segments,
        post_answer_evidence_windows=evidence_windows,
    )
