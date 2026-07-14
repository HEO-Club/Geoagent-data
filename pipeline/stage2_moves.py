"""stage2：按时间重叠对齐 TranscriptSegment 与 TimedScreenAction，构建 Move。"""

from __future__ import annotations

import re
from typing import Optional

from pipeline.schemas import (
    AgentRole,
    Move,
    PreprocessResult,
    TimedScreenAction,
    TranscriptSegment,
    VideoInput,
)

# 语气 / 转折语义边界（切分 Move 的旁白单元）
_BOUNDARY_SPLIT_RE = re.compile(
    r"(?<=[。！？；!?;])|"
    r"(?=(?:但是|不过|然而|接着|然后|接下来|另外|此外|所以|因此|"
    r"but\s+|however\s+|next\s+|then\s+|so\s+))",
    re.IGNORECASE,
)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """两区间重叠时长（秒）；无重叠返回 0。"""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _filter_by_range(
    segments: list[TranscriptSegment],
    time_range: tuple[float, float],
) -> list[TranscriptSegment]:
    start, end = time_range
    return [
        seg
        for seg in segments
        if _overlap(seg.start, seg.end, start, end) > 0
    ]


def _filter_actions_by_range(
    actions: list[TimedScreenAction],
    time_range: tuple[float, float],
) -> list[TimedScreenAction]:
    start, end = time_range
    return [
        act
        for act in actions
        if _overlap(act.start_time, act.end_time, start, end) > 0
    ]


def _split_segment_by_semantics(seg: TranscriptSegment) -> list[TranscriptSegment]:
    """按语气/转折把单条文字稿切成更小语义单元；时间按字符比例分配。"""
    text = seg.text.strip()
    if not text:
        return []

    parts = [p.strip() for p in _BOUNDARY_SPLIT_RE.split(text) if p and p.strip()]
    if len(parts) <= 1:
        return [TranscriptSegment(start=seg.start, end=seg.end, text=text)]

    total_chars = sum(len(p) for p in parts)
    duration = max(seg.end - seg.start, 1e-6)
    cursor = float(seg.start)
    result: list[TranscriptSegment] = []
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            part_end = float(seg.end)
        else:
            part_end = cursor + duration * (len(part) / total_chars)
        result.append(TranscriptSegment(start=cursor, end=part_end, text=part))
        cursor = part_end
    return result


def _best_overlapping_action(
    unit_start: float,
    unit_end: float,
    actions: list[TimedScreenAction],
) -> Optional[TimedScreenAction]:
    """按时间重叠选取最佳屏幕操作；不得按下标配对。"""
    best: Optional[TimedScreenAction] = None
    best_ov = 0.0
    for act in actions:
        ov = _overlap(unit_start, unit_end, act.start_time, act.end_time)
        if ov > best_ov:
            best_ov = ov
            best = act
    return best


def build_moves(
    transcript_segment: list[TranscriptSegment],
    screen_actions: list[TimedScreenAction],
    agent_role: AgentRole,
    time_range: tuple[float, float],
) -> list[Move]:
    """按时间重叠对齐 TranscriptSegment 与 TimedScreenAction，
    再按语气/转折等语义边界切分 Move。

    不得按列表下标一一配对。
    """
    start, end = time_range
    if end < start:
        raise ValueError(f"非法 time_range: {time_range}")

    in_range_segs = _filter_by_range(transcript_segment, time_range)
    in_range_actions = _filter_actions_by_range(screen_actions, time_range)

    # 语义边界切分
    units: list[TranscriptSegment] = []
    for seg in sorted(in_range_segs, key=lambda s: s.start):
        units.extend(_split_segment_by_semantics(seg))

    moves: list[Move] = []
    for unit in units:
        # 裁剪到 agent 时间区间
        u_start = max(float(unit.start), start)
        u_end = min(float(unit.end), end)
        if u_end <= u_start:
            continue

        matched = _best_overlapping_action(u_start, u_end, in_range_actions)
        screen_action: Optional[str] = None
        clues: list[str] = []
        if matched is not None:
            screen_action = matched.description
            clues = list(matched.visible_clues)

        moves.append(
            Move(
                start_time=u_start,
                end_time=u_end,
                narration=unit.text.strip(),
                screen_action=screen_action,
                visible_clues=clues,
                agent_role=agent_role,
            )
        )

    # 无旁白但有屏幕操作：仍产出 Move（narration 为空串），避免丢失操作
    covered_actions: set[int] = set()
    for i, act in enumerate(in_range_actions):
        for mv in moves:
            if (
                mv.screen_action == act.description
                and _overlap(mv.start_time, mv.end_time, act.start_time, act.end_time) > 0
            ):
                covered_actions.add(i)
                break

    for i, act in enumerate(in_range_actions):
        if i in covered_actions:
            continue
        a_start = max(float(act.start_time), start)
        a_end = min(float(act.end_time), end)
        if a_end <= a_start:
            continue
        moves.append(
            Move(
                start_time=a_start,
                end_time=a_end,
                narration="",
                screen_action=act.description,
                visible_clues=list(act.visible_clues),
                agent_role=agent_role,
            )
        )

    moves.sort(key=lambda m: (m.start_time, m.end_time))
    return moves


def build_all_agent_moves(
    video_input: VideoInput,
    preprocess_result: PreprocessResult,
    screen_actions_by_role: dict[AgentRole, list[TimedScreenAction]],
) -> dict[AgentRole, list[Move]]:
    """为三个 Agent 分别构建 Move 列表。"""
    result: dict[AgentRole, list[Move]] = {}
    for seg in preprocess_result.agent_segments:
        role = seg.agent_role
        time_range = (seg.start_time, seg.end_time)
        actions = screen_actions_by_role.get(role, [])
        result[role] = build_moves(
            transcript_segment=video_input.transcript,
            screen_actions=actions,
            agent_role=role,
            time_range=time_range,
        )
    return result
