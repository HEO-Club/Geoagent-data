"""stage2：以 TimedScreenAction 会话为主轴合并旁白，构建粗粒度 Move。"""

from __future__ import annotations

import re

from pipeline.schemas import (
    AgentRole,
    Move,
    PreprocessResult,
    TimedScreenAction,
    TranscriptSegment,
    VideoInput,
)

# 纯 UI / 社交噪声（无地理训练价值）
_NON_TRAINABLE_UI_RE = re.compile(
    r"置顶|消息列表|聊天记录|弹幕|点赞|评论区|播放器|进度条|"
    r"滚动(?:页面|条)?|移动鼠标|切换(?:浏览器)?标签|拖拽窗口|"
    r"点击空白|hover",
    re.I,
)
_GEO_SIGNAL_RE = re.compile(
    r"高地|平原|山脉?|桥|河|江|湖|海岸|峡谷|丘陵|盆地|地形|地貌|"
    r"地理|空间关系|位置关系|卫星|地图|排除|候选|附近|收窄|"
    r"俯视|远景|背景|河岸|纠正|误认|排查|对比|屋顶|植被|建筑|"
    r"拍摄地|拍摄|哪里拍|公里|不远|平原|山区|定位",
    re.I,
)
_SETUP_ONLY_RE = re.compile(
    r"沟通|求助|网友|粉丝|聊天|家乡|籍贯|离家|回忆|半年|故事",
    re.I,
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


def _join_narration_parts(parts: list[tuple[float, str]]) -> str:
    """按时间序拼接旁白；中文句子间不强制插空格。"""
    texts = [t.strip() for _, t in sorted(parts, key=lambda x: x[0]) if t.strip()]
    if not texts:
        return ""
    out = texts[0]
    for t in texts[1:]:
        if out and t and out[-1] not in "。！？；!?;\n" and t[0] not in "，,。！？":
            out += t
        else:
            out += t
    return out


def is_non_trainable_move(
    narration: str,
    screen_action: str | None,
    visible_clues: list[str] | None = None,
) -> bool:
    """判断 Move 是否无地理训练价值（置顶/纯聊天/纯社交开场等）。

    有明确地理信号时即使夹杂 UI 词也保留。
    """
    narr = (narration or "").strip()
    screen = (screen_action or "").strip()
    clues = " ".join(visible_clues or [])
    blob = f"{narr} {screen} {clues}"
    if not blob.strip():
        return True
    if _GEO_SIGNAL_RE.search(blob):
        return False
    if _NON_TRAINABLE_UI_RE.search(screen) or _NON_TRAINABLE_UI_RE.search(clues):
        return True
    if _NON_TRAINABLE_UI_RE.search(narr) and not _GEO_SIGNAL_RE.search(narr):
        return True
    # 纯社交开场：有求助/聊天话术、无地貌/地图断言
    if narr and _SETUP_ONLY_RE.search(narr) and not _GEO_SIGNAL_RE.search(narr):
        if not screen or _NON_TRAINABLE_UI_RE.search(screen) or _SETUP_ONLY_RE.search(
            screen
        ):
            return True
    return False


def build_moves(
    transcript_segment: list[TranscriptSegment],
    screen_actions: list[TimedScreenAction],
    agent_role: AgentRole,
    time_range: tuple[float, float],
) -> list[Move]:
    """以每条 TimedScreenAction 为核合并重叠旁白为 1 个 Move（宁粗无碎）。

    无 SA 覆盖的旁白按原 TranscriptSegment 保留，默认不按语气/转折细切。
    剔除置顶/纯聊天等无地理训练价值段。不得按列表下标一一配对。
    """
    start, end = time_range
    if end < start:
        raise ValueError(f"非法 time_range: {time_range}")

    in_range_segs = sorted(
        _filter_by_range(transcript_segment, time_range),
        key=lambda s: s.start,
    )
    in_range_actions = sorted(
        _filter_actions_by_range(screen_actions, time_range),
        key=lambda a: a.start_time,
    )

    moves: list[Move] = []
    covered_seg_indices: set[int] = set()

    for act in in_range_actions:
        a_start = max(float(act.start_time), start)
        a_end = min(float(act.end_time), end)
        if a_end <= a_start:
            continue
        narr_parts: list[tuple[float, str]] = []
        for i, seg in enumerate(in_range_segs):
            if _overlap(float(seg.start), float(seg.end), a_start, a_end) <= 0:
                continue
            covered_seg_indices.add(i)
            narr_parts.append((float(seg.start), seg.text))
        narration = _join_narration_parts(narr_parts)
        clues = list(act.visible_clues)
        if is_non_trainable_move(narration, act.description, clues):
            continue
        moves.append(
            Move(
                start_time=a_start,
                end_time=a_end,
                narration=narration,
                screen_action=act.description,
                visible_clues=clues,
                agent_role=agent_role,
            )
        )

    # 无屏幕操作覆盖的旁白：整段保留，不细切；仍过滤无用段
    for i, seg in enumerate(in_range_segs):
        if i in covered_seg_indices:
            continue
        u_start = max(float(seg.start), start)
        u_end = min(float(seg.end), end)
        if u_end <= u_start:
            continue
        text = (seg.text or "").strip()
        if not text:
            continue
        if is_non_trainable_move(text, None, []):
            continue
        moves.append(
            Move(
                start_time=u_start,
                end_time=u_end,
                narration=text,
                screen_action=None,
                visible_clues=[],
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
