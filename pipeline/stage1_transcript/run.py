"""阶段1：视频 → 带时间戳字幕。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.config import get_settings
from pipeline.llm import call_audio_text, call_structured
from pipeline.media.audio import extract_audio_range
from pipeline.media.keyframes import (
    extract_keyframes_range,
    video_duration_sec,
)
from pipeline.schemas.transcript import Stage1Result, TranscriptSegment

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SEC = 30.0
DEFAULT_MAX_FRAMES = 4


class _WindowSpeech(BaseModel):
    """无语音时，从画面可见字幕恢复的单窗口文本。"""

    text: str = Field(description="关键帧中可见的中文字幕正文；不可见则空串")


def load_anchor_transcript(path: str | Path) -> list[TranscriptSegment]:
    """加载旧字幕；仅用于时间锚，正文将被丢弃。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "segments" in raw:
        items = raw["segments"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"无法解析字幕 JSON: {path}")
    return [TranscriptSegment.model_validate(x) for x in items]


def build_time_windows(
    duration_sec: float,
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
    anchor: list[TranscriptSegment] | None = None,
) -> list[tuple[float, float]]:
    """构造分窗时间区间。"""
    if duration_sec <= 0:
        return []
    window_sec = max(5.0, float(window_sec))
    if anchor:
        sorted_anchor = sorted(anchor, key=lambda s: s.start)
        windows: list[tuple[float, float]] = []
        chunk_start: float | None = None
        chunk_end: float | None = None
        for seg in sorted_anchor:
            s = max(0.0, float(seg.start))
            e = min(duration_sec, float(seg.end))
            if e <= s:
                continue
            if chunk_start is None:
                chunk_start, chunk_end = s, e
                continue
            assert chunk_end is not None
            if e - chunk_start <= window_sec * 1.25:
                chunk_end = max(chunk_end, e)
            else:
                windows.append((chunk_start, chunk_end))
                chunk_start, chunk_end = s, e
        if chunk_start is not None and chunk_end is not None:
            windows.append((chunk_start, chunk_end))
        if windows:
            if windows[0][0] > 0.5:
                windows.insert(0, (0.0, windows[0][0]))
            if windows[-1][1] < duration_sec - 0.5:
                windows.append((windows[-1][1], duration_sec))
            return windows

    windows = []
    t = 0.0
    while t < duration_sec - 1e-9:
        end = min(duration_sec, t + window_sec)
        windows.append((t, end))
        t = end
    return windows


def _pick_frames(paths: list[str], max_frames: int = DEFAULT_MAX_FRAMES) -> list[str]:
    """从抽帧列表均匀取最多 max_frames 张。"""
    if not paths:
        return []
    if len(paths) <= max_frames:
        return list(paths)
    if max_frames == 1:
        return [paths[len(paths) // 2]]
    idxs = [
        round(i * (len(paths) - 1) / (max_frames - 1))
        for i in range(max_frames)
    ]
    return [paths[i] for i in idxs]


def transcribe_window(
    video_path: str,
    start: float,
    end: float,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> TranscriptSegment:
    """对单时间窗优先做音频 ASR；无结果时退回关键帧 VLM。"""
    if end <= start:
        return TranscriptSegment(start=start, end=end, text="")
    settings = get_settings()

    try:
        audio_path = extract_audio_range(video_path, (start, end))
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "stage1 audio extract failed %.2f-%.2f: %s; use VLM fallback",
            start,
            end,
            exc,
        )
    else:
        text = call_audio_text(audio_path).strip()
        if text:
            return TranscriptSegment(
                start=float(start), end=float(end), text=text
            )
        logger.info(
            "stage1 ASR returned empty %.2f-%.2f; use VLM fallback",
            start,
            end,
        )

    if not settings.STAGE1_ALLOW_VLM_FALLBACK:
        logger.info(
            "stage1 VLM fallback disabled %.2f-%.2f; keep empty audio window",
            start,
            end,
        )
        return TranscriptSegment(start=start, end=end, text="")

    duration = end - start
    fps = min(1.0, max(0.2, (max_frames + 1) / max(duration, 1e-6)))
    frames = extract_keyframes_range(video_path, (start, end), fps=fps)
    images = _pick_frames(frames, max_frames=max_frames)
    prompt = (
        "你正在为地理定位讲解视频恢复画面中可见的字幕。\n"
        f"本窗时间：{start:.2f}s – {end:.2f}s。\n"
        "音频 ASR 在本窗没有得到文本，请只读取所附关键帧中实际可见的"
        "中文字幕、自动字幕或讲解正文。\n"
        "规则：不得根据画面猜测口播；无清晰可见文字则 text 为空串；"
        "不要输出时间戳或解释性前缀。\n"
        "只输出结构化字段 text。"
    )
    if not images:
        return TranscriptSegment(start=start, end=end, text="")
    result = call_structured(prompt, _WindowSpeech, images=images, lane="vlm")
    return TranscriptSegment(
        start=float(start), end=float(end), text=(result.text or "").strip()
    )


def _merge_adjacent(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """吸收空窗并轻量去重。"""
    if not segments:
        return []
    out: list[TranscriptSegment] = []
    for seg in sorted(segments, key=lambda s: s.start):
        text = seg.text.strip()
        if not out:
            out.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))
            continue
        prev = out[-1]
        if not text:
            out[-1] = TranscriptSegment(
                start=prev.start, end=max(prev.end, seg.end), text=prev.text
            )
            continue
        if not prev.text:
            out[-1] = TranscriptSegment(
                start=prev.start, end=max(prev.end, seg.end), text=text
            )
            continue
        gap = seg.start - prev.end
        same = prev.text == text or text in prev.text or prev.text in text
        if gap <= 0.35 and same:
            out[-1] = TranscriptSegment(
                start=prev.start,
                end=max(prev.end, seg.end),
                text=prev.text if len(prev.text) >= len(text) else text,
            )
        else:
            out.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))
    return [s for s in out if s.text.strip()]


def _video_id_from_path(video_path: str) -> str:
    return Path(video_path).stem


def _write_segments(path: Path, segments: list[TranscriptSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [s.model_dump(mode="json") for s in segments],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_stage1(
    video_path: str,
    *,
    anchor_transcript_path: str | None = None,
    out_path: str | None = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[TranscriptSegment]:
    """视频 → 带时间戳字幕；可断点续跑。

    Args:
        video_path: 视频路径。
        anchor_transcript_path: 可选旧字幕，仅作时间锚。
        out_path: 正式字幕输出路径；默认 ``TRANSCRIPTS_DIR/{id}.json``。
        window_sec: 无锚点时的固定窗长。
        max_frames: 每窗最多关键帧数。

    Returns:
        合并后的 TranscriptSegment 列表。
    """
    settings = get_settings()
    video_id = _video_id_from_path(video_path)
    duration = video_duration_sec(video_path)
    anchor = (
        load_anchor_transcript(anchor_transcript_path)
        if anchor_transcript_path
        else None
    )
    if duration <= 0 and anchor:
        duration = max((s.end for s in anchor), default=0.0)
    if duration <= 0:
        raise ValueError(f"无法确定视频时长: {video_path}")

    windows = build_time_windows(duration, window_sec=window_sec, anchor=anchor)
    logger.info(
        "stage1: %s duration=%.1fs windows=%d", video_path, duration, len(windows)
    )

    formal_out = Path(out_path) if out_path else (
        Path(settings.TRANSCRIPTS_DIR) / f"{video_id}.json"
    )
    intermediate = Path(settings.INTERMEDIATE_DIR) / video_id / "stage1_transcript.json"
    raw_path = formal_out.with_suffix(formal_out.suffix + ".raw.json")

    segments: list[TranscriptSegment] = []
    if raw_path.is_file():
        try:
            raw_items = json.loads(raw_path.read_text(encoding="utf-8"))
            if (
                isinstance(raw_items, list)
                and len(raw_items) <= len(windows)
                and all(isinstance(x, dict) for x in raw_items)
            ):
                candid = [TranscriptSegment.model_validate(x) for x in raw_items]
                ok = True
                for i, seg in enumerate(candid):
                    w0, w1 = windows[i]
                    if abs(seg.start - w0) > 0.05 or abs(seg.end - w1) > 0.05:
                        ok = False
                        break
                if ok:
                    segments = candid
                    logger.info(
                        "resume stage1 from %s (%d/%d)",
                        raw_path,
                        len(segments),
                        len(windows),
                    )
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("ignore bad raw checkpoint %s: %s", raw_path, exc)
            segments = []

    for i, (start, end) in enumerate(windows):
        if i < len(segments):
            continue
        seg = transcribe_window(video_path, start, end, max_frames=max_frames)
        segments.append(seg)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(
                [s.model_dump(mode="json") for s in segments],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    merged = _merge_adjacent(segments)
    _write_segments(formal_out, merged)
    result = Stage1Result(video_id=video_id, video_path=video_path, segments=merged)
    intermediate.parent.mkdir(parents=True, exist_ok=True)
    intermediate.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return merged
