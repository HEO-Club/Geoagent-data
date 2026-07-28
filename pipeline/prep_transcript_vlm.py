"""分窗多模态重转录：用关键帧 VLM 生成带时间戳文字稿。

可选旧 ASR 仅作时间窗锚点，不采信其正文。输出 TranscriptSegment 列表 JSON。
本模块为离线数据准备，不进入 stage3–5 默认主链。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from pipeline.config import get_settings
from pipeline.llm import call_structured
from pipeline.schemas import TranscriptSegment
from pipeline.stage1_parse import extract_keyframes

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SEC = 30.0
DEFAULT_MAX_FRAMES = 4
_DEFAULT_OUT_DIR = Path("data/transcripts_vlm")


class _WindowSpeech(BaseModel):
    """单时间窗口播转录。"""

    text: str = Field(
        description="该时间窗内博主口播的中文转录；无语音则空串"
    )


def _video_duration_sec(video_path: str) -> float:
    """读取视频时长（秒）。"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0 and n > 0:
            return n / fps
        # 兜底：部分容器无 FRAME_COUNT
        ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        return max(ms / 1000.0, 0.0)
    finally:
        cap.release()


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
    anchor: Optional[list[TranscriptSegment]] = None,
) -> list[tuple[float, float]]:
    """构造分窗时间区间。

    有 anchor 时按其段起止拼成约 window_sec 的块；否则固定步长切分。
    """
    if duration_sec <= 0:
        return []
    window_sec = max(5.0, float(window_sec))
    if anchor:
        sorted_anchor = sorted(anchor, key=lambda s: s.start)
        windows: list[tuple[float, float]] = []
        chunk_start: Optional[float] = None
        chunk_end: Optional[float] = None
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
            # 覆盖片头/片尾空隙
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
        int(round(i * (len(paths) - 1) / (max_frames - 1)))
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
    """对单时间窗抽关键帧并 VLM 转录。"""
    if end <= start:
        return TranscriptSegment(start=start, end=end, text="")
    duration = end - start
    fps = min(1.0, max(0.2, (max_frames + 1) / max(duration, 1e-6)))
    frames = extract_keyframes(video_path, (start, end), fps=fps)
    images = _pick_frames(frames, max_frames=max_frames)
    prompt = (
        "你正在为地理定位讲解视频做分窗口播转录。\n"
        f"本窗时间：{start:.2f}s – {end:.2f}s。\n"
        "根据所附关键帧（屏幕录制/口播画面）写出该时间窗内博主口播的中文正文。\n"
        "规则：\n"
        "- 只转录本窗口播；无清晰语音则 text 为空串。\n"
        "- 专有名词以画面可见文字为准校正常见 ASR 错字；"
        "不得编造片尾才公布的最终答案级 POI/坐标。\n"
        "- 不要输出时间戳、说话人标签或解释性前缀。\n"
        "只输出结构化字段 text。"
    )
    if not images:
        return TranscriptSegment(start=start, end=end, text="")
    result = call_structured(prompt, _WindowSpeech, images=images, lane="vlm")
    text = (result.text or "").strip()
    return TranscriptSegment(start=float(start), end=float(end), text=text)


def _merge_adjacent(
    segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """吸收空窗并轻量去重；不把不同正文拼成超长单段（保留分窗时间轴）。"""
    if not segments:
        return []
    out: list[TranscriptSegment] = []
    for seg in sorted(segments, key=lambda s: s.start):
        text = seg.text.strip()
        if not out:
            out.append(
                TranscriptSegment(start=seg.start, end=seg.end, text=text)
            )
            continue
        prev = out[-1]
        # 空段：只并入时间轴，不污染正文
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
        # 紧邻且正文实质相同 → 去重合并；否则保留分窗边界
        gap = seg.start - prev.end
        same = prev.text == text or text in prev.text or prev.text in text
        if gap <= 0.35 and same:
            out[-1] = TranscriptSegment(
                start=prev.start,
                end=max(prev.end, seg.end),
                text=prev.text if len(prev.text) >= len(text) else text,
            )
        else:
            out.append(
                TranscriptSegment(start=seg.start, end=seg.end, text=text)
            )
    return [s for s in out if s.text.strip()]


def _inject_anchor_answer_cues(
    vlm: list[TranscriptSegment],
    anchor: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """若 VLM 稿无法定位答案宣布句，从锚点 ASR 回填最晚匹配段（结构线索）。

    不替换其它 VLM 正文；仅保证 stage0 `locate_answer_timestamp` 可工作。
    """
    from pipeline.stage0_preprocess import (
        _ANSWER_PATTERNS,
        _segment_matches,
        locate_answer_timestamp,
    )

    try:
        locate_answer_timestamp(vlm)
        return vlm
    except ValueError:
        pass

    cues = [s for s in anchor if _segment_matches(s, _ANSWER_PATTERNS)]
    if not cues:
        logger.warning("VLM 与锚点均无答案宣布句，无法回填")
        return vlm

    chosen = max(cues, key=lambda s: float(s.start))
    logger.info(
        "inject anchor answer cue at %.2fs (%d chars)",
        chosen.start,
        len(chosen.text),
    )
    # 去掉与 cue 高度重叠的 VLM 空/短窗，避免重复时间轴
    out: list[TranscriptSegment] = []
    for seg in vlm:
        overlap = max(
            0.0,
            min(seg.end, chosen.end) - max(seg.start, chosen.start),
        )
        if overlap > 0.5 * max(seg.end - seg.start, 1e-6):
            continue
        out.append(seg)
    out.append(
        TranscriptSegment(
            start=float(chosen.start),
            end=float(chosen.end),
            text=chosen.text.strip(),
        )
    )
    return sorted(out, key=lambda s: s.start)


def prep_transcript_vlm(
    video_path: str,
    *,
    output_path: str | Path,
    anchor_transcript_path: Optional[str] = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[TranscriptSegment]:
    """分窗 VLM 重转录并写入 JSON。

    Args:
        video_path: 视频路径。
        output_path: 输出 TranscriptSegment 列表 JSON。
        anchor_transcript_path: 可选旧字幕，仅作时间锚。
        window_sec: 无锚点时的固定窗长。
        max_frames: 每窗最多关键帧数。

    Returns:
        合并后的 TranscriptSegment 列表。
    """
    duration = _video_duration_sec(video_path)
    if duration <= 0:
        # 部分环境 CAP_PROP 失败：用锚点末尾或默认
        anchor_only = (
            load_anchor_transcript(anchor_transcript_path)
            if anchor_transcript_path
            else []
        )
        duration = max((s.end for s in anchor_only), default=0.0)
    if duration <= 0:
        raise ValueError(f"无法确定视频时长: {video_path}")

    anchor = (
        load_anchor_transcript(anchor_transcript_path)
        if anchor_transcript_path
        else None
    )
    windows = build_time_windows(
        duration, window_sec=window_sec, anchor=anchor
    )
    logger.info(
        "prep_transcript_vlm: %s duration=%.1fs windows=%d",
        video_path,
        duration,
        len(windows),
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out.with_suffix(out.suffix + ".raw.json")

    segments: list[TranscriptSegment] = []
    if raw_path.is_file():
        try:
            raw_items = json.loads(raw_path.read_text(encoding="utf-8"))
            if (
                isinstance(raw_items, list)
                and len(raw_items) <= len(windows)
                and all(isinstance(x, dict) for x in raw_items)
            ):
                segments = [
                    TranscriptSegment.model_validate(x) for x in raw_items
                ]
                # 仅当已缓存窗与当前窗边界一致时续跑
                ok = True
                for i, seg in enumerate(segments):
                    w0, w1 = windows[i]
                    if abs(seg.start - w0) > 0.05 or abs(seg.end - w1) > 0.05:
                        ok = False
                        break
                if ok:
                    logger.info(
                        "resume from %s (%d/%d windows)",
                        raw_path,
                        len(segments),
                        len(windows),
                    )
                else:
                    segments = []
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("ignore bad raw checkpoint %s: %s", raw_path, exc)
            segments = []

    for i, (start, end) in enumerate(windows):
        if i < len(segments):
            continue
        seg = transcribe_window(
            video_path, start, end, max_frames=max_frames
        )
        segments.append(seg)
        logger.info(
            "window %d/%d [%.1f-%.1f] chars=%d",
            i + 1,
            len(windows),
            start,
            end,
            len(seg.text),
        )
        raw_path.write_text(
            json.dumps(
                [s.model_dump(mode="json") for s in segments],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    merged = _merge_adjacent(segments)
    if anchor:
        merged = _inject_anchor_answer_cues(merged, anchor)
    payload = [s.model_dump(mode="json") for s in merged]
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return merged


def main() -> None:
    """CLI：单视频分窗 VLM 重转录。"""
    parser = argparse.ArgumentParser(
        description="分窗多模态重转录 → TranscriptSegment JSON"
    )
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument(
        "--out",
        default=None,
        help="输出 JSON 路径（默认 data/transcripts_vlm/{stem}.json）",
    )
    parser.add_argument(
        "--anchor-transcript",
        default=None,
        help="可选旧字幕路径（仅时间锚，不采信正文）",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=DEFAULT_WINDOW_SEC,
        help="无锚点时的固定窗长（秒）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="每窗最多关键帧数",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    stem = Path(args.video).stem
    out = args.out or str(_DEFAULT_OUT_DIR / f"{stem}.json")
    # 触达配置，确保 CACHE_DIR 等可用
    get_settings()
    segs = prep_transcript_vlm(
        args.video,
        output_path=out,
        anchor_transcript_path=args.anchor_transcript,
        window_sec=args.window_sec,
        max_frames=args.max_frames,
    )
    print(json.dumps({"output": out, "segments": len(segs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
