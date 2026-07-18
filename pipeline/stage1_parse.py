"""stage1：按 Agent 时间区间抽关键帧，并识别 TimedScreenAction。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.config import get_settings
from pipeline.llm import call_structured
from pipeline.schemas import TimedScreenAction


class _ScreenActionItem(BaseModel):
    """VLM 单条屏幕操作（内部结构化输出）。"""

    start_time: float
    end_time: float
    description: str
    visible_clues: list[str] = Field(default_factory=list)


class _ScreenActionBatch(BaseModel):
    """VLM 批量屏幕操作输出。"""

    actions: list[_ScreenActionItem] = Field(default_factory=list)


def _sample_keyframes_for_vlm(keyframes: list[str], max_frames: int) -> list[str]:
    """均匀抽样关键帧供 VLM；保留首尾。"""
    if max_frames <= 0 or len(keyframes) <= max_frames:
        return list(keyframes)
    if max_frames == 1:
        return [keyframes[len(keyframes) // 2]]
    last = len(keyframes) - 1
    indexes = sorted(
        {
            int(round(i * last / (max_frames - 1)))
            for i in range(max_frames)
        }
    )
    return [keyframes[i] for i in indexes]


def extract_keyframes(
    video_path: str,
    time_range: tuple[float, float],
    fps: float = 1.0,
) -> list[str]:
    """按 Agent 时间区间抽帧（不按 Move；Move 在 stage2 才生成）。

    Args:
        video_path: 视频文件路径。
        time_range: (start_sec, end_sec)。
        fps: 抽帧频率（帧/秒），默认 1.0。

    Returns:
        关键的关键帧图片路径列表（按时间升序）。
    """
    if fps <= 0:
        raise ValueError(f"fps 必须为正数，收到: {fps}")

    start, end = time_range
    if end < start:
        raise ValueError(f"非法 time_range: {time_range}")
    if end == start:
        return []

    import cv2  # 延迟导入，便于测试 mock

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if video_fps <= 0:
            video_fps = 25.0

        settings = get_settings()
        out_dir = Path(settings.CACHE_DIR) / "keyframes" / Path(video_path).stem
        out_dir.mkdir(parents=True, exist_ok=True)

        interval = 1.0 / fps
        paths: list[str] = []
        t = float(start)
        while t < end - 1e-9:
            frame_idx = int(round(t * video_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                t += interval
                continue
            out_path = out_dir / f"t{t:.3f}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"写入关键帧失败: {out_path}")
            paths.append(str(out_path.resolve()))
            t += interval
        return paths
    finally:
        cap.release()


def _estimate_keyframe_timestamps(
    time_range: tuple[float, float],
    n_frames: int,
    fps: float,
) -> list[float]:
    """根据抽帧参数估计各关键帧对应时间戳。"""
    if n_frames <= 0:
        return []
    start, end = time_range
    interval = 1.0 / fps if fps > 0 else 1.0
    stamps: list[float] = []
    t = float(start)
    for _ in range(n_frames):
        if t >= end:
            break
        stamps.append(t)
        t += interval
    # 若帧数与估算不一致，均匀铺开兜底
    if len(stamps) != n_frames:
        if n_frames == 1:
            return [float(start)]
        step = (end - start) / n_frames
        return [float(start + i * step) for i in range(n_frames)]
    return stamps


def detect_screen_actions(
    keyframes: list[str],
    narration_context: str,
    time_range: tuple[float, float],
) -> list[TimedScreenAction]:
    """VLM 识别屏幕操作，产出带时间戳的 TimedScreenAction 列表。

    Args:
        keyframes: 关键帧图片路径。
        narration_context: 对应时间区间内的旁白上下文。
        time_range: 所属 Agent 时间区间。

    Returns:
        TimedScreenAction 列表；无关键帧时返回空列表。
    """
    if not keyframes:
        return []

    start, end = time_range
    if end < start:
        raise ValueError(f"非法 time_range: {time_range}")

    settings = get_settings()
    vlm_frames = _sample_keyframes_for_vlm(
        keyframes, int(settings.STAGE1_VLM_MAX_FRAMES)
    )

    prompt = (
        "你是地理定位讲解视频的屏幕操作标注器。"
        "根据给定关键帧与旁白上下文，识别博主在屏幕上的有意义操作"
        "（如缩放查看、OCR 读文字、搜索、打开地图、街景浏览等）。"
        "忽略纯鼠标移动、无意义滚动、切换标签等 UI 噪声。"
        "每条操作必须给出 start_time/end_time（秒，落在给定时间区间内）、"
        "description（简洁中文或中英混合）、visible_clues（可见线索列表，可空）。"
        f"\n时间区间: [{start:.3f}, {end:.3f})"
        f"\n旁白上下文:\n{narration_context}\n"
        f"\n说明: 关键帧已从区间内均匀抽样 {len(vlm_frames)}/{len(keyframes)} 张。"
        "只输出结构化字段，不要编造视频中未出现的操作。"
    )

    batch = call_structured(
        prompt,
        _ScreenActionBatch,
        images=vlm_frames,
    )
    if not isinstance(batch, _ScreenActionBatch):
        batch = _ScreenActionBatch.model_validate(batch)

    results: list[TimedScreenAction] = []
    # 缺省按 1fps 估计关键帧时间，供越界结果兜底
    fallback_fps = 1.0
    fallback_times = _estimate_keyframe_timestamps(
        time_range, len(keyframes), fallback_fps
    )

    for i, item in enumerate(batch.actions):
        s = float(item.start_time)
        e = float(item.end_time)
        # 约束到 time_range；非法则用关键帧估计时间兜底
        if e < s or s < start - 1e-3 or e > end + 1e-3:
            if i < len(fallback_times):
                s = fallback_times[i]
                e = min(end, s + 1.0 / fallback_fps)
            else:
                continue
        s = max(s, start)
        e = min(max(e, s), end)
        results.append(
            TimedScreenAction(
                start_time=s,
                end_time=e,
                description=item.description.strip(),
                visible_clues=list(item.visible_clues),
            )
        )

    results.sort(key=lambda a: (a.start_time, a.end_time))
    return results
