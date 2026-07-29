"""媒体工具：关键帧抽取。"""

from __future__ import annotations

from pathlib import Path

from pipeline.config import get_settings


def extract_keyframes(
    video_path: str,
    timestamps: list[float],
    *,
    out_dir: str | None = None,
) -> list[str]:
    """按时间戳抽帧，返回图像路径列表。

    Args:
        video_path: 视频文件路径。
        timestamps: 秒级时间戳列表。
        out_dir: 输出目录；默认 ``CACHE_DIR/keyframes/{stem}``。
    """
    import cv2

    if not timestamps:
        return []

    settings = get_settings()
    dest = Path(out_dir) if out_dir else (
        Path(settings.CACHE_DIR) / "keyframes" / Path(video_path).stem
    )
    dest.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if video_fps <= 0:
            video_fps = 25.0
        paths: list[str] = []
        for t in timestamps:
            frame_idx = int(round(max(0.0, float(t)) * video_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            out_path = dest / f"t{float(t):.3f}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"写入关键帧失败: {out_path}")
            paths.append(str(out_path.resolve()))
        return paths
    finally:
        cap.release()


def extract_keyframes_range(
    video_path: str,
    time_range: tuple[float, float],
    *,
    fps: float = 1.0,
    out_dir: str | None = None,
) -> list[str]:
    """按时间区间与频率抽帧（阶段1 分窗用）。"""
    if fps <= 0:
        raise ValueError(f"fps 必须为正数，收到: {fps}")
    start, end = time_range
    if end < start:
        raise ValueError(f"非法 time_range: {time_range}")
    if end == start:
        return []

    timestamps: list[float] = []
    interval = 1.0 / fps
    t = float(start)
    while t < end - 1e-9:
        timestamps.append(t)
        t += interval
    return extract_keyframes(video_path, timestamps, out_dir=out_dir)


def video_duration_sec(video_path: str) -> float:
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
        ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        return max(ms / 1000.0, 0.0)
    finally:
        cap.release()
