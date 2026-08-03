"""媒体工具：从视频中按时间窗抽取压缩音频。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pipeline.config import get_settings


def extract_audio_range(
    video_path: str,
    time_range: tuple[float, float],
    *,
    out_dir: str | None = None,
) -> str:
    """抽取单声道 16kHz MP3，供窗口级 ASR 使用。

    使用 ``imageio-ffmpeg`` 自带的 ffmpeg，避免依赖系统 PATH。已经存在且
    非空的窗口文件会直接复用，因此 Stage 1 可以断点续跑。
    """
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    start, end = (float(time_range[0]), float(time_range[1]))
    if start < 0 or end <= start:
        raise ValueError(f"非法 time_range: {time_range}")

    settings = get_settings()
    dest = (
        Path(out_dir)
        if out_dir
        else Path(settings.CACHE_DIR) / "audio" / video.stem
    )
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"t{start:.3f}-{end:.3f}.mp3"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return str(out_path.resolve())

    import imageio_ffmpeg  # type: ignore[import-untyped]

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{end - start:.3f}",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        "-y",
        str(out_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=max(60.0, (end - start) * 3.0),
    )
    if completed.returncode != 0 or not out_path.is_file():
        out_path.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"音频抽取失败: {detail[-500:]}")
    if out_path.stat().st_size <= 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"音频抽取为空: {video_path} {time_range}")
    return str(out_path.resolve())
