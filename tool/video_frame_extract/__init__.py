"""video_frame_extract：从已找到视频中提取指定时间或场景帧。"""

from __future__ import annotations

from tool.video_frame_extract.frame_retrieve import execute as frame_retrieve

OPERATIONS = {
    'frame_retrieve': frame_retrieve,
}

__all__ = [
    "OPERATIONS",
    'frame_retrieve',
]
