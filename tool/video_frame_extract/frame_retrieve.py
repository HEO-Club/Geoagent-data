"""video_frame_extract.frame_retrieve：从已找到的视频中提取指定时间或视角的帧"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 video_frame_extract.frame_retrieve。尚未接入真实执行器。"""

    return not_implemented(
        'video_frame_extract',
        'frame_retrieve',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
