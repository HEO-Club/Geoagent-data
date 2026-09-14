"""media_search.video_search：检索指定地点、关键词或视角的视频"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.media_search._search import execute_video_search


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 media_search.video_search：在公开图库检索视频。"""

    return execute_video_search(purpose=purpose, inputs=inputs, ctx=ctx)
