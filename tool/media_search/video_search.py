"""media_search.video_search：检索指定地点、关键词或视角的视频"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 media_search.video_search。尚未接入真实执行器。"""

    return not_implemented(
        'media_search',
        'video_search',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
