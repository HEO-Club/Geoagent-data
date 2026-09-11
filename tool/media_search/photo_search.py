"""media_search.photo_search：检索历史照片、航拍照片或公共图库"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.media_search._search import execute_photo_search


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 media_search.photo_search：检索历史照片或公共图库。"""

    return execute_photo_search(purpose=purpose, inputs=inputs, ctx=ctx)
