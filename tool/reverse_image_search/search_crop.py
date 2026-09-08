"""reverse_image_search.search_crop：提交指定裁剪区域进行局部反向搜索"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.reverse_image_search._search import execute_search_crop


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 reverse_image_search.search_crop：本地裁剪后再反向搜图。"""

    return execute_search_crop(purpose=purpose, inputs=inputs, ctx=ctx)
