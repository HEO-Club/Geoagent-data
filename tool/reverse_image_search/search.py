"""reverse_image_search.search：提交图片并返回相似图片或来源页面"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.reverse_image_search._search import execute_search


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 reverse_image_search.search：提交整图并返回匹配 URL。"""

    return execute_search(purpose=purpose, inputs=inputs, ctx=ctx)
