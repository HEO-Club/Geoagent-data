"""reverse_image_search.search：提交图片并返回相似图片或来源页面"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 reverse_image_search.search。尚未接入真实执行器。"""

    return not_implemented(
        'reverse_image_search',
        'search',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
