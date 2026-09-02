"""reverse_image_search.search_crop：提交指定裁剪区域进行局部反向搜索"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 reverse_image_search.search_crop。尚未接入真实执行器。"""

    return not_implemented(
        'reverse_image_search',
        'search_crop',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
