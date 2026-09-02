"""image_compare.compare：对两张或多张图片执行程序化视觉差异比较"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 image_compare.compare。尚未接入真实执行器。"""

    return not_implemented(
        'image_compare',
        'compare',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
