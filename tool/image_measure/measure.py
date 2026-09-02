"""image_measure.measure：测量像素距离、角度、比例或可计算视觉量"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 image_measure.measure。尚未接入真实执行器。"""

    return not_implemented(
        'image_measure',
        'measure',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
