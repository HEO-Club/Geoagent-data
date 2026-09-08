"""image_measure.measure：测量像素距离、角度、比例或可计算视觉量"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.image_measure._measure import execute_measure


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 image_measure.measure：像素量测，仅在 reference 可解析时换算实尺。"""

    return execute_measure(purpose=purpose, inputs=inputs, ctx=ctx)
