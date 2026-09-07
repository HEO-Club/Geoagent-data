"""distance_bearing_calculator.distance：计算地点、要素或视点之间的距离"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 distance_bearing_calculator.distance。尚未接入真实执行器。"""

    return not_implemented(
        'distance_bearing_calculator',
        'distance',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
