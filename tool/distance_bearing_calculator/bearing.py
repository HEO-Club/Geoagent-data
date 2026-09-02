"""distance_bearing_calculator.bearing：计算方向、方位角或朝向"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 distance_bearing_calculator.bearing。尚未接入真实执行器。"""

    return not_implemented(
        'distance_bearing_calculator',
        'bearing',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
