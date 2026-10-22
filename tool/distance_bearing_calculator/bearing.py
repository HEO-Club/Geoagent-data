"""distance_bearing_calculator.bearing：计算方向、方位角或朝向"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.distance_bearing_calculator._geodesic import execute_bearing


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 distance_bearing_calculator.bearing：WGS84 椭球真北方位角。"""

    return execute_bearing(purpose=purpose, inputs=inputs, ctx=ctx)
