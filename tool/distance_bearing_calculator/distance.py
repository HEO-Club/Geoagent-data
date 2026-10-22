"""distance_bearing_calculator.distance：计算地点、要素或视点之间的距离"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.distance_bearing_calculator._geodesic import execute_distance


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 distance_bearing_calculator.distance：WGS84 椭球测地 / 沿几何 / 宽度。"""

    return execute_distance(purpose=purpose, inputs=inputs, ctx=ctx)
