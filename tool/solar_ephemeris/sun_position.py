"""solar_ephemeris.sun_position：计算太阳高度角和方位角"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.solar_ephemeris._ephemeris import execute_sun_position


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 solar_ephemeris.sun_position：pvlib NREL SPA 本地太阳位置。"""

    return execute_sun_position(purpose=purpose, inputs=inputs, ctx=ctx)
