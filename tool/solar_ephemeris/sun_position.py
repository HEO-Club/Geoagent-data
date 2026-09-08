"""solar_ephemeris.sun_position：计算太阳高度角和方位角"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 solar_ephemeris.sun_position。尚未接入真实执行器。"""

    return not_implemented(
        'solar_ephemeris',
        'sun_position',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
