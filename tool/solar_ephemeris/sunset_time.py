"""solar_ephemeris.sunset_time：查询一个或多个地点的日落和暮光时间"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 solar_ephemeris.sunset_time。尚未接入真实执行器。"""

    return not_implemented(
        'solar_ephemeris',
        'sunset_time',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
