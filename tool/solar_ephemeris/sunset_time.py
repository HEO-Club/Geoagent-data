"""solar_ephemeris.sunset_time：查询一个或多个地点的日落和暮光时间"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.solar_ephemeris._ephemeris import execute_sunset_time


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 solar_ephemeris.sunset_time：pvlib NREL SPA 本地日出日没与暮光。"""

    return execute_sunset_time(purpose=purpose, inputs=inputs, ctx=ctx)
