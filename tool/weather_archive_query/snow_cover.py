"""weather_archive_query.snow_cover：查询积雪范围和时间变化"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.weather_archive_query._archive import execute_snow_cover


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 weather_archive_query.snow_cover：积雪深度时序与 NDSI 覆盖图层分栏。"""

    return execute_snow_cover(purpose=purpose, inputs=inputs, ctx=ctx)
