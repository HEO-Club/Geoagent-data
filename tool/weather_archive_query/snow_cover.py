"""weather_archive_query.snow_cover：查询积雪范围和时间变化"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 weather_archive_query.snow_cover。尚未接入真实执行器。"""

    return not_implemented(
        'weather_archive_query',
        'snow_cover',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
