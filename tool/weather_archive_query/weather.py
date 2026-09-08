"""weather_archive_query.weather：查询历史天气、降水、温度或能见度"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 weather_archive_query.weather。尚未接入真实执行器。"""

    return not_implemented(
        'weather_archive_query',
        'weather',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
