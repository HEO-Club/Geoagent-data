"""weather_archive_query.weather：查询历史天气、降水、温度或能见度"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.weather_archive_query._archive import execute_weather


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 weather_archive_query.weather：Open-Meteo 再分析时序。"""

    return execute_weather(purpose=purpose, inputs=inputs, ctx=ctx)
