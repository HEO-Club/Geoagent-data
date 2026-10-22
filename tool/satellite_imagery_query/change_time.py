"""satellite_imagery_query.change_time：切换历史年份、季节或水期影像"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.satellite_imagery_query._query import execute_change_time


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 satellite_imagery_query.change_time：按新时间窗重搜同一区域目录。"""

    return execute_change_time(purpose=purpose, inputs=inputs, ctx=ctx)
