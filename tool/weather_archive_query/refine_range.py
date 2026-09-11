"""weather_archive_query.refine_range：在已有结果上按连续时间窗细化气象范围"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.weather_archive_query._archive import execute_refine_range


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 weather_archive_query.refine_range：本地筛选已有气象时序。"""

    return execute_refine_range(purpose=purpose, inputs=inputs, ctx=ctx)
