"""weather_archive_query.cloud_cover：查询指定日期和区域的历史云量或卫星云图"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.weather_archive_query._archive import execute_cloud_cover


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 weather_archive_query.cloud_cover：云量时序加代表日 GIBS 云图。"""

    return execute_cloud_cover(purpose=purpose, inputs=inputs, ctx=ctx)
