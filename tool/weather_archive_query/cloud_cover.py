"""weather_archive_query.cloud_cover：查询指定日期和区域的历史云量或卫星云图"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 weather_archive_query.cloud_cover。尚未接入真实执行器。"""

    return not_implemented(
        'weather_archive_query',
        'cloud_cover',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
