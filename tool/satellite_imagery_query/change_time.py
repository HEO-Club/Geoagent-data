"""satellite_imagery_query.change_time：切换历史年份、季节或水期影像"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 satellite_imagery_query.change_time。尚未接入真实执行器。"""

    return not_implemented(
        'satellite_imagery_query',
        'change_time',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
