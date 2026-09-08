"""satellite_imagery_query.retrieve：获取指定区域和时间的卫星或航片"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 satellite_imagery_query.retrieve。尚未接入真实执行器。"""

    return not_implemented(
        'satellite_imagery_query',
        'retrieve',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
