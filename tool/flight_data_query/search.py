"""flight_data_query.search：按日期、区域、机场或航线查询航班记录"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 flight_data_query.search。尚未接入真实执行器。"""

    return not_implemented(
        'flight_data_query',
        'search',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
