"""flight_data_query.nearby_traffic：统计指定时间和空间范围内的航空器活动"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.flight_data_query._query import execute_nearby_traffic


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 flight_data_query.nearby_traffic：OpenSky 状态向量快照。"""

    return execute_nearby_traffic(purpose=purpose, inputs=inputs, ctx=ctx)
