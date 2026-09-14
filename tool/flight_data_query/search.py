"""flight_data_query.search：按日期、区域、机场或航线查询航班记录"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.flight_data_query._query import execute_search


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 flight_data_query.search：OpenSky 实际航班档案。"""

    return execute_search(purpose=purpose, inputs=inputs, ctx=ctx)
