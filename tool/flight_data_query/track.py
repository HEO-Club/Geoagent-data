"""flight_data_query.track：查询指定航班或区域内航迹"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.flight_data_query._query import execute_track


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 flight_data_query.track：OpenSky 实际航迹，不插值。"""

    return execute_track(purpose=purpose, inputs=inputs, ctx=ctx)
