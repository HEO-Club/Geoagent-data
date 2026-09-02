"""flight_data_query.track：查询指定航班或区域内航迹"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 flight_data_query.track。尚未接入真实执行器。"""

    return not_implemented(
        'flight_data_query',
        'track',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
