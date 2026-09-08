"""satellite_imagery_compare.compare_time：调用影像服务进行多时相对比"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 satellite_imagery_compare.compare_time。尚未接入真实执行器。"""

    return not_implemented(
        'satellite_imagery_compare',
        'compare_time',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
