"""streetview_query.change_time：切换街景年份或历史图层"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 streetview_query.change_time。尚未接入真实执行器。"""

    return not_implemented(
        'streetview_query',
        'change_time',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
