"""streetview_query.change_time：切换街景年份或历史图层"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.streetview_query._streetview import execute_change_time


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 streetview_query.change_time：只从实际 captured_at 中选择年份。"""

    return execute_change_time(purpose=purpose, inputs=inputs, ctx=ctx)
