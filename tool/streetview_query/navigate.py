"""streetview_query.navigate：沿道路或方向移动街景视点"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.streetview_query._streetview import execute_navigate


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 streetview_query.navigate：沿 Mapillary sequence 换相邻视点。"""

    return execute_navigate(purpose=purpose, inputs=inputs, ctx=ctx)
