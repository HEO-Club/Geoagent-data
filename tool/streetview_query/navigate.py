"""streetview_query.navigate：沿道路或方向移动街景视点"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 streetview_query.navigate。尚未接入真实执行器。"""

    return not_implemented(
        'streetview_query',
        'navigate',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
