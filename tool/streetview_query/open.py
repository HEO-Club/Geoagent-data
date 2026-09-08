"""streetview_query.open：打开指定地点或候选点的街景"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 streetview_query.open。尚未接入真实执行器。"""

    return not_implemented(
        'streetview_query',
        'open',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
