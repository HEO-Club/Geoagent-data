"""route_query.route：查询道路连接、路线或沿线关系"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 route_query.route。尚未接入真实执行器。"""

    return not_implemented(
        'route_query',
        'route',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
