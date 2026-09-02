"""satellite_imagery_query.oblique_view：获取倾斜或三维地形视角"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 satellite_imagery_query.oblique_view。尚未接入真实执行器。"""

    return not_implemented(
        'satellite_imagery_query',
        'oblique_view',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
