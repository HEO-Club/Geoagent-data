"""terrain_analysis.terrain：查询或计算高程、坡度和地形剖面"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 terrain_analysis.terrain。尚未接入真实执行器。"""

    return not_implemented(
        'terrain_analysis',
        'terrain',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
