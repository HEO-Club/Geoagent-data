"""visibility_analysis.sightline：计算视线、遮挡、可视域或射线交点"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 visibility_analysis.sightline。尚未接入真实执行器。"""

    return not_implemented(
        'visibility_analysis',
        'sightline',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
