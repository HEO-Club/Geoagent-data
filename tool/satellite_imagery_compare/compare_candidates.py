"""satellite_imagery_compare.compare_candidates：在多个候选区域的卫星影像中按同一视觉/空间模板进行比对"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 satellite_imagery_compare.compare_candidates。尚未接入真实执行器。"""

    return not_implemented(
        'satellite_imagery_compare',
        'compare_candidates',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
