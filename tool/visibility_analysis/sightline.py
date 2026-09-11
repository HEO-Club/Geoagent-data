"""visibility_analysis.sightline：计算视线、遮挡、可视域或射线交点"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.visibility_analysis._sightline import execute_sightline


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 visibility_analysis.sightline：点到点 DEM 视线，不做区域可视域。"""

    return execute_sightline(purpose=purpose, inputs=inputs, ctx=ctx)
