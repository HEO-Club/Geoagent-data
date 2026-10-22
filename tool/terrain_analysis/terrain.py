"""terrain_analysis.terrain：查询或计算高程、坡度和地形剖面"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.terrain_analysis._terrain import execute_terrain


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 terrain_analysis.terrain：Horn DEM 高程/坡度/坡向/剖面。"""

    return execute_terrain(purpose=purpose, inputs=inputs, ctx=ctx)
