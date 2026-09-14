"""spatial_filter.geometry_filter：通过程序按几何形状和空间关系筛选要素"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.spatial_filter._spatial import execute_geometry_filter


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """在已有 WGS84 要素上执行空间关系筛选。"""

    return execute_geometry_filter(
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
