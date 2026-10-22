"""spatial_filter.geometry_filter：通过程序按几何形状和空间关系筛选要素"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.spatial_filter._filter import execute_geometry_filter


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 spatial_filter.geometry_filter：本地按空间关系筛选已有矢量要素。"""

    return execute_geometry_filter(purpose=purpose, inputs=inputs, ctx=ctx)
