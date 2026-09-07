"""spatial_filter.geometry_filter：通过程序按几何形状和空间关系筛选要素"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 spatial_filter.geometry_filter。尚未接入真实执行器。"""

    return not_implemented(
        'spatial_filter',
        'geometry_filter',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
