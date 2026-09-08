"""osm_result_process.filter：在 OSM 查询结果中按属性或几何条件筛选"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 osm_result_process.filter。尚未接入真实执行器。"""

    return not_implemented(
        'osm_result_process',
        'filter',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
