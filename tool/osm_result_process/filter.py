"""osm_result_process.filter：在 OSM 查询结果中按属性或几何条件筛选"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.osm_result_process._processor import execute_filter


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """在已返回 OSM payload 上执行本地属性与可选空间筛选。"""

    return execute_filter(
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
