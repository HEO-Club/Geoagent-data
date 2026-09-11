"""osm_result_process.filter：在 OSM 查询结果中按属性或几何条件筛选"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.osm_result_process._process import execute_filter


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 osm_result_process.filter：本地筛选已有 OSM 要素。"""

    return execute_filter(purpose=purpose, inputs=inputs, ctx=ctx)
