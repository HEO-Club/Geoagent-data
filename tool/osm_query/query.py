"""osm_query.query：按区域、标签和空间关系查询 OSM 要素"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.osm_query._overpass import execute_query


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行结构化 OSM 查询；默认由执行器生成受限 Overpass QL。"""

    return execute_query(
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
