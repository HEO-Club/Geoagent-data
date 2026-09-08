"""osm_query.count：统计符合条件的 OSM 要素数量或分布"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.osm_query._overpass import execute_count


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """统计真实 Overpass 回执或 result_store 中已有要素。"""

    return execute_count(
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
