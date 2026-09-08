"""poi_search.poi_search：按名称、类别和区域搜索兴趣点"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.poi_search._search import execute_poi_search


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行按名称、类别和区域的 POI 候选搜索。"""

    return execute_poi_search(
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
