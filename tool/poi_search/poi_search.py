"""poi_search.poi_search：按名称、类别和区域搜索兴趣点"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行按名称、类别和区域的 POI 候选搜索。"""

    extras = ctx.extras if ctx is not None else {}
    if extras.get("overpass_client") is not None or extras.get("geocode_client") is not None:
        from tool.poi_search._search import execute_poi_search

        return execute_poi_search(purpose=purpose, inputs=inputs, ctx=ctx)
    from tool.poi_search._amap import execute_poi_search

    return execute_poi_search(purpose=purpose, inputs=inputs, ctx=ctx)
