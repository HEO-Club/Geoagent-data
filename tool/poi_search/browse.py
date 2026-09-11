"""poi_search.browse：在指定区域内浏览地图并返回候选要素"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行明确区域内的受限 POI 浏览。"""

    extras = ctx.extras if ctx is not None else {}
    if extras.get("overpass_client") is not None or extras.get("geocode_client") is not None:
        from tool.poi_search._search import execute_browse

        return execute_browse(purpose=purpose, inputs=inputs, ctx=ctx)
    from tool.poi_search._amap import execute_browse

    return execute_browse(purpose=purpose, inputs=inputs, ctx=ctx)
