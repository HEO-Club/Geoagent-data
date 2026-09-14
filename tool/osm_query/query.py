"""osm_query.query：按区域、标签和空间关系查询 OSM 要素"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行结构化 OSM 查询；注入客户端时走 result_store 链路。"""

    extras = ctx.extras if ctx is not None else {}
    if extras.get("overpass_client") is not None:
        from tool.osm_query._overpass import execute_query

        return execute_query(purpose=purpose, inputs=inputs, ctx=ctx)
    from tool.osm_query._query import execute_query

    return execute_query(purpose=purpose, inputs=inputs, ctx=ctx)
