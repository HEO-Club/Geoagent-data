"""route_query.route：查询道路连接、路线或沿线关系"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.route_query._route import execute_route


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 route_query.route：高德路径规划，可选自建 OSRM。"""

    return execute_route(purpose=purpose, inputs=inputs, ctx=ctx)
