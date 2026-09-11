"""satellite_imagery_query.oblique_view：获取倾斜或三维地形视角"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.satellite_imagery_query._query import execute_oblique_view


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 satellite_imagery_query.oblique_view：正射预览加相机参数，不是真斜摄。"""

    return execute_oblique_view(purpose=purpose, inputs=inputs, ctx=ctx)
