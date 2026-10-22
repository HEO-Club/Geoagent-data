"""osm_query.intersect：查找两类线的交叉点。"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.osm_query._relation import execute_intersect


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """返回两类线的交叉点。"""

    return execute_intersect(purpose=purpose, inputs=inputs, ctx=ctx)
