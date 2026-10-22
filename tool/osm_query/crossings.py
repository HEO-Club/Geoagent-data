"""osm_query.crossings：查找落在一条线上的桥、坝、渡口或道口。"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.osm_query._relation import execute_crossings


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """返回落在一条线附近的桥、坝、渡口或道口。"""

    return execute_crossings(purpose=purpose, inputs=inputs, ctx=ctx)
