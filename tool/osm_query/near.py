"""osm_query.near：按距离查找相邻地物。"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.osm_query._relation import execute_near


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """返回与参照地物相距不超过给定米数的主体地物。"""

    return execute_near(purpose=purpose, inputs=inputs, ctx=ctx)
