"""osm_query.query：按区域、标签和空间关系查询 OSM 要素"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 osm_query.query。尚未接入真实执行器。"""

    return not_implemented(
        'osm_query',
        'query',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
