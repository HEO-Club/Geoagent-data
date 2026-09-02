"""osm_query.count：统计符合条件的 OSM 要素数量或分布"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 osm_query.count。尚未接入真实执行器。"""

    return not_implemented(
        'osm_query',
        'count',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
