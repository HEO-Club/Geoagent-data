"""poi_search.poi_search：按名称、类别和区域搜索兴趣点"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 poi_search.poi_search。尚未接入真实执行器。"""

    return not_implemented(
        'poi_search',
        'poi_search',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
