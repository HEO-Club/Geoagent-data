"""poi_search.browse：在指定区域内浏览地图并返回候选要素"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 poi_search.browse。尚未接入真实执行器。"""

    return not_implemented(
        'poi_search',
        'browse',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
