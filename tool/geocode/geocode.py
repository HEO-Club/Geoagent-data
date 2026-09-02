"""geocode.geocode：在地名、地址和坐标表达之间查询映射"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 geocode.geocode。尚未接入真实执行器。"""

    return not_implemented(
        'geocode',
        'geocode',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
