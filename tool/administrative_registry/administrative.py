"""administrative_registry.administrative：查询地点的行政归属、标准地名或水体名称"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 administrative_registry.administrative。尚未接入真实执行器。"""

    return not_implemented(
        'administrative_registry',
        'administrative',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
