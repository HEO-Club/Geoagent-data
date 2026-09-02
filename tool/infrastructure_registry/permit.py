"""infrastructure_registry.permit：查询许可、登记、编号或行业记录"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 infrastructure_registry.permit。尚未接入真实执行器。"""

    return not_implemented(
        'infrastructure_registry',
        'permit',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
