"""infrastructure_registry.construction：查询建筑、桥梁或设施的建设和历史记录"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 infrastructure_registry.construction。尚未接入真实执行器。"""

    return not_implemented(
        'infrastructure_registry',
        'construction',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
