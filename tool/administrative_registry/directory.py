"""administrative_registry.directory：查询特定类别设施、机构或对象名录"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 administrative_registry.directory。尚未接入真实执行器。"""

    return not_implemented(
        'administrative_registry',
        'directory',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
