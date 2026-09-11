"""infrastructure_registry.permit：查询许可、登记、编号或行业记录"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.infrastructure_registry._registry import execute_permit


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 infrastructure_registry.permit：地区许可与编号档案检索。"""

    return execute_permit(purpose=purpose, inputs=inputs, ctx=ctx)
