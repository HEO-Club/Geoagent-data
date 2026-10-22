"""administrative_registry.directory：查询特定类别设施、机构或对象名录"""

from __future__ import annotations

from typing import Any

from tool.administrative_registry._registry import execute_directory
from tool.contract import Observation, RuntimeContext


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 administrative_registry.directory：GeoNames 对象名录检索。"""

    return execute_directory(purpose=purpose, inputs=inputs, ctx=ctx)
