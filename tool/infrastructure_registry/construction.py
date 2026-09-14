"""infrastructure_registry.construction：查询建筑、桥梁或设施的建设和历史记录"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.infrastructure_registry._registry import execute_construction


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 infrastructure_registry.construction：地区建设历史档案检索。"""

    return execute_construction(purpose=purpose, inputs=inputs, ctx=ctx)
