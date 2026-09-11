"""administrative_registry.administrative：查询地点的行政归属、标准地名或水体名称"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.administrative_registry._registry import execute_administrative


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 administrative_registry.administrative：GeoNames 标准地名与行政层级。"""

    return execute_administrative(purpose=purpose, inputs=inputs, ctx=ctx)
