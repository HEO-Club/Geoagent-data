"""satellite_imagery_compare.compare_time：调用影像服务进行多时相对比"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.satellite_imagery_compare._compare import execute_compare_time


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 satellite_imagery_compare.compare_time：同区域多时相网格对齐后做像素差。"""

    return execute_compare_time(purpose=purpose, inputs=inputs, ctx=ctx)
