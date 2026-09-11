"""final_answer.submit：提交最终地点字符串或地点数组"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.final_answer._submit import execute_submit


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 final_answer.submit：校验并登记已有地点，不补造答案。"""

    return execute_submit(purpose=purpose, inputs=inputs, ctx=ctx)
