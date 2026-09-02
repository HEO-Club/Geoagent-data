"""final_answer.submit：提交最终地点字符串或地点数组"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 final_answer.submit。尚未接入真实执行器。"""

    return not_implemented(
        'final_answer',
        'submit',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
