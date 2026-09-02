"""llm_query.consult：把问题或上下文提交给外部推理模型并取得回答"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 llm_query.consult。尚未接入真实执行器。"""

    return not_implemented(
        'llm_query',
        'consult',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
