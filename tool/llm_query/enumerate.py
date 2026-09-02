"""llm_query.enumerate：要求外部模型补充候选、对象清单或结构化条目"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 llm_query.enumerate。尚未接入真实执行器。"""

    return not_implemented(
        'llm_query',
        'enumerate',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
