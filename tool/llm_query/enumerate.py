"""llm_query.enumerate：要求外部模型补充候选、对象清单或结构化条目"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.llm_query._query import execute_enumerate


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 llm_query.enumerate：要求外部模型生成待检验候选清单。"""

    return execute_enumerate(purpose=purpose, inputs=inputs, ctx=ctx)
