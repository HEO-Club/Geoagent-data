"""llm_query.consult：把问题或上下文提交给外部推理模型并取得回答"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.llm_query._query import execute_consult


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 llm_query.consult：向外部模型提交明确问题并取得建议。"""

    return execute_consult(purpose=purpose, inputs=inputs, ctx=ctx)
