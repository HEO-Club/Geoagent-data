"""web_search.keyword_search：按关键词和可选站点范围检索网页"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.web_search._search import execute_keyword_search


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 web_search.keyword_search：开放网页关键词检索。"""

    return execute_keyword_search(purpose=purpose, inputs=inputs, ctx=ctx)
