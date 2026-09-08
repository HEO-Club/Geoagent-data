"""web_search.keyword_search：按关键词和可选站点范围检索网页"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 web_search.keyword_search。尚未接入真实执行器。"""

    return not_implemented(
        'web_search',
        'keyword_search',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
