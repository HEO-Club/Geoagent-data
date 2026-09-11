"""web_search.site_search：在指定站点或内容平台内部检索"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.web_search._search import execute_site_search


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 web_search.site_search：在指定站点内检索。"""

    return execute_site_search(purpose=purpose, inputs=inputs, ctx=ctx)
