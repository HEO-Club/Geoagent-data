"""web_search.site_search：在指定站点或内容平台内部检索"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 web_search.site_search。尚未接入真实执行器。"""

    return not_implemented(
        'web_search',
        'site_search',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
