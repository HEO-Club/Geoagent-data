"""web_page_read.open_result：打开并读取选中的搜索结果页面"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 web_page_read.open_result。尚未接入真实执行器。"""

    return not_implemented(
        'web_page_read',
        'open_result',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
