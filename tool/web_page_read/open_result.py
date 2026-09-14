"""web_page_read.open_result：打开并读取选中的搜索结果页面"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.web_page_read._read import execute_open_result


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 web_page_read.open_result：打开 url 或搜索结果并抽取正文。"""

    return execute_open_result(purpose=purpose, inputs=inputs, ctx=ctx)
