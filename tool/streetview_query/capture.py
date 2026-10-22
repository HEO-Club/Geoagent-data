"""streetview_query.capture：获取指定视角的街景画面"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.streetview_query._streetview import execute_capture


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 streetview_query.capture：按朝向截取当前全景画面，用后删除本地文件。"""

    return execute_capture(purpose=purpose, inputs=inputs, ctx=ctx)
