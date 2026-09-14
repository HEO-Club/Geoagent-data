"""streetview_query.open：打开指定地点或候选点的街景"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.streetview_query._streetview import execute_open


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 streetview_query.open：按坐标检索 Mapillary 影像并建立会话。"""

    return execute_open(purpose=purpose, inputs=inputs, ctx=ctx)
