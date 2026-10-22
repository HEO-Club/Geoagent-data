"""satellite_imagery_query.retrieve：获取指定区域和时间的卫星或航片"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.satellite_imagery_query._query import execute_retrieve


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 satellite_imagery_query.retrieve：Copernicus 目录检索后取裁剪真彩预览。"""

    return execute_retrieve(purpose=purpose, inputs=inputs, ctx=ctx)
