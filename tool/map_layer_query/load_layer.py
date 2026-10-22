"""map_layer_query.load_layer：按范围加载地图图层；当前不支持时间轴、历史地图和 WFS。"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.map_layer_query._layers import execute_load_layer


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 map_layer_query.load_layer：默认 terrestris WMS 出图；当前不支持时间轴、历史地图和 WFS。"""

    return execute_load_layer(purpose=purpose, inputs=inputs, ctx=ctx)
