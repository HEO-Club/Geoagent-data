"""map_layer_query：默认 terrestris OSM WMS 出图；当前不支持时间轴、历史地图和 WFS。"""

from __future__ import annotations

from tool.map_layer_query.load_layer import execute as load_layer

OPERATIONS = {
    'load_layer': load_layer,
}

__all__ = [
    "OPERATIONS",
    'load_layer',
]
