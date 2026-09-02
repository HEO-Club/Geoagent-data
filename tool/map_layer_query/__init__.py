"""map_layer_query：加载水系、地形、行政或历史地图图层。"""

from __future__ import annotations

from tool.map_layer_query.load_layer import execute as load_layer

OPERATIONS = {
    'load_layer': load_layer,
}

__all__ = [
    "OPERATIONS",
    'load_layer',
]
