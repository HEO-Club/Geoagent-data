"""spatial_filter：按几何和空间关系过滤要素。"""

from __future__ import annotations

from tool.spatial_filter.geometry_filter import execute as geometry_filter

OPERATIONS = {
    'geometry_filter': geometry_filter,
}

__all__ = [
    "OPERATIONS",
    'geometry_filter',
]
