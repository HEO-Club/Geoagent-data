"""osm_query：在 OSM/Overpass 中查询或统计要素。"""

from __future__ import annotations

from tool.osm_query.count import execute as count
from tool.osm_query.query import execute as query

OPERATIONS = {
    'query': query,
    'count': count,
}

__all__ = [
    "OPERATIONS",
    'count',
    'query',
]
