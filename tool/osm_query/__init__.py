"""osm_query：按类型、距离、沿线或交叉查询地图地物。"""

from __future__ import annotations

from tool.osm_query.count import execute as count
from tool.osm_query.crossings import execute as crossings
from tool.osm_query.intersect import execute as intersect
from tool.osm_query.near import execute as near
from tool.osm_query.query import execute as query

OPERATIONS = {
    "query": query,
    "count": count,
    "near": near,
    "crossings": crossings,
    "intersect": intersect,
}

__all__ = [
    "OPERATIONS",
    "count",
    "crossings",
    "intersect",
    "near",
    "query",
]
