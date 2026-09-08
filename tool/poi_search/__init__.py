"""poi_search：搜索或浏览地图 POI。"""

from __future__ import annotations

from tool.poi_search.browse import execute as browse
from tool.poi_search.poi_search import execute as poi_search

OPERATIONS = {
    'poi_search': poi_search,
    'browse': browse,
}

__all__ = [
    "OPERATIONS",
    'browse',
    'poi_search',
]
