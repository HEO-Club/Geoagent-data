"""poi_search：搜索或浏览地图 POI。"""

from __future__ import annotations

from tool.poi_search.poi_search import execute as poi_search
from tool.poi_search.browse import execute as browse

OPERATIONS = {
    'poi_search': poi_search,
    'browse': browse,
}

__all__ = [
    "OPERATIONS",
    'poi_search',
    'browse',
]
