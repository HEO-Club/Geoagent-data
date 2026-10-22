"""web_search：执行开放网页或指定站点搜索。地名的行政归属和标准名称也用本工具检索，坐标再交给 geocode。"""

from __future__ import annotations

from tool.web_search.keyword_search import execute as keyword_search
from tool.web_search.site_search import execute as site_search

OPERATIONS = {
    'keyword_search': keyword_search,
    'site_search': site_search,
}

__all__ = [
    "OPERATIONS",
    'keyword_search',
    'site_search',
]
