"""web_search：执行开放网页或指定站点搜索。"""

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
