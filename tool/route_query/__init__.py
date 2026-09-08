"""route_query：查询道路连接和路线关系。"""

from __future__ import annotations

from tool.route_query.route import execute as route

OPERATIONS = {
    'route': route,
}

__all__ = [
    "OPERATIONS",
    'route',
]
