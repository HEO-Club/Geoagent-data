"""route_query：当前不可用；原用途为查询道路连接和路线关系。"""

from __future__ import annotations

from tool.route_query.route import execute as route

OPERATIONS = {
    'route': route,
}

__all__ = [
    "OPERATIONS",
    'route',
]
