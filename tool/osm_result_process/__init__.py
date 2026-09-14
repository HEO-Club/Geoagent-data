"""osm_result_process：筛选或导出已有 OSM 结果。"""

from __future__ import annotations

from tool.osm_result_process.export import execute as export
from tool.osm_result_process.filter import execute as filter

OPERATIONS = {
    'filter': filter,
    'export': export,
}

__all__ = [
    "OPERATIONS",
    'export',
    'filter',
]
