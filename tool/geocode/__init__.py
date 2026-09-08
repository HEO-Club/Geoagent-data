"""geocode：在地名、地址和坐标之间转换。"""

from __future__ import annotations

from tool.geocode.geocode import execute as geocode

OPERATIONS = {
    'geocode': geocode,
}

__all__ = [
    "OPERATIONS",
    'geocode',
]
