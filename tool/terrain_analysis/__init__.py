"""terrain_analysis：计算高程、坡度和地形剖面。"""

from __future__ import annotations

from tool.terrain_analysis.terrain import execute as terrain

OPERATIONS = {
    'terrain': terrain,
}

__all__ = [
    "OPERATIONS",
    'terrain',
]
