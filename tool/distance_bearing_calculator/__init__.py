"""distance_bearing_calculator：计算距离或方位角。"""

from __future__ import annotations

from tool.distance_bearing_calculator.distance import execute as distance
from tool.distance_bearing_calculator.bearing import execute as bearing

OPERATIONS = {
    'distance': distance,
    'bearing': bearing,
}

__all__ = [
    "OPERATIONS",
    'distance',
    'bearing',
]
