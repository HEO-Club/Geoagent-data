"""solar_ephemeris：计算日落时间和太阳位置。"""

from __future__ import annotations

from tool.solar_ephemeris.sunset_time import execute as sunset_time
from tool.solar_ephemeris.sun_position import execute as sun_position

OPERATIONS = {
    'sunset_time': sunset_time,
    'sun_position': sun_position,
}

__all__ = [
    "OPERATIONS",
    'sunset_time',
    'sun_position',
]
