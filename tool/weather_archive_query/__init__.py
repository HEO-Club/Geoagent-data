"""weather_archive_query：查询历史天气、云量或积雪。"""

from __future__ import annotations

from tool.weather_archive_query.cloud_cover import execute as cloud_cover
from tool.weather_archive_query.weather import execute as weather
from tool.weather_archive_query.snow_cover import execute as snow_cover
from tool.weather_archive_query.refine_range import execute as refine_range

OPERATIONS = {
    'cloud_cover': cloud_cover,
    'weather': weather,
    'snow_cover': snow_cover,
    'refine_range': refine_range,
}

__all__ = [
    "OPERATIONS",
    'cloud_cover',
    'weather',
    'snow_cover',
    'refine_range',
]
