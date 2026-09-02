"""flight_data_query：查询航班档案、航迹和附近航空活动。"""

from __future__ import annotations

from tool.flight_data_query.search import execute as search
from tool.flight_data_query.track import execute as track
from tool.flight_data_query.nearby_traffic import execute as nearby_traffic

OPERATIONS = {
    'search': search,
    'track': track,
    'nearby_traffic': nearby_traffic,
}

__all__ = [
    "OPERATIONS",
    'search',
    'track',
    'nearby_traffic',
]
