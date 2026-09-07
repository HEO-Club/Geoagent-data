"""satellite_imagery_compare：执行多时相或多候选影像比较。"""

from __future__ import annotations

from tool.satellite_imagery_compare.compare_candidates import execute as compare_candidates
from tool.satellite_imagery_compare.compare_time import execute as compare_time

OPERATIONS = {
    'compare_candidates': compare_candidates,
    'compare_time': compare_time,
}

__all__ = [
    "OPERATIONS",
    'compare_candidates',
    'compare_time',
]
