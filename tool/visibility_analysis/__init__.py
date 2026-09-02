"""visibility_analysis：计算视线、遮挡和可视域。"""

from __future__ import annotations

from tool.visibility_analysis.sightline import execute as sightline

OPERATIONS = {
    'sightline': sightline,
}

__all__ = [
    "OPERATIONS",
    'sightline',
]
