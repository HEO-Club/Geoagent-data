"""infrastructure_registry：当前不可用；原用途为查询建设历史、许可和登记记录。"""

from __future__ import annotations

from tool.infrastructure_registry.construction import execute as construction
from tool.infrastructure_registry.permit import execute as permit

OPERATIONS = {
    'construction': construction,
    'permit': permit,
}

__all__ = [
    "OPERATIONS",
    'construction',
    'permit',
]
