"""administrative_registry：查询行政归属、标准地名和对象名录。"""

from __future__ import annotations

from tool.administrative_registry.administrative import execute as administrative
from tool.administrative_registry.directory import execute as directory

OPERATIONS = {
    'administrative': administrative,
    'directory': directory,
}

__all__ = [
    "OPERATIONS",
    'administrative',
    'directory',
]
