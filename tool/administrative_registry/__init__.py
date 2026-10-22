"""administrative_registry：当前不可用；原用途为查询行政归属、标准地名和对象名录。地名行政归属改用 web_search。"""

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
