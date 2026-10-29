"""location_verify：用街景比对 + 文字一致性验证最终定位答案。"""

from __future__ import annotations

from tool.location_verify.verify import execute as verify

OPERATIONS = {
    'verify': verify,
}

__all__ = [
    "OPERATIONS",
    'verify',
]
