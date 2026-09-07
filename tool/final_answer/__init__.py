"""final_answer：提交最终地点。"""

from __future__ import annotations

from tool.final_answer.submit import execute as submit

OPERATIONS = {
    'submit': submit,
}

__all__ = [
    "OPERATIONS",
    'submit',
]
