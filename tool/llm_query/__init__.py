"""llm_query：向外部模型咨询或生成候选清单。"""

from __future__ import annotations

from tool.llm_query.consult import execute as consult
from tool.llm_query.enumerate import execute as enumerate

OPERATIONS = {
    'consult': consult,
    'enumerate': enumerate,
}

__all__ = [
    "OPERATIONS",
    'consult',
    'enumerate',
]
