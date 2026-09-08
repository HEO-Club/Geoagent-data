"""shadow_analysis：根据太阳和物体参数计算理论阴影。"""

from __future__ import annotations

from tool.shadow_analysis.shadow_model import execute as shadow_model

OPERATIONS = {
    'shadow_model': shadow_model,
}

__all__ = [
    "OPERATIONS",
    'shadow_model',
]
