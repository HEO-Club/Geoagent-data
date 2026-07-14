"""太阳/阴影纬度估算 adapter（确定性启发式，可选 pysolar）。"""

from __future__ import annotations

import math
import re
from typing import Any, Optional


def _parse_local_time(value: str | None) -> float | None:
    if not value:
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", value.strip())
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour + minute / 60.0


def _estimate_latitude_range(
    shadow_direction_deg: float | None,
    local_hour: float | None,
) -> tuple[list[float] | None, str | None]:
    """基于阴影方位与地方时的粗略纬度区间启发式。"""
    if shadow_direction_deg is None and local_hour is None:
        return None, "缺少 shadow_direction_deg 与 estimated_local_time，无法估计"

    # 简化：正午附近阴影短且指向南北，给出宽纬度带
    if local_hour is not None and 11.0 <= local_hour <= 13.0:
        base = 0.0
        if shadow_direction_deg is not None:
            # 北半球正午阴影大致朝北（~0°）
            deviation = min(abs(shadow_direction_deg % 360 - 0.0), abs(shadow_direction_deg % 360 - 360.0))
            base = max(0.0, 45.0 - deviation / 2.0)
        return [max(-60.0, base - 15.0), min(60.0, base + 15.0)], None

    if shadow_direction_deg is not None:
        rad = math.radians(shadow_direction_deg % 360)
        # 启发式：用 cos 映射到 [-50, 50]
        mid = 50.0 * math.cos(rad)
        return [max(-66.0, mid - 20.0), min(66.0, mid + 20.0)], None

    return [-40.0, 40.0], "仅根据地方时给出宽区间"


def execute(params: dict[str, Any], image_path: str) -> dict[str, Any]:
    """执行 sun_position_calc（本地确定性，不访问付费 API）。"""
    _ = image_path
    try:
        shadow = params.get("shadow_direction_deg")
        shadow_f = float(shadow) if shadow is not None else None
        local_hour = _parse_local_time(params.get("estimated_local_time"))
        lat_range, note = _estimate_latitude_range(shadow_f, local_hour)
        if lat_range is None:
            return {
                "status": "empty",
                "error_message": None,
                "possible_latitude_range": None,
                "note": note,
            }
        return {
            "status": "success",
            "error_message": None,
            "possible_latitude_range": lat_range,
            "note": note,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": str(exc),
            "possible_latitude_range": None,
            "note": None,
        }
