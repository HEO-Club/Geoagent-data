"""COARSE（Agent1）训练轨迹 Tool 允许/禁止策略。

固定三工具 + 视觉地图/卫星/地形类可进链；
web_search / map_query / reverse_image_search / submit_answer 禁止。
"""

from __future__ import annotations

import re

# 种子禁止：外部检索 / 地名 API / terminal
COARSE_FORBIDDEN_SEED_TOOLS: frozenset[str] = frozenset(
    {"web_search", "map_query", "reverse_image_search", "submit_answer"}
)

# 画面核心三工具
COARSE_CORE_TOOLS: frozenset[str] = frozenset(
    {"zoom_inspect", "ocr", "sun_position_calc"}
)

# 视觉地图/卫星/地形（非地名 API）
COARSE_GEO_VISUAL_TOOLS: frozenset[str] = frozenset(
    {
        "compare_images_for_geolocation",
        "lookup_historical_map_layout",
        "lookup_historical_satellite_map",
        "find_specific_features_in_satellite_map",
        "annotate_geographic_environment_on_image",
        "detect_terrain_features",
        "analyze_terrain_ambiguity",
        "analyze_terrain_visual_illusion",
    }
)

# 动态注册同名语义：允许进 COARSE 链
_COARSE_GEO_VISUAL_NAME_RE = re.compile(
    r"compare_images|satellite|terrain|annotate_geographic|"
    r"lookup_historical|detect_terrain|analyze_terrain|"
    r"find_specific_features",
    re.IGNORECASE,
)


def is_coarse_forbidden_tool(tool_name: str) -> bool:
    """是否为 COARSE 轨迹硬禁止 Tool。"""
    return tool_name in COARSE_FORBIDDEN_SEED_TOOLS


def is_coarse_allowed_tool(tool_name: str) -> bool:
    """是否允许出现在 COARSE 训练轨迹（禁止集优先）。"""
    if is_coarse_forbidden_tool(tool_name):
        return False
    if tool_name in COARSE_CORE_TOOLS or tool_name in COARSE_GEO_VISUAL_TOOLS:
        return True
    return bool(_COARSE_GEO_VISUAL_NAME_RE.search(tool_name))


def coarse_allowed_tool_names() -> frozenset[str]:
    """显式允许名集合（不含动态正则命中）。"""
    return COARSE_CORE_TOOLS | COARSE_GEO_VISUAL_TOOLS
