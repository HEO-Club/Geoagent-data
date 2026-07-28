"""COARSE 训练轨迹 Tool 允许/禁止策略单测。"""

from __future__ import annotations

from pipeline.coarse_tool_policy import (
    COARSE_FORBIDDEN_SEED_TOOLS,
    is_coarse_allowed_tool,
    is_coarse_forbidden_tool,
)


def test_forbidden_seeds() -> None:
    for name in COARSE_FORBIDDEN_SEED_TOOLS:
        assert is_coarse_forbidden_tool(name)
        assert not is_coarse_allowed_tool(name)


def test_core_and_geo_visual_allowed() -> None:
    assert is_coarse_allowed_tool("zoom_inspect")
    assert is_coarse_allowed_tool("compare_images_for_geolocation")
    assert is_coarse_allowed_tool("lookup_historical_satellite_map")
    assert is_coarse_allowed_tool("find_specific_features_in_satellite_map")
    assert is_coarse_allowed_tool("annotate_geographic_environment_on_image")
    assert is_coarse_allowed_tool("detect_terrain_features")
    assert is_coarse_allowed_tool("analyze_terrain_ambiguity")


def test_dynamic_satellite_name_allowed() -> None:
    assert is_coarse_allowed_tool("compare_images_satellite_custom")
    assert is_coarse_allowed_tool("inspect_satellite_layout_v2")


def test_unrelated_dynamic_not_allowed() -> None:
    assert not is_coarse_allowed_tool("search_tall_buildings_by_criteria")
    assert not is_coarse_allowed_tool("extract_geolocation_clues_from_chat_records")
