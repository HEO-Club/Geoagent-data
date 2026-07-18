"""初始 tool_registry.json 加载测试。"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.schemas import AgentRole, SEED_TOOL_NAMES, ToolDefinition, ToolTier

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "tool_registry.json"


def test_registry_file_exists() -> None:
    assert REGISTRY_PATH.is_file()


def test_load_and_index_by_name() -> None:
    items = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry = {item["name"]: ToolDefinition.model_validate(item) for item in items}
    assert set(registry) == SEED_TOOL_NAMES
    assert registry["submit_answer"].is_terminal is True
    assert registry["submit_answer"].observation_fields == []
    assert AgentRole.COARSE in registry["web_search"].allowed_agents
    assert AgentRole.FINE in registry["map_query"].allowed_agents
    assert AgentRole.VERIFIER in registry["map_query"].allowed_agents
    assert registry["sun_position_calc"].allowed_agents == [AgentRole.COARSE]
    # 已显式升档的 production seeds；其余仍为 draft
    expected_production = {
        "sun_position_calc": "pipeline.tools.sun_position.execute",
        "map_query": "pipeline.tools.map_query.execute",
        "web_search": "pipeline.tools.web_search.execute",
        "reverse_image_search": "pipeline.tools.reverse_image_search.execute",
        "ocr": "pipeline.tools.ocr.execute",
        "zoom_inspect": "pipeline.tools.zoom_inspect.execute",
    }
    for name, ref in expected_production.items():
        assert registry[name].tier is ToolTier.PRODUCTION
        assert registry[name].executor_ref == ref
    assert all(
        t.tier is ToolTier.DRAFT
        for name, t in registry.items()
        if name not in expected_production
    )
