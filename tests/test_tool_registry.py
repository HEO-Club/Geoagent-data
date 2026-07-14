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
    assert all(t.tier is ToolTier.DRAFT for t in registry.values())
