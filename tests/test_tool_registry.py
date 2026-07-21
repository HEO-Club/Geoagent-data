"""初始 tool_registry.json 加载测试。"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.schemas import AgentRole, SEED_TOOL_NAMES, ToolDefinition

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "tool_registry.json"


def test_registry_file_exists() -> None:
    assert REGISTRY_PATH.is_file()


def test_load_and_index_by_name() -> None:
    items = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry = {item["name"]: ToolDefinition.model_validate(item) for item in items}
    # 种子 ⊆ 注册表；允许流水线增量写入非种子 Tool
    assert SEED_TOOL_NAMES.issubset(set(registry))
    assert registry["submit_answer"].is_terminal is True
    assert registry["submit_answer"].observation_fields == []
    assert AgentRole.COARSE in registry["web_search"].allowed_agents
    assert AgentRole.FINE in registry["map_query"].allowed_agents
    assert AgentRole.VERIFIER in registry["map_query"].allowed_agents
    assert registry["sun_position_calc"].allowed_agents == [AgentRole.COARSE]
    # schema-only：无 tier / executor_ref
    for name, tool in registry.items():
        dumped = tool.model_dump()
        assert "tier" not in dumped
        assert "executor_ref" not in dumped
        assert tool.name == name
