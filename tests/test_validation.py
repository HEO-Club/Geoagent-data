"""validation.py：params / observation / map_query 条件规则测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.schemas import AgentRole, ToolDefinition
from pipeline.tools.validation import (
    apply_param_defaults,
    validate_action_params,
    validate_observation,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = {
    t.name: t
    for t in (
        ToolDefinition.model_validate(item)
        for item in json.loads((ROOT / "tool_registry.json").read_text(encoding="utf-8"))
    )
}


def test_apply_param_defaults_fills_missing_optional() -> None:
    tool = REGISTRY["web_search"]
    out = apply_param_defaults(tool, {"query": "x", "purpose": "broad_discovery"})
    assert out["top_k"] == 3
    assert out["query"] == "x"


def test_reject_extra_params() -> None:
    tool = REGISTRY["web_search"]
    with pytest.raises(ValidationError):
        validate_action_params(
            tool,
            {"query": "x", "purpose": "broad_discovery", "extra": 1},
        )


def test_web_search_purpose_role_constraints() -> None:
    tool = REGISTRY["web_search"]
    validate_action_params(
        tool,
        {"query": "church", "purpose": "broad_discovery"},
        agent_role=AgentRole.COARSE,
    )
    with pytest.raises(ValidationError):
        validate_action_params(
            tool,
            {"query": "church", "purpose": "verification"},
            agent_role=AgentRole.COARSE,
        )
    with pytest.raises(ValidationError):
        validate_action_params(
            tool,
            {"query": "church", "purpose": "precise_lookup"},
            agent_role=AgentRole.VERIFIER,
        )
    validate_action_params(
        tool,
        {"query": "church", "purpose": "precise_lookup"},
        agent_role=AgentRole.FINE,
    )


def test_map_query_params_cross_constraints() -> None:
    tool = REGISTRY["map_query"]
    validate_action_params(tool, {"query": "Paris"})
    validate_action_params(tool, {"latlng": [48.86, 2.29]})
    validate_action_params(tool, {"query": "Paris", "latlng": [48.86, 2.29]})
    with pytest.raises(ValidationError):
        validate_action_params(tool, {})


def test_map_query_observation_status_rules() -> None:
    tool = REGISTRY["map_query"]
    ok = validate_observation(
        tool,
        {
            "status": "success",
            "error_message": None,
            "formatted_address": "Paris",
            "resolved_latlng": [48.8584, 2.2945],
            "place_type": "tourist_attraction",
        },
    )
    assert ok is not None
    assert ok["resolved_latlng"] == [48.8584, 2.2945]

    validate_observation(
        tool,
        {
            "status": "empty",
            "error_message": None,
            "formatted_address": None,
            "resolved_latlng": None,
            "place_type": None,
        },
    )
    validate_observation(
        tool,
        {
            "status": "error",
            "error_message": "upstream failed",
            "formatted_address": None,
            "resolved_latlng": None,
            "place_type": None,
        },
    )

    with pytest.raises(ValidationError):
        validate_observation(
            tool,
            {
                "status": "success",
                "error_message": None,
                "formatted_address": None,
                "resolved_latlng": None,
                "place_type": None,
            },
        )
    with pytest.raises(ValidationError):
        validate_observation(
            tool,
            {
                "status": "empty",
                "error_message": "should be null",
                "formatted_address": None,
                "resolved_latlng": None,
                "place_type": None,
            },
        )
    with pytest.raises(ValidationError):
        validate_observation(
            tool,
            {
                "status": "error",
                "error_message": None,
                "formatted_address": None,
                "resolved_latlng": None,
                "place_type": None,
            },
        )


def test_map_query_rejects_legacy_latlng_observation_key() -> None:
    tool = REGISTRY["map_query"]
    with pytest.raises(ValidationError):
        validate_observation(
            tool,
            {
                "status": "success",
                "error_message": None,
                "formatted_address": "Paris",
                "latlng": [48.8584, 2.2945],
                "place_type": "tourist_attraction",
            },
        )


def test_terminal_observation_must_be_none() -> None:
    tool = REGISTRY["submit_answer"]
    assert validate_observation(tool, None) is None
    with pytest.raises(ValidationError):
        validate_observation(tool, {"status": "success"})


def test_f1_seed_map_query_param_obs_names_disjoint() -> None:
    tool = REGISTRY["map_query"]
    param_names = {p.name for p in tool.params}
    obs_names = {o.name for o in tool.observation_fields}
    assert "latlng" in param_names
    assert "resolved_latlng" in obs_names
    assert param_names.isdisjoint(obs_names)
