"""种子 Tool observation schema 校验烟测（无真实 executor / 无外部 API）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.schemas import ToolDefinition
from pipeline.tools.validation import validate_observation

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = {
    t["name"]: ToolDefinition.model_validate(t)
    for t in json.loads((ROOT / "tool_registry.json").read_text(encoding="utf-8"))
}


def test_web_search_observation_schema() -> None:
    obs = {
        "status": "success",
        "error_message": None,
        "results": [
            {"title": "T", "snippet": "S", "url": "https://e.com"},
        ],
    }
    validate_observation(REGISTRY["web_search"], obs)


def test_map_query_observation_schema() -> None:
    obs = {
        "status": "success",
        "error_message": None,
        "formatted_address": "Paris",
        "resolved_latlng": [48.8584, 2.2945],
        "place_type": "tourist_attraction",
    }
    validate_observation(REGISTRY["map_query"], obs)


def test_reverse_image_search_observation_schema() -> None:
    obs = {
        "status": "success",
        "error_message": None,
        "matches": [
            {"title": "Match", "snippet": "similar page", "url": "https://ex.com"},
        ],
    }
    validate_observation(REGISTRY["reverse_image_search"], obs)


def test_ocr_observation_schema() -> None:
    obs = {
        "status": "success",
        "error_message": None,
        "texts": ["HELLO", "GEO"],
    }
    validate_observation(REGISTRY["ocr"], obs)


def test_ocr_empty_texts() -> None:
    obs = {
        "status": "empty",
        "error_message": None,
        "texts": [],
    }
    validate_observation(REGISTRY["ocr"], obs)


def test_zoom_inspect_observation_schema() -> None:
    obs = {
        "status": "success",
        "error_message": None,
        "description": "red brick facade with arched window",
    }
    validate_observation(REGISTRY["zoom_inspect"], obs)


def test_sun_position_observation_schema() -> None:
    obs = {
        "status": "success",
        "error_message": None,
        "possible_latitude_range": [30.0, 50.0],
        "note": "approximate from shadow azimuth",
    }
    validate_observation(REGISTRY["sun_position_calc"], obs)


def test_submit_answer_has_no_observation_fields() -> None:
    assert REGISTRY["submit_answer"].is_terminal is True
    assert REGISTRY["submit_answer"].observation_fields == []


def test_invalid_observation_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_observation(
            REGISTRY["zoom_inspect"],
            {"status": "success", "error_message": None},  # 缺 description
        )
