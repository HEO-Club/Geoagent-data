"""各 Tool adapter 单元测试（外部 API 全部 mock）。"""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.tools import map_query, ocr, reverse_image_search, sun_position, web_search, zoom_inspect
from pipeline.tools.validation import validate_observation
from pipeline.schemas import ToolDefinition
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = {
    t["name"]: ToolDefinition.model_validate(t)
    for t in json.loads((ROOT / "tool_registry.json").read_text(encoding="utf-8"))
}


def test_web_search_mocked() -> None:
    class _C:
        def search(self, query: str, max_results: int) -> list[dict[str, str]]:
            assert query == "tower"
            return [{"title": "T", "snippet": "S", "url": "https://e.com"}]

    web_search.set_client(_C())
    try:
        obs = web_search.execute(
            {"query": "tower", "top_k": 3, "purpose": "broad_discovery"},
            "img.jpg",
        )
        assert obs["status"] == "success"
        validate_observation(REGISTRY["web_search"], obs)
    finally:
        web_search.set_client(None)


def test_map_query_mocked_resolved_latlng() -> None:
    class _C:
        def query(self, query: str | None, latlng: list[float] | None) -> dict[str, Any]:
            assert query == "Eiffel Tower"
            return {
                "status": "success",
                "error_message": None,
                "formatted_address": "Paris",
                "resolved_latlng": [48.8584, 2.2945],
                "place_type": "tourist_attraction",
            }

    map_query.set_client(_C())
    try:
        obs = map_query.execute({"query": "Eiffel Tower"}, "img.jpg")
        assert "latlng" not in obs
        assert obs["resolved_latlng"] == [48.8584, 2.2945]
        validate_observation(REGISTRY["map_query"], obs)
    finally:
        map_query.set_client(None)


def test_reverse_image_search_mocked() -> None:
    class _C:
        def search(self, image_path: str, bbox: list[float] | None) -> list[dict[str, str]]:
            return [{"title": "M", "snippet": "N", "url": "https://e.com/m"}]

    reverse_image_search.set_client(_C())
    try:
        obs = reverse_image_search.execute({}, "img.jpg")
        validate_observation(REGISTRY["reverse_image_search"], obs)
    finally:
        reverse_image_search.set_client(None)


def test_ocr_mocked() -> None:
    class _E:
        def run(self, image_path: str, bbox: list[float] | None) -> list[str]:
            return ["STREET", "CAFE"]

    ocr.set_engine(_E())
    try:
        obs = ocr.execute({}, "img.jpg")
        validate_observation(REGISTRY["ocr"], obs)
    finally:
        ocr.set_engine(None)


def test_zoom_inspect_mocked() -> None:
    zoom_inspect.set_describe_fn(lambda image_path, bbox: "red brick facade with arched window")
    try:
        obs = zoom_inspect.execute({"bbox": [0.1, 0.2, 0.3, 0.4]}, "img.jpg")
        validate_observation(REGISTRY["zoom_inspect"], obs)
    finally:
        zoom_inspect.set_describe_fn(None)


def test_sun_position_deterministic() -> None:
    obs = sun_position.execute(
        {"shadow_direction_deg": 0.0, "estimated_local_time": "12:00"},
        "img.jpg",
    )
    assert obs["status"] == "success"
    validate_observation(REGISTRY["sun_position_calc"], obs)


def test_adapters_block_real_api_in_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline.config import clear_settings_cache

    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    clear_settings_cache()
    web_search.set_client(None)
    obs = web_search.execute({"query": "x", "top_k": 1, "purpose": "broad_discovery"}, "i.jpg")
    assert obs["status"] == "error"
    clear_settings_cache()
