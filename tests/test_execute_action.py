"""execute_action：权限、terminal、draft/production、缓存测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas import Action, AgentRole, ObservationSource, ToolDefinition
from pipeline.tools.base import execute_action


@pytest.fixture()
def env_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = Path(__file__).resolve().parents[1]
    dst = tmp_path / "tool_registry.json"
    dst.write_text((root / "tool_registry.json").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("TOOL_REGISTRY_PATH", str(dst))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / ".cache"))
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DRAFT_TOOL_MAX_RETRY", "2")
    clear_settings_cache()
    yield dst
    clear_settings_cache()


def _promote_local(registry_path: Path, name: str, ref: str) -> None:
    items = json.loads(registry_path.read_text(encoding="utf-8"))
    for item in items:
        if item["name"] == name:
            item["tier"] = "production"
            item["executor_ref"] = ref
    # 校验可导入
    ToolDefinition.model_validate(next(i for i in items if i["name"] == name))
    registry_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def test_permission_denied_for_wrong_agent(env_registry: Path, tmp_path: Path) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"fake")
    result = execute_action(
        Action(tool="map_query", params={"query": "Paris"}),
        str(img),
        AgentRole.COARSE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "error"
    assert result.error_message and "无权" in result.error_message


def test_terminal_skipped(env_registry: Path, tmp_path: Path) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"fake")
    result = execute_action(
        Action(
            tool="submit_answer",
            params={
                "latitude": 1.0,
                "longitude": 2.0,
                "location_name": "Somewhere",
                "confidence": 0.5,
                "reasoning": "enough clues aligned for submit",
            },
        ),
        str(img),
        AgentRole.FINE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "skipped"
    assert result.observation is None
    assert result.source is None


def test_production_sun_position_and_cache(env_registry: Path, tmp_path: Path) -> None:
    _promote_local(env_registry, "sun_position_calc", "pipeline.tools.sun_position.execute")
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"fake-image-bytes")
    action = Action(
        tool="sun_position_calc",
        params={"shadow_direction_deg": 180.0, "estimated_local_time": "12:00"},
    )
    r1 = execute_action(action, str(img), AgentRole.COARSE, registry_path=str(env_registry))
    assert r1.status == "success"
    assert r1.source is ObservationSource.REAL_EXECUTION
    assert r1.observation is not None
    assert "possible_latitude_range" in r1.observation
    assert r1.cache_hit is False

    r2 = execute_action(action, str(img), AgentRole.COARSE, registry_path=str(env_registry))
    assert r2.cache_hit is True
    assert r2.observation == r1.observation


def test_draft_vlm_synthesis_mocked(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"x")

    class _Obs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "success",
                "error_message": None,
                "results": [
                    {"title": "A", "snippet": "B", "url": "https://example.com/a"},
                    {"title": "C", "snippet": "D", "url": "https://example.com/c"},
                ],
            }

    def fake_structured(prompt: str, response_model: type, images: list[str] | None = None, **kwargs: Any) -> Any:
        _ = prompt, response_model, images, kwargs
        return _Obs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(tool="web_search", params={"query": "spire", "purpose": "broad_discovery"}),
        str(img),
        AgentRole.COARSE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "success"
    assert result.source is ObservationSource.VLM_SYNTHESIZED
    assert result.observation is not None
    assert result.observation["results"][0]["title"] == "A"


def test_production_invalid_observation_not_masked_by_vlm(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _promote_local(env_registry, "sun_position_calc", "pipeline.tools.sun_position.execute")
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"x")

    def bad_execute(params: dict[str, Any], image_path: str) -> dict[str, Any]:
        _ = params, image_path
        return {"status": "success", "error_message": None}  # 缺字段

    monkeypatch.setattr("pipeline.tools.sun_position.execute", bad_execute)

    result = execute_action(
        Action(tool="sun_position_calc", params={"shadow_direction_deg": 10.0}),
        str(img),
        AgentRole.COARSE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "error"
    assert result.source is ObservationSource.REAL_EXECUTION
    assert result.observation is None
    assert "校验失败" in (result.error_message or "")


def test_fine_cannot_use_sun_position(env_registry: Path, tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    img.write_bytes(b"1")
    result = execute_action(
        Action(tool="sun_position_calc", params={}),
        str(img),
        AgentRole.FINE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "error"
