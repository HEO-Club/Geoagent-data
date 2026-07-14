"""stage4：generate_observations 与 Action→Observation 路径测试（外部 API 全部 mock）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas import (
    Action,
    AgentRole,
    Move,
    NormalizationMode,
    NormalizedStep,
    ObservationSource,
    ToolDefinition,
)
from pipeline.stage4_observe import generate_observations
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
    ToolDefinition.model_validate(next(i for i in items if i["name"] == name))
    registry_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _move(*, screen_action: str | None = "搜索", role: AgentRole = AgentRole.COARSE) -> Move:
    return Move(
        start_time=0.0,
        end_time=1.0,
        narration="旁白。",
        screen_action=screen_action,
        visible_clues=[],
        agent_role=role,
    )


def _step(
    actions: list[Action],
    *,
    mode: NormalizationMode = NormalizationMode.MATCHED,
    screen_action: str | None = "搜索",
    role: AgentRole = AgentRole.COARSE,
) -> NormalizedStep:
    return NormalizedStep(
        move=_move(screen_action=screen_action, role=role),
        thought_draft="草稿思考。",
        actions=actions,
        normalization_mode=mode,
        matched_tool_confidence=0.9 if actions else None,
        fallback_reason=None if actions else "screen_action 为空",
    )


def test_thought_only_produces_no_results(env_registry: Path, tmp_path: Path) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"fake")
    steps = [
        _step([], mode=NormalizationMode.THOUGHT_ONLY, screen_action=None),
    ]
    results = generate_observations(
        steps,
        str(img),
        AgentRole.COARSE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert results == []


def test_expands_composed_actions(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _promote_local(env_registry, "sun_position_calc", "pipeline.tools.sun_position.execute")
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"img")

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

    monkeypatch.setattr(
        "pipeline.tools.base.call_structured",
        lambda *a, **k: _Obs(),
    )

    steps = [
        _step(
            [
                Action(
                    tool="web_search",
                    params={"query": "spire", "purpose": "broad_discovery"},
                ),
                Action(
                    tool="sun_position_calc",
                    params={"shadow_direction_deg": 180.0, "estimated_local_time": "12:00"},
                ),
            ],
            mode=NormalizationMode.COMPOSED,
        ),
        _step([], mode=NormalizationMode.THOUGHT_ONLY, screen_action=None),
        _step(
            [
                Action(
                    tool="submit_answer",
                    params={
                        "latitude": 1.0,
                        "longitude": 2.0,
                        "location_name": "Somewhere",
                        "confidence": 0.5,
                        "reasoning": "enough clues aligned for submit",
                    },
                )
            ],
            role=AgentRole.FINE,
        ),
    ]

    # FINE terminal 与 COARSE 混用：按角色分批调用更贴近真实；此处分别测展开顺序
    coarse_results = generate_observations(
        steps[:2],
        str(img),
        AgentRole.COARSE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert len(coarse_results) == 2
    assert coarse_results[0].source is ObservationSource.VLM_SYNTHESIZED
    assert coarse_results[0].status == "success"
    assert coarse_results[1].source is ObservationSource.REAL_EXECUTION
    assert coarse_results[1].status == "success"

    fine_results = generate_observations(
        steps[2:],
        str(img),
        AgentRole.FINE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert len(fine_results) == 1
    assert fine_results[0].status == "skipped"
    assert fine_results[0].observation is None
    assert fine_results[0].source is None


def test_draft_retry_exhausted_returns_error(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"x")
    calls = {"n": 0}

    class _BadObs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {"status": "success", "error_message": None}  # 缺 results 等字段

    def fake_structured(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        calls["n"] += 1
        return _BadObs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(tool="web_search", params={"query": "spire", "purpose": "broad_discovery"}),
        str(img),
        AgentRole.COARSE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert calls["n"] == 2  # DRAFT_TOOL_MAX_RETRY=2
    assert result.status == "error"
    assert result.source is ObservationSource.VLM_SYNTHESIZED
    assert result.observation is None


def test_draft_retry_succeeds_on_second_attempt(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"x")
    calls = {"n": 0}

    class _Bad:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {"status": "success", "error_message": None}

    class _Good:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "empty",
                "error_message": None,
                "results": None,
            }

    def fake_structured(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        calls["n"] += 1
        return _Bad() if calls["n"] == 1 else _Good()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(tool="web_search", params={"query": "spire", "purpose": "broad_discovery"}),
        str(img),
        AgentRole.COARSE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert calls["n"] == 2
    assert result.status == "empty"
    assert result.source is ObservationSource.VLM_SYNTHESIZED
    assert result.observation is not None
    assert result.observation["results"] is None


def test_production_error_not_cached(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipeline.tools.sun_position as sun_mod

    _promote_local(env_registry, "sun_position_calc", "pipeline.tools.sun_position.execute")
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"bytes-for-hash")
    calls = {"n": 0}
    real_execute = sun_mod.execute

    def flaky(params: dict[str, Any], image_path: str) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("upstream timeout")
        return real_execute(params, image_path)

    monkeypatch.setattr(sun_mod, "execute", flaky)
    action = Action(
        tool="sun_position_calc",
        params={"shadow_direction_deg": 90.0},
    )
    r1 = execute_action(action, str(img), AgentRole.COARSE, registry_path=str(env_registry))
    assert r1.status == "error"
    assert r1.cache_hit is False

    # 错误结果不得写入 cache；第二次应重新执行并成功
    r2 = execute_action(action, str(img), AgentRole.COARSE, registry_path=str(env_registry))
    assert calls["n"] == 2
    assert r2.status == "success"
    assert r2.cache_hit is False
    assert r2.source is ObservationSource.REAL_EXECUTION


def test_unknown_tool_error(env_registry: Path, tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    img.write_bytes(b"1")
    result = execute_action(
        Action(tool="not_a_real_tool", params={}),
        str(img),
        AgentRole.COARSE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "error"
    assert "未知 tool" in (result.error_message or "")
