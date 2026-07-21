"""stage4：generate_observations 与 LLM 合成路径测试（外部 API 全部 mock）。"""

from __future__ import annotations

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
)
from pipeline.stage4_observe import (
    ObservationSynthesisExhausted,
    generate_observations,
)
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
    monkeypatch.setenv("OBS_SYNTH_MAX_RETRY", "2")
    clear_settings_cache()
    yield dst
    clear_settings_cache()


def _move(*, screen_action: str | None = "搜索", role: AgentRole = AgentRole.COARSE) -> Move:
    return Move(
        start_time=0.0,
        end_time=1.0,
        narration="旁白线索。",
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


def test_expands_composed_actions_with_narration(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"img")
    seen_narrations: list[str] = []

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = response_model, images, kwargs
        for line in prompt.splitlines():
            if line.startswith("Narration:"):
                seen_narrations.append(line.split(":", 1)[1].strip())

        class _Obs:
            def model_dump(self, mode: str = "json") -> dict[str, Any]:
                if "zoom_inspect" in prompt or "bbox" in prompt.lower():
                    # 动态模型字段由 tool 决定；按 prompt 中 Tool 名区分
                    pass
                tool_line = next(
                    (ln for ln in prompt.splitlines() if ln.startswith("Tool:")),
                    "",
                )
                if "zoom_inspect" in tool_line:
                    return {
                        "status": "success",
                        "error_message": None,
                        "description": "arched stone window on facade",
                    }
                if "sun_position_calc" in tool_line:
                    return {
                        "status": "success",
                        "error_message": None,
                        "possible_latitude_range": [30.0, 50.0],
                        "note": None,
                    }
                return {"status": "success", "error_message": None}

        return _Obs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)

    steps = [
        _step(
            [
                Action(
                    tool="zoom_inspect",
                    params={"bbox": [0.1, 0.2, 0.3, 0.4]},
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

    coarse_results = generate_observations(
        steps[:2],
        str(img),
        AgentRole.COARSE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert len(coarse_results) == 2
    assert all(r.source is ObservationSource.LLM_SYNTHESIZED for r in coarse_results)
    assert all(r.status == "success" for r in coarse_results)
    assert seen_narrations
    assert all("旁白线索" in n for n in seen_narrations)

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


def test_synth_retry_exhausted_raises(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"x")
    calls = {"n": 0}

    class _BadObs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {"status": "success", "error_message": None}  # 缺 description

    def fake_structured(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        calls["n"] += 1
        return _BadObs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    # execute_action 层仍返回 error（不抛）
    result = execute_action(
        Action(tool="zoom_inspect", params={"bbox": [0.1, 0.2, 0.3, 0.4]}),
        str(img),
        AgentRole.COARSE,
        narration="放大。",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert calls["n"] == 2  # OBS_SYNTH_MAX_RETRY=2
    assert result.status == "error"
    assert result.source is ObservationSource.LLM_SYNTHESIZED
    assert result.observation is None

    # generate_observations 层必须拒样
    steps = [
        _step(
            [Action(tool="zoom_inspect", params={"bbox": [0.1, 0.2, 0.3, 0.4]})],
        )
    ]
    with pytest.raises(ObservationSynthesisExhausted, match="不得入库"):
        generate_observations(
            steps,
            str(img),
            AgentRole.COARSE,
            registry_path=str(env_registry),
            use_cache=False,
        )
    assert calls["n"] == 4  # 又一轮重试 2 次


def test_synth_retry_succeeds_on_second_attempt(
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
                "description": "",
            }

    def fake_structured(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        calls["n"] += 1
        return _Bad() if calls["n"] == 1 else _Good()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(tool="zoom_inspect", params={"bbox": [0.1, 0.2, 0.3, 0.4]}),
        str(img),
        AgentRole.COARSE,
        narration="局部。",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert calls["n"] == 2
    assert result.status == "empty"
    assert result.source is ObservationSource.LLM_SYNTHESIZED
    assert result.observation is not None
    assert result.observation["description"] == ""


def test_error_not_cached(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"bytes-for-hash")
    calls = {"n": 0}

    class _Bad:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {"status": "success", "error_message": None}

    class _Good:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "success",
                "error_message": None,
                "possible_latitude_range": [10.0, 30.0],
                "note": None,
            }

    def fake_structured(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        calls["n"] += 1
        return _Bad() if calls["n"] == 1 else _Good()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    monkeypatch.setenv("OBS_SYNTH_MAX_RETRY", "1")
    clear_settings_cache()

    action = Action(
        tool="sun_position_calc",
        params={"shadow_direction_deg": 90.0},
    )
    r1 = execute_action(
        action,
        str(img),
        AgentRole.COARSE,
        narration="阴影。",
        registry_path=str(env_registry),
    )
    assert r1.status == "error"
    assert r1.cache_hit is False

    r2 = execute_action(
        action,
        str(img),
        AgentRole.COARSE,
        narration="阴影。",
        registry_path=str(env_registry),
    )
    assert calls["n"] == 2
    assert r2.status == "success"
    assert r2.cache_hit is False
    assert r2.source is ObservationSource.LLM_SYNTHESIZED


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
