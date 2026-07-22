"""execute_action：权限、terminal、LLM 合成、缓存与重试测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas import Action, AgentRole, ObservationSource
from pipeline.tools.base import (
    execute_action,
    observation_contains_video_overlay,
    sanitize_narration_for_obs,
)


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


def test_llm_synthesis_and_cache(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"fake-image-bytes")
    narrations: list[str] = []

    class _Obs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "success",
                "error_message": None,
                "possible_latitude_range": [20.0, 40.0],
                "note": "from shadow",
            }

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = response_model, images, kwargs
        # 从 prompt 中抽出 Narration 行便于断言
        for line in prompt.splitlines():
            if line.startswith("Narration:"):
                narrations.append(line.split(":", 1)[1].strip())
        return _Obs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    action = Action(
        tool="sun_position_calc",
        params={"shadow_direction_deg": 180.0, "estimated_local_time": "12:00"},
    )
    r1 = execute_action(
        action,
        str(img),
        AgentRole.COARSE,
        narration="阴影朝北。",
        registry_path=str(env_registry),
    )
    assert r1.status == "success"
    assert r1.source is ObservationSource.LLM_SYNTHESIZED
    assert r1.observation is not None
    assert "possible_latitude_range" in r1.observation
    assert r1.cache_hit is False
    assert narrations and "阴影朝北" in narrations[0]

    r2 = execute_action(
        action,
        str(img),
        AgentRole.COARSE,
        narration="阴影朝北。",
        registry_path=str(env_registry),
    )
    assert r2.cache_hit is True
    assert r2.observation == r1.observation
    assert r2.source is ObservationSource.LLM_SYNTHESIZED


def test_synth_retry_exhausted_returns_error(
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
    result = execute_action(
        Action(tool="zoom_inspect", params={"bbox": [0.1, 0.2, 0.3, 0.4]}),
        str(img),
        AgentRole.COARSE,
        narration="放大查看立面。",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert calls["n"] == 2  # OBS_SYNTH_MAX_RETRY=2
    assert result.status == "error"
    assert result.source is ObservationSource.LLM_SYNTHESIZED
    assert result.observation is None


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
        narration="局部细节。",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert calls["n"] == 2
    assert result.status == "empty"
    assert result.source is ObservationSource.LLM_SYNTHESIZED
    assert result.observation is not None
    assert result.observation["description"] == ""


def test_error_result_not_cached(
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
    # OBS_SYNTH_MAX_RETRY=2：第一次调用整次 execute 会重试到成功，故拆成两次独立调用
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


def test_sanitize_narration_strips_places_for_coarse() -> None:
    raw = "阴影朝北，感觉像在许昌市附近，大概 34.0, 113.8。"
    cleaned = sanitize_narration_for_obs(AgentRole.COARSE, raw)
    assert "许昌" not in cleaned
    assert "34.0" not in cleaned
    assert "阴影朝北" in cleaned


def test_sanitize_narration_keeps_place_words_for_fine() -> None:
    raw = "对照地图核对许昌市候选点。"
    cleaned = sanitize_narration_for_obs(AgentRole.FINE, raw)
    assert "许昌市" in cleaned


def test_coarse_synth_prompt_uses_sanitized_narration(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"fake-image-bytes")
    prompts: list[str] = []

    class _Obs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "success",
                "error_message": None,
                "possible_latitude_range": [20.0, 40.0],
                "note": "from shadow",
            }

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = response_model, images, kwargs
        prompts.append(prompt)
        return _Obs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(
            tool="sun_position_calc",
            params={"shadow_direction_deg": 180.0, "estimated_local_time": "12:00"},
        ),
        str(img),
        AgentRole.COARSE,
        narration="阴影朝北，这是河南许昌市的建筑。",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "success"
    assert prompts
    narr_line = next(ln for ln in prompts[0].splitlines() if ln.startswith("Narration:"))
    assert "许昌" not in narr_line
    assert "阴影朝北" in narr_line
    assert "Do NOT copy city names" in prompts[0]
    assert "H9" in prompts[0] or "overlays" in prompts[0]


def _write_solid_jpeg(path: Path, *, w: int = 64, h: int = 64) -> None:
    import cv2
    import numpy as np

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 120)
    assert cv2.imwrite(str(path), img)


def test_observation_overlay_detector_generic() -> None:
    hits = observation_contains_video_overlay(
        {
            "status": "success",
            "description": "top-left youtube channel logo watermark visible",
        }
    )
    assert hits
    # 裸平台名不足以判 overlay（避免检索摘要误杀）
    bare = observation_contains_video_overlay(
        {
            "status": "success",
            "snippets": ["youtube video of stone railing hills"],
        }
    )
    assert not bare
    clean = observation_contains_video_overlay(
        {"status": "success", "description": "stone railing and distant hills"}
    )
    assert not clean


def test_web_search_skips_overlay_gate(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非画面 Tool 即使摘要含平台名也不触发 H9 重试。"""
    img = tmp_path / "frame.jpg"
    _write_solid_jpeg(img)
    n = {"i": 0}

    class _Obs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "success",
                "error_message": None,
                "results": [
                    {
                        "title": "youtube travel clip",
                        "snippet": "mentions youtube as source site",
                        "url": "https://example.com/a",
                    }
                ],
            }

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = prompt, response_model, images, kwargs
        n["i"] += 1
        return _Obs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(
            tool="web_search",
            params={"query": "stone railing hills", "purpose": "precise_lookup"},
        ),
        str(img),
        AgentRole.FINE,
        narration="检索地貌",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "success"
    assert n["i"] == 1


def test_zoom_crops_image_before_synth(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    _write_solid_jpeg(img, w=100, h=100)
    seen_images: list[str] = []

    class _Obs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "success",
                "error_message": None,
                "description": "stone railing detail",
            }

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = prompt, response_model, kwargs
        seen_images.extend(images or [])
        return _Obs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]}),
        str(img),
        AgentRole.COARSE,
        narration="放大栏杆",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "success"
    assert seen_images
    assert seen_images[0] != str(img.resolve()) or "_cropped" in seen_images[0] or "cropped" in seen_images[0]


def test_overlay_observation_triggers_retry(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    _write_solid_jpeg(img)
    n = {"i": 0}

    class _Obs:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return self._payload

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = response_model, images, kwargs
        n["i"] += 1
        if n["i"] == 1:
            assert "H9" in prompt or "overlays" in prompt
            return _Obs(
                {
                    "status": "success",
                    "error_message": None,
                    "description": "screen shows bilibili watermark at corner",
                }
            )
        assert "Previous Observation was rejected" in prompt or "H9 violation" in prompt
        return _Obs(
            {
                "status": "success",
                "error_message": None,
                "description": "open hills and a stone railing",
            }
        )

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(tool="zoom_inspect", params={"bbox": [0.0, 0.0, 1.0, 1.0]}),
        str(img),
        AgentRole.COARSE,
        narration="看地貌",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "success"
    assert n["i"] == 2
    assert "bilibili" not in str(result.observation)
