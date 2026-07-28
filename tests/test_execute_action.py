"""execute_action：权限、terminal、LLM 合成、缓存与重试测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas import Action, AgentRole, ObservationSource
from pipeline.image_utils import expand_bbox_xywh
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
    assert narrations and narrations[0] in {"(empty)", "阴影朝北"}

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
    # COARSE 不传自由旁白，避免地名/新事实经 Narration 反哺
    assert cleaned == ""


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
    assert "Narration: (empty)" in narr_line or "阴影朝北" not in narr_line
    assert "VisualObs" in prompts[0] or "H9" in prompts[0]
    assert "H9" in prompts[0] or "overlays" in prompts[0]
    assert "ATTENTION HINT" in prompts[0] or "bbox" in prompts[0].lower()
    assert "Tool modality" in prompts[0] or "satellite API" in prompts[0]


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


def test_content_region_crop_before_action_bbox(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """内容区先裁，再相对裁 Action bbox；标注不得直接成为事实。"""
    import cv2
    import numpy as np

    from pipeline.evidence_routing import (
        ContentRegion,
        ContentType,
        EvidenceIntent,
        SemanticRoute,
    )

    img_path = tmp_path / "frame.jpg"
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    canvas[20:160, 20:180] = (40, 90, 40)  # 内容区
    assert cv2.imwrite(str(img_path), canvas)

    seen_images: list[str] = []

    class _Obs:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def model_dump(self, **_k: Any) -> dict[str, Any]:
            return self._data

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = kwargs, prompt
        if response_model.__name__ == "_ObservationGroundingCheck":
            return response_model(
                fully_entailed_by_source_claims=True,
                unsupported_spans=[],
                target_visibility_consistent=True,
                reason="ok",
            )
        assert images
        seen_images.append(images[0])
        return _Obs(
            {
                "status": "success",
                "error_message": None,
                "description": "green terrain patch inside content region",
            }
        )

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    intent = EvidenceIntent(
        target_object="嵌入照片",
        content_type=ContentType.PRIMARY_SCENE,
        target_features=["地形"],
        source_concepts=["地形"],
        source_claims=["画面中可见地形"],
        video_fact_ids=["vf0"],
        suggested_bbox=[0.1, 0.1, 0.8, 0.7],
        route=SemanticRoute.COARSE,
    )
    region = ContentRegion(
        content_type=ContentType.PRIMARY_SCENE,
        content_bbox=[0.1, 0.1, 0.8, 0.7],
        target_visible=True,
    )
    result = execute_action(
        Action(tool="zoom_inspect", params={"bbox": [0.2, 0.2, 0.5, 0.5]}),
        str(img_path),
        AgentRole.COARSE,
        narration="【标注：高地】请观察",
        registry_path=str(env_registry),
        use_cache=False,
        evidence_intent=intent,
        content_region=region,
    )
    assert result.status == "success"
    assert seen_images
    assert seen_images[0] != str(img_path)


def test_coarse_source_contract_rejects_unentailed_detail(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任何未被逐视频来源声明蕴含的细节都应耗尽后降为 empty。"""
    import cv2
    import numpy as np

    from pipeline.evidence_routing import EvidenceIntent, ContentType, SemanticRoute

    img_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

    class _Obs:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def model_dump(self, **_k: Any) -> dict[str, Any]:
            return self._data

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **_k: Any,
    ) -> Any:
        _ = prompt, images
        if response_model.__name__ == "_ObservationGroundingCheck":
            return response_model(
                fully_entailed_by_source_claims=False,
                unsupported_spans=["未声明的新设施"],
                target_visibility_consistent=True,
                reason="来源声明未提及该设施",
            )
        return _Obs(
            {
                "status": "success",
                "error_message": None,
                "description": "远处可见未声明的新设施",
            }
        )

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    intent = EvidenceIntent(
        target_object="来源声明指定目标",
        content_type=ContentType.PRIMARY_SCENE,
        target_features=["目标甲"],
        source_concepts=["目标甲"],
        source_claims=["画面中明确出现目标甲"],
        video_fact_ids=["vf0"],
        route=SemanticRoute.COARSE,
    )
    result = execute_action(
        Action(tool="zoom_inspect", params={"bbox": [0.0, 0.0, 1.0, 0.5]}),
        str(img_path),
        AgentRole.COARSE,
        narration="画面中明确出现目标甲",
        registry_path=str(env_registry),
        use_cache=False,
        evidence_intent=intent,
    )
    assert result.status == "empty"
    assert result.error_message and "ungrounded_video_fact" in result.error_message


def test_coarse_keeps_entailed_visibility_confirmation(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """来源蕴含的可见确认即使「无增量」标记也不得再降为 empty。"""
    import cv2
    import numpy as np

    from pipeline.evidence_routing import ContentType, EvidenceIntent, SemanticRoute

    img_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

    class _Obs:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def model_dump(self, **_k: Any) -> dict[str, Any]:
            return self._data

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **_k: Any,
    ) -> Any:
        _ = prompt, images
        if response_model.__name__ == "_ObservationGroundingCheck":
            return response_model(
                fully_entailed_by_source_claims=True,
                unsupported_spans=[],
                target_visibility_consistent=True,
                adds_incremental_information=False,
                reason="短确认",
            )
        return _Obs(
            {
                "status": "success",
                "error_message": None,
                "description": "画面可见目标甲对应的高地边缘",
            }
        )

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    intent = EvidenceIntent(
        target_object="目标甲",
        content_type=ContentType.PRIMARY_SCENE,
        target_features=["目标甲"],
        source_concepts=["目标甲", "高地"],
        source_claims=["画面中明确出现目标甲，位于高地"],
        video_fact_ids=["vf0"],
        route=SemanticRoute.COARSE,
    )
    result = execute_action(
        Action(tool="zoom_inspect", params={"bbox": [0.0, 0.0, 1.0, 0.5]}),
        str(img_path),
        AgentRole.COARSE,
        narration="画面中明确出现目标甲",
        registry_path=str(env_registry),
        use_cache=False,
        evidence_intent=intent,
    )
    assert result.status == "success"
    assert result.observation is not None
    assert "高地" in str(result.observation)


def test_expand_bbox_xywh_margin() -> None:
    out = expand_bbox_xywh([0.2, 0.2, 0.2, 0.2], margin=0.08)
    assert out[0] == pytest.approx(0.12)
    assert out[1] == pytest.approx(0.12)
    assert out[2] == pytest.approx(0.36)
    assert out[3] == pytest.approx(0.36)
    clamped = expand_bbox_xywh([0.0, 0.0, 0.1, 0.1], margin=0.08)
    assert clamped[0] == pytest.approx(0.0)
    assert clamped[1] == pytest.approx(0.0)


def test_web_search_retrieval_no_images_allows_place_in_prompt(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RetrievalObs：不传图；旁白地名可进入合成 prompt。"""
    img = tmp_path / "frame.jpg"
    _write_solid_jpeg(img)
    seen: dict[str, Any] = {"images": "unset", "prompt": ""}

    class _Obs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "success",
                "error_message": None,
                "results": [
                    {
                        "title": "津湾广场",
                        "snippet": "天津津湾广场简介",
                        "url": "https://example.com/jwan",
                    }
                ],
            }

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = response_model, kwargs
        seen["images"] = images
        seen["prompt"] = prompt
        return _Obs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(
            tool="web_search",
            params={"query": "天津 津湾广场", "purpose": "precise_lookup"},
        ),
        str(img),
        AgentRole.FINE,
        narration="检索确认天津津湾广场候选",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "success"
    assert seen["images"] is None
    assert "RetrievalObs" in seen["prompt"]
    assert "津湾广场" in seen["prompt"]
    assert "MAY be written" in seen["prompt"] or "顺推" in seen["prompt"]


def test_satellite_retrieval_works_without_real_image_file(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """卫星匹配走文本合成，不依赖有效图片 URL。"""
    missing = tmp_path / "does_not_exist.jpg"
    seen_images: list[Any] = []

    class _Obs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "success",
                "error_message": None,
                "matched_features": [
                    {
                        "feature_name": "雕刻扶手",
                        "location_bbox": [0.1, 0.1, 0.2, 0.2],
                        "confidence": 0.8,
                        "visual_description": "卫星图上可见栏杆状线性结构",
                    }
                ],
                "overall_match_assessment": "部分匹配",
            }

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = prompt, response_model, kwargs
        seen_images.append(images)
        return _Obs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(
            tool="find_specific_features_in_satellite_map",
            params={
                "target_satellite_image": "sat_placeholder.jpg",
                "reference_features": ["雕刻扶手"],
            },
        ),
        str(missing),
        AgentRole.FINE,
        narration="在园区卫星图中查找雕刻扶手与同类建筑",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "success"
    assert seen_images == [None]


def test_zoom_expands_bbox_before_crop(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = tmp_path / "frame.jpg"
    _write_solid_jpeg(img, w=100, h=100)
    cropped_bboxes: list[list[float]] = []
    real_crop = __import__(
        "pipeline.image_utils", fromlist=["crop_image_by_bbox"]
    ).crop_image_by_bbox

    def spy_crop(
        image_path: str,
        bbox: list[float],
        *,
        cache_dir: str | None = None,
    ) -> str:
        cropped_bboxes.append([float(x) for x in bbox])
        return real_crop(image_path, bbox, cache_dir=cache_dir)

    class _Obs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "success",
                "error_message": None,
                "description": "open ground",
            }

    def fake_structured(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        return _Obs()

    monkeypatch.setattr("pipeline.tools.base.crop_image_by_bbox", spy_crop)
    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    result = execute_action(
        Action(tool="zoom_inspect", params={"bbox": [0.2, 0.2, 0.2, 0.2]}),
        str(img),
        AgentRole.COARSE,
        narration="放大",
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert result.status == "success"
    # 至少一次裁切使用了外扩后的框（相对内容区二次裁切）
    assert any(
        abs(b[0] - 0.12) < 1e-6 and abs(b[2] - 0.36) < 1e-6 for b in cropped_bboxes
    )