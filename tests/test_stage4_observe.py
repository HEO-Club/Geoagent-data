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
    ObservationExecutionResult,
    ObservationSource,
)
from pipeline.stage4_observe import (
    generate_observations,
    resolve_image_for_step,
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
    # COARSE 不注入自由旁白；合成靠 EvidenceIntent.source_claims
    assert seen_narrations
    assert all(n in ("", "(empty)") for n in seen_narrations)

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

    # 全角色：合成耗尽标 error（诚实失败），流水线继续
    steps = [
        _step(
            [Action(tool="zoom_inspect", params={"bbox": [0.1, 0.2, 0.3, 0.4]})],
        )
    ]
    coarse_out = generate_observations(
        steps,
        str(img),
        AgentRole.COARSE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert len(coarse_out) == 1
    assert coarse_out[0].status == "error"
    assert coarse_out[0].observation is not None
    assert coarse_out[0].observation.get("status") == "error"
    assert "no in-scene geography" not in str(coarse_out[0].observation.get("description") or "")
    assert calls["n"] == 4  # 又一轮重试 2 次

    fine_out = generate_observations(
        steps,
        str(img),
        AgentRole.FINE,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert len(fine_out) == 1
    assert fine_out[0].status == "error"
    assert calls["n"] == 6


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


def test_resolve_image_for_step_picks_nearest_keyframe(tmp_path: Path) -> None:
    f0 = tmp_path / "t0.000.jpg"
    f5 = tmp_path / "t5.000.jpg"
    f10 = tmp_path / "t10.000.jpg"
    for p in (f0, f5, f10):
        p.write_bytes(b"x")
    step = NormalizedStep(
        move=Move(
            start_time=4.5,
            end_time=5.5,
            narration="n",
            screen_action="放大",
            visible_clues=[],
            agent_role=AgentRole.COARSE,
        ),
        thought_draft="t",
        actions=[Action(tool="zoom_inspect", params={"bbox": [0, 0, 1, 1]})],
        normalization_mode=NormalizationMode.MATCHED,
        matched_tool_confidence=0.9,
        fallback_reason=None,
    )
    chosen = resolve_image_for_step(
        step,
        image_path=str(f0),
        keyframes=[str(f0), str(f5), str(f10)],
    )
    assert Path(chosen).name == "t5.000.jpg"


def test_generate_observations_uses_per_step_keyframes(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2
    import numpy as np

    frames = []
    for t in (0.0, 8.0):
        p = tmp_path / f"t{t:.3f}.jpg"
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        assert cv2.imwrite(str(p), img)
        frames.append(str(p))
    seen: list[str] = []

    class _Obs:
        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "status": "success",
                "error_message": None,
                "description": "scene hills",
            }

    def fake_structured(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = prompt, response_model, kwargs
        seen.extend(images or [])
        return _Obs()

    monkeypatch.setattr("pipeline.tools.base.call_structured", fake_structured)
    steps = [
        NormalizedStep(
            move=Move(
                start_time=0.0,
                end_time=1.0,
                narration="早",
                screen_action="放大",
                visible_clues=[],
                agent_role=AgentRole.COARSE,
            ),
            thought_draft="a",
            actions=[Action(tool="zoom_inspect", params={"bbox": [0.2, 0.2, 0.3, 0.3]})],
            normalization_mode=NormalizationMode.MATCHED,
            matched_tool_confidence=0.9,
            fallback_reason=None,
        ),
        NormalizedStep(
            move=Move(
                start_time=7.5,
                end_time=8.5,
                narration="晚",
                screen_action="放大",
                visible_clues=[],
                agent_role=AgentRole.COARSE,
            ),
            thought_draft="b",
            actions=[Action(tool="zoom_inspect", params={"bbox": [0.5, 0.5, 0.3, 0.3]})],
            normalization_mode=NormalizationMode.MATCHED,
            matched_tool_confidence=0.9,
            fallback_reason=None,
        ),
    ]
    results = generate_observations(
        steps,
        frames[0],
        AgentRole.COARSE,
        keyframes=frames,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert len(results) == 2
    assert len(seen) == 2
    # 两步应基于不同源帧裁剪（路径不同）
    assert seen[0] != seen[1]


def test_resolve_image_prefers_mid_window_for_primary_scene(tmp_path: Path) -> None:
    from pipeline.evidence_routing import (
        ContentType,
        EvidenceIntent,
        SemanticRoute,
        embed_evidence_intent,
    )

    frames = []
    for t in (0.0, 100.0, 180.0, 200.0):
        p = tmp_path / f"t{t:.3f}.jpg"
        p.write_bytes(b"x")
        frames.append(str(p))
    intent = EvidenceIntent(
        target_object="老照片地貌",
        content_type=ContentType.PRIMARY_SCENE,
        target_features=["高地", "桥"],
        route=SemanticRoute.COARSE,
    )
    step = NormalizedStep(
        move=Move(
            start_time=175.0,
            end_time=195.0,
            narration="高地与桥",
            screen_action="看照片",
            visible_clues=[],
            agent_role=AgentRole.COARSE,
        ),
        thought_draft=embed_evidence_intent("草稿", intent),
        actions=[Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.5, 0.5]})],
        normalization_mode=NormalizationMode.MATCHED,
        matched_tool_confidence=0.9,
        fallback_reason=None,
    )
    chosen = resolve_image_for_step(
        step, image_path=frames[0], keyframes=frames
    )
    # 应落在 Move 窗口附近，而非全局 t0
    assert Path(chosen).name != "t0.000.jpg"


def test_interface_only_returns_empty_without_narration_fabrication(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pipeline.evidence_routing import (
        ContentType,
        EvidenceIntent,
        SemanticRoute,
        embed_evidence_intent,
    )

    img = tmp_path / "t0.000.jpg"
    img.write_bytes(b"x")
    calls = {"n": 0}

    def boom(*_a: Any, **_k: Any) -> Any:
        calls["n"] += 1
        raise AssertionError("interface_only 不应调用合成 LLM")

    monkeypatch.setattr("pipeline.tools.base.call_structured", boom)
    intent = EvidenceIntent(
        target_object="界面",
        content_type=ContentType.INTERFACE_ONLY,
        route=SemanticRoute.NON_TRAINING,
    )
    step = NormalizedStep(
        move=Move(
            start_time=0.0,
            end_time=1.0,
            narration="高地桥平原",  # 旁白有地貌词，但内容区是 interface
            screen_action="聊天置顶",
            visible_clues=[],
            agent_role=AgentRole.COARSE,
        ),
        thought_draft=embed_evidence_intent("ui", intent),
        actions=[Action(tool="zoom_inspect", params={"bbox": [0.0, 0.0, 1.0, 1.0]})],
        normalization_mode=NormalizationMode.MATCHED,
        matched_tool_confidence=0.9,
        fallback_reason=None,
    )
    results = generate_observations(
        [step],
        str(img),
        AgentRole.COARSE,
        keyframes=[str(img)],
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert calls["n"] == 0
    assert results[0].status == "empty"
    assert "高地" not in str(results[0].observation)


def test_pick_agent1_representative_skips_t0_ui(tmp_path: Path) -> None:
    from pipeline.evidence_routing import (
        ContentType,
        EvidenceIntent,
        SemanticRoute,
        embed_evidence_intent,
    )
    from pipeline.stage4_observe import pick_agent1_representative_image

    frames = []
    for t in (0.0, 50.0, 180.0):
        p = tmp_path / f"t{t:.3f}.jpg"
        p.write_bytes(b"x")
        frames.append(str(p))
    intent = EvidenceIntent(
        target_object="老照片",
        content_type=ContentType.PRIMARY_SCENE,
        route=SemanticRoute.COARSE,
    )
    step = NormalizedStep(
        move=Move(
            start_time=170.0,
            end_time=190.0,
            narration="看照片",
            screen_action="放大",
            visible_clues=[],
            agent_role=AgentRole.COARSE,
        ),
        thought_draft=embed_evidence_intent("t", intent),
        actions=[],
        normalization_mode=NormalizationMode.THOUGHT_ONLY,
        matched_tool_confidence=None,
        fallback_reason="x",
    )
    picked = pick_agent1_representative_image(
        [], [step], keyframes=frames, fallback=frames[0]
    )
    assert Path(picked).name != "t0.000.jpg"


def test_pick_agent1_skips_early_meta_primary_scene(tmp_path: Path) -> None:
    """开场元叙事即使标成 PRIMARY_SCENE 也不得当代表帧。"""
    from pipeline.evidence_routing import (
        ContentType,
        EvidenceIntent,
        SemanticRoute,
        embed_evidence_intent,
    )
    from pipeline.stage4_observe import pick_agent1_representative_image

    frames = []
    for t in (5.0, 39.0, 148.0):
        p = tmp_path / f"t{t:.3f}.jpg"
        p.write_bytes(b"x")
        frames.append(str(p))
    meta_intent = EvidenceIntent(
        target_object="拍摄地",
        content_type=ContentType.PRIMARY_SCENE,
        target_features=["照片", "半年"],
        route=SemanticRoute.COARSE,
    )
    geo_intent = EvidenceIntent(
        target_object="高地",
        content_type=ContentType.PRIMARY_SCENE,
        target_features=["高地", "屋顶"],
        route=SemanticRoute.COARSE,
    )
    meta_step = NormalizedStep(
        move=Move(
            start_time=0.0,
            end_time=4.6,
            narration="为了找到这张照片的拍摄地,我足足花了半年的时间。",
            screen_action="片头",
            visible_clues=[],
            agent_role=AgentRole.COARSE,
        ),
        thought_draft=embed_evidence_intent("meta", meta_intent),
        actions=[Action(tool="zoom_inspect", params={"bbox": [0, 0, 1, 1]})],
        normalization_mode=NormalizationMode.MATCHED,
        matched_tool_confidence=0.9,
        fallback_reason=None,
    )
    geo_step = NormalizedStep(
        move=Move(
            start_time=39.0,
            end_time=44.0,
            narration="细看照片可以看到下方建筑屋顶，拍摄点在高地。",
            screen_action="放大",
            visible_clues=["屋顶", "高地"],
            agent_role=AgentRole.COARSE,
        ),
        thought_draft=embed_evidence_intent("geo", geo_intent),
        actions=[Action(tool="zoom_inspect", params={"bbox": [0.2, 0.2, 0.5, 0.5]})],
        normalization_mode=NormalizationMode.MATCHED,
        matched_tool_confidence=0.9,
        fallback_reason=None,
    )
    empty_obs = ObservationExecutionResult(
        action=meta_step.actions[0],
        observation={
            "status": "empty",
            "error_message": None,
            "description": "no in-scene geography visible in content region",
        },
        source=ObservationSource.LLM_SYNTHESIZED,
        status="empty",
        error_message=None,
        cache_hit=False,
    )
    good_obs = ObservationExecutionResult(
        action=geo_step.actions[0],
        observation={
            "status": "success",
            "error_message": None,
            "description": "下方可见建筑屋顶，呈俯视",
        },
        source=ObservationSource.LLM_SYNTHESIZED,
        status="success",
        error_message=None,
        cache_hit=False,
    )
    picked = pick_agent1_representative_image(
        [empty_obs, good_obs],
        [meta_step, geo_step],
        keyframes=frames,
        fallback=frames[0],
    )
    assert Path(picked).name == "t39.000.jpg"


def test_visual_tool_retries_neighbor_frame_on_empty(
    env_registry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首帧 empty 时换近邻关键帧再合成，拿到 success。"""
    frames = []
    for name in ("t20.000.jpg", "t21.000.jpg", "t22.000.jpg"):
        p = tmp_path / name
        p.write_bytes(b"img")
        frames.append(str(p))

    calls: list[str] = []

    def fake_execute(
        action: Action,
        image_path: str,
        agent_role: AgentRole,
        **kwargs: Any,
    ) -> ObservationExecutionResult:
        _ = action, agent_role, kwargs
        calls.append(Path(image_path).name)
        # 选帧优先 Move 中点 t21；首帧 empty 后应回退到近邻帧。
        if Path(image_path).name == "t21.000.jpg":
            return ObservationExecutionResult(
                action=action,
                observation={
                    "status": "empty",
                    "error_message": None,
                    "description": "no in-scene geography visible in content region",
                },
                source=ObservationSource.LLM_SYNTHESIZED,
                status="empty",
                error_message=None,
                cache_hit=False,
            )
        return ObservationExecutionResult(
            action=action,
            observation={
                "status": "success",
                "error_message": None,
                "description": "elevated ground and river bank",
            },
            source=ObservationSource.LLM_SYNTHESIZED,
            status="success",
            error_message=None,
            cache_hit=False,
        )

    monkeypatch.setattr("pipeline.stage4_observe.execute_action", fake_execute)
    step = NormalizedStep(
        move=Move(
            start_time=20.0,
            end_time=22.0,
            narration="高地俯视",
            screen_action="放大",
            visible_clues=[],
            agent_role=AgentRole.COARSE,
        ),
        thought_draft="观察高地",
        actions=[Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})],
        normalization_mode=NormalizationMode.MATCHED,
        matched_tool_confidence=0.9,
        fallback_reason=None,
    )
    results = generate_observations(
        [step],
        frames[0],
        AgentRole.COARSE,
        keyframes=frames,
        registry_path=str(env_registry),
        use_cache=False,
    )
    assert len(results) == 1
    assert results[0].status == "success"
    assert calls[0] == "t21.000.jpg"
    assert len(calls) >= 2
    assert any(name in ("t20.000.jpg", "t22.000.jpg") for name in calls[1:])
