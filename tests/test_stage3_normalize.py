"""stage3：Move → NormalizedStep 规范化测试（LLM / registry 全部 mock）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas import (
    AgentRole,
    Move,
    NormalizationMode,
    ObservationField,
    ParamField,
)
from pipeline.stage3_normalize import (
    _GRuleFlags,
    _MatchLLMResponse,
    _NewToolProposal,
    _ProposedAction,
    _is_pure_ui,
    match_or_register_tool,
    normalize_to_steps,
)
from pipeline.tools.registry import load_registry


def _move(
    *,
    narration: str = "旁白推理。",
    screen_action: str | None = "在搜索框输入地标名",
    role: AgentRole = AgentRole.COARSE,
) -> Move:
    return Move(
        start_time=0.0,
        end_time=1.0,
        narration=narration,
        screen_action=screen_action,
        visible_clues=[],
        agent_role=role,
    )


def _g_all_true() -> _GRuleFlags:
    return _GRuleFlags(
        cannot_match_existing=True,
        cannot_compose=True,
        io_semantics_clear=True,
        reusable_in_geolocation=True,
        observation_schema_complete=True,
        not_pure_ui=True,
        not_one_off_for_video=True,
        not_similar_to_existing=True,
    )


def _new_tool_proposal(name: str = "detect_license_plate") -> _NewToolProposal:
    return _NewToolProposal(
        name=name,
        description="检测车牌区域文字线索供粗定位使用。",
        params=[
            ParamField(
                name="region_hint",
                type="string",
                required=True,
                description="车牌大致区域描述，如[front bumper]",
                example="front bumper",
            )
        ],
        observation_fields=[
            ObservationField(
                name="status",
                type="string",
                nullable=False,
                description="执行状态，取值 success/empty/error",
            ),
            ObservationField(
                name="error_message",
                type="string",
                nullable=True,
                description="status=error 时填写，否则为 null",
            ),
            ObservationField(
                name="plate_text",
                type="string",
                nullable=True,
                description="识别到的车牌文字；无法识别时为 null",
            ),
        ],
        allowed_agents=[AgentRole.COARSE],
    )


@pytest.fixture()
def tmp_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "tool_registry.json"
    dst = tmp_path / "tool_registry.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("TOOL_REGISTRY_PATH", str(dst))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / ".cache"))
    clear_settings_cache()
    yield dst
    clear_settings_cache()


class TestPureUiAndThoughtOnly:
    def test_is_pure_ui_detects_scroll(self) -> None:
        assert _is_pure_ui("滚动页面查看下方内容") is True
        assert _is_pure_ui("切换标签到另一个窗口") is True
        assert _is_pure_ui("在搜索框搜索教堂尖顶") is False

    def test_thought_only_when_screen_action_empty(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called = {"n": 0}

        def _boom(*_a: Any, **_k: Any) -> Any:
            called["n"] += 1
            raise AssertionError("thought_only 不应调用 LLM")

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _boom)
        steps = normalize_to_steps(
            [_move(screen_action=None, narration="只是口述推理。")],
            AgentRole.COARSE,
        )
        assert len(steps) == 1
        assert steps[0].normalization_mode is NormalizationMode.THOUGHT_ONLY
        assert steps[0].actions == []
        assert steps[0].thought_draft.startswith("视觉线索：")
        assert "只是口述推理" in steps[0].thought_draft
        assert steps[0].thought_draft != "只是口述推理。"
        assert called["n"] == 0

    def test_pure_ui_fallback_without_llm(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("纯 UI 不应调用 LLM")

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _boom)
        tools = list(load_registry(tmp_registry).values())
        actions, mode, reason = match_or_register_tool(
            "滚动页面",
            "随便看看",
            AgentRole.COARSE,
            tools,
            [_move(screen_action="滚动页面")],
        )
        assert actions == []
        assert mode is NormalizationMode.FALLBACK
        assert reason is not None
        assert "UI" in reason or "滚动" in reason


class TestMatchedAndComposed:
    def test_matched_web_search_with_purpose(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake(
            _prompt: str,
            response_model: type[Any],
            **_kwargs: Any,
        ) -> _MatchLLMResponse:
            assert response_model is _MatchLLMResponse
            return _MatchLLMResponse(
                decision="matched",
                actions=[
                    _ProposedAction(
                        tool="web_search",
                        params={
                            "query": "red brick church Europe",
                            "purpose": "broad_discovery",
                        },
                    )
                ],
                confidence=0.91,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [_move(screen_action="在搜索框输入教堂尖顶")],
            AgentRole.COARSE,
        )
        assert len(steps) == 1
        step = steps[0]
        assert step.normalization_mode is NormalizationMode.MATCHED
        assert len(step.actions) == 1
        assert step.actions[0].tool == "web_search"
        assert step.actions[0].params["purpose"] == "broad_discovery"
        assert step.actions[0].params.get("top_k") == 3  # default 补齐
        assert step.matched_tool_confidence == 0.91
        assert step.fallback_reason is None

    def test_composed_ocr_then_web_search(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="composed",
                actions=[
                    _ProposedAction(tool="ocr", params={"bbox": [0.1, 0.2, 0.5, 0.6]}),
                    _ProposedAction(
                        tool="web_search",
                        params={"query": "sign text", "purpose": "broad_discovery"},
                    ),
                ],
                confidence=0.75,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [_move(screen_action="OCR 路牌并搜索文字")],
            AgentRole.COARSE,
        )
        assert steps[0].normalization_mode is NormalizationMode.COMPOSED
        assert [a.tool for a in steps[0].actions] == ["ocr", "web_search"]

    def test_agent_permission_recovers_via_heuristic_web_search(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """COARSE 误选 map_query 时，启发式降级为 web_search，避免空坍缩。"""

        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="matched",
                actions=[
                    _ProposedAction(
                        tool="map_query",
                        params={"query": "Paris"},
                    )
                ],
                confidence=0.9,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [_move(screen_action="打开地图查询巴黎", role=AgentRole.COARSE)],
            AgentRole.COARSE,
        )
        assert steps[0].normalization_mode is NormalizationMode.MATCHED
        assert len(steps[0].actions) == 1
        assert steps[0].actions[0].tool == "web_search"
        assert steps[0].actions[0].params["purpose"] == "broad_discovery"
        assert steps[0].matched_tool_confidence == 0.55

    def test_web_search_wrong_purpose_recovers_via_heuristic(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="matched",
                actions=[
                    _ProposedAction(
                        tool="web_search",
                        params={"query": "x", "purpose": "verification"},
                    )
                ],
                confidence=0.5,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        tools = list(load_registry(tmp_registry).values())
        actions, mode, reason = match_or_register_tool(
            "搜索验证线索",
            "核对",
            AgentRole.COARSE,
            tools,
            [_move()],
        )
        assert mode is NormalizationMode.MATCHED
        assert len(actions) == 1
        assert actions[0].tool == "web_search"
        assert actions[0].params["purpose"] == "broad_discovery"
        assert reason is None


class TestToolRegisteredAndFallback:
    def test_tool_registered_registers_tool(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        proposal = _new_tool_proposal("detect_facade_ornament")

        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="tool_registered",
                actions=[
                    _ProposedAction(
                        tool="detect_facade_ornament",
                        params={"region_hint": "upper facade"},
                    )
                ],
                g_flags=_g_all_true(),
                new_tool=proposal,
                confidence=0.6,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [_move(screen_action="识别建筑立面纹饰图案")],
            AgentRole.COARSE,
        )
        assert steps[0].normalization_mode is NormalizationMode.TOOL_REGISTERED
        assert len(steps[0].actions) == 1
        assert steps[0].actions[0].tool == "detect_facade_ornament"
        reg = load_registry(tmp_registry)
        assert "detect_facade_ornament" in reg
        dumped = reg["detect_facade_ornament"].model_dump()
        assert "tier" not in dumped
        assert "executor_ref" not in dumped

    def test_tool_registered_rejected_when_g_flags_incomplete(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bad_flags = _g_all_true().model_copy(update={"cannot_match_existing": False})

        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="tool_registered",
                g_flags=bad_flags,
                new_tool=_new_tool_proposal("lookup_unique_crest"),
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        before = set(load_registry(tmp_registry).keys())
        steps = normalize_to_steps(
            [_move(screen_action="某种无法描述的操作")],
            AgentRole.COARSE,
        )
        assert steps[0].normalization_mode is NormalizationMode.FALLBACK
        assert steps[0].actions == []
        assert "G 规则" in (steps[0].fallback_reason or "")
        after = set(load_registry(tmp_registry).keys())
        assert after == before

    def test_llm_explicit_fallback(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="fallback",
                fallback_reason="无法映射为地理定位 Tool",
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [_move(screen_action="打开无关娱乐网页")],
            AgentRole.FINE,
        )
        assert steps[0].normalization_mode is NormalizationMode.FALLBACK
        assert steps[0].actions == []
        assert steps[0].fallback_reason == "无法映射为地理定位 Tool"

    def test_invalid_map_params_recover_via_heuristic(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="matched",
                actions=[
                    _ProposedAction(
                        tool="map_query",
                        params={},  # query 与 latlng 皆缺
                    )
                ],
                confidence=0.8,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [
                _move(
                    screen_action="查询地图坐标",
                    role=AgentRole.FINE,
                )
            ],
            AgentRole.FINE,
        )
        assert steps[0].normalization_mode is NormalizationMode.MATCHED
        assert len(steps[0].actions) == 1
        assert steps[0].actions[0].tool == "map_query"
        assert steps[0].actions[0].params.get("query")
        assert steps[0].matched_tool_confidence == 0.55

    def test_llm_connection_error_recovers_with_heuristic(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = {"n": 0}

        def _boom(*_a: Any, **_k: Any) -> Any:
            calls["n"] += 1
            raise ConnectionError("simulated llm outage")

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _boom)
        steps = normalize_to_steps(
            [_move(screen_action="放大尖顶查看细节", narration="尖顶有十字架")],
            AgentRole.COARSE,
        )
        assert calls["n"] == 2  # 重试一次
        assert steps[0].normalization_mode is NormalizationMode.MATCHED
        assert steps[0].actions[0].tool == "zoom_inspect"
        assert steps[0].matched_tool_confidence == 0.55

    def test_llm_error_non_toolish_still_fallback(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(*_a: Any, **_k: Any) -> Any:
            raise ConnectionError("down")

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _boom)
        steps = normalize_to_steps(
            [_move(screen_action="打开无关娱乐网页", role=AgentRole.FINE)],
            AgentRole.FINE,
        )
        assert steps[0].normalization_mode is NormalizationMode.FALLBACK
        assert steps[0].actions == []
        assert "LLM 决策失败" in (steps[0].fallback_reason or "")

    def test_explicit_fallback_toolish_recovers_search(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="fallback",
                fallback_reason="模型犹豫",
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [_move(screen_action="搜索红色砖砌教堂", narration="欧洲风格")],
            AgentRole.COARSE,
        )
        assert steps[0].normalization_mode is NormalizationMode.MATCHED
        assert steps[0].actions[0].tool == "web_search"
        assert steps[0].actions[0].params["purpose"] == "broad_discovery"


class TestNoObservation:
    def test_normalized_step_has_no_observation_fields(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="matched",
                actions=[
                    _ProposedAction(
                        tool="zoom_inspect",
                        params={"bbox": [0.0, 0.0, 0.5, 0.5]},
                    )
                ],
                confidence=0.88,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [_move(screen_action="放大尖顶")],
            AgentRole.COARSE,
        )
        dumped = steps[0].model_dump()
        assert "observation" not in dumped
        assert "actions" in dumped
        assert dumped["normalization_mode"] == "matched"
