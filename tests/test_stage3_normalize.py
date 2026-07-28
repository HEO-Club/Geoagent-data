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
    start: float = 0.0,
    end: float = 1.0,
) -> Move:
    return Move(
        start_time=start,
        end_time=end,
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


def _dispatch_stage3_llm(
    response_model: type,
    *,
    extraction: object | None = None,
    match: object | None = None,
) -> object:
    """测试辅助：区分 VideoContextExtraction 与 Tool 匹配调用。"""
    from pipeline.evidence_routing import (
        ExtractedVideoFact,
        VideoContextExtraction,
    )
    from pipeline.stage3_normalize import _VideoContextGroundingReview

    if response_model is VideoContextExtraction:
        if extraction is not None:
            return extraction
        return VideoContextExtraction(
            facts=[
                ExtractedVideoFact(
                    move_index=0,
                    claim="排除无关候选",
                    concepts=["排除无关候选"],
                    kind="exclude",
                    excluded_candidates=["无关候选"],
                )
            ]
        )
    if response_model is _VideoContextGroundingReview:
        return _VideoContextGroundingReview(working_scope_supported=True)
    from pipeline.stage3_normalize import _WorkingScopeDerivation

    if response_model is _WorkingScopeDerivation:
        return _WorkingScopeDerivation(region="", supporting_raw_clue_positions=[])
    if match is not None:
        return match
    raise AssertionError(f"unexpected response_model: {response_model}")


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
        from pipeline.evidence_routing import strip_evidence_intent

        assert len(steps) == 1
        assert steps[0].normalization_mode is NormalizationMode.THOUGHT_ONLY
        assert steps[0].actions == []
        draft = strip_evidence_intent(steps[0].thought_draft)
        assert "只是口述推理" in draft
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
        # COARSE：web_search 分解为有画面依据的训练 Tool
        assert step.normalization_mode is NormalizationMode.MATCHED
        assert len(step.actions) == 1
        assert step.actions[0].tool == "zoom_inspect"
        assert "bbox" in step.actions[0].params
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
        # ocr 保留；web_search 分解为 zoom（检索/视觉语义）
        assert steps[0].normalization_mode is NormalizationMode.COMPOSED
        assert {a.tool for a in steps[0].actions} == {"zoom_inspect", "ocr"}
        assert "web_search" not in {a.tool for a in steps[0].actions}

    def test_agent_permission_recovers_via_heuristic_web_search(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """COARSE 误选 map_query 时，启发式后分解为训练 Tool，避免空坍缩。"""

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
        assert steps[0].actions[0].tool == "zoom_inspect"
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
        assert actions[0].tool == "zoom_inspect"
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
        # 仍可注册进 registry，但 Agent1 落盘 Action 须分解为固定训练 Tool
        assert len(steps[0].actions) == 1
        assert steps[0].actions[0].tool == "zoom_inspect"
        assert steps[0].normalization_mode is NormalizationMode.MATCHED
        reg = load_registry(tmp_registry)
        assert "detect_facade_ornament" in reg
        dumped = reg["detect_facade_ornament"].model_dump()
        assert "tier" not in dumped
        assert "executor_ref" not in dumped

    def test_annotate_dynamic_tool_kept_on_coarse(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """视觉地理标注类允许原样进入 COARSE 训练链。"""

        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="matched",
                actions=[
                    _ProposedAction(
                        tool="annotate_geographic_environment_on_image",
                        params={
                            "target_image": "old_photo_001.jpg",
                            "geographic_features": ["高地桥", "平原"],
                        },
                    )
                ],
                confidence=0.88,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [_move(screen_action="标注桥与平原的地理环境")],
            AgentRole.COARSE,
        )
        assert len(steps[0].actions) == 1
        assert (
            steps[0].actions[0].tool
            == "annotate_geographic_environment_on_image"
        )

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
        assert steps[0].actions[0].tool == "zoom_inspect"


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


class TestCoarseTrainingDecompose:
    def test_compare_images_kept_without_spurious_ocr(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """比对语义保留 compare_images；无文字语义不加 OCR。"""

        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="matched",
                actions=[
                    _ProposedAction(
                        tool="compare_images_for_geolocation",
                        params={"image_a": "a", "image_b": "b"},
                    )
                ],
                confidence=0.8,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [
                _move(
                    screen_action="对比卫星图与老照片桥梁跨度",
                    narration="看看桥和河面宽度",
                )
            ],
            AgentRole.COARSE,
        )
        tools = [a.tool for a in steps[0].actions]
        assert "compare_images_for_geolocation" in tools
        assert "web_search" not in tools
        assert "ocr" not in tools

    def test_satellite_annotate_not_zoom_fallback(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """卫星标注语义即使 LLM 误选 zoom，也应改写为 geo Tool。"""

        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="matched",
                actions=[
                    _ProposedAction(
                        tool="zoom_inspect",
                        params={"bbox": [0.05, 0.1, 0.9, 0.8]},
                    )
                ],
                confidence=0.55,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [
                _move(
                    screen_action="在卫星地图上标注拍摄朝向及依山亭位置",
                    narration="反而不远处依山亭的东侧山脚下",
                )
            ],
            AgentRole.COARSE,
        )
        tools = [a.tool for a in steps[0].actions]
        assert "zoom_inspect" not in tools
        assert any(
            t
            in {
                "annotate_geographic_environment_on_image",
                "lookup_historical_satellite_map",
                "detect_terrain_features",
            }
            for t in tools
        )

    def test_diversify_zoom_avoids_identical_bbox(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """连续模板 zoom 经 diversify 后 bbox 不应全部相同。"""

        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="matched",
                actions=[
                    _ProposedAction(
                        tool="zoom_inspect",
                        params={"bbox": [0.05, 0.1, 0.9, 0.8]},
                    )
                ],
                confidence=0.9,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        moves = [
            _move(
                screen_action=f"放大观察桥梁细节{i}",
                narration="看桥墩结构",
                start=float(i),
                end=float(i) + 0.5,
            )
            for i in range(5)
        ]
        steps = normalize_to_steps(moves, AgentRole.COARSE)
        bboxes = []
        for s in steps:
            if s.actions and s.actions[0].tool == "zoom_inspect":
                bboxes.append(tuple(s.actions[0].params["bbox"]))
        assert len(bboxes) >= 2
        assert len(set(bboxes)) >= 2

    def test_shadow_narration_can_add_sun(
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
                        params={"query": "shadow latitude", "purpose": "broad_discovery"},
                    )
                ],
                confidence=0.8,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [
                _move(
                    screen_action="根据阴影推算太阳位置",
                    narration="阴影朝北，估算日照",
                )
            ],
            AgentRole.COARSE,
        )
        tools = [a.tool for a in steps[0].actions]
        assert "sun_position_calc" in tools
        assert "web_search" not in tools

    def test_fine_keeps_web_search(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """分解守卫仅作用于 COARSE。"""

        def _fake(*_a: Any, **_k: Any) -> _MatchLLMResponse:
            return _MatchLLMResponse(
                decision="matched",
                actions=[
                    _ProposedAction(
                        tool="web_search",
                        params={"query": "q", "purpose": "precise_lookup"},
                    )
                ],
                confidence=0.9,
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        steps = normalize_to_steps(
            [_move(screen_action="搜索地标", role=AgentRole.FINE)],
            AgentRole.FINE,
        )
        assert steps[0].actions[0].tool == "web_search"


class TestEvidenceRoutingStage3:
    """语义重路由、旁白冲突、差异化 bbox、禁止 Tool 不恢复。"""

    def test_late_coarse_geo_reroutes_from_fine_window(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pipeline.stage3_normalize import normalize_all_agent_steps

        def _fake(_prompt: str, response_model: type, **_k: Any) -> object:
            from pipeline.evidence_routing import (
                ExtractedVideoFact,
                VideoContextExtraction,
            )

            return _dispatch_stage3_llm(
                response_model,
                extraction=VideoContextExtraction(
                    facts=[
                        ExtractedVideoFact(
                            move_index=0,
                            claim="应排除无高地的桥",
                            concepts=["高地", "桥", "平原"],
                            kind="exclude",
                            excluded_candidates=["无高地的桥"],
                        ),
                        ExtractedVideoFact(
                            move_index=1,
                            claim="核对公园扶手花纹与坐标是否一致。",
                            concepts=["扶手", "坐标"],
                            kind="observe",
                        ),
                    ]
                ),
                match=_MatchLLMResponse(
                    decision="matched",
                    actions=[
                        _ProposedAction(
                            tool="zoom_inspect",
                            params={"bbox": [0.1, 0.1, 0.8, 0.7]},
                        )
                    ],
                    confidence=0.9,
                ),
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        late = Move(
            start_time=180.0,
            end_time=200.0,
            narration="重新分析：高地俯瞰长桥，桥跨宽河，对岸是平原，应排除无高地的桥。",
            screen_action="查看卫星图对比",
            visible_clues=["高地", "桥", "平原"],
            agent_role=AgentRole.FINE,
        )
        park = Move(
            start_time=220.0,
            end_time=230.0,
            narration="核对公园扶手花纹与坐标是否一致。",
            screen_action="放大扶手做匹配",
            visible_clues=["扶手"],
            agent_role=AgentRole.FINE,
        )
        out = normalize_all_agent_steps(
            {
                AgentRole.COARSE: [],
                AgentRole.FINE: [late, park],
                AgentRole.VERIFIER: [],
            },
            answer_timestamp=300.0,
        )
        assert out[AgentRole.COARSE], "广域地貌纠错应重路由到 COARSE"
        assert any(
            a.tool
            in (
                "zoom_inspect",
                "ocr",
                "sun_position_calc",
                "compare_images_for_geolocation",
                "lookup_historical_satellite_map",
                "annotate_geographic_environment_on_image",
                "detect_terrain_features",
            )
            for s in out[AgentRole.COARSE]
            for a in s.actions
        )
        # 公园扶手/坐标类精定位步不得被 video_fact 吞进 COARSE
        assert out[AgentRole.FINE], "精定位粒度 Move 应保留在 FINE"
        assert any("扶手" in (s.move.narration or "") for s in out[AgentRole.FINE])

    def test_narration_ui_conflict_does_not_inherit_chat_semantics(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pipeline.evidence_routing import parse_evidence_intent
        from pipeline.stage3_normalize import normalize_all_agent_steps

        def _fake(_prompt: str, response_model: type, **_k: Any) -> object:
            from pipeline.evidence_routing import (
                ExtractedVideoFact,
                VideoContextExtraction,
            )

            return _dispatch_stage3_llm(
                response_model,
                extraction=VideoContextExtraction(
                    facts=[
                        ExtractedVideoFact(
                            move_index=0,
                            claim="远景有一座长桥跨过宽河",
                            concepts=["长桥", "宽河", "高地"],
                            kind="exclude",
                            excluded_candidates=["聊天框"],
                        )
                    ]
                ),
                match=_MatchLLMResponse(
                    decision="matched",
                    actions=[
                        _ProposedAction(
                            tool="zoom_inspect",
                            params={"bbox": [0.1, 0.1, 0.8, 0.7]},
                        )
                    ],
                    confidence=0.9,
                ),
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        move = Move(
            start_time=40.0,
            end_time=50.0,
            narration="远景有一座长桥跨过宽河，背景是高地。",
            screen_action="置顶聊天消息并滚动评论区",
            visible_clues=["聊天框"],
            agent_role=AgentRole.COARSE,
        )
        out = normalize_all_agent_steps(
            {
                AgentRole.COARSE: [move],
                AgentRole.FINE: [],
                AgentRole.VERIFIER: [],
            },
            answer_timestamp=300.0,
        )
        assert out[AgentRole.COARSE]
        intent = parse_evidence_intent(out[AgentRole.COARSE][0].thought_draft)
        assert intent is not None
        assert intent.screen_action_untrusted is True
        tools = {a.tool for s in out[AgentRole.COARSE] for a in s.actions}
        assert "web_search" not in tools
        assert "map_query" not in tools

    def test_bbox_defaults_are_not_feature_specialized(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """bbox 不得按特定地貌词硬编码分支（防过拟合）。"""
        from pipeline.evidence_routing import (
            ExtractedVideoFact,
            VideoContextExtraction,
        )
        from pipeline.stage3_normalize import normalize_all_agent_steps

        def _fake(_prompt: str, response_model: type, **_k: Any) -> object:
            return _dispatch_stage3_llm(
                response_model,
                extraction=VideoContextExtraction(
                    facts=[
                        ExtractedVideoFact(
                            move_index=0,
                            claim="排除候选甲",
                            concepts=["目标甲"],
                            kind="exclude",
                            excluded_candidates=["候选甲"],
                        ),
                        ExtractedVideoFact(
                            move_index=1,
                            claim="排除候选乙",
                            concepts=["目标乙"],
                            kind="exclude",
                            excluded_candidates=["候选乙"],
                        ),
                    ]
                ),
                match=_MatchLLMResponse(
                    decision="matched",
                    actions=[
                        _ProposedAction(
                            tool="zoom_inspect",
                            params={"bbox": [0.2, 0.2, 0.6, 0.6]},
                        )
                    ],
                    confidence=0.9,
                ),
            )

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", _fake)
        m1 = Move(
            start_time=10.0,
            end_time=20.0,
            narration="排除候选甲。",
            screen_action="放大查看目标甲",
            visible_clues=["目标甲"],
            agent_role=AgentRole.COARSE,
        )
        m2 = Move(
            start_time=25.0,
            end_time=35.0,
            narration="排除候选乙。",
            screen_action="放大查看目标乙",
            visible_clues=["目标乙"],
            agent_role=AgentRole.COARSE,
        )
        out = normalize_all_agent_steps(
            {
                AgentRole.COARSE: [m1, m2],
                AgentRole.FINE: [],
                AgentRole.VERIFIER: [],
            },
            answer_timestamp=300.0,
        )
        zooms = [
            a.params.get("bbox")
            for s in out[AgentRole.COARSE]
            for a in s.actions
            if a.tool == "zoom_inspect"
        ]
        assert len(zooms) >= 1

    def test_pure_ui_produces_no_training_actions(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pipeline.stage3_normalize import normalize_all_agent_steps

        monkeypatch.setattr(
            "pipeline.stage3_normalize._extract_video_chain_context",
            lambda _moves: __import__(
                "pipeline.evidence_routing", fromlist=["VideoChainContext"]
            ).VideoChainContext(),
        )
        ui = Move(
            start_time=1.0,
            end_time=2.0,
            narration="",
            screen_action="打开聊天置顶并点赞",
            visible_clues=["聊天界面"],
            agent_role=AgentRole.COARSE,
        )
        out = normalize_all_agent_steps(
            {
                AgentRole.COARSE: [ui],
                AgentRole.FINE: [],
                AgentRole.VERIFIER: [],
            },
            answer_timestamp=300.0,
        )
        assert out[AgentRole.COARSE] == []

    def test_working_scope_and_geo_chain_filter(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """逐视频抽取工作范围；置顶剔除；试错+成功均保留。"""
        from pipeline.evidence_routing import (
            ExtractedRawClue,
            ExtractedVideoFact,
            ExtractedWorkingScope,
            RawClueRole,
            VideoContextExtraction,
            parse_video_context,
            strip_evidence_intent,
        )
        from pipeline.stage3_normalize import _VideoContextGroundingReview
        from pipeline.stage3_normalize import normalize_all_agent_steps

        def fake_structured(_prompt: str, response_model: type, **_kwargs: object) -> object:
            if response_model is VideoContextExtraction:
                return VideoContextExtraction(
                    raw_clues=[
                        ExtractedRawClue(
                            move_index=0,
                            text="拍摄地未出示例省",
                            clue_role=RawClueRole.PHOTO_LOCATION_CONSTRAINT,
                        ),
                        ExtractedRawClue(
                            move_index=0,
                            text="求助者来自示例省示例市",
                            clue_role=RawClueRole.PERSON_OR_SOCIAL_ATTRIBUTE,
                        ),
                    ],
                    working_scope=ExtractedWorkingScope(
                        region="示例省",
                        supporting_move_indices=[0],
                    ),
                    facts=[
                        ExtractedVideoFact(
                            move_index=2,
                            claim="目标甲横跨目标乙",
                            concepts=["目标甲", "目标乙"],
                            kind="observe",
                        ),
                        ExtractedVideoFact(
                            move_index=3,
                            claim="远端不是候选甲",
                            concepts=["远端", "候选甲"],
                            kind="correct",
                            supporting_move_indices=[2],
                            excluded_candidates=["候选甲"],
                        ),
                        ExtractedVideoFact(
                            move_index=4,
                            claim="排除候选乙",
                            concepts=["候选乙"],
                            kind="exclude",
                            supporting_move_indices=[2, 3],
                            excluded_candidates=["候选乙"],
                            proposed_candidates=["候选丙"],
                        ),
                    ],
                )
            if response_model is _VideoContextGroundingReview:
                return _VideoContextGroundingReview(
                    working_scope_supported=True,
                )
            raise AssertionError(response_model)

        monkeypatch.setattr(
            "pipeline.stage3_normalize.call_structured",
            fake_structured,
        )
        moves = [
            Move(
                start_time=1.0,
                end_time=3.0,
                narration="外部沟通说明拍摄地未出示例省；求助者来自示例省示例市。",
                screen_action="打开聊天",
                visible_clues=["示例省示例市"],
                agent_role=AgentRole.COARSE,
            ),
            Move(
                start_time=10.0,
                end_time=12.0,
                narration="先放着置顶，以后再想。",
                screen_action="置顶消息",
                visible_clues=[],
                agent_role=AgentRole.COARSE,
            ),
            Move(
                start_time=20.0,
                end_time=25.0,
                narration="画面明确显示目标甲横跨目标乙。",
                screen_action="查看老照片",
                visible_clues=["目标甲", "目标乙"],
                agent_role=AgentRole.COARSE,
            ),
            Move(
                start_time=40.0,
                end_time=48.0,
                narration="重新分析：远端不是候选甲，应纠正原假设。",
                screen_action="对比地貌",
                visible_clues=["远端", "候选甲"],
                agent_role=AgentRole.COARSE,
            ),
            Move(
                start_time=60.0,
                end_time=70.0,
                narration="排除候选乙后，很可能位于候选丙。",
                screen_action="排查候选",
                visible_clues=["候选乙", "候选丙"],
                agent_role=AgentRole.COARSE,
            ),
        ]
        out = normalize_all_agent_steps(
            {
                AgentRole.COARSE: moves,
                AgentRole.FINE: [],
                AgentRole.VERIFIER: [],
            },
            answer_timestamp=300.0,
        )
        coarse = out[AgentRole.COARSE]
        assert coarse
        narrs = " ".join(s.move.narration or "" for s in coarse)
        assert "置顶" not in narrs
        assert "目标甲" in narrs  # 被 supporting 引用的 observe 须保留
        assert "候选甲" in narrs
        assert "候选乙" in narrs
        # 推进步之前的 observe 也须保留（视觉地基）
        assert any(abs(s.move.start_time - 20.0) < 1e-6 for s in coarse)
        ctx = parse_video_context(coarse[0].thought_draft)
        assert ctx is not None
        assert ctx.working_scope is not None
        assert ctx.working_scope.region == "示例省内"
        assert ctx.working_scope.bound_kind.value == "inside"
        assert any("示例省" in c.text for c in ctx.raw_given_clues)
        # 推导候选可在 context，但不伪装进 thought 正文 known clue
        body = strip_evidence_intent(coarse[0].thought_draft)
        assert "候选丙" not in body or "候选丙" in (ctx.candidate_hypotheses or [])
        assert ctx.video_facts
        assert any(f.kind in ("correct", "exclude", "candidate") for f in ctx.video_facts)

    def test_distill_keeps_observe_facts_when_no_progress(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """无 exclude/correct 推进时，不得把 COARSE 蒸成空链。"""
        from pipeline.evidence_routing import (
            ExtractedVideoFact,
            VideoContextExtraction,
        )
        from pipeline.stage3_normalize import normalize_all_agent_steps

        def fake_structured(_prompt: str, response_model: type, **_kwargs: object) -> object:
            return _dispatch_stage3_llm(
                response_model,
                extraction=VideoContextExtraction(
                    facts=[
                        ExtractedVideoFact(
                            move_index=0,
                            claim="画面可见屋顶与远桥",
                            concepts=["屋顶", "远桥"],
                            kind="observe",
                        ),
                        ExtractedVideoFact(
                            move_index=1,
                            claim="对岸是大片平原",
                            concepts=["平原"],
                            kind="observe",
                        ),
                    ]
                ),
                match=_MatchLLMResponse(
                    decision="matched",
                    actions=[
                        _ProposedAction(
                            tool="zoom_inspect",
                            params={"bbox": [0.2, 0.2, 0.5, 0.5]},
                        )
                    ],
                    confidence=0.9,
                ),
            )

        monkeypatch.setattr(
            "pipeline.stage3_normalize.call_structured",
            fake_structured,
        )
        moves = [
            Move(
                start_time=10.0,
                end_time=12.0,
                narration="画面可见屋顶与远桥。",
                screen_action="查看老照片",
                visible_clues=["屋顶", "远桥"],
                agent_role=AgentRole.COARSE,
            ),
            Move(
                start_time=20.0,
                end_time=22.0,
                narration="对岸是大片平原。",
                screen_action="放大远景",
                visible_clues=["平原"],
                agent_role=AgentRole.COARSE,
            ),
        ]
        out = normalize_all_agent_steps(
            {
                AgentRole.COARSE: moves,
                AgentRole.FINE: [],
                AgentRole.VERIFIER: [],
            },
            answer_timestamp=300.0,
        )
        assert len(out[AgentRole.COARSE]) >= 1
        narrs = " ".join(s.move.narration or "" for s in out[AgentRole.COARSE])
        assert "屋顶" in narrs or "平原" in narrs

    def test_video_context_extraction_failure_raises(
        self,
        tmp_registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """真实路径下抽取失败必须抛错，不得静默 fallback。"""
        from pipeline.stage3_normalize import normalize_all_agent_steps

        def boom(*_a: object, **_k: object) -> object:
            raise ConnectionError("Connection error")

        monkeypatch.setattr("pipeline.stage3_normalize.call_structured", boom)
        moves = [
            Move(
                start_time=1.0,
                end_time=2.0,
                narration="外部沟通说明拍摄者来自示例省。",
                screen_action="打开聊天",
                visible_clues=["示例省"],
                agent_role=AgentRole.COARSE,
            ),
            Move(
                start_time=3.0,
                end_time=4.0,
                narration="排除候选甲。",
                screen_action="查看画面",
                visible_clues=["候选甲"],
                agent_role=AgentRole.COARSE,
            ),
        ]
        with pytest.raises(RuntimeError, match="拒绝降级到低质量 fallback"):
            normalize_all_agent_steps(
                {
                    AgentRole.COARSE: moves,
                    AgentRole.FINE: [],
                    AgentRole.VERIFIER: [],
                },
                answer_timestamp=300.0,
            )


def test_gap_fill_recovers_missed_highland_and_drops_meta() -> None:
    """LLM 漏抽时补全高地观察；开场耗时元叙事不得进入事实池。"""
    from pipeline.evidence_routing import (
        ExtractedVideoFact,
        SubjectScope,
        VideoContextExtraction,
        context_from_extraction,
        drop_meta_setup_facts,
        gap_fill_missing_geo_facts,
    )

    moves = [
        Move(
            start_time=0.0,
            end_time=4.6,
            narration="为了找到这张照片的拍摄地,我足足花了半年的时间。",
            screen_action="片头",
            visible_clues=[],
            agent_role=AgentRole.COARSE,
        ),
        Move(
            start_time=35.5,
            end_time=39.0,
            narration="细看照片,可以很清楚看到下方的建筑屋顶,",
            screen_action="放大",
            visible_clues=["屋顶"],
            agent_role=AgentRole.COARSE,
        ),
        Move(
            start_time=39.18,
            end_time=44.0,
            narration="所以父亲的照片是在一个高处的地点,看起来很像在山上。",
            screen_action="观察",
            visible_clues=["高处", "山上"],
            agent_role=AgentRole.COARSE,
        ),
        Move(
            start_time=155.0,
            end_time=166.0,
            narration="重新分析后,这应该只是一段河岸而已。",
            screen_action="纠正",
            visible_clues=["河岸"],
            agent_role=AgentRole.COARSE,
        ),
    ]
    extraction = VideoContextExtraction(
        facts=[
            ExtractedVideoFact(
                move_index=0,
                claim="为了找到这张照片的拍摄地,我足足花了半年的时间。",
                concepts=["照片", "半年"],
                kind="observe",
                subject_scope=SubjectScope.CAMERA_POSITION,
                spatial_anchor="拍摄地",
            ),
            ExtractedVideoFact(
                move_index=3,
                claim="重新分析后,这应该只是一段河岸而已。",
                concepts=["河岸"],
                kind="correct",
                corrected_to="河岸",
                subject_scope=SubjectScope.SCENE_REGION,
            ),
        ]
    )
    cleaned = drop_meta_setup_facts(extraction)
    assert all(f.move_index != 0 for f in cleaned.facts)
    filled = gap_fill_missing_geo_facts(moves, cleaned)
    idxs = {f.move_index for f in filled.facts}
    assert 1 in idxs and 2 in idxs and 3 in idxs
    assert 0 not in idxs
    ctx = context_from_extraction(moves, filled)
    scopes = {f.subject_scope for f in ctx.video_facts}
    assert SubjectScope.CAMERA_POSITION in scopes or any(
        "屋顶" in (f.quote + "".join(f.tokens)) for f in ctx.video_facts
    )


def test_camera_and_scene_scopes_coexist_without_cross_revoke() -> None:
    """近处与远处空间事实可并存；背景纠正不进入拍摄点候选池。"""
    from pipeline.evidence_routing import (
        ExtractedVideoFact,
        SubjectScope,
        VideoContextExtraction,
        context_from_extraction,
    )

    moves = [
        Move(
            start_time=1.0,
            end_time=2.0,
            narration="拍摄点下方可见高地，呈俯视关系。",
            screen_action="放大近景",
            visible_clues=["高地", "俯视"],
            agent_role=AgentRole.COARSE,
        ),
        Move(
            start_time=3.0,
            end_time=4.0,
            narration="远处对岸其实不是山，而是河岸平原。",
            screen_action="查看远景",
            visible_clues=["对岸", "河岸", "平原"],
            agent_role=AgentRole.COARSE,
        ),
    ]
    extraction = VideoContextExtraction(
        facts=[
            ExtractedVideoFact(
                move_index=0,
                claim="拍摄点下方可见高地，呈俯视关系。",
                concepts=["高地", "俯视"],
                kind="observe",
                subject_scope=SubjectScope.CAMERA_POSITION,
                spatial_anchor="拍摄点",
            ),
            ExtractedVideoFact(
                move_index=1,
                claim="远处对岸其实不是山，而是河岸平原。",
                concepts=["对岸", "河岸", "平原"],
                kind="correct",
                corrected_from="山",
                corrected_to="河岸平原",
                subject_scope=SubjectScope.SCENE_REGION,
                spatial_anchor="远处",
            ),
        ]
    )
    ctx = context_from_extraction(moves, extraction)
    assert len(ctx.video_facts) == 2
    scopes = {f.subject_scope for f in ctx.video_facts}
    assert SubjectScope.CAMERA_POSITION in scopes
    assert SubjectScope.SCENE_REGION in scopes
    # 背景纠正不得进入拍摄点候选池
    assert ctx.candidate_hypotheses == []
