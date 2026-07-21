"""stage5：轨迹重构、handoff、返工与 groundtruth 隔离测试（LLM 全部 mock）。"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from pipeline.schemas import (
    Action,
    AgentRole,
    LocationHypothesis,
    Move,
    NormalizationMode,
    NormalizedStep,
    ObservationExecutionResult,
    ObservationSource,
    RevisionSource,
    SubmitAnswerResult,
    VerificationResult,
)
from pipeline.stage5_reconstruct import (
    _CoarseOutputBundle,
    _CoarseReasoningCheck,
    _CoarseToolSuitability,
    _ExternalHints,
    _RewrittenTrajectory,
    _TaoStyleCheck,
    _VerifierOutputBundle,
    reconstruct_all_trajectories,
    reconstruct_revision_trajectories,
    reconstruct_single_trajectory,
)


def _hyp() -> LocationHypothesis:
    return LocationHypothesis(
        possible_countries=["France"],
        possible_regions=["Île-de-France"],
        reasoning_summary="Landmark geometry suggests France.",
        confidence=0.7,
        key_clues_remaining=["exact plaza"],
    )


def _submit() -> SubmitAnswerResult:
    return SubmitAnswerResult(
        latitude=48.8584,
        longitude=2.2945,
        location_name="Eiffel Tower",
        confidence=0.9,
        reasoning="Matched tower silhouette after map_query.",
    )


def _move(
    *,
    start: float = 0.0,
    end: float = 1.0,
    role: AgentRole = AgentRole.COARSE,
    narration: str = "旁白。",
) -> Move:
    return Move(
        start_time=start,
        end_time=end,
        narration=narration,
        screen_action="操作",
        visible_clues=[],
        agent_role=role,
    )


def _step(
    actions: list[Action],
    *,
    thought: str = "草稿思考。",
    role: AgentRole = AgentRole.COARSE,
    start: float = 0.0,
    end: float = 1.0,
    mode: NormalizationMode = NormalizationMode.MATCHED,
) -> NormalizedStep:
    return NormalizedStep(
        move=_move(start=start, end=end, role=role),
        thought_draft=thought,
        actions=actions,
        normalization_mode=mode if actions else NormalizationMode.THOUGHT_ONLY,
        matched_tool_confidence=0.9 if actions else None,
        fallback_reason=None if actions else "screen_action 为空",
    )


def _obs(
    action: Action,
    *,
    observation: Optional[dict[str, Any]] = None,
    status: str = "success",
    source: Optional[ObservationSource] = ObservationSource.LLM_SYNTHESIZED,
) -> ObservationExecutionResult:
    if action.tool == "submit_answer":
        return ObservationExecutionResult(
            action=action,
            observation=None,
            source=None,
            status="skipped",
            error_message=None,
            cache_hit=False,
        )
    return ObservationExecutionResult(
        action=action,
        observation=observation or {"status": "success", "error_message": None},
        source=source,
        status=status,  # type: ignore[arg-type]
        error_message=None,
        cache_hit=False,
    )


@pytest.fixture()
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """拦截 call_structured，按 response_model 返回合法结构化结果。"""
    calls: list[dict[str, Any]] = []

    def _fake(
        prompt: str,
        response_model: type[BaseModel],
        images: Optional[list[str]] = None,
        video: Optional[str] = None,
        model: Optional[str] = None,
    ) -> BaseModel:
        calls.append(
            {
                "prompt": prompt,
                "response_model": response_model,
                "images": images,
                "video": video,
                "model": model,
            }
        )
        # 统计 scaffold 中的 Step 数
        n_steps = prompt.count("### Step ")
        thoughts = [f"前向思考 {i + 1}" for i in range(max(n_steps, 1))]

        if response_model is _CoarseOutputBundle:
            return _CoarseOutputBundle(thoughts=thoughts, coarse_output=_hyp())
        if response_model is _VerifierOutputBundle:
            return _VerifierOutputBundle(
                thoughts=thoughts,
                verifier_output=VerificationResult(
                    verdict="pass",
                    failed_checks=[],
                    suggested_recheck="none",
                    return_to_agent=None,
                ),
            )
        if response_model is _RewrittenTrajectory:
            return _RewrittenTrajectory(thoughts=thoughts)
        if response_model is SubmitAnswerResult:
            return _submit()
        if response_model is _TaoStyleCheck:
            return _TaoStyleCheck(is_standard_tao=True, issues=[])
        if response_model is _CoarseReasoningCheck:
            return _CoarseReasoningCheck(
                identifies_geo_human_features=True,
                narrows_scope_progressively=True,
                has_reasoning_gap=False,
                thought_action_aligned=True,
                issues=[],
            )
        if response_model is _CoarseToolSuitability:
            # 默认：名称含 compare/inspect/feature → 适合；否则否
            suitable = any(
                k in prompt.lower()
                for k in ("compare", "inspect", "feature", "shadow", "植被", "建筑")
            )
            return _CoarseToolSuitability(
                suitable_for_coarse_reasoning=suitable,
                reason="mock",
            )
        if response_model is _ExternalHints:
            hints: list[str] = []
            if "河南信阳" in prompt or "网友" in prompt:
                hints = ["河南信阳"]
            return _ExternalHints(hints=hints)
        raise AssertionError(f"未预期的 response_model: {response_model}")

    monkeypatch.setattr(
        "pipeline.stage5_reconstruct.call_structured",
        _fake,
    )
    holder = MagicMock()
    holder.calls = calls
    return holder


def test_coarse_trajectory_output_and_no_handoff(mock_llm: MagicMock) -> None:
    action = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    steps = [_step([action], thought="看塔尖。")]
    observations = [
        _obs(
            action,
            observation={
                "status": "success",
                "error_message": None,
                "description": "iron lattice tower",
            },
        )
    ]
    traj = reconstruct_single_trajectory(
        steps,
        observations,
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    assert traj.agent_role == AgentRole.COARSE
    assert traj.coarse_handoff is None
    assert traj.fine_handoff is None
    assert traj.coarse_output is not None
    assert traj.coarse_output.possible_countries == ["France"]
    assert len(traj.steps) == 1
    assert traj.steps[0].thought.startswith("前向思考")
    assert traj.steps[0].observation is not None
    # scaffold 改写 prompt 不得含 groundtruth
    scaffold_calls = [
        c for c in mock_llm.calls if c["response_model"] is _CoarseOutputBundle
    ]
    assert scaffold_calls
    assert "groundtruth" not in scaffold_calls[0]["prompt"].lower()
    assert "风格规范" in scaffold_calls[0]["prompt"] or "GOOD 示例" in scaffold_calls[0]["prompt"]
    assert any(c["response_model"] is _TaoStyleCheck for c in mock_llm.calls)
    assert any(c["response_model"] is _CoarseReasoningCheck for c in mock_llm.calls)
    assert "特征识别" in traj.system_prompt or "递进" in traj.system_prompt


def test_coarse_projects_out_web_search(mock_llm: MagicMock) -> None:
    """COARSE 投影剔除 web_search，保留前后允许步骤且 Obs 对齐。"""
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    search = Action(
        tool="web_search",
        params={"query": "tropical arcade street", "purpose": "broad_discovery"},
    )
    ocr = Action(tool="ocr", params={"bbox": [0.2, 0.2, 0.3, 0.3]})
    steps = [
        _step([zoom], thought="看建筑。"),
        _step([search], thought="搜索。"),
        _step([ocr], thought="读文字。"),
    ]
    observations = [
        _obs(zoom, observation={"status": "success", "error_message": None, "description": "arcade"}),
        _obs(
            search,
            observation={
                "status": "success",
                "error_message": None,
                "results": [{"title": "x", "snippet": "y", "url": "http://a"}],
            },
        ),
        _obs(ocr, observation={"status": "success", "error_message": None, "texts": ["Cafe"]}),
    ]
    traj = reconstruct_single_trajectory(
        steps,
        observations,
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    tools = [s.action.tool for s in traj.steps]
    assert tools == ["zoom_inspect", "ocr"]
    assert "web_search" not in tools
    scaffold = next(
        c["prompt"] for c in mock_llm.calls if c["response_model"] is _CoarseOutputBundle
    )
    assert "tool=web_search" not in scaffold
    assert scaffold.count("### Step ") == 2


def test_coarse_projection_empty_raises(mock_llm: MagicMock) -> None:
    """仅 web_search 时投影为空应失败。"""
    search = Action(
        tool="web_search",
        params={"query": "q", "purpose": "broad_discovery"},
    )
    with pytest.raises(ValueError, match="投影后无可重构"):
        reconstruct_single_trajectory(
            [_step([search], thought="搜。")],
            [
                _obs(
                    search,
                    observation={
                        "status": "success",
                        "error_message": None,
                        "results": [],
                    },
                )
            ],
            AgentRole.COARSE,
            answer_timestamp=10.0,
            image_path="frame.jpg",
        )


def test_coarse_reasoning_gap_triggers_rewrite(
    mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """递进链跳步触发带 issues 重写。"""
    chain_n = {"n": 0}

    def _fake(
        prompt: str,
        response_model: type[BaseModel],
        images: Optional[list[str]] = None,
        video: Optional[str] = None,
        model: Optional[str] = None,
    ) -> BaseModel:
        _ = images, video, model
        n_steps = max(prompt.count("### Step "), 1)
        thoughts = [f"前向思考 {i + 1}" for i in range(n_steps)]
        if response_model is _ExternalHints:
            return _ExternalHints(hints=[])
        if response_model is _TaoStyleCheck:
            return _TaoStyleCheck(is_standard_tao=True, issues=[])
        if response_model is _CoarseReasoningCheck:
            chain_n["n"] += 1
            # 仅第一次检查；重写后不再检查（与现实现一致）
            return _CoarseReasoningCheck(
                identifies_geo_human_features=False,
                narrows_scope_progressively=False,
                has_reasoning_gap=True,
                thought_action_aligned=False,
                issues=["单一弱特征跳步"],
            )
        if response_model is _CoarseOutputBundle:
            return _CoarseOutputBundle(thoughts=thoughts, coarse_output=_hyp())
        raise AssertionError(response_model)

    monkeypatch.setattr("pipeline.stage5_reconstruct.call_structured", _fake)
    action = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    traj = reconstruct_single_trajectory(
        [_step([action], thought="看。")],
        [_obs(action, observation={"status": "success", "error_message": None, "description": "x"})],
        AgentRole.COARSE,
        answer_timestamp=10.0,
        image_path="frame.jpg",
    )
    assert traj.coarse_output is not None
    assert chain_n["n"] == 1


def test_external_hints_go_into_user_query(mock_llm: MagicMock) -> None:
    """网友给定地名抽入 user_query，不含来源套话。"""
    action = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    steps = [
        NormalizedStep(
            move=_move(narration="有网友说是河南信阳附近拍的。"),
            thought_draft="看塔。",
            actions=[action],
            normalization_mode=NormalizationMode.MATCHED,
            matched_tool_confidence=0.9,
            fallback_reason=None,
        )
    ]
    traj = reconstruct_single_trajectory(
        steps,
        [
            _obs(
                action,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "tower",
                },
            )
        ],
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    assert "河南信阳" in traj.user_query
    assert "已知线索" in traj.user_query
    assert "网友" not in traj.user_query
    scaffold = next(
        c["prompt"] for c in mock_llm.calls if c["response_model"] is _CoarseOutputBundle
    )
    assert "河南信阳" in scaffold


def test_fine_prompt_allows_early_precise(mock_llm: MagicMock) -> None:
    map_action = Action(tool="map_query", params={"query": "Eiffel Tower"})
    submit = Action(tool="submit_answer", params=_submit().model_dump())
    steps = [
        _step([map_action], role=AgentRole.FINE, thought="查地图。"),
        _step([submit], role=AgentRole.FINE, thought="提交。", start=1.0, end=2.0),
    ]
    observations = [
        _obs(
            map_action,
            observation={
                "status": "success",
                "error_message": None,
                "formatted_address": "Paris",
                "resolved_latlng": [48.8584, 2.2945],
                "place_type": "landmark",
            },
        ),
        _obs(submit),
    ]
    traj = reconstruct_single_trajectory(
        steps,
        observations,
        AgentRole.FINE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        coarse_handoff=_hyp(),
    )
    assert traj.fine_output is not None
    assert "缩小范围无上限" in traj.system_prompt or "可尽早" in traj.system_prompt
    rewrite = next(
        c["prompt"]
        for c in mock_llm.calls
        if c["response_model"] is _RewrittenTrajectory
    )
    assert "缩小范围无上限" in rewrite or "尽早写出" in rewrite


def test_tao_style_failure_triggers_rewrite(mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """形态自检失败时强制重写一次。"""
    calls: list[type] = []
    style_calls = {"n": 0}

    def _fake(
        prompt: str,
        response_model: type[BaseModel],
        images: Optional[list[str]] = None,
        video: Optional[str] = None,
        model: Optional[str] = None,
    ) -> BaseModel:
        _ = images, video, model
        calls.append(response_model)
        n_steps = max(prompt.count("### Step "), 1)
        thoughts = [f"前向思考 {i + 1}" for i in range(n_steps)]
        if response_model is _TaoStyleCheck:
            style_calls["n"] += 1
            # 第一次失败触发重写
            ok = style_calls["n"] > 1
            return _TaoStyleCheck(
                is_standard_tao=ok,
                issues=[] if ok else ["旁白叙事体"],
            )
        if response_model is _ExternalHints:
            return _ExternalHints(hints=[])
        if response_model is _CoarseReasoningCheck:
            return _CoarseReasoningCheck(
                identifies_geo_human_features=True,
                narrows_scope_progressively=True,
                has_reasoning_gap=False,
                thought_action_aligned=True,
                issues=[],
            )
        if response_model is _CoarseOutputBundle:
            return _CoarseOutputBundle(thoughts=thoughts, coarse_output=_hyp())
        raise AssertionError(response_model)

    monkeypatch.setattr("pipeline.stage5_reconstruct.call_structured", _fake)
    action = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    traj = reconstruct_single_trajectory(
        [_step([action], thought="看塔尖。")],
        [
            _obs(
                action,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "tower",
                },
            )
        ],
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    assert traj.coarse_output is not None
    assert calls.count(_CoarseOutputBundle) == 2
    assert style_calls["n"] >= 1


def test_fine_requires_submit_answer_and_terminal_none(mock_llm: MagicMock) -> None:
    map_action = Action(
        tool="map_query",
        params={"query": "Eiffel Tower"},
    )
    submit = Action(
        tool="submit_answer",
        params=_submit().model_dump(),
    )
    steps = [
        _step([map_action], role=AgentRole.FINE, thought="查地图。"),
        _step([submit], role=AgentRole.FINE, thought="提交。", start=1.0, end=2.0),
    ]
    observations = [
        _obs(
            map_action,
            observation={
                "status": "success",
                "error_message": None,
                "formatted_address": "Paris",
                "resolved_latlng": [48.8584, 2.2945],
                "place_type": "tourist_attraction",
            },
        ),
        _obs(submit),
    ]
    traj = reconstruct_single_trajectory(
        steps,
        observations,
        AgentRole.FINE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        coarse_handoff=_hyp(),
    )
    assert traj.fine_output is not None
    assert traj.fine_output.location_name == "Eiffel Tower"
    assert traj.steps[-1].action.tool == "submit_answer"
    assert traj.steps[-1].observation is None
    assert traj.steps[-1].observation_source is None
    assert traj.coarse_handoff is not None
    assert traj.fine_handoff is None


def test_fine_synthesizes_submit_answer_when_missing(mock_llm: MagicMock) -> None:
    """脚手架末步非 submit_answer 时，stage5 基于证据合成 terminal 步。"""
    map_action = Action(tool="map_query", params={"query": "Eiffel Tower"})
    steps = [
        _step([map_action], role=AgentRole.FINE, thought="查地图锁定地标。"),
    ]
    observations = [
        _obs(
            map_action,
            observation={
                "status": "success",
                "error_message": None,
                "formatted_address": "Paris",
                "resolved_latlng": [48.8584, 2.2945],
                "place_type": "tourist_attraction",
            },
        )
    ]
    traj = reconstruct_single_trajectory(
        steps,
        observations,
        AgentRole.FINE,
        answer_timestamp=50.0,
        image_path="a.jpg",
        coarse_handoff=_hyp(),
    )
    assert traj.steps[-1].action.tool == "submit_answer"
    assert traj.steps[-1].observation is None
    assert traj.fine_output is not None
    assert traj.fine_output.location_name == "Eiffel Tower"
    assert len(traj.steps) == 2
    # 先合成 SubmitAnswerResult，再抽线索，再改写 thoughts
    assert mock_llm.calls[0]["response_model"] is SubmitAnswerResult
    assert "groundtruth:" not in mock_llm.calls[0]["prompt"].lower()
    rewrite = next(
        c for c in mock_llm.calls if c["response_model"] is _RewrittenTrajectory
    )
    assert rewrite["prompt"].count("### Step ") == 2


def test_fine_empty_units_cannot_synthesize_submit(mock_llm: MagicMock) -> None:
    """全 thought_only 时无法展开 Action，仍应失败。"""
    steps = [_step([], role=AgentRole.FINE, thought="只有旁白。")]
    with pytest.raises(ValueError, match="无可重构的 Action"):
        reconstruct_single_trajectory(
            steps,
            [],
            AgentRole.FINE,
            answer_timestamp=50.0,
            image_path="a.jpg",
            coarse_handoff=_hyp(),
        )


def _verifier_map_obs(action: Action) -> ObservationExecutionResult:
    return _obs(
        action,
        observation={
            "status": "success",
            "error_message": None,
            "formatted_address": "Paris",
            "resolved_latlng": [48.8584, 2.2945],
            "place_type": "tourist_attraction",
        },
    )


def _verifier_web_action() -> Action:
    return Action(
        tool="web_search",
        params={
            "query": "verify Eiffel Tower visual features",
            "top_k": 3,
            "purpose": "verification",
        },
    )


def _verifier_web_obs(action: Action) -> ObservationExecutionResult:
    return _obs(
        action,
        observation={
            "status": "success",
            "error_message": None,
            "results": [
                {
                    "title": "Landmark",
                    "snippet": "iron lattice",
                    "url": "https://example.com/a",
                }
            ],
        },
    )


def test_verifier_uses_fine_handoff_as_candidate(mock_llm: MagicMock) -> None:
    map_action = Action(
        tool="map_query",
        params={"latlng": [48.8584, 2.2945]},
    )
    web_action = _verifier_web_action()
    steps = [
        _step([map_action], role=AgentRole.VERIFIER, thought="核对坐标。"),
        _step([web_action], role=AgentRole.VERIFIER, start=1.0, end=2.0, thought="检索佐证。"),
    ]
    observations = [_verifier_map_obs(map_action), _verifier_web_obs(web_action)]
    traj = reconstruct_single_trajectory(
        steps,
        observations,
        AgentRole.VERIFIER,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        coarse_handoff=_hyp(),
        fine_handoff=_submit(),
    )
    assert len(traj.steps) == 2
    assert traj.verifier_output is not None
    assert traj.verifier_output.verdict == "pass"
    assert traj.fine_handoff is not None
    prompt = mock_llm.calls[0]["prompt"]
    assert "候选" in prompt or "fine_handoff" in prompt
    assert "48.8584" in prompt  # 候选坐标可出现
    assert "groundtruth" not in prompt.lower()


def test_verifier_synthesizes_scaffold_when_no_video_actions(
    mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无视频 Action 时，基于 fine_handoff 合成 map_query+web_search 验证链。"""
    calls: list[str] = []

    def _fake_execute(
        action: Action, image_path: str, agent_role: AgentRole, **kwargs: object
    ) -> ObservationExecutionResult:
        _ = kwargs
        assert agent_role == AgentRole.VERIFIER
        assert image_path == "frame.jpg"
        calls.append(action.tool)
        if action.tool == "map_query":
            return _obs(
                action,
                observation={
                    "status": "success",
                    "error_message": None,
                    "formatted_address": "Paris",
                    "resolved_latlng": [48.8584, 2.2945],
                    "place_type": "tourist_attraction",
                },
            )
        if action.tool == "web_search":
            assert action.params.get("purpose") == "verification"
            return _obs(
                action,
                observation={
                    "status": "success",
                    "error_message": None,
                    "results": [
                        {
                            "title": "Landmark guide",
                            "snippet": "iron lattice tower",
                            "url": "https://example.com/a",
                        }
                    ],
                },
            )
        raise AssertionError(f"unexpected tool {action.tool}")

    monkeypatch.setattr(
        "pipeline.stage5_reconstruct.execute_action",
        _fake_execute,
    )
    traj = reconstruct_single_trajectory(
        [],
        [],
        AgentRole.VERIFIER,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        fine_handoff=_submit(),
    )
    assert calls == ["map_query", "web_search"]
    assert len(traj.steps) == 2
    assert traj.steps[0].action.tool == "map_query"
    assert traj.steps[1].action.tool == "web_search"
    assert traj.steps[0].observation is not None
    assert traj.steps[1].observation is not None
    assert traj.verifier_output is not None
    assert traj.fine_handoff is not None
    prompt = mock_llm.calls[0]["prompt"]
    assert "时序" in prompt or "禁止" in prompt


def test_reconstruct_all_handoff_chain(mock_llm: MagicMock) -> None:
    coarse_action = Action(tool="zoom_inspect", params={"bbox": [0.0, 0.0, 0.5, 0.5]})
    fine_map = Action(tool="map_query", params={"query": "tower paris"})
    fine_submit = Action(tool="submit_answer", params=_submit().model_dump())
    ver_map = Action(tool="map_query", params={"latlng": [48.8584, 2.2945]})
    ver_web = _verifier_web_action()

    all_steps = {
        AgentRole.COARSE: [_step([coarse_action], role=AgentRole.COARSE)],
        AgentRole.FINE: [
            _step([fine_map], role=AgentRole.FINE),
            _step([fine_submit], role=AgentRole.FINE, start=1.0, end=2.0),
        ],
        AgentRole.VERIFIER: [
            _step([ver_map], role=AgentRole.VERIFIER),
            _step([ver_web], role=AgentRole.VERIFIER, start=1.0, end=2.0),
        ],
    }
    all_obs = {
        AgentRole.COARSE: [
            _obs(
                coarse_action,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "tower",
                },
            )
        ],
        AgentRole.FINE: [
            _obs(
                fine_map,
                observation={
                    "status": "success",
                    "error_message": None,
                    "formatted_address": "Paris",
                    "resolved_latlng": [48.8584, 2.2945],
                    "place_type": "tourist_attraction",
                },
            ),
            _obs(fine_submit),
        ],
        AgentRole.VERIFIER: [
            _verifier_map_obs(ver_map),
            _verifier_web_obs(ver_web),
        ],
    }
    result = reconstruct_all_trajectories(
        all_steps, all_obs, answer_timestamp=80.0, image_path="img.jpg"
    )
    assert set(result.keys()) == {
        AgentRole.COARSE,
        AgentRole.FINE,
        AgentRole.VERIFIER,
    }
    assert result[AgentRole.FINE].coarse_handoff == result[AgentRole.COARSE].coarse_output
    assert result[AgentRole.VERIFIER].fine_handoff == result[AgentRole.FINE].fine_output
    assert len(result[AgentRole.VERIFIER].steps) == 2


def test_thought_only_merged_into_next_action(mock_llm: MagicMock) -> None:
    action = Action(tool="ocr", params={})
    steps = [
        _step([], thought="先观察整体。", mode=NormalizationMode.THOUGHT_ONLY),
        _step([action], thought="再读招牌。"),
    ]
    observations = [
        _obs(
            action,
            observation={"status": "success", "error_message": None, "texts": ["Cafe"]},
        )
    ]
    traj = reconstruct_single_trajectory(
        steps,
        observations,
        AgentRole.COARSE,
        answer_timestamp=10.0,
        image_path="x.jpg",
    )
    assert len(traj.steps) == 1
    # scaffold prompt 应包含 thought_only 草稿
    scaffold = next(
        c["prompt"] for c in mock_llm.calls if c["response_model"] is _CoarseOutputBundle
    )
    assert "先观察整体" in scaffold


def test_system_feedback_revision_to_fine(mock_llm: MagicMock) -> None:
    hyp = _hyp()
    submit = _submit()
    parents = {
        AgentRole.COARSE: reconstruct_single_trajectory(
            [
                _step(
                    [Action(tool="zoom_inspect", params={"bbox": [0, 0, 1, 1]})],
                    role=AgentRole.COARSE,
                )
            ],
            [
                _obs(
                    Action(tool="zoom_inspect", params={"bbox": [0, 0, 1, 1]}),
                    observation={
                        "status": "success",
                        "error_message": None,
                        "description": "x",
                    },
                )
            ],
            AgentRole.COARSE,
            50.0,
            "i.jpg",
        ),
        AgentRole.FINE: reconstruct_single_trajectory(
            [
                _step(
                    [Action(tool="submit_answer", params=submit.model_dump())],
                    role=AgentRole.FINE,
                )
            ],
            [_obs(Action(tool="submit_answer", params=submit.model_dump()))],
            AgentRole.FINE,
            50.0,
            "i.jpg",
            coarse_handoff=hyp,
        ),
        AgentRole.VERIFIER: reconstruct_single_trajectory(
            [
                _step(
                    [Action(tool="map_query", params={"query": "x"})],
                    role=AgentRole.VERIFIER,
                ),
                _step(
                    [_verifier_web_action()],
                    role=AgentRole.VERIFIER,
                    start=1.0,
                    end=2.0,
                ),
            ],
            [
                _obs(
                    Action(tool="map_query", params={"query": "x"}),
                    observation={
                        "status": "empty",
                        "error_message": None,
                        "formatted_address": None,
                        "resolved_latlng": None,
                        "place_type": None,
                    },
                    status="empty",
                ),
                _verifier_web_obs(_verifier_web_action()),
            ],
            AgentRole.VERIFIER,
            50.0,
            "i.jpg",
            fine_handoff=submit,
        ),
    }
    # 覆盖 verifier_output 为 fail（手动替换）
    parents[AgentRole.VERIFIER] = parents[AgentRole.VERIFIER].model_copy(
        update={
            "verifier_output": VerificationResult(
                verdict="fail",
                failed_checks=["address mismatch"],
                suggested_recheck="recheck map",
                return_to_agent=2,
            )
        }
    )

    fine_submit = Action(tool="submit_answer", params=submit.model_dump())
    all_steps = {
        AgentRole.COARSE: parents[AgentRole.COARSE] and [
            _step(
                [Action(tool="zoom_inspect", params={"bbox": [0, 0, 1, 1]})],
                role=AgentRole.COARSE,
            )
        ],
        AgentRole.FINE: [_step([fine_submit], role=AgentRole.FINE)],
        AgentRole.VERIFIER: [
            _step(
                [Action(tool="map_query", params={"query": "x"})],
                role=AgentRole.VERIFIER,
            ),
            _step(
                [_verifier_web_action()],
                role=AgentRole.VERIFIER,
                start=1.0,
                end=2.0,
            ),
        ],
    }
    all_obs = {
        AgentRole.COARSE: [
            _obs(
                Action(tool="zoom_inspect", params={"bbox": [0, 0, 1, 1]}),
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "x",
                },
            )
        ],
        AgentRole.FINE: [_obs(fine_submit)],
        AgentRole.VERIFIER: [
            _obs(
                Action(tool="map_query", params={"query": "x"}),
                observation={
                    "status": "empty",
                    "error_message": None,
                    "formatted_address": None,
                    "resolved_latlng": None,
                    "place_type": None,
                },
                status="empty",
            ),
            _verifier_web_obs(_verifier_web_action()),
        ],
    }

    revs = reconstruct_revision_trajectories(
        parents,
        VerificationResult(
            verdict="fail",
            failed_checks=["address mismatch"],
            suggested_recheck="recheck",
            return_to_agent=2,
        ),
        all_steps,
        all_obs,
        answer_timestamp=50.0,
        image_path="i.jpg",
        revision_round=1,
        max_revision_rounds=2,
    )
    assert len(revs) == 1
    rev = revs[0]
    assert rev.is_revision is True
    assert rev.agent_role == AgentRole.FINE
    assert rev.revision_source == RevisionSource.SYSTEM_FEEDBACK
    assert rev.revision_input is not None
    assert rev.revision_input.verdict == "fail"
    assert rev.parent_trajectory_id == parents[AgentRole.FINE].id


def test_revision_rejected_when_over_max_rounds(mock_llm: MagicMock) -> None:
    hyp = _hyp()
    submit = _submit()
    fine_action = Action(tool="submit_answer", params=submit.model_dump())
    parents = {
        AgentRole.FINE: reconstruct_single_trajectory(
            [_step([fine_action], role=AgentRole.FINE)],
            [_obs(fine_action)],
            AgentRole.FINE,
            10.0,
            "a.jpg",
            coarse_handoff=hyp,
        ),
        AgentRole.COARSE: reconstruct_single_trajectory(
            [
                _step(
                    [Action(tool="ocr", params={})],
                    role=AgentRole.COARSE,
                )
            ],
            [
                _obs(
                    Action(tool="ocr", params={}),
                    observation={
                        "status": "success",
                        "error_message": None,
                        "texts": ["x"],
                    },
                )
            ],
            AgentRole.COARSE,
            10.0,
            "a.jpg",
        ),
    }
    revs = reconstruct_revision_trajectories(
        parents,
        VerificationResult(
            verdict="fail",
            failed_checks=["x"],
            suggested_recheck="y",
            return_to_agent=2,
        ),
        {
            AgentRole.FINE: [_step([fine_action], role=AgentRole.FINE)],
            AgentRole.COARSE: [
                _step([Action(tool="ocr", params={})], role=AgentRole.COARSE)
            ],
            AgentRole.VERIFIER: [],
        },
        {
            AgentRole.FINE: [_obs(fine_action)],
            AgentRole.COARSE: [
                _obs(
                    Action(tool="ocr", params={}),
                    observation={
                        "status": "success",
                        "error_message": None,
                        "texts": ["x"],
                    },
                )
            ],
            AgentRole.VERIFIER: [],
        },
        answer_timestamp=10.0,
        image_path="a.jpg",
        revision_round=3,
        max_revision_rounds=2,
    )
    assert revs == []


def test_video_observed_revision(mock_llm: MagicMock) -> None:
    hyp = _hyp()
    submit = _submit()
    fine_action = Action(tool="submit_answer", params=submit.model_dump())
    fine_step = _step(
        [fine_action],
        role=AgentRole.FINE,
        start=20.0,
        end=30.0,
        thought="纠错后提交。",
    )
    parents = {
        AgentRole.COARSE: reconstruct_single_trajectory(
            [
                _step(
                    [Action(tool="ocr", params={})],
                    role=AgentRole.COARSE,
                    start=0.0,
                    end=5.0,
                )
            ],
            [
                _obs(
                    Action(tool="ocr", params={}),
                    observation={
                        "status": "success",
                        "error_message": None,
                        "texts": ["x"],
                    },
                )
            ],
            AgentRole.COARSE,
            40.0,
            "v.jpg",
        ),
        AgentRole.FINE: reconstruct_single_trajectory(
            [fine_step],
            [_obs(fine_action)],
            AgentRole.FINE,
            40.0,
            "v.jpg",
            coarse_handoff=hyp,
        ),
    }
    # 修正 coarse_output 以便 handoff
    parents[AgentRole.COARSE] = parents[AgentRole.COARSE].model_copy(
        update={"coarse_output": hyp}
    )

    revs = reconstruct_revision_trajectories(
        parents,
        VerificationResult(
            verdict="pass",
            failed_checks=[],
            suggested_recheck="none",
            return_to_agent=None,
        ),
        {
            AgentRole.FINE: [fine_step],
            AgentRole.COARSE: [
                _step(
                    [Action(tool="ocr", params={})],
                    role=AgentRole.COARSE,
                    start=0.0,
                    end=5.0,
                )
            ],
            AgentRole.VERIFIER: [],
        },
        {
            AgentRole.FINE: [_obs(fine_action)],
            AgentRole.COARSE: [
                _obs(
                    Action(tool="ocr", params={}),
                    observation={
                        "status": "success",
                        "error_message": None,
                        "texts": ["x"],
                    },
                )
            ],
            AgentRole.VERIFIER: [],
        },
        answer_timestamp=40.0,
        image_path="v.jpg",
        revision_round=1,
        max_revision_rounds=2,
        video_revision_segments=[(18.0, 32.0)],
    )
    assert len(revs) == 1
    assert revs[0].revision_source == RevisionSource.VIDEO_OBSERVED
    assert revs[0].revision_input is None
    assert revs[0].is_revision is True


def test_video_observed_revision_falls_back_without_overlap(
    mock_llm: MagicMock,
) -> None:
    """revision 时段与 Move 无重叠时，回退全量可展开步，避免空跑。"""
    hyp = _hyp()
    submit = _submit()
    fine_action = Action(tool="submit_answer", params=submit.model_dump())
    fine_step = _step(
        [fine_action],
        role=AgentRole.FINE,
        start=20.0,
        end=30.0,
        thought="提交。",
    )
    parents = {
        AgentRole.COARSE: reconstruct_single_trajectory(
            [
                _step(
                    [Action(tool="ocr", params={})],
                    role=AgentRole.COARSE,
                    start=0.0,
                    end=5.0,
                )
            ],
            [
                _obs(
                    Action(tool="ocr", params={}),
                    observation={
                        "status": "success",
                        "error_message": None,
                        "texts": ["x"],
                    },
                )
            ],
            AgentRole.COARSE,
            40.0,
            "v.jpg",
        ),
        AgentRole.FINE: reconstruct_single_trajectory(
            [fine_step],
            [_obs(fine_action)],
            AgentRole.FINE,
            40.0,
            "v.jpg",
            coarse_handoff=hyp,
        ),
    }
    parents[AgentRole.COARSE] = parents[AgentRole.COARSE].model_copy(
        update={"coarse_output": hyp}
    )

    revs = reconstruct_revision_trajectories(
        parents,
        VerificationResult(
            verdict="pass",
            failed_checks=[],
            suggested_recheck="none",
            return_to_agent=None,
        ),
        {
            AgentRole.FINE: [fine_step],
            AgentRole.COARSE: [
                _step(
                    [Action(tool="ocr", params={})],
                    role=AgentRole.COARSE,
                    start=0.0,
                    end=5.0,
                )
            ],
            AgentRole.VERIFIER: [],
        },
        {
            AgentRole.FINE: [_obs(fine_action)],
            AgentRole.COARSE: [
                _obs(
                    Action(tool="ocr", params={}),
                    observation={
                        "status": "success",
                        "error_message": None,
                        "texts": ["x"],
                    },
                )
            ],
            AgentRole.VERIFIER: [],
        },
        answer_timestamp=40.0,
        image_path="v.jpg",
        revision_round=1,
        max_revision_rounds=2,
        # 与 FINE Move(20-30) 无重叠
        video_revision_segments=[(90.0, 100.0)],
    )
    assert len(revs) == 1
    assert revs[0].revision_source == RevisionSource.VIDEO_OBSERVED
    assert revs[0].agent_role == AgentRole.FINE
    assert revs[0].is_revision is True


def test_thin_verifier_augmented_with_web_search(
    mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """视频侧仅一步 map_query 时，补齐 web_search 验证深度。"""
    calls: list[str] = []

    def _fake_execute(
        action: Action, image_path: str, agent_role: AgentRole, **kwargs: object
    ) -> ObservationExecutionResult:
        _ = image_path, agent_role, kwargs
        calls.append(action.tool)
        assert action.tool == "web_search"
        return _verifier_web_obs(action)

    monkeypatch.setattr(
        "pipeline.stage5_reconstruct.execute_action",
        _fake_execute,
    )
    map_action = Action(tool="map_query", params={"latlng": [48.8584, 2.2945]})
    traj = reconstruct_single_trajectory(
        [_step([map_action], role=AgentRole.VERIFIER)],
        [_verifier_map_obs(map_action)],
        AgentRole.VERIFIER,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        fine_handoff=_submit(),
    )
    assert calls == ["web_search"]
    assert len(traj.steps) == 2
    assert traj.steps[0].action.tool == "map_query"
    assert traj.steps[1].action.tool == "web_search"


def test_signature_has_no_groundtruth_param() -> None:
    import inspect

    for fn in (
        reconstruct_single_trajectory,
        reconstruct_all_trajectories,
        reconstruct_revision_trajectories,
    ):
        params = inspect.signature(fn).parameters
        assert "groundtruth" not in params
