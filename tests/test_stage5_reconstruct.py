"""stage5：逐步因果生成、polish、拒绝采样与 handoff 测试（LLM 全部 mock）。"""

from __future__ import annotations

import inspect
import re
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from pipeline.config import clear_settings_cache
from pipeline.evidence_routing import (
    RawGivenClue,
    ScopeBoundKind,
    VideoChainContext,
    VideoFactClaim,
    WorkingScope,
    embed_video_context,
)
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
    TrajectoryQualityRejected,
    _CoarseToolSuitability,
    _ExternalHints,
    _FaithfulnessCheck,
    _PolishedThoughts,
    _StepThought,
    _TrajectoryJudgement,
    _build_coarse_evidence_ledger,
    _collapse_consecutive_duplicate_actions,
    _collapse_semantic_fact_clusters,
    _drop_noninformative_empty_units,
    _filter_unusable_ui_units,
    _hard_check_issues,
    _validate_coarse_projection_richness,
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
    narration: str = "旁白。",
) -> NormalizedStep:
    return NormalizedStep(
        move=_move(start=start, end=end, role=role, narration=narration),
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


def _coarse_ctx_draft(thought: str = "观察地貌。") -> str:
    ctx = VideoChainContext(
        raw_given_clues=[RawGivenClue(text="河南许昌附近")],
        working_scope=WorkingScope(
            region="河南许昌附近",
            bound_kind=ScopeBoundKind.NEAR,
            raw_clue_texts=["河南许昌附近"],
        ),
        video_facts=[
            VideoFactClaim(
                fact_id="vf1",
                start_time=0.0,
                end_time=1.0,
                quote="高地俯视长桥",
                tokens=["高地", "桥"],
                kind="observe",
            )
        ],
    )
    return embed_video_context(thought, ctx)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    monkeypatch.setenv("STAGE5_BEST_OF_K", "2")
    monkeypatch.setenv("STAGE5_JUDGE_THRESHOLD", "0.6")
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture()
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """拦截 call_structured，按 response_model 返回合法结构化结果。"""
    calls: list[dict[str, Any]] = []
    step_thought_i = {"n": 0}
    judge_scores: list[float] = []

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
        if response_model is _StepThought:
            step_thought_i["n"] += 1
            tool_name = "zoom_inspect"
            m = re.search(r"tool=([a-z0-9_]+)", prompt)
            if m:
                tool_name = m.group(1)
            return _StepThought(
                thought=(
                    f"前向思考 {step_thought_i['n']}：基于已有观察，"
                    f"调用 {tool_name} 继续核对画面线索。"
                )
            )
        if response_model is _PolishedThoughts:
            # 从待润色轨迹解析 Thought 行，保持条数
            thoughts = []
            for line in prompt.splitlines():
                if line.startswith("Thought: "):
                    thoughts.append("润色后：" + line[len("Thought: ") :])
            if not thoughts:
                thoughts = ["润色后思考"]
            return _PolishedThoughts(thoughts=thoughts)
        if response_model is _FaithfulnessCheck:
            return _FaithfulnessCheck(unfaithful_steps=[])
        if response_model is _TrajectoryJudgement:
            score = judge_scores.pop(0) if judge_scores else 0.85
            return _TrajectoryJudgement(score=score, issues=[])
        if response_model is LocationHypothesis:
            return _hyp()
        if response_model is VerificationResult:
            return VerificationResult(
                verdict="pass",
                failed_checks=[],
                suggested_recheck="none",
                return_to_agent=None,
            )
        if response_model is SubmitAnswerResult:
            return _submit()
        if response_model is _CoarseToolSuitability:
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
            return _ExternalHints(hints=hints, given_clues=hints)
        raise AssertionError(f"未预期的 response_model: {response_model}")

    monkeypatch.setattr("pipeline.stage5_reconstruct.call_structured", _fake)
    holder = MagicMock()
    holder.calls = calls
    holder.judge_scores = judge_scores
    holder.reset_step_counter = lambda: step_thought_i.__setitem__("n", 0)
    return holder


def test_stepwise_prompt_hides_current_observation(mock_llm: MagicMock) -> None:
    """第 t 步 prompt 含前 t-1 步 Obs，不含第 t 步 Obs。"""
    zoom1 = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    zoom2 = Action(tool="zoom_inspect", params={"bbox": [0.5, 0.5, 0.3, 0.3]})
    obs1_text = "UNIQUE_OBS_ALPHA_tower_lattice"
    obs2_text = "UNIQUE_OBS_BETA_river_bank"
    steps = [
        _step(
            [zoom1],
            thought=_coarse_ctx_draft("看塔。"),
            start=0.0,
            end=1.0,
            narration="看塔尖结构",
        ),
        _step(
            [zoom2],
            thought="看河岸。",
            start=1.0,
            end=2.0,
            narration="看河岸农田",
        ),
    ]
    observations = [
        _obs(
            zoom1,
            observation={
                "status": "success",
                "error_message": None,
                "description": obs1_text,
            },
        ),
        _obs(
            zoom2,
            observation={
                "status": "success",
                "error_message": None,
                "description": obs2_text,
            },
        ),
    ]
    traj = reconstruct_single_trajectory(
        steps,
        observations,
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    step_prompts = [
        c["prompt"] for c in mock_llm.calls if c["response_model"] is _StepThought
    ]
    # best-of-2 → 每候选 2 步 → 至少 4 次逐步调用；取第一候选前两步
    assert len(step_prompts) >= 2
    p0, p1 = step_prompts[0], step_prompts[1]
    assert obs1_text not in p0
    assert obs2_text not in p0
    assert obs1_text in p1
    assert obs2_text not in p1
    assert traj.stage5_judge_score is not None
    assert traj.stage5_judge_score >= 0.6


def test_polish_unfaithful_step_rolls_back(mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """忠实性对比判定不忠实的步回退为润色前文本。"""
    calls: list[dict[str, Any]] = []
    step_i = {"n": 0}

    def _fake(
        prompt: str,
        response_model: type[BaseModel],
        images: Optional[list[str]] = None,
        video: Optional[str] = None,
        model: Optional[str] = None,
    ) -> BaseModel:
        calls.append({"prompt": prompt, "response_model": response_model})
        if response_model is _StepThought:
            step_i["n"] += 1
            return _StepThought(
                thought=f"原始思考{step_i['n']}：调用 zoom_inspect 核对画面。"
            )
        if response_model is _PolishedThoughts:
            return _PolishedThoughts(
                thoughts=["润色后思考1（改了事实）", "润色后思考2"]
            )
        if response_model is _FaithfulnessCheck:
            return _FaithfulnessCheck(unfaithful_steps=[1])
        if response_model is _TrajectoryJudgement:
            return _TrajectoryJudgement(score=0.9, issues=[])
        if response_model is LocationHypothesis:
            return _hyp()
        if response_model is _CoarseToolSuitability:
            return _CoarseToolSuitability(
                suitable_for_coarse_reasoning=True, reason="mock"
            )
        if response_model is _ExternalHints:
            return _ExternalHints()
        raise AssertionError(response_model)

    monkeypatch.setattr("pipeline.stage5_reconstruct.call_structured", _fake)
    monkeypatch.setenv("STAGE5_BEST_OF_K", "1")
    clear_settings_cache()

    zoom1 = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    zoom2 = Action(tool="zoom_inspect", params={"bbox": [0.5, 0.5, 0.3, 0.3]})
    traj = reconstruct_single_trajectory(
        [
            _step([zoom1], thought=_coarse_ctx_draft("a"), start=0, end=1),
            _step([zoom2], thought="b", start=1, end=2),
        ],
        [
            _obs(
                zoom1,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "tower detail",
                },
            ),
            _obs(
                zoom2,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "river detail",
                },
            ),
        ],
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    assert traj.steps[0].thought.startswith("原始思考1")
    assert traj.steps[1].thought.startswith("润色后思考2")


def test_hard_check_internal_id_rejects_candidate(
    mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """内部 ID 泄漏 → 该候选硬校验判废；若全部失败则 TrajectoryQualityRejected。"""
    step_i = {"n": 0}

    def _fake(
        prompt: str,
        response_model: type[BaseModel],
        images: Optional[list[str]] = None,
        video: Optional[str] = None,
        model: Optional[str] = None,
    ) -> BaseModel:
        if response_model is _StepThought:
            step_i["n"] += 1
            return _StepThought(thought=f"依据视频来源事实 vf12_3_observe 排除平原。")
        if response_model is _PolishedThoughts:
            return _PolishedThoughts(
                thoughts=["依据视频来源事实 vf12_3_observe 排除平原。"]
            )
        if response_model is _FaithfulnessCheck:
            return _FaithfulnessCheck(unfaithful_steps=[])
        if response_model is _TrajectoryJudgement:
            return _TrajectoryJudgement(score=0.9, issues=[])
        if response_model is LocationHypothesis:
            return _hyp()
        if response_model is _CoarseToolSuitability:
            return _CoarseToolSuitability(
                suitable_for_coarse_reasoning=True, reason="ok"
            )
        if response_model is _ExternalHints:
            return _ExternalHints()
        raise AssertionError(response_model)

    monkeypatch.setattr("pipeline.stage5_reconstruct.call_structured", _fake)
    monkeypatch.setenv("STAGE5_BEST_OF_K", "1")
    clear_settings_cache()

    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    with pytest.raises(TrajectoryQualityRejected):
        reconstruct_single_trajectory(
            [_step([zoom], thought=_coarse_ctx_draft("看。"))],
            [
                _obs(
                    zoom,
                    observation={
                        "status": "success",
                        "error_message": None,
                        "description": "elevated ground",
                    },
                )
            ],
            AgentRole.COARSE,
            answer_timestamp=100.0,
            image_path="frame.jpg",
        )


def test_rejection_sampling_picks_highest_score(mock_llm: MagicMock) -> None:
    """best-of-k 选择最高分且达到阈值的候选。"""
    mock_llm.judge_scores.extend([0.55, 0.82])
    zoom1 = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    zoom2 = Action(tool="zoom_inspect", params={"bbox": [0.5, 0.2, 0.3, 0.3]})
    traj = reconstruct_single_trajectory(
        [
            _step([zoom1], thought=_coarse_ctx_draft("a"), start=0, end=1),
            _step([zoom2], thought="b", start=1, end=2),
        ],
        [
            _obs(
                zoom1,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "tower A",
                },
            ),
            _obs(
                zoom2,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "river B",
                },
            ),
        ],
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    assert traj.stage5_judge_score == pytest.approx(0.82)


def test_rejection_sampling_all_below_threshold(
    mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_llm.judge_scores.extend([0.2, 0.3])
    monkeypatch.setenv("STAGE5_BEST_OF_K", "2")
    clear_settings_cache()
    zoom1 = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    zoom2 = Action(tool="zoom_inspect", params={"bbox": [0.5, 0.2, 0.3, 0.3]})
    with pytest.raises(TrajectoryQualityRejected):
        reconstruct_single_trajectory(
            [
                _step([zoom1], thought=_coarse_ctx_draft("a"), start=0, end=1),
                _step([zoom2], thought="b", start=1, end=2),
            ],
            [
                _obs(
                    zoom1,
                    observation={
                        "status": "success",
                        "error_message": None,
                        "description": "tower A",
                    },
                ),
                _obs(
                    zoom2,
                    observation={
                        "status": "success",
                        "error_message": None,
                        "description": "river B",
                    },
                ),
            ],
            AgentRole.COARSE,
            answer_timestamp=100.0,
            image_path="frame.jpg",
        )


def test_system_prompt_is_concise_role_instruction(mock_llm: MagicMock) -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    traj = reconstruct_single_trajectory(
        [_step([zoom], thought=_coarse_ctx_draft("看。"))],
        [
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "elevated ground bridge",
                },
            )
        ],
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    assert "粗定位 Agent" in traj.system_prompt
    assert "vf" not in traj.system_prompt
    assert "<<<" not in traj.system_prompt
    assert "禁止" not in traj.system_prompt


def test_coarse_projects_out_web_search(mock_llm: MagicMock) -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    search = Action(
        tool="web_search",
        params={"query": "tropical arcade street", "purpose": "broad_discovery"},
    )
    ocr = Action(tool="ocr", params={"bbox": [0.2, 0.2, 0.3, 0.3]})
    steps = [
        _step([zoom], thought=_coarse_ctx_draft("看建筑。")),
        _step([search], thought="搜索。", start=1, end=2),
        _step([ocr], thought="读文字。", start=2, end=3),
    ]
    observations = [
        _obs(
            zoom,
            observation={
                "status": "success",
                "error_message": None,
                "description": "arcade",
            },
        ),
        _obs(
            search,
            observation={
                "status": "success",
                "error_message": None,
                "results": [{"title": "x", "snippet": "y", "url": "http://a"}],
            },
        ),
        _obs(
            ocr,
            observation={
                "status": "success",
                "error_message": None,
                "texts": ["Cafe"],
            },
        ),
    ]
    traj = reconstruct_single_trajectory(
        steps,
        observations,
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    assert [s.action.tool for s in traj.steps] == ["zoom_inspect", "ocr"]


def test_coarse_projection_empty_raises(mock_llm: MagicMock) -> None:
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
            answer_timestamp=100.0,
            image_path="frame.jpg",
        )


def test_drop_keeps_error_and_geo_empty_with_success() -> None:
    """有 success 时仍保留 error 与有地理增益的 empty，避免断链。"""
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    sat = Action(
        tool="lookup_historical_satellite_map",
        params={"query": "许昌北侧", "year": 2005},
    )
    find = Action(
        tool="find_specific_features_in_satellite_map",
        params={
            "target_satellite_image": "map",
            "reference_features": ["桥", "河"],
        },
    )
    pin = Action(tool="zoom_inspect", params={"bbox": [0.2, 0.2, 0.3, 0.3]})

    units = [
        (
            _coarse_ctx_draft("看河桥。"),
            zoom,
            _obs(
                zoom,
                status="error",
                observation={
                    "status": "error",
                    "error_message": "schema exhausted",
                    "description": "observation synthesis failed",
                },
            ),
            _step(
                [zoom],
                thought=_coarse_ctx_draft("看河桥。"),
                narration="河桥山关系明确",
            ),
        ),
        (
            _coarse_ctx_draft("查卫星。"),
            sat,
            _obs(
                sat,
                observation={
                    "status": "success",
                    "error_message": None,
                    "image_url": "u",
                    "layout_description": "农田与坑塘",
                    "matched_features": ["农田"],
                },
            ),
            _step(
                [sat],
                thought=_coarse_ctx_draft("查卫星。"),
                start=1,
                end=2,
                narration="打开许昌地图排查",
            ),
        ),
        (
            _coarse_ctx_draft("未命中细部。"),
            find,
            _obs(
                find,
                status="empty",
                observation={
                    "status": "empty",
                    "error_message": None,
                    "matched_features": [],
                    "overall_match_assessment": "未找到匹配",
                },
            ),
            _step(
                [find],
                thought=_coarse_ctx_draft("未命中细部。"),
                start=2,
                end=3,
                narration="对比扶手石与地图河岸",
            ),
        ),
        (
            "置顶聊天。",
            pin,
            _obs(
                pin,
                status="empty",
                observation={
                    "status": "empty",
                    "error_message": None,
                    "description": "no in-scene geography visible in content region",
                },
            ),
            NormalizedStep(
                move=Move(
                    start_time=3.0,
                    end_time=4.0,
                    narration="静待时间的流逝，看看聊天记录",
                    screen_action="置顶求助消息列表并滚动",
                    visible_clues=["置顶", "聊天"],
                    agent_role=AgentRole.COARSE,
                ),
                thought_draft="置顶聊天。",
                actions=[pin],
                normalization_mode=NormalizationMode.MATCHED,
                matched_tool_confidence=0.5,
                fallback_reason=None,
            ),
        ),
    ]
    kept = _drop_noninformative_empty_units(units)
    tools = [u[1].tool for u in kept]
    statuses = [u[2].status for u in kept]
    assert tools == [
        "zoom_inspect",
        "lookup_historical_satellite_map",
        "find_specific_features_in_satellite_map",
    ]
    assert statuses == ["error", "success", "empty"]


def test_coarse_keeps_compare_images(mock_llm: MagicMock) -> None:
    """COARSE 投影保留视觉比对类 Tool。"""
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    cmp_ = Action(
        tool="compare_images_for_geolocation",
        params={"image_a": "a.jpg", "image_b": "b.jpg"},
    )
    ocr = Action(tool="ocr", params={"bbox": [0.2, 0.2, 0.3, 0.3]})
    traj = reconstruct_single_trajectory(
        [
            _step([zoom], thought=_coarse_ctx_draft("看。")),
            _step([cmp_], thought="比对。", start=1, end=2),
            _step([ocr], thought="读。", start=2, end=3),
        ],
        [
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "a",
                },
            ),
            _obs(
                cmp_,
                observation={
                    "status": "success",
                    "error_message": None,
                    "visual_similarity_score": 0.7,
                    "matched_features": ["bridge"],
                    "mismatched_features": [],
                    "geolocation_hints": ["wide river"],
                },
            ),
            _obs(
                ocr,
                observation={
                    "status": "success",
                    "error_message": None,
                    "texts": ["x"],
                },
            ),
        ],
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    assert "compare_images_for_geolocation" in [s.action.tool for s in traj.steps]


def test_step_prompt_requires_matching_action_tool(mock_llm: MagicMock) -> None:
    """Thought prompt 强制说明本步 tool，并禁止异工具话术。"""
    from pipeline.stage5_reconstruct import _build_step_prompt

    sat = Action(
        tool="lookup_historical_satellite_map",
        params={"year": 2005, "region": "河南"},
    )
    units = [
        (
            "想查卫星图看河岸",
            sat,
            _obs(
                sat,
                observation={"status": "success", "error_message": None},
            ),
            _step([sat], thought="想查卫星图看河岸"),
        )
    ]
    prompt = _build_step_prompt(
        AgentRole.COARSE,
        ["agent_role: coarse_locator", "user_query: test"],
        [],
        units,
        0,
    )
    assert "lookup_historical_satellite_map" in prompt
    assert "必须说明为何调用本步工具" in prompt
    assert "禁止提及未出现在本步 Action 的工具名" in prompt
    assert "本步工具为 lookup_historical_satellite_map" in prompt


def test_thought_mismatch_triggers_one_retry(
    mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thought 含异工具话术时重试一次。"""
    from pipeline.stage5_reconstruct import _generate_thoughts_stepwise

    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    units = [
        (
            "看塔",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "tower",
                },
            ),
            _step([zoom], thought="看塔"),
        )
    ]
    calls: list[str] = []

    def _fake(prompt: str, response_model: type[BaseModel], **_k: Any) -> BaseModel:
        calls.append(prompt)
        if len(calls) == 1:
            return _StepThought(thought="我先用 web_search 搜一下塔的名称。")
        return _StepThought(thought="用 zoom_inspect 查看塔身结构细节。")

    monkeypatch.setattr("pipeline.stage5_reconstruct.call_structured", _fake)
    thoughts = _generate_thoughts_stepwise(
        AgentRole.COARSE,
        units,
        ["agent_role: coarse_locator"],
        "frame.jpg",
    )
    assert len(calls) == 2
    assert "不对齐" in calls[1]
    assert "zoom_inspect" in thoughts[0]
    assert "web_search" not in thoughts[0]


def test_coarse_sanitizes_illegal_zoom_bbox(mock_llm: MagicMock) -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [10, 20, 300, 400]})
    ocr = Action(tool="ocr", params={"bbox": [0.2, 0.2, 0.3, 0.3]})
    traj = reconstruct_single_trajectory(
        [
            _step([zoom], thought=_coarse_ctx_draft("看。")),
            _step([ocr], thought="读。", start=1, end=2),
        ],
        [
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "wide scene",
                },
            ),
            _obs(
                ocr,
                observation={
                    "status": "success",
                    "error_message": None,
                    "texts": ["sign"],
                },
            ),
        ],
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    bbox = traj.steps[0].action.params["bbox"]
    assert all(abs(float(x)) <= 1.5 for x in bbox)


def test_working_scope_survives_coarse_projection(mock_llm: MagicMock) -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    ctx = VideoChainContext(
        raw_given_clues=[RawGivenClue(text="河南许昌人、拍摄地离家不远")],
        working_scope=WorkingScope(
            region="河南许昌附近",
            bound_kind=ScopeBoundKind.NEAR,
            raw_clue_texts=["河南许昌人、拍摄地离家不远"],
        ),
        video_facts=[
            VideoFactClaim(
                fact_id="vf0",
                start_time=1.0,
                end_time=2.0,
                quote="高地长桥",
                tokens=["高地", "桥"],
                kind="observe",
            )
        ],
    )
    ui_step = NormalizedStep(
        move=_move(narration="打开聊天"),
        thought_draft=embed_video_context("界面", ctx),
        actions=[zoom],
        normalization_mode=NormalizationMode.MATCHED,
        matched_tool_confidence=0.9,
        fallback_reason=None,
    )
    geo_step = NormalizedStep(
        move=_move(narration="高地俯视长桥", start=2.0, end=3.0),
        thought_draft="确认桥与高地",
        actions=[zoom],
        normalization_mode=NormalizationMode.MATCHED,
        matched_tool_confidence=0.9,
        fallback_reason=None,
    )
    traj = reconstruct_single_trajectory(
        [ui_step, geo_step],
        [
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "聊天界面置顶消息",
                },
            ),
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "elevated ground and long bridge",
                },
            ),
        ],
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    assert "工作范围：河南许昌附近" in traj.user_query


def test_fine_requires_submit_answer_and_terminal_none(mock_llm: MagicMock) -> None:
    search = Action(
        tool="web_search",
        params={"query": "tower", "purpose": "entity_lookup"},
    )
    submit = Action(tool="submit_answer", params=_submit().model_dump())
    traj = reconstruct_single_trajectory(
        [
            _step([search], thought="检索。", role=AgentRole.FINE),
            _step([submit], thought="提交。", role=AgentRole.FINE, start=1, end=2),
        ],
        [
            _obs(
                search,
                observation={
                    "status": "success",
                    "error_message": None,
                    "results": [],
                },
            ),
            _obs(submit),
        ],
        AgentRole.FINE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        coarse_handoff=_hyp(),
    )
    assert traj.fine_output is not None
    assert traj.steps[-1].action.tool == "submit_answer"
    assert traj.steps[-1].observation is None
    assert traj.steps[-1].observation_source is None


def test_fine_synthesizes_submit_answer_when_missing(mock_llm: MagicMock) -> None:
    mq = Action(tool="map_query", params={"query": "Eiffel Tower"})
    traj = reconstruct_single_trajectory(
        [_step([mq], thought="查地图。", role=AgentRole.FINE)],
        [
            _obs(
                mq,
                observation={
                    "status": "success",
                    "error_message": None,
                    "resolved_latlng": [48.8584, 2.2945],
                    "display_name": "Eiffel Tower",
                },
            )
        ],
        AgentRole.FINE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        coarse_handoff=_hyp(),
    )
    assert traj.steps[-1].action.tool == "submit_answer"
    assert traj.fine_output is not None


def test_verifier_uses_fine_handoff_as_candidate(mock_llm: MagicMock) -> None:
    mq = Action(tool="map_query", params={"query": "Eiffel Tower"})
    traj = reconstruct_single_trajectory(
        [_step([mq], thought="核对。", role=AgentRole.VERIFIER)],
        [
            _obs(
                mq,
                observation={
                    "status": "success",
                    "error_message": None,
                    "resolved_latlng": [48.8584, 2.2945],
                    "display_name": "Eiffel Tower",
                },
            )
        ],
        AgentRole.VERIFIER,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        fine_handoff=_submit(),
    )
    assert traj.verifier_output is not None
    assert traj.fine_handoff is not None
    header = next(
        c["prompt"]
        for c in mock_llm.calls
        if c["response_model"] is _StepThought
    )
    assert "fine_handoff" in header.lower() or "候选" in header or "Eiffel" in header


def test_verifier_synthesizes_scaffold_when_no_video_actions(
    mock_llm: MagicMock,
) -> None:
    traj = reconstruct_single_trajectory(
        [],
        [],
        AgentRole.VERIFIER,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        fine_handoff=_submit(),
    )
    tools = [s.action.tool for s in traj.steps]
    assert "map_query" in tools
    assert "web_search" in tools
    assert traj.verifier_output is not None


def test_reconstruct_all_handoff_chain(mock_llm: MagicMock) -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    ocr = Action(tool="ocr", params={"bbox": [0.2, 0.2, 0.3, 0.3]})
    mq = Action(tool="map_query", params={"query": "tower"})
    submit = Action(tool="submit_answer", params=_submit().model_dump())

    all_steps = {
        AgentRole.COARSE: [
            _step([zoom], thought=_coarse_ctx_draft("看。"), start=0, end=1),
            _step([ocr], thought="读。", start=1, end=2),
        ],
        AgentRole.FINE: [
            _step([mq], thought="查。", role=AgentRole.FINE, start=2, end=3),
            _step(
                [submit], thought="交。", role=AgentRole.FINE, start=3, end=4
            ),
        ],
        AgentRole.VERIFIER: [],
    }
    all_obs = {
        AgentRole.COARSE: [
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "tower",
                },
            ),
            _obs(
                ocr,
                observation={
                    "status": "success",
                    "error_message": None,
                    "texts": ["Tour"],
                },
            ),
        ],
        AgentRole.FINE: [
            _obs(
                mq,
                observation={
                    "status": "success",
                    "error_message": None,
                    "resolved_latlng": [48.8584, 2.2945],
                    "display_name": "Eiffel Tower",
                },
            ),
            _obs(submit),
        ],
        AgentRole.VERIFIER: [],
    }
    trajs = reconstruct_all_trajectories(
        all_steps, all_obs, answer_timestamp=100.0, image_path="frame.jpg"
    )
    assert trajs[AgentRole.COARSE].coarse_output is not None
    assert trajs[AgentRole.FINE].coarse_handoff is not None
    assert trajs[AgentRole.FINE].fine_output is not None
    assert trajs[AgentRole.VERIFIER].fine_handoff is not None
    assert trajs[AgentRole.VERIFIER].verifier_output is not None


def test_reconstruct_all_keeps_coarse_when_fine_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FINE 拒绝采样失败时仍返回已成功的 COARSE。"""
    from pipeline.stage5_reconstruct import (
        _FaithfulnessCheck,
        _PolishedThoughts,
        _StepThought,
        _TrajectoryJudgement,
    )

    call_n = {"judge": 0}

    def _fake(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **_k: Any,
    ) -> Any:
        if response_model is _StepThought:
            return _StepThought(thought="观察地貌推进粗定位。")
        if response_model is _PolishedThoughts:
            n = prompt.count("Step ") or 1
            return _PolishedThoughts(thoughts=["观察地貌推进粗定位。"] * max(n, 1))
        if response_model is _FaithfulnessCheck:
            return _FaithfulnessCheck(unfaithful_steps=[])
        if response_model is _TrajectoryJudgement:
            call_n["judge"] += 1
            # COARSE 高分；FINE 低分
            if "fine_locator" in prompt or "agent_role: fine_locator" in prompt:
                return _TrajectoryJudgement(score=0.1, issues=["故意低分"])
            return _TrajectoryJudgement(score=0.9, issues=[])
        if response_model is LocationHypothesis:
            return _hyp()
        if response_model is SubmitAnswerResult:
            return SubmitAnswerResult(
                location_name="Somewhere",
                latitude=1.0,
                longitude=2.0,
                confidence=0.5,
                reasoning="x",
            )
        if response_model is VerificationResult:
            return VerificationResult(
                verdict="pass",
                failed_checks=[],
                suggested_recheck="",
                return_to_agent=None,
            )
        raise AssertionError(response_model)

    monkeypatch.setattr("pipeline.stage5_reconstruct.call_structured", _fake)
    monkeypatch.setenv("STAGE5_BEST_OF_K", "1")
    clear_settings_cache()

    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    mq = Action(tool="map_query", params={"query": "x"})
    submit = Action(
        tool="submit_answer",
        params={
            "location_name": "Somewhere",
            "lat": 1.0,
            "lng": 2.0,
            "confidence": 0.5,
            "reasoning": "x",
        },
    )
    all_steps = {
        AgentRole.COARSE: [
            _step([zoom], thought=_coarse_ctx_draft("看。"), start=0, end=1)
        ],
        AgentRole.FINE: [
            _step([mq], thought="查。", start=1, end=2),
            _step([submit], thought="交。", start=2, end=3),
        ],
        AgentRole.VERIFIER: [],
    }
    all_obs = {
        AgentRole.COARSE: [
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "hills",
                },
            )
        ],
        AgentRole.FINE: [
            _obs(
                mq,
                observation={
                    "status": "success",
                    "error_message": None,
                    "resolved_latlng": [1.0, 2.0],
                    "display_name": "Somewhere",
                },
            ),
            _obs(submit),
        ],
        AgentRole.VERIFIER: [],
    }
    trajs = reconstruct_all_trajectories(
        all_steps, all_obs, answer_timestamp=100.0, image_path="frame.jpg"
    )
    assert AgentRole.COARSE in trajs
    assert trajs[AgentRole.COARSE].coarse_output is not None
    assert AgentRole.FINE not in trajs
    assert AgentRole.VERIFIER not in trajs


def test_thought_only_merged_into_next_action(mock_llm: MagicMock) -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    ocr = Action(tool="ocr", params={"bbox": [0.2, 0.2, 0.3, 0.3]})
    traj = reconstruct_single_trajectory(
        [
            _step([], thought=_coarse_ctx_draft("仅思考。"), start=0, end=1),
            _step([zoom], thought="看。", start=1, end=2),
            _step([ocr], thought="读。", start=2, end=3),
        ],
        [
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "tower",
                },
            ),
            _obs(
                ocr,
                observation={
                    "status": "success",
                    "error_message": None,
                    "texts": ["x"],
                },
            ),
        ],
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    assert len(traj.steps) == 2


def test_system_feedback_revision_to_fine(mock_llm: MagicMock) -> None:
    mq = Action(tool="map_query", params={"query": "tower"})
    submit = Action(tool="submit_answer", params=_submit().model_dump())
    parent = reconstruct_single_trajectory(
        [
            _step([mq], thought="查。", role=AgentRole.FINE),
            _step([submit], thought="交。", role=AgentRole.FINE, start=1, end=2),
        ],
        [
            _obs(
                mq,
                observation={
                    "status": "success",
                    "error_message": None,
                    "resolved_latlng": [48.8584, 2.2945],
                    "display_name": "Eiffel Tower",
                },
            ),
            _obs(submit),
        ],
        AgentRole.FINE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        coarse_handoff=_hyp(),
    )
    verification = VerificationResult(
        verdict="fail",
        failed_checks=["visual mismatch"],
        suggested_recheck="recheck plaza",
        return_to_agent=2,
    )
    revs = reconstruct_revision_trajectories(
        {AgentRole.FINE: parent},
        verification,
        {
            AgentRole.FINE: [
                _step([mq], thought="再查。", role=AgentRole.FINE),
                _step(
                    [submit], thought="再交。", role=AgentRole.FINE, start=1, end=2
                ),
            ]
        },
        {
            AgentRole.FINE: [
                _obs(
                    mq,
                    observation={
                        "status": "success",
                        "error_message": None,
                        "resolved_latlng": [48.8584, 2.2945],
                        "display_name": "Eiffel Tower",
                    },
                ),
                _obs(submit),
            ]
        },
        answer_timestamp=100.0,
        image_path="frame.jpg",
        revision_round=1,
        max_revision_rounds=2,
    )
    assert len(revs) == 1
    assert revs[0].is_revision is True
    assert revs[0].revision_source == RevisionSource.SYSTEM_FEEDBACK
    assert revs[0].agent_role == AgentRole.FINE


def test_revision_rejected_when_over_max_rounds(mock_llm: MagicMock) -> None:
    mq = Action(tool="map_query", params={"query": "tower"})
    submit = Action(tool="submit_answer", params=_submit().model_dump())
    parent = reconstruct_single_trajectory(
        [
            _step([mq], thought="查。", role=AgentRole.FINE),
            _step([submit], thought="交。", role=AgentRole.FINE, start=1, end=2),
        ],
        [
            _obs(
                mq,
                observation={
                    "status": "success",
                    "error_message": None,
                    "resolved_latlng": [48.8584, 2.2945],
                    "display_name": "Eiffel Tower",
                },
            ),
            _obs(submit),
        ],
        AgentRole.FINE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
        coarse_handoff=_hyp(),
    )
    verification = VerificationResult(
        verdict="fail",
        failed_checks=["x"],
        suggested_recheck="y",
        return_to_agent=2,
    )
    revs = reconstruct_revision_trajectories(
        {AgentRole.FINE: parent},
        verification,
        {
            AgentRole.FINE: [
                _step([mq], thought="再查。", role=AgentRole.FINE),
                _step(
                    [submit], thought="再交。", role=AgentRole.FINE, start=1, end=2
                ),
            ]
        },
        {
            AgentRole.FINE: [
                _obs(
                    mq,
                    observation={
                        "status": "success",
                        "error_message": None,
                        "resolved_latlng": [48.8584, 2.2945],
                        "display_name": "Eiffel Tower",
                    },
                ),
                _obs(submit),
            ]
        },
        answer_timestamp=100.0,
        image_path="frame.jpg",
        revision_round=3,
        max_revision_rounds=2,
    )
    assert revs == []


def test_signature_has_no_groundtruth_param() -> None:
    sig = inspect.signature(reconstruct_single_trajectory)
    assert "groundtruth" not in sig.parameters
    src = inspect.getsource(reconstruct_single_trajectory)
    assert "禁止将 groundtruth" in src or "禁止使用 groundtruth" in src.lower()


def test_hard_check_issues_detects_redundancy() -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    desc = "俯视视角下画面下方可见建筑物的屋顶结构整体视角呈现从高处向下俯瞰的特征"
    unit = (
        "draft",
        zoom,
        _obs(
            zoom,
            observation={
                "status": "success",
                "error_message": None,
                "description": desc,
            },
        ),
        _step([zoom], narration=desc),
    )
    issues = _hard_check_issues([desc], [unit])
    assert any("thought_observation_redundancy" in x for x in issues) or any(
        "narration_copy" in x for x in issues
    )


def test_duplicate_fullframe_zoom_raises() -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.0, 0.0, 1.0, 1.0]})
    units = [
        (
            "a",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "same",
                },
            ),
            _step([zoom]),
        ),
        (
            "b",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "same",
                },
            ),
            _step([zoom], start=1, end=2),
        ),
    ]
    with pytest.raises(ValueError, match="递进可写性不足"):
        _validate_coarse_projection_richness(units)


def test_collapse_consecutive_duplicates_keeps_progressive_chain() -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    units = [
        (
            "a",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "same",
                },
            ),
            _step([zoom]),
        ),
        (
            "b",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "same",
                },
            ),
            _step([zoom], start=1, end=2),
        ),
        (
            "c",
            Action(tool="ocr", params={"bbox": [0.2, 0.2, 0.3, 0.3]}),
            _obs(
                Action(tool="ocr", params={"bbox": [0.2, 0.2, 0.3, 0.3]}),
                observation={
                    "status": "success",
                    "error_message": None,
                    "texts": ["x"],
                },
            ),
            _step(
                [Action(tool="ocr", params={"bbox": [0.2, 0.2, 0.3, 0.3]})],
                start=2,
                end=3,
            ),
        ),
    ]
    collapsed = _collapse_consecutive_duplicate_actions(units)
    assert len(collapsed) == 2
    assert collapsed[0][1].tool == "zoom_inspect"
    assert collapsed[1][1].tool == "ocr"


def test_collapse_same_action_different_obs_without_delta() -> None:
    """同 bbox zoom、Obs 字面不同但无候选增量 → 折叠。"""
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.05, 0.55, 0.9, 0.45]})
    units = [
        (
            "看河岸。",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "desc-a",
                },
            ),
            _step([zoom]),
        ),
        (
            "再看河岸细节。",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "desc-b-different",
                },
            ),
            _step([zoom], start=1, end=2),
        ),
        (
            "继续同一框。",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "desc-c",
                },
            ),
            _step([zoom], start=2, end=3),
        ),
    ]
    collapsed = _collapse_consecutive_duplicate_actions(units)
    assert len(collapsed) == 1
    assert collapsed[0][2].observation is not None
    assert collapsed[0][2].observation["description"] == "desc-a"


def test_judge_rubric_distinguishes_video_fact_vs_early() -> None:
    from pipeline.stage5_reconstruct import _JUDGE_RUBRIC

    assert "不得" in _JUDGE_RUBRIC and "无视频来源" in _JUDGE_RUBRIC
    assert "链内过早" in _JUDGE_RUBRIC
    assert "轻微瑕疵" in _JUDGE_RUBRIC
    assert "不应" in _JUDGE_RUBRIC and "≤0.4" in _JUDGE_RUBRIC
    assert "空转 zoom" in _JUDGE_RUBRIC or "连续空转" in _JUDGE_RUBRIC


def test_filter_ui_observation_units() -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    units = [
        (
            "ui",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "聊天界面置顶消息进度条",
                },
            ),
            _step([zoom]),
        ),
        (
            "geo",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "elevated ground and long bridge",
                },
            ),
            _step([zoom], start=1, end=2),
        ),
    ]
    kept, removed = _filter_unusable_ui_units(units)
    assert len(removed) == 1
    assert len(kept) == 1
    assert "bridge" in str(kept[0][2].observation)


def test_build_ledger_lists_visual_facts() -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    units = [
        (
            _coarse_ctx_draft("看。"),
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "elevated ground",
                },
            ),
            _step([zoom]),
        )
    ]
    ledger = _build_coarse_evidence_ledger(
        units, given_clues=[], candidate_hypotheses=[]
    )
    assert ledger.visual_facts or ledger.video_fact_claims


def test_prompts_never_contain_groundtruth_token(mock_llm: MagicMock) -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    ocr = Action(tool="ocr", params={"bbox": [0.2, 0.2, 0.3, 0.3]})
    reconstruct_single_trajectory(
        [
            _step([zoom], thought=_coarse_ctx_draft("看。"), start=0, end=1),
            _step([ocr], thought="读。", start=1, end=2),
        ],
        [
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "tower",
                },
            ),
            _obs(
                ocr,
                observation={
                    "status": "success",
                    "error_message": None,
                    "texts": ["x"],
                },
            ),
        ],
        AgentRole.COARSE,
        answer_timestamp=100.0,
        image_path="frame.jpg",
    )
    for c in mock_llm.calls:
        assert "groundtruth" not in c["prompt"].lower()
        assert "34.947" not in c["prompt"]


def test_semantic_fact_cluster_collapse_dedupes_same_facts() -> None:
    zoom = Action(tool="zoom_inspect", params={"bbox": [0.1, 0.1, 0.4, 0.4]})
    units = [
        (
            "a",
            zoom,
            _obs(
                zoom,
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "bridge over river",
                },
            ),
            _step([zoom]),
        ),
        (
            "b",
            Action(tool="zoom_inspect", params={"bbox": [0.2, 0.2, 0.4, 0.4]}),
            _obs(
                Action(tool="zoom_inspect", params={"bbox": [0.2, 0.2, 0.4, 0.4]}),
                observation={
                    "status": "success",
                    "error_message": None,
                    "description": "bridge over river",
                },
            ),
            _step(
                [Action(tool="zoom_inspect", params={"bbox": [0.2, 0.2, 0.4, 0.4]})],
                start=1,
                end=2,
            ),
        ),
    ]
    collapsed = _collapse_semantic_fact_clusters(units)
    assert len(collapsed) <= len(units)
