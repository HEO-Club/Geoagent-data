"""stage6：距离验证、泄漏检查、质量评分测试（外部 API / LLM 全部 mock）。

按 SPEC 5.12：stage6 只做 GT 相关检查；质量基分取自 stage5_judge_score。
"""

from __future__ import annotations

import re
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas import (
    Action,
    AgentRole,
    LocationHypothesis,
    SubmitAnswerResult,
    Trajectory,
    TrajectoryStep,
    VerificationResult,
)
from pipeline.stage6_verify import (
    LeakageJudgeResult,
    PlaceHints,
    verify_and_score,
)

GT = (48.8584, 2.2945)  # Eiffel Tower
GT_ZH = (34.9475818, 113.517007)  # Zhengzhou park


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    monkeypatch.setenv("ANSWER_LEAK_CHECK_ENABLED", "true")
    monkeypatch.setenv("DISTANCE_ERROR_THRESHOLD_KM", "25")
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture()
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """mock 泄漏 LLM judge。"""
    calls: list[tuple[type, str]] = []

    def _fake(prompt: str, response_model: type, **kwargs: object) -> Any:
        calls.append((response_model, prompt))
        if response_model is LeakageJudgeResult:
            leaked = False
            reasons: list[str] = []
            if "agent_role: verifier" in prompt and "fine_handoff" in prompt:
                if re.search(r"正确答案|真值|ground\s*truth|官方答案", prompt, re.I):
                    if "--- Step" in prompt:
                        thought_blob = prompt
                        if re.search(
                            r"(正确答案|真值|官方答案).{0,40}(郑州黄河文化公园|Eiffel)",
                            thought_blob,
                        ) or re.search(
                            r"(郑州黄河文化公园|Eiffel).{0,40}(正确答案|真值)",
                            thought_blob,
                        ):
                            return LeakageJudgeResult(
                                leaked=True, reasons=["把 GT 当作已知正确答案"]
                            )
                return LeakageJudgeResult(leaked=False, reasons=[])

            if re.search(
                r"(正确答案就是|真值就是|我知道答案|官方答案就是)",
                prompt,
            ):
                return LeakageJudgeResult(
                    leaked=True, reasons=["后见之明 / 直接使用 GT"]
                )

            if "agent_role: coarse" in prompt:
                for needle in ("郑州黄河文化公园", "Eiffel Tower"):
                    if needle in prompt and (
                        f"就是{needle}" in prompt
                        or f"我认为就是{needle}" in prompt
                    ):
                        return LeakageJudgeResult(
                            leaked=True,
                            reasons=["COARSE 以最终精准 POI 作结论"],
                        )
            return LeakageJudgeResult(leaked=leaked, reasons=reasons)
        raise AssertionError(f"unexpected response_model: {response_model}")

    monkeypatch.setattr("pipeline.stage6_verify.call_structured", _fake)
    holder = MagicMock()
    holder.calls = calls
    return holder


def _hyp(
    *,
    countries: Optional[list[str]] = None,
    regions: Optional[list[str]] = None,
) -> LocationHypothesis:
    return LocationHypothesis(
        possible_countries=countries or ["France"],
        possible_regions=regions or ["Île-de-France"],
        reasoning_summary="Iron lattice and river suggest France.",
        confidence=0.7,
        key_clues_remaining=["plaza"],
    )


def _submit(*, lat: float = 48.8584, lng: float = 2.2945) -> SubmitAnswerResult:
    return SubmitAnswerResult(
        latitude=lat,
        longitude=lng,
        location_name="Eiffel Tower",
        confidence=0.9,
        reasoning="Matched landmark.",
    )


def _traj(
    role: AgentRole,
    *,
    thoughts: Optional[list[str]] = None,
    coarse_output: Optional[LocationHypothesis] = None,
    fine_output: Optional[SubmitAnswerResult] = None,
    verifier_output: Optional[VerificationResult] = None,
    fine_handoff: Optional[SubmitAnswerResult] = None,
    coarse_handoff: Optional[LocationHypothesis] = None,
    user_query: str = "q",
    stage5_judge_score: Optional[float] = None,
) -> Trajectory:
    thoughts = thoughts or ["前向推理。"]
    steps = [
        TrajectoryStep(
            thought=t,
            action=Action(tool="ocr", params={}),
            observation={"status": "success", "error_message": None, "texts": []},
            observation_source=None,
        )
        for t in thoughts
    ]
    if role == AgentRole.FINE:
        steps[-1] = TrajectoryStep(
            thought=thoughts[-1],
            action=Action(
                tool="submit_answer",
                params=(fine_output or _submit()).model_dump(),
            ),
            observation=None,
            observation_source=None,
        )
        if coarse_handoff is None:
            coarse_handoff = _hyp()
    if role == AgentRole.VERIFIER and fine_handoff is None:
        fine_handoff = _submit()

    return Trajectory(
        id=f"t-{role.value}",
        agent_role=role,
        system_prompt="sys",
        user_query=user_query,
        image_path="a.jpg",
        steps=steps,
        coarse_handoff=coarse_handoff,
        fine_handoff=fine_handoff,
        coarse_output=coarse_output,
        fine_output=fine_output,
        verifier_output=verifier_output,
        stage5_judge_score=stage5_judge_score,
    )


def _fr_geocode(_c: tuple[float, float]) -> PlaceHints:
    return PlaceHints(
        country="France",
        region="Île-de-France",
        city="Paris",
        display_name="Eiffel Tower, Paris, France",
    )


def _zh_geocode(_c: tuple[float, float]) -> PlaceHints:
    return PlaceHints(
        country="China",
        region="Henan",
        city="Zhengzhou",
        display_name="郑州黄河文化公园, 河南省郑州市",
    )


def test_fine_distance_pass(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.FINE,
        fine_output=_submit(lat=48.86, lng=2.30),
        thoughts=["观察塔身。", "提交坐标。"],
        stage5_judge_score=0.85,
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is True
    assert report.distance_error_km is not None
    assert report.distance_error_km < 25
    assert report.leakage_detected is False
    assert report.quality_score > 0.5


def test_fine_distance_hard_fail(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.FINE,
        fine_output=_submit(lat=40.0, lng=-74.0),
        thoughts=["观察。", "提交。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert report.distance_error_km is not None
    assert report.distance_error_km > 25
    assert any("距离误差" in r for r in report.hard_fail_reasons)


def test_coarse_country_coverage(mock_llm: MagicMock) -> None:
    ok = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["France", "Belgium"]),
        thoughts=["宏观特征像西欧，排除东亚候选。"],
    )
    report_ok = verify_and_score(ok, GT, reverse_geocode=_fr_geocode)
    assert report_ok.passed is True

    bad = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["Japan"]),
        thoughts=["像东亚，排除西欧候选。"],
    )
    report_bad = verify_and_score(bad, GT, reverse_geocode=_fr_geocode)
    assert report_bad.passed is False
    assert any("未覆盖真值国家" in r for r in report_bad.hard_fail_reasons)


def test_llm_leakage_coarse_precise_poi_conclusion(mock_llm: MagicMock) -> None:
    """COARSE 以最终精准 POI 作结论 → 泄漏（角色越界 / 直接用答案）。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=LocationHypothesis(
            possible_countries=["中国"],
            possible_regions=["河南"],
            reasoning_summary="线索指向郑州黄河文化公园附近。",
            confidence=0.5,
            key_clues_remaining=[],
        ),
        thoughts=["我认为就是郑州黄河文化公园。"],
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.leakage_detected is True
    assert report.passed is False
    leak_prompts = [p for m, p in mock_llm.calls if m is LeakageJudgeResult]
    assert leak_prompts
    assert "直接使用 groundtruth" in leak_prompts[0] or "后见之明" in leak_prompts[0]
    assert "定位到准确地点本身" in leak_prompts[0]


def test_llm_allows_non_gt_candidate_city(mock_llm: MagicMock) -> None:
    """策略 B：仅候选「许昌」不算直接使用 GT。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=LocationHypothesis(
            possible_countries=["中国"],
            possible_regions=["河南许昌"],
            reasoning_summary="根据建筑与植被，拍摄地可能在河南许昌一带。",
            confidence=0.6,
            key_clues_remaining=["具体场景"],
        ),
        thoughts=["根据建筑与植被排除不符候选，可能在河南许昌一带。"],
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.leakage_detected is False
    assert report.passed is True


def test_fine_early_precise_location_not_leak(mock_llm: MagicMock) -> None:
    """FINE 非终端步已写准 POI/近 GT 坐标，但前向连贯 → 不算泄漏。"""
    traj = _traj(
        AgentRole.FINE,
        fine_output=_submit(),
        thoughts=[
            "塔影与广场布局吻合，假设为 Eiffel Tower，坐标约 48.8584, 2.2945，先 map_query 核实。",
            "核实后提交。",
        ],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.leakage_detected is False
    assert not any("过早" in r for r in report.hard_fail_reasons)


def test_oracle_gt_thought_is_leak(mock_llm: MagicMock) -> None:
    """明确「正确答案就是 GT」→ 泄漏。"""
    traj = _traj(
        AgentRole.FINE,
        fine_output=_submit(),
        thoughts=[
            "正确答案就是 Eiffel Tower，无需再查。",
            "提交。",
        ],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.leakage_detected is True


def test_user_query_hint_reuse_not_leak(mock_llm: MagicMock) -> None:
    """user_query 含外部线索且 Thought 使用 → 不算泄漏。"""
    traj = _traj(
        AgentRole.FINE,
        fine_output=_submit(lat=32.15, lng=114.08),
        user_query="请在粗定位假设基础上精确定位。\n已知线索：河南信阳",
        thoughts=[
            "结合已知线索河南信阳与画面山丘农田，先检索附近地标。",
            "提交候选坐标。",
        ],
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.leakage_detected is False


def test_coarse_coordinate_leakage(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["坐标大约是 48.8584, 2.2945。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.leakage_detected is True
    assert any("坐标" in r for r in report.hard_fail_reasons)


def test_style_issues_not_stage6_hard_fail(mock_llm: MagicMock) -> None:
    """旁白叙事体等风格问题不再由 stage6 hard-fail（由 stage5 judge 负责）。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["为了找到这张照片的拍摄地,我足足花了半年的时间。当我知道答案的那一刻起。"],
        stage5_judge_score=0.2,
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert not any("TAO" in r or "旁白" in r for r in report.hard_fail_reasons)
    assert report.quality_score == pytest.approx(0.2)


def test_verifier_handoff_restatement_not_leak(mock_llm: MagicMock) -> None:
    """VERIFIER 复述 fine_handoff 候选地名/坐标不算泄漏。"""
    handoff = _submit(lat=34.9458, lng=113.5517)
    handoff = handoff.model_copy(update={"location_name": "郑州黄河文化公园"})
    traj = _traj(
        AgentRole.VERIFIER,
        fine_handoff=handoff,
        verifier_output=VerificationResult(
            verdict="fail",
            failed_checks=["visual mismatch"],
            suggested_recheck="recheck bridge",
            return_to_agent=2,
        ),
        thoughts=[
            "Agent2 候选为郑州黄河文化公园，坐标 [34.9458, 113.5517]。先 map_query 核对。"
        ],
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.leakage_detected is False


def test_verifier_consistency_mismatch(mock_llm: MagicMock) -> None:
    handoff = _submit(lat=40.0, lng=-74.0)  # far from GT
    traj = _traj(
        AgentRole.VERIFIER,
        fine_handoff=handoff,
        verifier_output=VerificationResult(
            verdict="pass",
            failed_checks=[],
            suggested_recheck="",
            return_to_agent=None,
        ),
        thoughts=["候选看起来合理。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert any("不一致" in r for r in report.hard_fail_reasons)


def test_verifier_pass_when_candidate_near(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.VERIFIER,
        fine_handoff=_submit(),
        verifier_output=VerificationResult(
            verdict="pass",
            failed_checks=[],
            suggested_recheck="",
            return_to_agent=None,
        ),
        thoughts=["候选与地图一致。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is True


def test_verifier_allows_candidate_coords_near_gt_in_thought(
    mock_llm: MagicMock,
) -> None:
    traj = _traj(
        AgentRole.VERIFIER,
        fine_handoff=_submit(),
        verifier_output=VerificationResult(
            verdict="pass",
            failed_checks=[],
            suggested_recheck="",
            return_to_agent=None,
        ),
        thoughts=["候选坐标约 48.8584, 2.2945，与卫星图一致。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.leakage_detected is False


def test_coarse_accepts_chinese_country_alias(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=LocationHypothesis(
            possible_countries=["中国"],
            possible_regions=["河南省"],
            reasoning_summary="华北平原特征。",
            confidence=0.6,
            key_clues_remaining=[],
        ),
        thoughts=["华北平原特征，排除华南。"],
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.passed is True


def test_quality_uses_stage5_judge_score(mock_llm: MagicMock) -> None:
    """无 hard-fail 时 quality 基分取自 stage5_judge_score。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["铁塔与河流组合，排除无河岸候选。"],
        stage5_judge_score=0.73,
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is True
    assert report.quality_score == pytest.approx(0.73)


def test_quality_defaults_to_one_without_stage5_score(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["铁塔与河流组合，排除无河岸候选。"],
        stage5_judge_score=None,
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is True
    assert report.quality_score == pytest.approx(1.0)


def test_hard_fail_caps_quality_at_point_three(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["Japan"]),
        thoughts=["像东亚。"],
        stage5_judge_score=0.95,
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert report.quality_score <= 0.3


def test_leak_check_can_be_disabled(
    mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANSWER_LEAK_CHECK_ENABLED", "false")
    clear_settings_cache()
    traj = _traj(
        AgentRole.FINE,
        fine_output=_submit(),
        thoughts=["正确答案就是 Eiffel Tower，无需再查。", "提交。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.leakage_detected is False
    assert mock_llm.calls == []


def test_report_distance_none_for_coarse(mock_llm: MagicMock) -> None:
    traj = _traj(AgentRole.COARSE, coarse_output=_hyp())
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.distance_error_km is None


def test_tuple_reverse_geocode_still_works(mock_llm: MagicMock) -> None:
    traj = _traj(AgentRole.COARSE, coarse_output=_hyp())
    report = verify_and_score(
        traj, GT, reverse_geocode=lambda _c: ("France", "Île-de-France")
    )
    assert report.passed is True


def test_coarse_forbidden_web_search_hard_fail(mock_llm: MagicMock) -> None:
    traj = _traj(AgentRole.COARSE, coarse_output=_hyp())
    traj.steps[0] = TrajectoryStep(
        thought="检索地名。",
        action=Action(tool="web_search", params={"query": "tower"}),
        observation={"status": "success", "error_message": None, "results": []},
        observation_source=None,
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert any("禁止 Tool" in r for r in report.hard_fail_reasons)


def test_coarse_allows_compare_and_satellite_tools(mock_llm: MagicMock) -> None:
    """视觉比对/卫星类不再因「禁止 Tool」hard-fail。"""
    traj = _traj(AgentRole.COARSE, coarse_output=_hyp())
    traj.steps[0] = TrajectoryStep(
        thought="比对两张图的桥梁跨度。",
        action=Action(
            tool="compare_images_for_geolocation",
            params={"image_a": "a.jpg", "image_b": "b.jpg"},
        ),
        observation={
            "status": "success",
            "error_message": None,
            "visual_similarity_score": 0.8,
            "matched_features": ["bridge"],
            "mismatched_features": [],
            "geolocation_hints": ["wide river"],
        },
        observation_source=None,
    )
    traj.steps.append(
        TrajectoryStep(
            thought="查看历史卫星图上的河岸布局。",
            action=Action(
                tool="lookup_historical_satellite_map",
                params={"year": 2005, "region": "France"},
            ),
            observation={"status": "success", "error_message": None},
            observation_source=None,
        )
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert not any("禁止 Tool" in r for r in report.hard_fail_reasons)


def test_coarse_forbidden_map_query_hard_fail(mock_llm: MagicMock) -> None:
    traj = _traj(AgentRole.COARSE, coarse_output=_hyp())
    traj.steps[0] = TrajectoryStep(
        thought="解析坐标。",
        action=Action(tool="map_query", params={"query": "Paris"}),
        observation={
            "status": "success",
            "error_message": None,
            "resolved_latlng": [48.8, 2.3],
            "formatted_address": "Paris",
        },
        observation_source=None,
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert any("禁止 Tool" in r for r in report.hard_fail_reasons)


def test_coarse_region_coverage_hard_fail(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["France"], regions=["Brittany"]),
        thoughts=["像布列塔尼。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert any("未覆盖真值一级行政区" in r for r in report.hard_fail_reasons)


def test_legacy_video_chain_gates_not_hard_fail(mock_llm: MagicMock) -> None:
    """旧视频链程序化门禁不再 hard-fail。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=[
            "画面可见铁塔与河流。",  # observe-only，无排除词
            "继续观察立面细节。",
        ],
        stage5_judge_score=0.8,
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is True
    joined = " ".join(report.hard_fail_reasons)
    assert "reused_fact_without_delta" not in joined
    assert "candidate_provenance_gap" not in joined
    assert "no_candidate_progress" not in joined
    assert "redundant_step" not in joined
