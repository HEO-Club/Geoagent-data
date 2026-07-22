"""stage6：距离验证、泄漏检查、质量评分测试（外部 API / LLM 全部 mock）。"""

from __future__ import annotations

import re
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from pipeline.config import Settings, clear_settings_cache
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
    CoarseReasoningChainJudgeResult,
    LeakageJudgeResult,
    PlaceHints,
    TaoStyleJudgeResult,
    _JudgeResult,
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
    """同时 mock 合理性 judge 与泄漏 judge。"""
    calls: list[tuple[type, str]] = []

    def _fake(prompt: str, response_model: type, **kwargs: object) -> Any:
        calls.append((response_model, prompt))
        if response_model is TaoStyleJudgeResult:
            # 只检查轨迹 Thought 行，忽略 checklist 中的 BAD 反例
            bad = False
            for line in prompt.splitlines():
                if "thought=" not in line:
                    continue
                if any(
                    x in line
                    for x in ("我足足花了", "求助者希望", "粉丝向我", "当我知道答案")
                ):
                    bad = True
                    break
            return TaoStyleJudgeResult(
                is_standard_geo_tao=not bad,
                issues=["旁白叙事体"] if bad else [],
            )
        if response_model is CoarseReasoningChainJudgeResult:
            # 只检查轨迹 Thought 行，忽略 checklist 中的 BAD 反例
            gap = False
            misaligned = False
            satellite_verify = False
            clue_only = False
            for line in prompt.splitlines():
                # 兼容 thought= 与分字段 thought:
                if "thought=" not in line and not line.strip().startswith("thought:"):
                    continue
                if any(
                    x in line
                    for x in ("直接就是法国", "所以这就是越南", "无需再看")
                ):
                    gap = True
                if "随便调用" in line:
                    misaligned = True
                # 把已知线索当最终答案做卫星/地图验证
                if ("卫星" in line or "地图验证" in line) and (
                    "许昌" in line or "已知线索" in line or "验证就是" in line
                ):
                    satellite_verify = True
                    clue_only = True
            return CoarseReasoningChainJudgeResult(
                identifies_geo_human_features=not gap and not satellite_verify,
                narrows_scope_progressively=not gap and not satellite_verify,
                has_reasoning_gap=gap or satellite_verify,
                thought_action_aligned=not misaligned,
                coarse_scope_within_role=True,
                feature_driven_narrowing=not clue_only and not gap,
                clues_only_as_auxiliary=not clue_only,
                regions_are_administrative_and_consistent=True,
                issues=(
                    ["把线索当答案做卫星验证"]
                    if satellite_verify
                    else (["跳步"] if gap else (["Thought-Action不对齐"] if misaligned else []))
                ),
            )
        if response_model is LeakageJudgeResult:
            leaked = False
            reasons: list[str] = []
            # 新语义：直接用 GT / 后见之明；命中地点本身不算泄漏
            if "agent_role: verifier" in prompt and "fine_handoff" in prompt:
                if re.search(r"正确答案|真值|ground\s*truth|官方答案", prompt, re.I):
                    # 检查 Thought 段是否含这些话术
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

            # 明确后见之明 / 粘贴真值话术
            if re.search(
                r"(正确答案就是|真值就是|我知道答案|官方答案就是)",
                prompt,
            ):
                return LeakageJudgeResult(
                    leaked=True, reasons=["后见之明 / 直接使用 GT"]
                )

            # COARSE 以最终精准 POI 作结论
            if "agent_role: coarse" in prompt:
                for needle in ("郑州黄河文化公园", "Eiffel Tower"):
                    if needle in prompt and (
                        f"就是{needle}" in prompt
                        or f"我认为就是{needle}" in prompt
                        or f"线索指向{needle}" in prompt
                    ):
                        return LeakageJudgeResult(
                            leaked=True,
                            reasons=[f"COARSE 以精准 POI 作结论: {needle}"],
                        )

            # FINE：仅写准地点/坐标且无「正确答案」话术 → 不泄漏
            if "agent_role: fine_locator" in prompt:
                return LeakageJudgeResult(leaked=False, reasons=[])

            # user_query 已知线索复用 → 不泄漏
            if "已知线索" in prompt and (
                "河南信阳" in prompt or "河南许昌" in prompt
            ):
                return LeakageJudgeResult(leaked=False, reasons=[])

            return LeakageJudgeResult(leaked=leaked, reasons=reasons)
        if response_model is _JudgeResult:
            return _JudgeResult(reasonable=True, issues=[], score=0.9)
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
        thoughts=["宏观特征像西欧。"],
    )
    report_ok = verify_and_score(ok, GT, reverse_geocode=_fr_geocode)
    assert report_ok.passed is True

    bad = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["Japan"]),
        thoughts=["像东亚。"],
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
        thoughts=["根据线索，可能在河南许昌一带。"],
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
    # 距离会 hard-fail，但泄漏不应触发
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


def test_tao_style_narration_hard_fail(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["为了找到这张照片的拍摄地,我足足花了半年的时间。当我知道答案的那一刻起。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert any("TAO" in r or "旁白" in r for r in report.hard_fail_reasons)


def test_tao_style_pass_for_geo_reasoning(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["画面可见铁塔与宽阔河流，先放大立面确认建筑细节。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert not any("TAO" in r for r in report.hard_fail_reasons)


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
    report = verify_and_score(
        traj,
        GT_ZH,
        reverse_geocode=lambda _g: PlaceHints(
            country="中国",
            region="河南省",
            city="郑州市",
            display_name="黄河文化公园, 惠济区, 郑州市, 河南省, 中国",
        ),
    )
    # 距离远于阈值且 verdict=fail → 一致性可通过；关键是泄漏不应因复述候选而触发
    assert report.leakage_detected is False


def test_verifier_consistency_mismatch(mock_llm: MagicMock) -> None:
    far = _submit(lat=40.0, lng=-74.0)
    traj = _traj(
        AgentRole.VERIFIER,
        fine_handoff=far,
        verifier_output=VerificationResult(
            verdict="pass",
            failed_checks=[],
            suggested_recheck="none",
            return_to_agent=None,
        ),
        thoughts=["候选与图像自洽。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert report.distance_error_km is not None
    assert any("不一致" in r for r in report.hard_fail_reasons)


def test_verifier_pass_when_candidate_near(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.VERIFIER,
        fine_handoff=_submit(),
        verifier_output=VerificationResult(
            verdict="pass",
            failed_checks=[],
            suggested_recheck="none",
            return_to_agent=None,
        ),
        thoughts=["候选与铁塔特征一致。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is True
    assert report.distance_error_km is not None
    assert report.distance_error_km < 1.0


def test_verifier_allows_candidate_coords_near_gt_in_thought(
    mock_llm: MagicMock,
) -> None:
    cand = _submit(lat=48.86, lng=2.29)
    traj = _traj(
        AgentRole.VERIFIER,
        fine_handoff=cand,
        verifier_output=VerificationResult(
            verdict="pass",
            failed_checks=[],
            suggested_recheck="none",
            return_to_agent=None,
        ),
        thoughts=["核对候选 48.86, 2.29 与图像是否自洽。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.leakage_detected is False
    assert report.passed is True


def test_coarse_accepts_chinese_country_alias(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        thoughts=["东亚宏观特征，缩小到中国境内省份级别。"],
    )
    traj.coarse_output = LocationHypothesis(
        possible_countries=["中国"],
        possible_regions=["河南省", "许昌市"],
        reasoning_summary="宏观气候与文字线索指向中国中部。",
        confidence=0.6,
        key_clues_remaining=["具体城市"],
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.passed is True
    assert not any("possible_countries" in r for r in report.hard_fail_reasons)


def test_reasonableness_judge_excludes_gt_as_answer(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["西欧特征。"],
    )
    verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    reason_prompts = [p for m, p in mock_llm.calls if m is _JudgeResult]
    assert reason_prompts
    assert "不要把任何坐标当作必须输出的标准答案" in reason_prompts[0]


def test_leak_check_can_be_disabled(
    mock_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANSWER_LEAK_CHECK_ENABLED", "false")
    clear_settings_cache()
    settings = Settings()
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["这是 Paris。"],
    )
    report = verify_and_score(
        traj,
        GT,
        settings=settings,
        reverse_geocode=_fr_geocode,
    )
    assert report.leakage_detected is False


def test_report_distance_none_for_coarse(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["宏观判断。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.distance_error_km is None
    assert isinstance(report.quality_score, float)
    assert 0.0 <= report.quality_score <= 1.0


def test_tuple_reverse_geocode_still_works(mock_llm: MagicMock) -> None:
    """兼容旧式 (country, region) 注入。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["西欧。"],
    )
    report = verify_and_score(
        traj,
        GT,
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert report.passed is True


def test_coarse_progressive_reasoning_pass(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["画面可见铁塔与宽阔河流，先放大立面确认建筑细节。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert not any("跳步" in r or "递进" in r for r in report.hard_fail_reasons)
    chain_prompts = [p for m, p in mock_llm.calls if m is CoarseReasoningChainJudgeResult]
    assert chain_prompts
    assert "groundtruth" not in chain_prompts[0].lower() or "不含真值" in chain_prompts[0]


def test_coarse_reasoning_gap_hard_fail(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["看到铁塔，直接就是法国，无需再看。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert any("跳步" in r or "递进" in r for r in report.hard_fail_reasons)


def test_coarse_thought_action_misaligned_hard_fail(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["随便调用一下工具看看。"],
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert any("不对齐" in r for r in report.hard_fail_reasons)


def test_coarse_forbidden_web_search_hard_fail(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["先检索宏观线索。"],
    )
    traj.steps[0] = TrajectoryStep(
        thought=traj.steps[0].thought,
        action=Action(
            tool="web_search",
            params={"query": "q", "purpose": "broad_discovery"},
        ),
        observation={"status": "success", "error_message": None, "results": []},
        observation_source=None,
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert any("禁止 Tool" in r and "web_search" in r for r in report.hard_fail_reasons)


def test_coarse_forbidden_compare_images_hard_fail(mock_llm: MagicMock) -> None:
    """轨迹仍含 compare_images* → 程序化 hard-fail。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["比对候选卫星图与画面特征。"],
    )
    traj.steps[0] = TrajectoryStep(
        thought=traj.steps[0].thought,
        action=Action(
            tool="compare_images_for_geolocation",
            params={"image_a": "a", "image_b": "b"},
        ),
        observation={"status": "success", "error_message": None, "similarity": 0.8},
        observation_source=None,
    )
    report = verify_and_score(traj, GT, reverse_geocode=_fr_geocode)
    assert report.passed is False
    assert any(
        "禁止 Tool" in r and "compare_images" in r for r in report.hard_fail_reasons
    )


def test_coarse_known_clue_as_hypothesis_not_gt_fail(mock_llm: MagicMock) -> None:
    """已知线索作假设收窄 → 不因「真值」误杀；裁判 prompt 含线索校准说明。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["China"], regions=["Henan"]),
        thoughts=[
            "画面为华北平原农田与低缓丘陵；结合已知线索河南许昌作为候选假设，"
            "先放大路旁建筑细节以排除/收窄周边县域。"
        ],
        user_query="请粗定位照片拍摄地。\n已知线索：河南许昌",
    )
    traj.steps[0] = TrajectoryStep(
        thought=traj.steps[0].thought,
        action=Action(tool="zoom_inspect", params={"bbox": [0.0, 0.0, 1.0, 1.0]}),
        observation={"status": "success", "error_message": None, "description": "fields"},
        observation_source=None,
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.passed is True
    assert not any("真值" in r for r in report.hard_fail_reasons)
    chain_prompts = [
        p for m, p in mock_llm.calls if m is CoarseReasoningChainJudgeResult
    ]
    assert chain_prompts
    assert "已知线索" in chain_prompts[0]
    assert (
        "不违规" in chain_prompts[0]
        or "辅助" in chain_prompts[0]
        or "特征驱动" in chain_prompts[0]
    )


def test_coarse_clue_satellite_verify_still_hard_fail(mock_llm: MagicMock) -> None:
    """把线索当最终答案做卫星验证 → 仍 hard-fail。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["China"]),
        thoughts=["已知线索是许昌，我对卫星图验证就是许昌。"],
        user_query="请粗定位。\n已知线索：河南许昌",
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.passed is False
    assert any(
        "跳步" in r
        or "递进" in r
        or "卫星" in r
        or "唯一" in r
        or "线索" in r
        for r in report.hard_fail_reasons
    )


def test_coarse_descriptive_regions_hard_fail(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(
            countries=["China"],
            regions=["华北平原南缘", "中原地区"],
        ),
        thoughts=["画面有开阔平原与低缓丘陵，先放大确认地貌。"],
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.passed is False
    assert any("possible_regions" in r or "描述" in r for r in report.hard_fail_reasons)


def test_coarse_region_coverage_hard_fail(mock_llm: MagicMock) -> None:
    """regions 非空但未覆盖 GT 一级行政区 → hard-fail。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["China"], regions=["Guangdong"]),
        thoughts=["画面植被偏南，先放大确认建筑。"],
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.passed is False
    assert any("一级行政区" in r or "possible_regions" in r for r in report.hard_fail_reasons)


def test_coarse_thin_duplicate_fullframe_hard_fail(mock_llm: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["China"], regions=["Henan"]),
        thoughts=["看全图地貌。", "再看一次全图。"],
    )
    obs = {"status": "success", "error_message": None, "description": "same view"}
    traj.steps = [
        TrajectoryStep(
            thought=traj.steps[0].thought,
            action=Action(tool="zoom_inspect", params={"bbox": [0.0, 0.0, 1.0, 1.0]}),
            observation=obs,
            observation_source=None,
        ),
        TrajectoryStep(
            thought=traj.steps[1].thought if len(traj.steps) > 1 else "再看。",
            action=Action(tool="zoom_inspect", params={"bbox": [0.0, 0.0, 1.0, 1.0]}),
            observation=obs,
            observation_source=None,
        ),
    ]
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    assert report.passed is False
    assert any("薄链" in r for r in report.hard_fail_reasons)


def test_coarse_judge_separates_thought_and_observation(mock_llm: MagicMock) -> None:
    """Obs 含水印不应仅凭 Obs 把 Thought 判成元叙事（分字段展示）。"""
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["China"], regions=["Henan"]),
        thoughts=["背景山脉与高架桥梁横跨开阔地带，先放大桥梁结构确认。"],
    )
    traj.steps[0] = TrajectoryStep(
        thought=traj.steps[0].thought,
        action=Action(tool="zoom_inspect", params={"bbox": [0.0, 0.0, 1.0, 0.4]}),
        observation={
            "status": "success",
            "error_message": None,
            "description": "bridge over open land; corner has youtube watermark",
        },
        observation_source=None,
    )
    report = verify_and_score(traj, GT_ZH, reverse_geocode=_zh_geocode)
    chain_prompts = [
        p for m, p in mock_llm.calls if m is CoarseReasoningChainJudgeResult
    ]
    assert chain_prompts
    assert "thought:" in chain_prompts[0]
    assert "observation:" in chain_prompts[0]
    # Thought 本身无元叙事话术时，mock 不应因 Obs 误杀
    assert report.passed is True
