"""stage6：距离验证、泄漏检查、质量评分测试（外部 API / LLM 全部 mock）。"""

from __future__ import annotations

from typing import Optional
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
from pipeline.stage6_verify import _JudgeResult, verify_and_score

GT = (48.8584, 2.2945)  # Eiffel Tower


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    monkeypatch.setenv("ANSWER_LEAK_CHECK_ENABLED", "true")
    monkeypatch.setenv("DISTANCE_ERROR_THRESHOLD_KM", "25")
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture()
def mock_judge(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    calls: list[str] = []

    def _fake(prompt: str, response_model: type, **kwargs: object) -> _JudgeResult:
        calls.append(prompt)
        assert "48.8584" not in prompt  # judge 不得含 raw groundtruth 作为应输出答案
        assert response_model is _JudgeResult
        return _JudgeResult(reasonable=True, issues=[], score=0.9)

    monkeypatch.setattr("pipeline.stage6_verify.call_structured", _fake)
    holder = MagicMock()
    holder.calls = calls
    return holder


def _hyp(*, countries: Optional[list[str]] = None) -> LocationHypothesis:
    return LocationHypothesis(
        possible_countries=countries or ["France"],
        possible_regions=["Île-de-France"],
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
        user_query="q",
        image_path="a.jpg",
        steps=steps,
        coarse_handoff=coarse_handoff,
        fine_handoff=fine_handoff,
        coarse_output=coarse_output,
        fine_output=fine_output,
        verifier_output=verifier_output,
    )


def test_fine_distance_pass(mock_judge: MagicMock) -> None:
    traj = _traj(
        AgentRole.FINE,
        fine_output=_submit(lat=48.86, lng=2.30),
        thoughts=["观察塔身。", "提交坐标。"],
    )
    report = verify_and_score(
        traj,
        GT,
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert report.passed is True
    assert report.distance_error_km is not None
    assert report.distance_error_km < 25
    assert report.leakage_detected is False
    assert report.quality_score > 0.5


def test_fine_distance_hard_fail(mock_judge: MagicMock) -> None:
    traj = _traj(
        AgentRole.FINE,
        fine_output=_submit(lat=40.0, lng=-74.0),  # NYC
        thoughts=["观察。", "提交。"],
    )
    report = verify_and_score(
        traj,
        GT,
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert report.passed is False
    assert report.distance_error_km is not None
    assert report.distance_error_km > 25
    assert any("距离误差" in r for r in report.hard_fail_reasons)


def test_coarse_country_coverage(mock_judge: MagicMock) -> None:
    ok = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["France", "Belgium"]),
        thoughts=["宏观特征像西欧。"],
    )
    report_ok = verify_and_score(
        ok,
        GT,
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert report_ok.passed is True

    bad = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(countries=["Japan"]),
        thoughts=["像东亚。"],
    )
    report_bad = verify_and_score(
        bad,
        GT,
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert report_bad.passed is False
    assert any("未覆盖真值国家" in r for r in report_bad.hard_fail_reasons)


def test_coarse_city_leakage(mock_judge: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["这肯定是 Paris 的铁塔。"],
    )
    report = verify_and_score(
        traj,
        GT,
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert report.leakage_detected is True
    assert report.passed is False


def test_coarse_coordinate_leakage(mock_judge: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["坐标大约是 48.8584, 2.2945。"],
    )
    report = verify_and_score(
        traj,
        GT,
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert report.leakage_detected is True
    assert any("坐标" in r for r in report.hard_fail_reasons)


def test_verifier_consistency_mismatch(mock_judge: MagicMock) -> None:
    # 候选远离真值，但 VERIFIER 判 pass → hard fail
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
    report = verify_and_score(
        traj,
        GT,
        reverse_geocode=lambda _c: ("France", None),
    )
    assert report.passed is False
    assert report.distance_error_km is not None
    assert any("不一致" in r for r in report.hard_fail_reasons)


def test_verifier_pass_when_candidate_near(mock_judge: MagicMock) -> None:
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
    report = verify_and_score(
        traj,
        GT,
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert report.passed is True
    assert report.distance_error_km is not None
    assert report.distance_error_km < 1.0


def test_judge_prompt_excludes_groundtruth_as_answer(mock_judge: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["西欧特征。"],
    )
    verify_and_score(
        traj,
        GT,
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert mock_judge.calls
    joined = "\n".join(mock_judge.calls)
    assert "应输出" not in joined or "不要把任何坐标当作必须输出的标准答案" in joined
    assert "48.8584" not in joined


def test_leak_check_can_be_disabled(mock_judge: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
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
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert report.leakage_detected is False


def test_report_distance_none_for_coarse(mock_judge: MagicMock) -> None:
    traj = _traj(
        AgentRole.COARSE,
        coarse_output=_hyp(),
        thoughts=["宏观判断。"],
    )
    report = verify_and_score(
        traj,
        GT,
        reverse_geocode=lambda _c: ("France", "Île-de-France"),
    )
    assert report.distance_error_km is None
    assert isinstance(report.quality_score, float)
    assert 0.0 <= report.quality_score <= 1.0
