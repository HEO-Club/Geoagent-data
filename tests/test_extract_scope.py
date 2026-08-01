"""working_scope 抽取与兜底清洗测试。"""

from __future__ import annotations

from pipeline.schemas.clues import (
    BoundKind,
    CandidateHypothesis,
    ClueExtractionResult,
    ClueRole,
    RawGivenClue,
    WorkingScope,
)
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao import extract_scope


def test_sanitize_soft_prior_rejects_inside_upgrade() -> None:
    soft = WorkingScope(region="许昌市内", bound_kind=BoundKind.near)
    cleaned = extract_scope.sanitize_working_scope(soft)
    assert cleaned is not None
    assert cleaned.bound_kind == BoundKind.near
    assert cleaned.region == "许昌附近"
    assert not cleaned.region.endswith("内")


def test_sanitize_empty_region_becomes_none() -> None:
    # WorkingScope 构造会拒空串，直接测 sanitize 对空白 strip 后空的路径无意义；
    # 用 near + 可 strip 成空的边界由 validator 拦，这里测 None 透传。
    assert extract_scope.sanitize_working_scope(None) is None


def test_sanitize_inside_keeps_phrase() -> None:
    hard = WorkingScope(region="河南许昌内", bound_kind=BoundKind.inside)
    cleaned = extract_scope.sanitize_working_scope(hard)
    assert cleaned is not None
    assert cleaned.region == "河南许昌内"
    assert cleaned.bound_kind == BoundKind.inside


def test_normalize_extraction_drops_bad_soft_scope() -> None:
    raw = ClueExtractionResult(
        raw_given_clues=[
            RawGivenClue(
                text="籍贯许昌，离家不远",
                role=ClueRole.person_or_social_attribute,
            )
        ],
        working_scope=WorkingScope(region="许昌内", bound_kind=BoundKind.near),
        candidate_hypotheses=[
            CandidateHypothesis(text="很可能是许昌北边80公里"),
        ],
    )
    out = extract_scope.normalize_extraction(raw)
    assert out.working_scope is not None
    assert out.working_scope.region == "许昌附近"
    assert len(out.candidate_hypotheses) == 1


def test_extract_working_scope_mock(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    expected = ClueExtractionResult(
        raw_given_clues=[
            RawGivenClue(
                text="拍摄地在河南许昌附近",
                role=ClueRole.photo_location_constraint,
            )
        ],
        working_scope=WorkingScope(region="河南许昌附近", bound_kind=BoundKind.near),
        candidate_hypotheses=[
            CandidateHypothesis(text="博主猜是郑州"),
        ],
    )

    def _fake_call(prompt: str, schema, **_k):  # type: ignore[no-untyped-def]
        assert "禁止使用或猜测 groundtruth" in prompt
        assert "candidate_hypotheses" in prompt
        assert schema is ClueExtractionResult
        return expected

    monkeypatch.setattr(extract_scope, "call_structured", _fake_call)
    transcript = [
        TranscriptSegment(start=0, end=2, text="拍摄地在河南许昌附近"),
        TranscriptSegment(start=2, end=4, text="我猜很可能是郑州"),
    ]
    result = extract_scope.extract_working_scope(transcript)
    assert result.working_scope is not None
    assert result.working_scope.region == "河南许昌附近"
    assert result.candidate_hypotheses[0].text == "博主猜是郑州"
