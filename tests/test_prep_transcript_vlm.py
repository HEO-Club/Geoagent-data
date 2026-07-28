"""分窗 VLM 重转录 prep 单测（禁止真实 API）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.config import clear_settings_cache
from pipeline.prep_transcript_vlm import (
    _WindowSpeech,
    _merge_adjacent,
    build_time_windows,
    prep_transcript_vlm,
)
from pipeline.schemas import TranscriptSegment


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / ".cache"))
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_build_time_windows_fixed() -> None:
    wins = build_time_windows(75.0, window_sec=30.0, anchor=None)
    assert wins[0] == (0.0, 30.0)
    assert wins[-1][1] == pytest.approx(75.0)
    assert len(wins) == 3


def test_build_time_windows_from_anchor() -> None:
    anchor = [
        TranscriptSegment(start=0.0, end=10.0, text="a"),
        TranscriptSegment(start=10.0, end=20.0, text="b"),
        TranscriptSegment(start=40.0, end=50.0, text="c"),
    ]
    wins = build_time_windows(50.0, window_sec=30.0, anchor=anchor)
    assert wins
    assert wins[0][0] == pytest.approx(0.0)
    assert any(w[0] <= 40.0 <= w[1] or w[0] == 40.0 for w in wins)


def test_merge_adjacent_joins_touching() -> None:
    segs = [
        TranscriptSegment(start=0.0, end=1.0, text="你好"),
        TranscriptSegment(start=1.1, end=2.0, text="世界"),
        TranscriptSegment(start=5.0, end=6.0, text="下一段"),
    ]
    merged = _merge_adjacent(segs)
    # 不同正文即使紧邻也保留分窗，避免整片塌成一条
    assert len(merged) == 3
    assert merged[0].text == "你好"
    assert merged[1].text == "世界"
    assert merged[2].text == "下一段"


def test_merge_adjacent_dedupes_same_text() -> None:
    segs = [
        TranscriptSegment(start=0.0, end=1.0, text="同一句"),
        TranscriptSegment(start=1.0, end=2.0, text="同一句"),
    ]
    merged = _merge_adjacent(segs)
    assert len(merged) == 1
    assert merged[0].text == "同一句"
    assert merged[0].end == pytest.approx(2.0)


def test_inject_anchor_answer_cues_when_vlm_lacks_phrase() -> None:
    from pipeline.prep_transcript_vlm import _inject_anchor_answer_cues
    from pipeline.stage0_preprocess import locate_answer_timestamp

    vlm = [
        TranscriptSegment(start=0.0, end=10.0, text="先看地貌和桥梁。"),
        TranscriptSegment(start=10.0, end=20.0, text="横跨黄河的桥不多。"),
    ]
    anchor = [
        TranscriptSegment(start=0.0, end=5.0, text="开场白。"),
        TranscriptSegment(
            start=18.0, end=22.0, text="答案就是郑州黄河文化公园。"
        ),
    ]
    with pytest.raises(ValueError):
        locate_answer_timestamp(vlm)
    fixed = _inject_anchor_answer_cues(vlm, anchor)
    assert locate_answer_timestamp(fixed) == pytest.approx(18.0)
    assert any("答案就是" in s.text for s in fixed)


def test_prep_transcript_vlm_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "out.json"
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        "pipeline.prep_transcript_vlm._video_duration_sec",
        lambda _p: 60.0,
    )
    monkeypatch.setattr(
        "pipeline.prep_transcript_vlm.extract_keyframes",
        lambda *_a, **_k: [str(tmp_path / "f.jpg")],
    )
    (tmp_path / "f.jpg").write_bytes(b"img")

    def _fake(
        prompt: str,
        response_model: type,
        images: list[str] | None = None,
        **_k: Any,
    ) -> Any:
        calls.append({"prompt": prompt, "images": images})
        assert response_model is _WindowSpeech
        assert images
        return _WindowSpeech(text=f"窗转录{len(calls)}")

    monkeypatch.setattr("pipeline.prep_transcript_vlm.call_structured", _fake)

    segs = prep_transcript_vlm(
        str(video),
        output_path=out,
        window_sec=30.0,
        max_frames=2,
    )
    assert out.is_file()
    assert len(segs) >= 1
    assert calls
    assert "分窗口播转录" in calls[0]["prompt"]
    assert "不得编造" in calls[0]["prompt"]
