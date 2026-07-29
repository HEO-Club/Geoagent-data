"""阶段1 测试（mock VLM / 抽帧）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage1_transcript import run as stage1


class _Speech:
    def __init__(self, text: str) -> None:
        self.text = text


def test_build_time_windows_fixed() -> None:
    wins = stage1.build_time_windows(100.0, window_sec=30.0)
    assert wins[0] == (0.0, 30.0)
    assert wins[-1][1] == 100.0
    assert len(wins) == 4


def test_merge_adjacent() -> None:
    segs = [
        TranscriptSegment(start=0, end=1, text="a"),
        TranscriptSegment(start=1.1, end=2, text=""),
        TranscriptSegment(start=2, end=3, text="b"),
    ]
    merged = stage1._merge_adjacent(segs)
    assert [s.text for s in merged] == ["a", "b"]


def test_run_stage1_with_mocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")

    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    monkeypatch.setattr(stage1, "video_duration_sec", lambda _p: 60.0)

    def fake_extract(*_a: Any, **_k: Any) -> list[str]:
        img = tmp_path / "f.jpg"
        img.write_bytes(b"x")
        return [str(img)]

    monkeypatch.setattr(stage1, "extract_keyframes_range", fake_extract)

    calls = {"n": 0}

    def fake_call(prompt: str, response_model: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return _Speech(text=f"window-{calls['n']}")

    monkeypatch.setattr(stage1, "call_structured", fake_call)

    segs = stage1.run_stage1(str(video), window_sec=30.0, max_frames=2)
    assert len(segs) == 2
    assert segs[0].text.startswith("window-")

    out = tmp_path / "transcripts" / "demo.json"
    assert out.is_file()
    inter = tmp_path / "intermediate" / "demo" / "stage1_transcript.json"
    assert inter.is_file()
    data = json.loads(inter.read_text(encoding="utf-8"))
    assert data["video_id"] == "demo"
    assert len(data["segments"]) == 2
