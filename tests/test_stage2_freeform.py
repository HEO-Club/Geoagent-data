"""阶段2 测试（mock LLM）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao import run as stage2


def test_run_stage2_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    monkeypatch.setattr(stage2, "video_duration_sec", lambda _p: 10.0)
    monkeypatch.setattr(stage2, "extract_keyframes", lambda *_a, **_k: [])

    class _Step:
        thought = "observe vegetation"
        tool = "inspect_plants"
        params = {"region": "center"}
        observation = {"species_hint": "bamboo"}

    class _Result:
        steps = [_Step()]
        notes = "dropped intro"

    monkeypatch.setattr(stage2, "call_structured", lambda *a, **k: _Result())

    transcript = [
        TranscriptSegment(start=0, end=5, text="这里有竹子"),
        TranscriptSegment(start=5, end=10, text="可能在南方"),
    ]
    traj = stage2.run_stage2(str(video), transcript)
    assert isinstance(traj, FreeFormTrajectory)
    assert traj.source_video == "vid"
    assert len(traj.steps) == 1
    assert traj.steps[0].tool == "inspect_plants"
    path = tmp_path / "intermediate" / "vid" / "stage2_freeform_tao.json"
    assert path.is_file()
    loaded = FreeFormTrajectory.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.notes == "dropped intro"
