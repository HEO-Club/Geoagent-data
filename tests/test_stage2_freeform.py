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
        thought = (
            "图中可见竹林与湿热植被；当前假设偏华南/西南，"
            "但缺气候区与物种分布交叉验证，因此检索该植被组合的典型分布区。"
        )
        tool = "inspect_plants"
        params = {"region": "center"}
        observation = {"species_hint": "bamboo"}

    class _Result:
        steps = [_Step()]
        notes = None

    captured: dict[str, str] = {}

    def _fake_call(prompt: str, *_a, **_k):  # type: ignore[no-untyped-def]
        captured["prompt"] = prompt
        return _Result()

    monkeypatch.setattr(stage2, "call_structured", _fake_call)

    transcript = [
        TranscriptSegment(start=0, end=5, text="这里有竹子"),
        TranscriptSegment(start=5, end=10, text="可能在南方"),
    ]
    traj = stage2.run_stage2(str(video), transcript)
    assert isinstance(traj, FreeFormTrajectory)
    assert traj.source_video == "vid"
    assert len(traj.steps) == 1
    assert traj.steps[0].tool == "inspect_plants"
    assert "字幕" not in traj.steps[0].thought
    assert traj.notes is None

    prompt = captured["prompt"]
    assert "禁止写入产物" in prompt
    assert "假设缺口" in prompt
    assert "为何调用本步 tool" in prompt
    assert "讲解内容参考" in prompt
    assert "notes 可简述删除了哪些无用部分" not in prompt
    assert "字幕（带时间戳）" not in prompt

    path = tmp_path / "intermediate" / "vid" / "stage2_freeform_tao.json"
    assert path.is_file()
    loaded = FreeFormTrajectory.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.notes is None
