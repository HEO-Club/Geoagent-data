"""编排 e2e（全 mock）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline import orchestrator
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.transcript import TranscriptSegment


def test_run_one_video_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "e2e.mp4"
    video.write_bytes(b"vid")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setenv("TOOL_TREES_PATH", str(tmp_path / "tool_trees.json"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    def fake_stage1(video_path: str, **kwargs: Any) -> list[TranscriptSegment]:
        segs = [TranscriptSegment(start=0, end=1, text="南方竹子")]
        inter = tmp_path / "intermediate" / "e2e"
        inter.mkdir(parents=True, exist_ok=True)
        from pipeline.schemas.transcript import Stage1Result

        (inter / "stage1_transcript.json").write_text(
            Stage1Result(
                video_id="e2e", video_path=video_path, segments=segs
            ).model_dump_json(),
            encoding="utf-8",
        )
        return segs

    def fake_stage2(
        video_path: str,
        transcript: list[TranscriptSegment],
        **kwargs: Any,
    ) -> FreeFormTrajectory:
        traj = FreeFormTrajectory(
            source_video="e2e",
            steps=[
                FreeFormStep(
                    thought="see bamboo",
                    tool="plant_check",
                    params={},
                    observation={"hint": "south"},
                )
            ],
        )
        path = tmp_path / "intermediate" / "e2e" / "stage2_freeform_tao.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traj.model_dump_json(), encoding="utf-8")
        return traj

    monkeypatch.setattr(orchestrator, "run_stage1", fake_stage1)
    monkeypatch.setattr(orchestrator, "run_stage2", fake_stage2)

    entry = orchestrator.run_one_video(
        str(video),
        image_path="img.jpg",
        stage3_matcher=lambda _n, _f: None,
    )
    assert entry.source_video == "e2e"
    assert any(m.role == "assistant" for m in entry.messages)
    manifest = orchestrator.load_manifest("e2e")
    assert manifest.stages["stage1"] == "done"
    assert manifest.stages["stage2"] == "done"
    assert manifest.stages["stage3"] == "done"

    n = orchestrator.merge_jsonl_shards(tmp_path / "output")
    assert n >= 1
    final = tmp_path / "output" / "geolocate_agent.jsonl"
    assert final.is_file()
