"""编排 e2e（全 mock）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline import orchestrator
from pipeline.schemas.audit import (
    AuditDecision,
    AuditSplitResult,
    GeoTaskSpec,
    TargetKind,
)
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

    def fake_audit(
        video_path: str,
        transcript: list[TranscriptSegment],
        **kwargs: Any,
    ) -> AuditSplitResult:
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"jpg")
        result = AuditSplitResult(
            video_id="e2e",
            decision=AuditDecision.accept,
            reason="单任务",
            tasks=[
                GeoTaskSpec(
                    task_id="e2e__t01",
                    time_start=0.0,
                    time_end=1.0,
                    target_kind=TargetKind.video_derived,
                    keyframe_timestamps=[0.5],
                    image_paths=[str(frame)],
                    segment_start_idx=0,
                    segment_end_idx=0,
                )
            ],
        )
        out = Path(kwargs["out_path"]) if kwargs.get("out_path") else (
            tmp_path / "intermediate" / "e2e" / "stage_audit_split.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

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
        path = Path(kwargs["out_path"]) if kwargs.get("out_path") else (
            tmp_path / "intermediate" / "e2e" / "stage2_freeform_tao.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traj.model_dump_json(), encoding="utf-8")
        return traj

    monkeypatch.setattr(orchestrator, "run_stage1", fake_stage1)
    monkeypatch.setattr(orchestrator, "run_audit_split", fake_audit)
    monkeypatch.setattr(orchestrator, "run_stage2", fake_stage2)

    entries = orchestrator.run_one_video(
        str(video),
        image_path="img.jpg",
        stage3_matcher=lambda _n, _f: None,
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry.source_video == "e2e"
    assert any(m.role == "assistant" for m in entry.messages)
    assert "[Image:" in entry.messages[1].content
    manifest = orchestrator.load_manifest("e2e")
    assert manifest.stages["stage1"] == "done"
    assert manifest.stages["stage_audit_split"] == "done"
    assert manifest.stages["task:e2e__t01:stage2"] == "done"
    assert manifest.stages["task:e2e__t01:stage3"] == "done"

    n = orchestrator.merge_jsonl_shards(tmp_path / "output")
    assert n >= 1
    final = tmp_path / "output" / "geolocate_agent.jsonl"
    assert final.is_file()
    shard = tmp_path / "output" / "shards" / "e2e__t01.jsonl"
    assert shard.is_file()


def test_run_one_video_reject_skips_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "skip.mp4"
    video.write_bytes(b"vid")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("TOOL_TREES_PATH", str(tmp_path / "tool_trees.json"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    def fake_stage1(video_path: str, **kwargs: Any) -> list[TranscriptSegment]:
        segs = [TranscriptSegment(start=0, end=1, text="科普")]
        inter = tmp_path / "intermediate" / "skip"
        inter.mkdir(parents=True, exist_ok=True)
        from pipeline.schemas.transcript import Stage1Result

        (inter / "stage1_transcript.json").write_text(
            Stage1Result(
                video_id="skip", video_path=video_path, segments=segs
            ).model_dump_json(),
            encoding="utf-8",
        )
        return segs

    def fake_audit(
        video_path: str,
        transcript: list[TranscriptSegment],
        **kwargs: Any,
    ) -> AuditSplitResult:
        result = AuditSplitResult(
            video_id="skip",
            decision=AuditDecision.reject,
            reason="非定位",
            tasks=[],
        )
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    called = {"stage2": 0}

    def fake_stage2(*_a: Any, **_k: Any) -> FreeFormTrajectory:
        called["stage2"] += 1
        raise AssertionError("reject 后不应调用 stage2")

    monkeypatch.setattr(orchestrator, "run_stage1", fake_stage1)
    monkeypatch.setattr(orchestrator, "run_audit_split", fake_audit)
    monkeypatch.setattr(orchestrator, "run_stage2", fake_stage2)

    entries = orchestrator.run_one_video(str(video), skip_completed=False)
    assert entries == []
    assert called["stage2"] == 0
    manifest = orchestrator.load_manifest("skip")
    assert manifest.stages["stage_audit_split"] == "rejected"
