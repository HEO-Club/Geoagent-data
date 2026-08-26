"""编排 e2e（全 mock）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline import orchestrator
from pipeline.schemas.audit import (
    AuditDecision,
    AuditSplitResult,
    GeoTaskSpec,
    TaskStatus,
    TargetKind,
)
from pipeline.schemas.confidence import ConfidenceJudgeDraft
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment


def _dummy_stage4_judge(**_kwargs: Any) -> ConfidenceJudgeDraft:
    """e2e 注入：高分、无硬门槛。"""
    return ConfidenceJudgeDraft(
        evidence_grounding=0.85,
        final_answer_support=0.8,
        tool_param_correctness=0.8,
        logical_consistency=0.85,
        input_quality_alignment=0.8,
        notes="e2e-mock",
    )


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
                ),
                GeoTaskSpec(
                    task_id="e2e__t02",
                    time_start=1.0,
                    time_end=2.0,
                    target_kind=TargetKind.still_image,
                    status=TaskStatus.rejected,
                    status_reason="答案不明确",
                ),
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
                ),
                FreeFormStep(
                    event_type="final",
                    thought="conclude",
                    tool="final_answer",
                    params={"location": "南方"},
                    observation=None,
                ),
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
        stage4_judge=_dummy_stage4_judge,
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry.source_video == "e2e"
    assert any(m.role == "assistant" for m in entry.messages)
    assert "[Image:" in entry.messages[1].content
    assert entry.quality_score is not None
    assert entry.quality_score >= 0.5
    manifest = orchestrator.load_manifest("e2e")
    assert manifest.stages["stage1"] == "done"
    assert manifest.stages["stage_audit_split"] == "done"
    assert manifest.stages["task:e2e__t01:stage2"] == "done"
    assert manifest.stages["task:e2e__t01:stage3"] == "done"
    assert manifest.stages["task:e2e__t01:stage4"] == "done"
    assert manifest.stages["task:e2e__t02:stage2"] == "rejected"
    assert manifest.stages["task:e2e__t02:stage3"] == "rejected"
    assert manifest.stages["task:e2e__t02:stage4"] == "rejected"
    task_dir = tmp_path / "intermediate" / "e2e" / "tasks" / "e2e__t01"
    assert (task_dir / "stage2_freeform_tao.json").is_file()
    assert (task_dir / "stage3_trajectory.json").is_file()
    assert (task_dir / "stage4_confidence.json").is_file()
    rejected_dir = (
        tmp_path / "intermediate" / "e2e" / "tasks" / "e2e__t02"
    )
    assert not (rejected_dir / "stage2_freeform_tao.json").exists()

    n = orchestrator.merge_jsonl_shards(tmp_path / "output")
    assert n >= 1
    final = tmp_path / "output" / "geolocate_agent.jsonl"
    assert final.is_file()
    shard = tmp_path / "output" / "shards" / "e2e__t01.jsonl"
    assert shard.is_file()
    from pipeline.schemas.dataset import DatasetEntry

    shard_entry = DatasetEntry.model_validate_json(
        shard.read_text(encoding="utf-8").splitlines()[0]
    )
    assert shard_entry.quality_score == pytest.approx(entry.quality_score)


def test_run_one_video_needs_review_continues_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """选图 needs_review 仍跑完 Stage 2–4 并写出 JSONL。"""
    video = tmp_path / "review.mp4"
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
        inter = tmp_path / "intermediate" / "review"
        inter.mkdir(parents=True, exist_ok=True)
        from pipeline.schemas.transcript import Stage1Result

        (inter / "stage1_transcript.json").write_text(
            Stage1Result(
                video_id="review", video_path=video_path, segments=segs
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
            video_id="review",
            decision=AuditDecision.accept,
            reason="地理定位视频",
            tasks=[
                GeoTaskSpec(
                    task_id="review__t01",
                    time_start=0.0,
                    time_end=1.0,
                    target_kind=TargetKind.still_image,
                    keyframe_timestamps=[0.5],
                    image_paths=[str(frame)],
                    status=TaskStatus.needs_review,
                    status_reason="选中帧仍含讲解覆盖、界面残留或质量低于阈值",
                    image_selection_note=(
                        "选图质量等级=needs_review\n选中张数=1\n"
                        "选图原因: 选中帧仍含讲解覆盖、界面残留或质量低于阈值"
                    ),
                    segment_start_idx=0,
                    segment_end_idx=0,
                ),
            ],
        )
        out = Path(kwargs["out_path"]) if kwargs.get("out_path") else (
            tmp_path / "intermediate" / "review" / "stage_audit_split.json"
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
            source_video="review",
            steps=[
                FreeFormStep(
                    thought="see bamboo",
                    tool="plant_check",
                    params={},
                    observation={"hint": "south"},
                ),
                FreeFormStep(
                    event_type="final",
                    thought="conclude",
                    tool="final_answer",
                    params={"location": "南方"},
                    observation=None,
                ),
            ],
        )
        path = Path(kwargs["out_path"]) if kwargs.get("out_path") else (
            tmp_path / "intermediate" / "review" / "stage2_freeform_tao.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
        return traj

    monkeypatch.setattr(orchestrator, "run_stage1", fake_stage1)
    monkeypatch.setattr(orchestrator, "run_audit_split", fake_audit)
    monkeypatch.setattr(orchestrator, "run_stage2", fake_stage2)

    entries = orchestrator.run_one_video(
        str(video),
        skip_completed=False,
        stage3_matcher=lambda _n, _f: None,
        stage4_judge=_dummy_stage4_judge,
    )
    assert len(entries) == 1
    manifest = orchestrator.load_manifest("review")
    assert manifest.stages["task:review__t01:stage2"] == "done"
    assert manifest.stages["task:review__t01:stage3"] == "done"
    assert manifest.stages["task:review__t01:stage4"] == "done"
    task_dir = tmp_path / "intermediate" / "review" / "tasks" / "review__t01"
    assert (task_dir / "stage2_freeform_tao.json").is_file()
    assert (task_dir / "stage3_trajectory.json").is_file()
    assert (task_dir / "stage4_confidence.json").is_file()
    conf = json.loads((task_dir / "stage4_confidence.json").read_text(encoding="utf-8"))
    assert "选图质量等级=needs_review" in conf["notes"]
    shard = tmp_path / "output" / "shards" / "review__t01.jsonl"
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


def test_run_one_video_continues_without_trajectory_image_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不再做轨迹–选图一致性门禁：Stage 2 后直接跑 Stage 3/4 并写 JSONL。"""
    video = tmp_path / "noconflict.mp4"
    video.write_bytes(b"vid")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setenv("TOOL_TREES_PATH", str(tmp_path / "tool_trees.json"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")

    def fake_stage1(video_path: str, **kwargs: Any) -> list[TranscriptSegment]:
        segs = [TranscriptSegment(start=0, end=1, text="红瓦屋顶")]
        inter = tmp_path / "intermediate" / "noconflict"
        inter.mkdir(parents=True, exist_ok=True)
        from pipeline.schemas.transcript import Stage1Result

        (inter / "stage1_transcript.json").write_text(
            Stage1Result(
                video_id="noconflict", video_path=video_path, segments=segs
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
            video_id="noconflict",
            decision=AuditDecision.accept,
            reason="单任务",
            tasks=[
                GeoTaskSpec(
                    task_id="noconflict__t01",
                    time_start=0.0,
                    time_end=1.0,
                    target_kind=TargetKind.still_image,
                    keyframe_timestamps=[0.5],
                    image_paths=[str(frame)],
                    visual_evidence_brief="红瓦屋顶与宽阔水面",
                    segment_start_idx=0,
                    segment_end_idx=0,
                ),
            ],
        )
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def fake_stage2(
        video_path: str,
        transcript: list[TranscriptSegment],
        **kwargs: Any,
    ) -> FreeFormTrajectory:
        traj = FreeFormTrajectory(
            source_video="noconflict",
            steps=[
                FreeFormStep(
                    event_type="reasoning",
                    thought="图中是红瓦屋顶临水场景",
                ),
                FreeFormStep(
                    event_type="final",
                    thought="结论",
                    tool="final_answer",
                    params={"location": "某地"},
                    observation=None,
                ),
            ],
        )
        path = Path(kwargs["out_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traj.model_dump_json(), encoding="utf-8")
        return traj

    monkeypatch.setattr(orchestrator, "run_stage1", fake_stage1)
    monkeypatch.setattr(orchestrator, "run_audit_split", fake_audit)
    monkeypatch.setattr(orchestrator, "run_stage2", fake_stage2)

    entries = orchestrator.run_one_video(
        str(video),
        skip_completed=False,
        stage3_matcher=lambda _n, _f: None,
        stage4_judge=_dummy_stage4_judge,
    )
    assert len(entries) == 1
    task_dir = (
        tmp_path / "intermediate" / "noconflict" / "tasks" / "noconflict__t01"
    )
    assert (task_dir / "stage2_freeform_tao.json").is_file()
    assert (task_dir / "stage3_trajectory.json").is_file()
    assert (task_dir / "stage4_confidence.json").is_file()
    assert not (task_dir / "image_trajectory_consistency.json").exists()
    manifest = orchestrator.load_manifest("noconflict")
    assert manifest.stages["task:noconflict__t01:stage2"] == "done"
    assert manifest.stages["task:noconflict__t01:stage3"] == "done"
    assert manifest.stages["task:noconflict__t01:stage4"] == "done"
    shard = tmp_path / "output" / "shards" / "noconflict__t01.jsonl"
    assert shard.is_file()


def test_run_one_video_empty_images_still_writes_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """image_paths 为空的 needs_review 题仍跑完 Stage 2–4 并入库。"""
    video = tmp_path / "emptyimg.mp4"
    video.write_bytes(b"vid")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setenv("TOOL_TREES_PATH", str(tmp_path / "tool_trees.json"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    def fake_stage1(video_path: str, **kwargs: Any) -> list[TranscriptSegment]:
        segs = [TranscriptSegment(start=0, end=1, text="定位题旁白")]
        inter = tmp_path / "intermediate" / "emptyimg"
        inter.mkdir(parents=True, exist_ok=True)
        from pipeline.schemas.transcript import Stage1Result

        (inter / "stage1_transcript.json").write_text(
            Stage1Result(
                video_id="emptyimg", video_path=video_path, segments=segs
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
            video_id="emptyimg",
            decision=AuditDecision.accept,
            reason="地理定位视频",
            tasks=[
                GeoTaskSpec(
                    task_id="emptyimg__t01",
                    time_start=0.0,
                    time_end=1.0,
                    target_kind=TargetKind.still_image,
                    image_paths=[],
                    status=TaskStatus.needs_review,
                    status_reason="出示窗内未找到干净待定位原图",
                    image_selection_note=(
                        "选图质量等级=needs_review\n选中张数=0\n"
                        "选图原因: 出示窗内未找到干净待定位原图\n选中帧: 无"
                    ),
                    segment_start_idx=0,
                    segment_end_idx=0,
                ),
            ],
        )
        out = Path(kwargs["out_path"]) if kwargs.get("out_path") else (
            tmp_path / "intermediate" / "emptyimg" / "stage_audit_split.json"
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
            source_video="emptyimg",
            steps=[
                FreeFormStep(
                    event_type="reasoning",
                    thought="根据旁白推理地点",
                ),
                FreeFormStep(
                    event_type="final",
                    thought="conclude",
                    tool="final_answer",
                    params={"location": "某市"},
                    observation=None,
                ),
            ],
        )
        path = Path(kwargs["out_path"]) if kwargs.get("out_path") else (
            tmp_path
            / "intermediate"
            / "emptyimg"
            / "tasks"
            / "emptyimg__t01"
            / "stage2_freeform_tao.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
        return traj

    monkeypatch.setattr(orchestrator, "run_stage1", fake_stage1)
    monkeypatch.setattr(orchestrator, "run_audit_split", fake_audit)
    monkeypatch.setattr(orchestrator, "run_stage2", fake_stage2)

    entries = orchestrator.run_one_video(
        str(video),
        skip_completed=False,
        stage3_matcher=lambda _n, _f: None,
        stage4_judge=_dummy_stage4_judge,
    )
    assert len(entries) == 1
    assert entries[0].messages[1].content == "Locate the place shown in the image."
    assert "[Image:" not in entries[0].messages[1].content
    traj_path = (
        tmp_path
        / "intermediate"
        / "emptyimg"
        / "tasks"
        / "emptyimg__t01"
        / "stage3_trajectory.json"
    )
    traj = Trajectory.model_validate_json(traj_path.read_text(encoding="utf-8"))
    assert traj.image_paths == []
    shard = tmp_path / "output" / "shards" / "emptyimg__t01.jsonl"
    assert shard.is_file()
