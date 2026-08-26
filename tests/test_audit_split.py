"""阶段1.5 审核切分测试（mock LLM / 截帧）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.schemas.audit import (
    AnswerStatus,
    AuditDecision,
    GeoTaskSpec,
    ProcessInterval,
    ProcessRole,
    TaskStatus,
    TargetKind,
)
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage_audit_split import run as audit


def _transcript() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=0.0, end=5.0, text="第一张图在哪里"),
        TranscriptSegment(start=5.0, end=10.0, text="第二张图又在哪"),
    ]


def _isolate_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """把 intermediate / cache / selected 全部指到临时目录。"""
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SELECTED_DIR", str(tmp_path / "selected"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()


def _fake_extract_factory(captured: dict[str, Any] | None = None):
    def fake_extract(
        video_path: str, stamps: list[float], *, out_dir: str
    ) -> list[str]:
        if captured is not None:
            captured.setdefault("calls", []).append(
                {"stamps": list(stamps), "out_dir": out_dir}
            )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        paths = []
        for i, t in enumerate(stamps or [0.0]):
            p = Path(out_dir) / f"t{float(t):.3f}.jpg"
            p.write_bytes(b"jpg")
            paths.append(str(p))
        return paths

    return fake_extract


def _route_call(
    draft: Any,
    *,
    frame_kind: audit.FrameKind | None = None,
    merge_tasks: list[Any] | None = None,
    call_log: list[str] | None = None,
    source_groups: list[int] | None = None,
    visual_evidence_brief: str = "",
    process_intervals: list[Any] | None = None,
):
    """按 schema 分流：审核草稿 / 证据简报 / 过程时间线 / 帧验收 / 源输入归并 / 合并复核。"""

    distinct_sources = (
        source_groups is not None and len({int(g) for g in source_groups}) > 1
    )

    def _fake(prompt: str, schema: Any, **_k: Any) -> Any:  # noqa: ANN401
        name = getattr(schema, "__name__", "")
        if call_log is not None:
            call_log.append(name)
        if name == "_LLMEvidenceBrief":
            return audit._LLMEvidenceBrief(
                visual_evidence_brief=visual_evidence_brief
            )
        if name == "_LLMProcessTimeline":
            return audit._LLMProcessTimeline(
                intervals=list(process_intervals or [])
            )
        if name == "_LLMFrameVerdict":
            kind = frame_kind or audit.FrameKind.target_photo
            return audit._LLMFrameVerdict(
                kind=kind,
                quality_score=0.9,
                answer_leakage=False,
                tutorial_overlay=False,
                clean_source=kind == audit.FrameKind.target_photo,
                evidence_role=(
                    audit.EvidenceRole.problem_input
                    if kind == audit.FrameKind.target_photo
                    else audit.EvidenceRole.process_tool
                ),
                chain_support_score=(
                    0.5 if kind == audit.FrameKind.target_photo else 0.1
                ),
                reason="mock",
            )
        if name == "_LLMContainmentVerdict":
            return audit._LLMContainmentVerdict(
                containment=audit.ContainmentKind.none,
                confidence=0.0,
                reason="mock none",
            )
        if name == "_LLMPhotoRelationVerdict":
            relation = (
                audit.PhotoRelation.different_photo
                if distinct_sources
                else audit.PhotoRelation.same_photo
            )
            return audit._LLMPhotoRelationVerdict(
                relation=relation,
                confidence=0.9,
                reason="mock",
            )
        if name == "_LLMSourceIdentityResult":
            images = _k.get("images") or []
            n = max(1, len(images))
            groups = source_groups
            if groups is None:
                groups = [0] * n
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(
                        index=i,
                        source_group=int(groups[i] if i < len(groups) else 0),
                        reason="mock",
                    )
                    for i in range(n)
                ]
            )
        if name == "_LLMSamePlaceGate":
            tasks = list(getattr(draft, "tasks", []) or [])
            force_same = merge_tasks is not None and len(tasks) >= 2
            pairs: list[audit._SamePlacePair] = []
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    pairs.append(
                        audit._SamePlacePair(
                            task_i=i + 1,
                            task_j=j + 1,
                            same_place_or_chain=force_same,
                            reason="mock",
                        )
                    )
            return audit._LLMSamePlaceGate(pairs=pairs, reason="mock")
        if name == "_LLMTaskMergeResult":

            class _M:
                tasks = list(
                    merge_tasks
                    if merge_tasks is not None
                    else (getattr(draft, "tasks", []) or [])
                )
                reason = "mock merge"

            return _M()
        return draft

    return _fake


def test_run_audit_split_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "reject.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 12.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Draft:
        decision = AuditDecision.reject
        reason = "纯科普无定位目标"
        has_unresolved_target = False
        tasks: list[Any] = []

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    assert result.decision == AuditDecision.reject
    assert result.tasks == []
    assert result.has_unresolved_target is False
    out = tmp_path / "intermediate" / "reject" / "stage_audit_split.json"
    assert out.is_file()
    loaded = audit.load_audit_split(out)
    assert loaded.decision == AuditDecision.reject


def test_force_reject_when_no_unresolved_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """has_unresolved_target=false 且无 tasks 时，即使 decision=accept 也强制 reject。"""
    video = tmp_path / "force.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 12.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Draft:
        decision = AuditDecision.accept
        reason = "误把科普当定位"
        has_unresolved_target = False
        tasks = []

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    assert result.decision == AuditDecision.reject
    assert result.tasks == []
    assert result.has_unresolved_target is False
    assert "has_unresolved_target=false" in result.reason


def test_accept_with_tasks_overrides_false_unresolved_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """decision=accept 且已给 tasks 时，不因 has_unresolved_target=false 整片拒识。"""
    video = tmp_path / "coerce.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="有待定位原图",
        has_unresolved_target=False,
        tasks=[
            audit._LLMGeoTaskDraft(
                time_start=0.0,
                time_end=8.0,
                display_time_start=1.0,
                display_time_end=4.0,
                target_kind=TargetKind.still_image,
                keyframe_timestamps=[2.0],
                answer_status=AnswerStatus.resolved,
                final_location_text="某地",
            )
        ],
        split_confidence=0.9,
    )
    monkeypatch.setattr(
        audit,
        "call_structured",
        _route_call(
            draft,
            frame_kind=audit.FrameKind.target_photo,
            visual_evidence_brief="红瓦屋顶",
        ),
    )
    result = audit.run_audit_split(str(video), _transcript())
    assert result.decision == AuditDecision.accept
    assert result.has_unresolved_target is True
    assert len(result.tasks) == 1


def test_still_image_clamps_to_single_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "single.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_MAX_KEYFRAMES_PER_TASK", "8")

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        audit, "extract_keyframes", _fake_extract_factory(captured)
    )

    class _Task:
        time_start = 2.0
        time_end = 8.0
        display_time_start = 2.0
        display_time_end = 8.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [3.0, 6.0, 7.0]
        multi_target_images = False
        segment_start_idx = 0
        segment_end_idx = 0
        task_summary = "第一题"
        expected_image_count = 1

    class _Draft:
        decision = AuditDecision.accept
        reason = "单定位任务"
        has_unresolved_target = True
        tasks = [_Task()]

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    assert result.decision == AuditDecision.accept
    task = result.tasks[0]
    assert task.task_id == "single__t01"
    assert len(task.keyframe_timestamps) == 1
    assert len(task.image_paths) == 1
    assert "single__t01_" in Path(task.image_paths[0]).name
    assert "selected" in Path(task.image_paths[0]).as_posix()


def test_multi_target_images_allows_multiple_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """两段出示被非目标帧隔开，且源输入判定为不同 → 可留 2 张。"""
    video = tmp_path / "multiimg.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_MAX_KEYFRAMES_PER_TASK", "4")
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "1.0")

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 120.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 0.0
        time_end = 100.0
        display_time_start = 0.0
        display_time_end = 5.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [0.0, 5.0]
        multi_target_images = True
        segment_start_idx = 0
        segment_end_idx = 1
        task_summary = "同题两图"
        answer_status = AnswerStatus.resolved
        final_location_text = "同一地点"

    class _Draft:
        decision = AuditDecision.accept
        reason = "同题多图"
        has_unresolved_target = True
        tasks = [_Task()]
        split_confidence = 0.95
        needs_split_review = False

    kinds_by_stamp = {
        0.0: audit.FrameKind.target_photo,
        1.0: audit.FrameKind.teaching_ui,
        2.0: audit.FrameKind.teaching_ui,
        3.0: audit.FrameKind.teaching_ui,
        4.0: audit.FrameKind.teaching_ui,
        5.0: audit.FrameKind.target_photo,
    }

    def _fake(prompt: str, schema: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        name = getattr(schema, "__name__", "")
        if name == "_LLMFrameVerdict":
            stamp = 0.0
            for line in prompt.splitlines():
                if line.startswith("候选时间："):
                    stamp = float(line.split("：", 1)[1].rstrip("s"))
                    break
            kind = kinds_by_stamp.get(round(stamp, 1), audit.FrameKind.teaching_ui)
            return audit._LLMFrameVerdict(
                kind=kind,
                quality_score=0.95,
                clean_source=kind == audit.FrameKind.target_photo,
                reason="mock",
            )
        if name == "_LLMContainmentVerdict":
            return audit._LLMContainmentVerdict(
                containment=audit.ContainmentKind.none, confidence=0.0
            )
        if name == "_LLMPhotoRelationVerdict":
            return audit._LLMPhotoRelationVerdict(
                relation=audit.PhotoRelation.different_photo,
                confidence=0.9,
                reason="不同原图",
            )
        return _Draft()

    monkeypatch.setattr(audit, "call_structured", _fake)

    result = audit.run_audit_split(str(video), _transcript())
    assert len(result.tasks) == 1
    assert len(result.tasks[0].image_paths) == 2
    assert result.tasks[0].multi_target_images is True
    assert result.tasks[0].expected_image_count == 2
    assert all(0.0 <= t <= 5.0 for t in result.tasks[0].keyframe_timestamps)


def test_teaching_ui_frame_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """teaching_ui 帧被剔除；仅保留 target_photo。"""
    video = tmp_path / "ui.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 0.0
        time_end = 10.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [1.0, 5.0]
        multi_target_images = False
        segment_start_idx = 0
        segment_end_idx = 0
        task_summary = "单图"

    class _Draft:
        decision = AuditDecision.accept
        reason = "一题"
        has_unresolved_target = True
        tasks = [_Task()]

    kinds = iter(
        [
            audit.FrameKind.teaching_ui,
            audit.FrameKind.target_photo,
        ]
    )

    def _fake(prompt: str, schema: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        name = getattr(schema, "__name__", "")
        if name == "_LLMFrameVerdict":

            class _V:
                kind = next(kinds, audit.FrameKind.target_photo)
                quality_score = 0.9
                answer_leakage = False
                tutorial_overlay = False
                clean_source = kind == audit.FrameKind.target_photo
                reason = "mock"

            return _V()
        if name == "_LLMSourceIdentityResult":
            images = kwargs.get("images") or []
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="mock")
                    for i in range(max(1, len(images)))
                ]
            )
        return _Draft()

    monkeypatch.setattr(audit, "call_structured", _fake)

    result = audit.run_audit_split(str(video), _transcript())
    assert len(result.tasks[0].image_paths) == 1


def test_multi_target_quota_no_longer_forces_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模型即便标 multi/expected=2，只找到 1 个合格源输入也可 accepted。"""
    video = tmp_path / "badmulti.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 120.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 0.0
        time_end = 100.0
        display_time_start = 0.0
        display_time_end = 5.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [1.0, 80.0]
        multi_target_images = True
        segment_start_idx = 0
        segment_end_idx = 1
        task_summary = "同题两图"
        expected_image_count = 2
        answer_status = AnswerStatus.resolved
        final_location_text = "地点"

    class _Draft:
        decision = AuditDecision.accept
        reason = "同题多图"
        has_unresolved_target = True
        tasks = [_Task()]
        split_confidence = 0.95
        needs_split_review = False

    kinds = iter(
        [
            audit.FrameKind.target_photo,
            audit.FrameKind.teaching_ui,
        ]
    )
    call_log: list[str] = []

    def _fake(prompt: str, schema: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        name = getattr(schema, "__name__", "")
        call_log.append(name)
        if name == "_LLMFrameVerdict":
            kind = next(kinds, audit.FrameKind.teaching_ui)
            return audit._LLMFrameVerdict(
                kind=kind,
                quality_score=0.95,
                clean_source=kind == audit.FrameKind.target_photo,
                reason="mock",
            )
        if name == "_LLMSourceIdentityResult":
            images = kwargs.get("images") or []
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="同组")
                    for i in range(len(images) or 1)
                ]
            )
        return _Draft()

    monkeypatch.setattr(audit, "call_structured", _fake)

    result = audit.run_audit_split(str(video), _transcript())
    assert result.tasks[0].status == TaskStatus.accepted
    assert len(result.tasks[0].image_paths) == 1
    assert result.tasks[0].expected_image_count == 1
    assert "预计" not in result.tasks[0].status_reason
    assert "_LLMKeyframeRetry" not in call_log


def test_video_derived_collapses_continuous_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """连续实拍多帧默认折成 1 个现场代表，不按簇数膨胀。"""
    video = tmp_path / "vder.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 30.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 5.0
        time_end = 10.0
        display_time_start = 5.0
        display_time_end = 10.0
        target_kind = TargetKind.video_derived
        keyframe_timestamps = [6.0, 8.0]
        multi_target_images = False
        segment_start_idx = 1
        segment_end_idx = 1
        task_summary = "场景"
        expected_image_count = 2
        answer_status = AnswerStatus.resolved
        final_location_text = "现场地点"

    class _Draft:
        decision = AuditDecision.accept
        reason = "视频场景定位"
        has_unresolved_target = True
        tasks = [_Task()]
        split_confidence = 0.95
        needs_split_review = False

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    assert len(result.tasks[0].image_paths) == 1
    assert result.tasks[0].multi_target_images is False
    assert result.tasks[0].expected_image_count == 1


def test_extract_failure_marks_only_task_for_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "nofallback.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 12.0)

    def extract_or_fail(
        video_path: str, stamps: list[float], *, out_dir: str
    ) -> list[str]:
        if "audit_sparse" in out_dir.replace("\\", "/"):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            p = Path(out_dir) / "sparse.jpg"
            p.write_bytes(b"jpg")
            return [str(p)]
        raise RuntimeError("task extract boom")

    monkeypatch.setattr(audit, "extract_keyframes", extract_or_fail)

    class _Task:
        time_start = 0.0
        time_end = 5.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [1.0]
        multi_target_images = False
        segment_start_idx = 0
        segment_end_idx = 0
        task_summary = "题"

    class _Draft:
        decision = AuditDecision.accept
        reason = "有目标"
        has_unresolved_target = True
        tasks = [_Task()]

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    assert result.tasks[0].status == TaskStatus.needs_review
    assert result.tasks[0].image_paths == []
    assert "未找到" in result.tasks[0].status_reason


def test_run_audit_split_multi_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "multi.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 30.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _T1:
        time_start = 0.0
        time_end = 5.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [1.0]
        multi_target_images = False
        segment_start_idx = 0
        segment_end_idx = 0
        task_summary = "题1"

    class _T2:
        time_start = 5.0
        time_end = 10.0
        display_time_start = 5.0
        display_time_end = 10.0
        target_kind = TargetKind.video_derived
        keyframe_timestamps = [6.0, 8.0]
        multi_target_images = False
        segment_start_idx = 1
        segment_end_idx = 1
        task_summary = "题2"
        expected_image_count = 2

    class _Draft:
        decision = AuditDecision.accept
        reason = "两题"
        has_unresolved_target = True
        tasks = [_T1(), _T2()]

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    assert [t.task_id for t in result.tasks] == ["multi__t01", "multi__t02"]
    assert result.tasks[1].target_kind == TargetKind.video_derived
    assert len(result.tasks[1].image_paths) == 1

    sliced = audit.slice_transcript_for_task(_transcript(), result.tasks[1])
    assert len(sliced) == 1
    assert "第二张" in sliced[0].text


def test_time_slice_excludes_segments_that_only_touch_task_boundary() -> None:
    transcript = [
        TranscriptSegment(start=30.0, end=60.0, text="前一题"),
        TranscriptSegment(start=60.0, end=90.0, text="本题上半段"),
        TranscriptSegment(start=90.0, end=120.0, text="本题下半段"),
        TranscriptSegment(start=120.0, end=150.0, text="后一题"),
    ]
    task = GeoTaskSpec(
        task_id="video__t03",
        time_start=60.0,
        time_end=120.0,
        target_kind=TargetKind.still_image,
    )
    sliced = audit.slice_transcript_for_task(transcript, task)
    assert [segment.text for segment in sliced] == ["本题上半段", "本题下半段"]


def test_frame_verify_hint_is_principle_based() -> None:
    """验收提示用过程角色原则：实拍输入 vs 工具/核验，不靠品类清单。"""
    hint = audit.FRAME_VERIFY_HINT
    assert "定位输入" in hint or "待定位" in hint
    assert "target_photo" in hint
    assert "teaching_ui" in hint
    assert "实拍" in hint
    assert "工具" in hint or "核验" in hint
    assert "箭头" in hint or "字幕" in hint
    assert "建筑外观" in hint
    assert "黑屏" in hint
    assert "不得因源实拍上有方位字" in hint or "不得因源实拍上有" in hint
    assert "evidence_role" in hint
    assert "problem_input" in hint
    assert "unused_broll" in hint
    assert "chain_support_score" in hint
    assert "卫星" not in hint
    assert "谷歌" not in hint


def test_audit_system_hint_lists_all_localization_inputs() -> None:
    hint = audit.AUDIT_SYSTEM_HINT
    assert "定位输入" in hint or "实拍" in hint
    assert "同一最终地点" in hint
    assert "出示粗窗" in hint or "display_time" in hint
    assert "蒸馏窗" in hint
    assert "过程角色" in hint
    assert "工具" in hint
    assert "揭晓" in hint
    assert "整条答案链" in hint
    assert "预定" in hint
    assert "最小源输入" in hint
    assert "后续人工解出" in hint
    assert "reject 理由" in hint
    assert "连续实拍" in hint or "连续现场" in hint
    assert "video_derived" in hint
    assert "题面明确" in hint
    assert "已知线索" in hint
    assert "卫星" not in hint
    assert "谷歌" not in hint


def test_display_window_limits_dense_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """蒸馏窗很长时，密采样只落在出示窗内。"""
    video = tmp_path / "longwin.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "1.0")
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 300.0)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        audit, "extract_keyframes", _fake_extract_factory(captured)
    )

    task = audit._LLMGeoTaskDraft(
        time_start=0.0,
        time_end=280.0,
        display_time_start=20.0,
        display_time_end=35.0,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[200.0],
        multi_target_images=False,
        expected_image_count=1,
        task_summary="单图",
        answer_status=AnswerStatus.resolved,
        final_location_text="目标地点",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="单题",
        has_unresolved_target=True,
        tasks=[task],
        split_confidence=0.95,
    )
    monkeypatch.setattr(audit, "call_structured", _route_call(draft))
    result = audit.run_audit_split(str(video), _transcript())
    task_calls = [
        c
        for c in captured.get("calls", [])
        if "audit_candidates" in c["out_dir"].replace("\\", "/")
    ]
    all_stamps = [t for c in task_calls for t in c["stamps"]]
    assert all_stamps
    assert all(20.0 <= t <= 35.0 for t in all_stamps)
    assert all(t < 100.0 for t in result.tasks[0].keyframe_timestamps)


def test_wrong_model_stamp_still_finds_display_window_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模型精确戳错误时，仍能靠出示窗密采样选到窗内帧。"""
    video = tmp_path / "wrongstamp.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 60.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())
    task = audit._LLMGeoTaskDraft(
        time_start=0.0,
        time_end=60.0,
        display_time_start=10.0,
        display_time_end=18.0,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[55.0],
        multi_target_images=False,
        expected_image_count=1,
        task_summary="单图",
        answer_status=AnswerStatus.resolved,
        final_location_text="目标地点",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="单题",
        has_unresolved_target=True,
        tasks=[task],
        split_confidence=0.95,
    )
    monkeypatch.setattr(audit, "call_structured", _route_call(draft))
    result = audit.run_audit_split(str(video), _transcript())
    assert result.tasks[0].status == TaskStatus.accepted
    assert len(result.tasks[0].keyframe_timestamps) == 1
    assert 10.0 <= result.tasks[0].keyframe_timestamps[0] <= 18.0


def test_vlm_verify_budget_is_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """廉价过滤后 VLM 验收次数不超过配置上限。"""
    video = tmp_path / "vlmcap.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_MAX_VLM_FRAME_VERIFIES", "3")
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "1.0")
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 60.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())
    call_log: list[str] = []
    task = audit._LLMGeoTaskDraft(
        time_start=0.0,
        time_end=60.0,
        display_time_start=0.0,
        display_time_end=40.0,
        target_kind=TargetKind.still_image,
        multi_target_images=False,
        expected_image_count=1,
        task_summary="单图",
        answer_status=AnswerStatus.resolved,
        final_location_text="目标地点",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="单题",
        has_unresolved_target=True,
        tasks=[task],
        split_confidence=0.95,
    )
    monkeypatch.setattr(
        audit,
        "call_structured",
        _route_call(draft, call_log=call_log),
    )
    audit.run_audit_split(str(video), _transcript())
    assert call_log.count("_LLMFrameVerdict") <= 3


def test_video_derived_continuous_window_keeps_one_representative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """video_derived 连续窗内密采样多帧，默认只留 1 个现场代表。"""
    video = tmp_path / "neardup.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "10.0")

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 100.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 0.0
        time_end = 100.0
        display_time_start = 40.0
        display_time_end = 80.0
        target_kind = TargetKind.video_derived
        keyframe_timestamps: list[float] = []
        multi_target_images = True
        segment_start_idx = 0
        segment_end_idx = 1
        task_summary = "多源镜头"
        expected_image_count = 3
        answer_status = AnswerStatus.resolved
        final_location_text = "现场地点"

    class _Draft:
        decision = AuditDecision.accept
        reason = "同题多输入"
        has_unresolved_target = True
        tasks = [_Task()]
        split_confidence = 0.95
        needs_split_review = False

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    stamps = result.tasks[0].keyframe_timestamps
    assert len(stamps) == 1
    assert all(40.0 <= s <= 80.0 for s in stamps)


def test_fold_presentation_episodes_breaks_on_non_target() -> None:
    frames = [
        audit.KeyframeAssessment(
            timestamp=1.0,
            image_path="a.jpg",
            kind="target_photo",
            quality_score=0.7,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.8,
        ),
        audit.KeyframeAssessment(
            timestamp=2.0,
            image_path="b.jpg",
            kind="target_photo",
            quality_score=0.9,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.9,
        ),
        audit.KeyframeAssessment(
            timestamp=3.0,
            image_path="c.jpg",
            kind="teaching_ui",
            quality_score=0.5,
            evidence_role="process_tool",
        ),
        audit.KeyframeAssessment(
            timestamp=4.0,
            image_path="d.jpg",
            kind="target_photo",
            quality_score=0.8,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.7,
        ),
    ]
    reps = audit._fold_presentation_episodes(frames)
    assert [r.timestamp for r in reps] == [2.0, 4.0]


def test_fold_presentation_episodes_breaks_on_unused_broll() -> None:
    """有证据对齐时 unused_broll 打断连续段，空镜不得当段代表。"""
    frames = [
        audit.KeyframeAssessment(
            timestamp=1.0,
            image_path="a.jpg",
            kind="target_photo",
            quality_score=0.7,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.8,
        ),
        audit.KeyframeAssessment(
            timestamp=2.0,
            image_path="b.jpg",
            kind="target_photo",
            quality_score=0.99,
            clean_source=True,
            evidence_role="unused_broll",
            chain_support_score=0.1,
        ),
        audit.KeyframeAssessment(
            timestamp=3.0,
            image_path="c.jpg",
            kind="target_photo",
            quality_score=0.75,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.9,
        ),
    ]
    reps = audit._fold_presentation_episodes(
        frames, require_evidence_role=True
    )
    assert [r.timestamp for r in reps] == [1.0, 3.0]
    assert all(r.evidence_role == "problem_input" for r in reps)


def test_source_identity_merges_same_group_keeps_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reps = [
        audit.KeyframeAssessment(
            timestamp=1.0,
            image_path="a.jpg",
            kind="target_photo",
            quality_score=0.7,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.8,
        ),
        audit.KeyframeAssessment(
            timestamp=5.0,
            image_path="b.jpg",
            kind="target_photo",
            quality_score=0.95,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.85,
        ),
    ]
    seen: dict[str, str] = {}

    def fake_call(prompt: str, schema: Any, **_k: Any) -> Any:
        name = getattr(schema, "__name__", "")
        seen["prompt"] = prompt
        if name == "_LLMContainmentVerdict":
            return audit._LLMContainmentVerdict(
                containment=audit.ContainmentKind.none,
                confidence=0.0,
            )
        if name == "_LLMPhotoRelationVerdict":
            assert "同一张照片" in prompt or "same_photo" in prompt
            assert "不得用来决定要几个" in prompt or "不决定张数" in prompt
            return audit._LLMPhotoRelationVerdict(
                relation=audit.PhotoRelation.same_photo,
                confidence=0.9,
                reason="同图变体",
            )
        raise AssertionError(f"unexpected schema {name}")

    monkeypatch.setattr(audit, "call_structured", fake_call)
    selected = audit._resolve_source_identity(
        reps,
        target_kind=TargetKind.still_image,
        visual_evidence_brief="红瓦屋顶与宽阔水面",
    )
    assert len(selected) == 1
    assert selected[0].timestamp == 5.0
    assert "红瓦屋顶与宽阔水面" in seen["prompt"]


def test_visual_duplicate_prefilter_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline.stage_audit_split import frame_prefilter as pref

    values = {"a.jpg": 0b1010, "b.jpg": 0b1011, "c.jpg": 0b11110000}
    monkeypatch.setattr(
        pref, "_image_dhash", lambda path: values[Path(path).name]
    )
    duplicate, value = pref.is_near_duplicate(
        "b.jpg", [values["a.jpg"]], max_distance=1
    )
    assert duplicate is True
    assert value == values["b.jpg"]
    duplicate, _ = pref.is_near_duplicate(
        "c.jpg", [values["a.jpg"]], max_distance=1
    )
    assert duplicate is False


def test_merge_same_final_location_into_one_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同最终地点的多草稿须合并为 1 个 task。"""
    video = tmp_path / "sameloc.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 100.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _T1:
        time_start = 0.0
        time_end = 50.0
        target_kind = TargetKind.video_derived
        keyframe_timestamps = [10.0, 40.0]
        multi_target_images = True
        segment_start_idx = 0
        segment_end_idx = 0
        task_summary = "河道线索定位酒店"

    class _T2:
        time_start = 50.0
        time_end = 90.0
        target_kind = TargetKind.video_derived
        keyframe_timestamps = [70.0]
        multi_target_images = False
        segment_start_idx = 1
        segment_end_idx = 1
        task_summary = "外观线索定位同一酒店"

    class _Merged:
        time_start = 0.0
        time_end = 90.0
        display_time_start = 0.0
        display_time_end = 90.0
        target_kind = TargetKind.video_derived
        keyframe_timestamps = [10.0, 40.0, 70.0]
        multi_target_images = True
        segment_start_idx = 0
        segment_end_idx = 1
        task_summary = "同酒店多源镜头"
        expected_image_count = 3

    class _Draft:
        decision = AuditDecision.accept
        reason = "两段线索"
        has_unresolved_target = True
        tasks = [_T1(), _T2()]

    monkeypatch.setattr(
        audit,
        "call_structured",
        _route_call(_Draft(), merge_tasks=[_Merged()]),
    )

    result = audit.run_audit_split(str(video), _transcript())
    assert len(result.tasks) == 1
    # 连续出示默认折成最小源输入集；多图由源输入判定决定，不由合并草稿的 expected 决定
    assert len(result.tasks[0].image_paths) >= 1
    assert result.tasks[0].expected_image_count == len(result.tasks[0].image_paths)


def test_all_rejected_frames_mark_task_for_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """候选全无效时标记该 task，不再让整条视频抛异常。"""
    video = tmp_path / "noretry.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 0.0
        time_end = 10.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [1.0, 5.0]
        multi_target_images = False
        segment_start_idx = 0
        segment_end_idx = 0
        task_summary = "单图"

    class _Draft:
        decision = AuditDecision.accept
        reason = "有目标"
        has_unresolved_target = True
        tasks = [_Task()]

    call_log: list[str] = []
    monkeypatch.setattr(
        audit,
        "call_structured",
        _route_call(
            _Draft(),
            frame_kind=audit.FrameKind.teaching_ui,
            call_log=call_log,
        ),
    )

    result = audit.run_audit_split(str(video), _transcript())
    assert result.tasks[0].status == TaskStatus.needs_review
    assert result.tasks[0].image_paths == []
    assert "_LLMKeyframeRetry" not in call_log
    assert not hasattr(audit, "_request_alt_timestamps")


def test_episode_quota_ignored_still_keeps_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一出示段内多帧合格、模型即使报 N=3 → 只留 1 张。"""
    video = tmp_path / "quota.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "1.0")
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    task = audit._LLMGeoTaskDraft(
        time_start=0.0,
        time_end=10.0,
        display_time_start=0.0,
        display_time_end=4.0,
        target_kind=TargetKind.still_image,
        multi_target_images=True,
        expected_image_count=3,
        task_summary="同段多帧",
        answer_status=AnswerStatus.resolved,
        final_location_text="地点",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="单题",
        has_unresolved_target=True,
        tasks=[task],
        split_confidence=0.95,
    )
    monkeypatch.setattr(audit, "call_structured", _route_call(draft))
    result = audit.run_audit_split(str(video), _transcript())
    assert result.tasks[0].status == TaskStatus.accepted
    assert len(result.tasks[0].image_paths) == 1
    assert result.tasks[0].multi_target_images is False


def test_clear_split_skips_second_model_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[1],
        task_summary="一条清晰答案链",
        answer_status=AnswerStatus.resolved,
        final_location_text="上海杨浦大桥",
    )

    def unexpected(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("清晰切分不应重复调用模型")

    monkeypatch.setattr(audit, "call_structured", unexpected)
    result = audit._maybe_review_task_split(
        [task],
        video_id="clear",
        transcript=_transcript(),
        overview_images=None,
        duration=20,
        split_confidence=0.95,
        model_requests_review=False,
        boundary_tolerance=20,
    )
    assert result == [task]


def test_objective_duplicate_answer_triggers_conservative_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[1],
        answer_status=AnswerStatus.resolved,
        final_location_text="同一地点",
    )
    second = audit._LLMGeoTaskDraft(
        time_start=10,
        time_end=20,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[11],
        answer_status=AnswerStatus.resolved,
        final_location_text="同一地点",
    )
    calls = {"count": 0}

    def review(_prompt: str, schema: Any, **_k: Any) -> Any:
        calls["count"] += 1
        assert schema is audit._LLMTaskMergeResult
        return audit._LLMTaskMergeResult(tasks=[first], reason="同题合并")

    monkeypatch.setattr(audit, "call_structured", review)
    result = audit._maybe_review_task_split(
        [first, second],
        video_id="duplicate",
        transcript=_transcript(),
        overview_images=None,
        duration=20,
        split_confidence=0.95,
        model_requests_review=False,
        boundary_tolerance=20,
    )
    assert calls["count"] == 1
    assert result == [first]


def test_paraphrased_same_place_triggers_llm_gate_then_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """措辞不同但同地：同地门禁命中后再走双向复核合并。"""
    first = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[1],
        answer_status=AnswerStatus.resolved,
        final_location_text="贵州省兴仁市振兴大道（靠近雕塑环岛的第一个路口）",
        task_summary="两张老照片定位同一路口",
    )
    second = audit._LLMGeoTaskDraft(
        time_start=10,
        time_end=20,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[11],
        answer_status=AnswerStatus.resolved,
        final_location_text="贵州省兴仁市振兴大道南侧（雕塑环岛旁建筑）",
        task_summary="第二张图精化同一地点",
    )
    schemas: list[str] = []

    def route(_prompt: str, schema: Any, **_k: Any) -> Any:
        schemas.append(getattr(schema, "__name__", ""))
        if schema is audit._LLMSamePlaceGate:
            return audit._LLMSamePlaceGate(
                pairs=[
                    audit._SamePlacePair(
                        task_i=1,
                        task_j=2,
                        same_place_or_chain=True,
                        reason="同题两图同一路口",
                    )
                ],
                reason="过拆",
            )
        assert schema is audit._LLMTaskMergeResult
        return audit._LLMTaskMergeResult(tasks=[first], reason="合并为同题多图")

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit._maybe_review_task_split(
        [first, second],
        video_id="paraphrase",
        transcript=_transcript(),
        overview_images=None,
        duration=20,
        split_confidence=0.95,
        model_requests_review=False,
        boundary_tolerance=20,
    )
    assert schemas == ["_LLMSamePlaceGate", "_LLMTaskMergeResult"]
    assert result == [first]


def test_truly_different_places_skip_merge_after_llm_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真正不同地点：同地门禁否定后，高置信切分不再双向复核。"""
    first = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[1],
        answer_status=AnswerStatus.resolved,
        final_location_text="上海外滩",
    )
    second = audit._LLMGeoTaskDraft(
        time_start=10,
        time_end=20,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[11],
        answer_status=AnswerStatus.resolved,
        final_location_text="北京天安门",
    )
    calls = {"gate": 0, "merge": 0}

    def route(_prompt: str, schema: Any, **_k: Any) -> Any:
        if schema is audit._LLMSamePlaceGate:
            calls["gate"] += 1
            return audit._LLMSamePlaceGate(
                pairs=[
                    audit._SamePlacePair(
                        task_i=1,
                        task_j=2,
                        same_place_or_chain=False,
                        reason="两地无关",
                    )
                ]
            )
        calls["merge"] += 1
        raise AssertionError("不同地点不应进入双向复核")

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit._maybe_review_task_split(
        [first, second],
        video_id="distinct",
        transcript=_transcript(),
        overview_images=None,
        duration=20,
        split_confidence=0.95,
        model_requests_review=False,
        boundary_tolerance=20,
    )
    assert calls == {"gate": 1, "merge": 0}
    assert result == [first, second]


def test_nearby_display_window_repairs_task_boundary() -> None:
    task = audit._LLMGeoTaskDraft(
        time_start=900,
        time_end=990,
        target_kind=TargetKind.still_image,
        display_time_start=892,
        display_time_end=910,
        answer_status=AnswerStatus.resolved,
        final_location_text="孔雀湖",
    )
    start, end = audit._normalize_task_window(
        task,
        duration=1200,
        transcript=[],
        boundary_tolerance=20,
    )
    assert start == 892
    assert end == 990
    display = audit._resolve_display_window(
        task, distill_start=start, distill_end=end
    )
    assert display == (892.0, 910.0)


def test_ambiguous_task_is_saved_and_skips_frame_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "ambiguous.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    extract_calls: list[str] = []

    def extract(
        video_path: str, stamps: list[float], *, out_dir: str
    ) -> list[str]:
        extract_calls.append(out_dir)
        return _fake_extract_factory()(video_path, stamps, out_dir=out_dir)

    monkeypatch.setattr(audit, "extract_keyframes", extract)
    task = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[1],
        answer_status=AnswerStatus.ambiguous,
        final_location_text="",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="有题但只有猜测",
        has_unresolved_target=True,
        tasks=[task],
        split_confidence=0.95,
    )
    monkeypatch.setattr(audit, "call_structured", lambda *_a, **_k: draft)

    result = audit.run_audit_split(str(video), _transcript())
    item = result.tasks[0]
    assert item.status == TaskStatus.rejected
    assert item.answer_status == AnswerStatus.ambiguous
    assert item.image_paths == []
    assert len(extract_calls) == 1  # 只有视频级稀疏审核帧
    task_file = (
        tmp_path
        / "intermediate"
        / "ambiguous"
        / "tasks"
        / "ambiguous__t01"
        / "task_audit.json"
    )
    assert task_file.is_file()


def test_single_still_selects_cleaner_later_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "quality.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())
    task = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        display_time_start=1.0,
        display_time_end=5.0,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[1, 5],
        answer_status=AnswerStatus.resolved,
        final_location_text="目标地点",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="清晰单题",
        tasks=[task],
        split_confidence=0.95,
    )

    def route(prompt: str, schema: Any, **kwargs: Any) -> Any:
        if schema is audit._LLMFrameVerdict:
            if "候选时间：1.0s" in prompt:
                return audit._LLMFrameVerdict(
                    kind=audit.FrameKind.target_photo,
                    quality_score=0.4,
                    tutorial_overlay=True,
                    clean_source=False,
                    reason="带讲解红线",
                )
            if "候选时间：5.0s" in prompt:
                return audit._LLMFrameVerdict(
                    kind=audit.FrameKind.target_photo,
                    quality_score=0.95,
                    clean_source=True,
                    reason="干净原图",
                )
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.teaching_ui,
                reason="工具界面",
            )
        if schema is audit._LLMSourceIdentityResult:
            images = kwargs.get("images") or []
            # 同段再展示：合并，保留质量更高者
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="同图")
                    for i in range(max(1, len(images)))
                ]
            )
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit.run_audit_split(str(video), _transcript())
    item = result.tasks[0]
    assert item.status == TaskStatus.accepted
    assert item.keyframe_timestamps == [5.0]
    assert len(item.image_paths) == 1
    assert any(
        assessment.timestamp == 5.0 and assessment.selected
        for assessment in item.frame_assessments
    )


def test_partial_task_checkpoints_resume_without_reauditing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "resume.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    intermediate = tmp_path / "intermediate"
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())
    first = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[1],
        answer_status=AnswerStatus.resolved,
        final_location_text="地点一",
    )
    second = audit._LLMGeoTaskDraft(
        time_start=10,
        time_end=20,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[11],
        answer_status=AnswerStatus.resolved,
        final_location_text="地点二",
    )
    root = intermediate / "resume"
    root.mkdir(parents=True)
    (root / "stage_audit_split_draft.json").write_text(
        json.dumps(
            {
                "video_id": "resume",
                "reason": "两题",
                "split_confidence": 0.9,
                "needs_split_review": False,
                "tasks": [
                    first.model_dump(mode="json"),
                    second.model_dump(mode="json"),
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    frame = tmp_path / "first.jpg"
    frame.write_bytes(b"jpg")
    first_checkpoint = root / "tasks" / "resume__t01" / "task_audit.json"
    first_checkpoint.parent.mkdir(parents=True)
    first_checkpoint.write_text(
        audit.GeoTaskSpec(
            task_id="resume__t01",
            time_start=0,
            time_end=10,
            target_kind=TargetKind.still_image,
            keyframe_timestamps=[1],
            image_paths=[str(frame)],
            answer_status=AnswerStatus.resolved,
            final_location_text="地点一",
        ).model_dump_json(),
        encoding="utf-8",
    )
    calls = {"frames": 0}

    def only_frame_calls(_prompt: str, schema: Any, **_k: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name in {
            "_LLMEvidenceBrief",
            "_LLMProcessTimeline",
            "_LLMSourceIdentityResult",
            "_LLMContainmentVerdict",
            "_LLMPhotoRelationVerdict",
        }:
            if name == "_LLMEvidenceBrief":
                return audit._LLMEvidenceBrief(visual_evidence_brief="")
            if name == "_LLMProcessTimeline":
                return audit._LLMProcessTimeline(intervals=[])
            if name == "_LLMContainmentVerdict":
                return audit._LLMContainmentVerdict(
                    containment=audit.ContainmentKind.none, confidence=0.0
                )
            if name == "_LLMPhotoRelationVerdict":
                return audit._LLMPhotoRelationVerdict(
                    relation=audit.PhotoRelation.same_photo, confidence=0.9
                )
            images = _k.get("images") or []
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="mock")
                    for i in range(max(1, len(images)))
                ]
            )
        assert schema is audit._LLMFrameVerdict
        calls["frames"] += 1
        return audit._LLMFrameVerdict(
            kind=audit.FrameKind.target_photo,
            quality_score=0.9,
            clean_source=True,
        )

    monkeypatch.setattr(audit, "call_structured", only_frame_calls)
    result = audit.run_audit_split(str(video), _transcript())
    assert len(result.tasks) == 2
    assert result.tasks[0].image_paths == [str(frame)]
    assert calls["frames"] > 0
    assert (root / "stage_audit_split.json").is_file()


def test_frame_verification_error_is_checkpointed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中转调用超时时也要保留候选帧进度，便于下次断点续跑。"""
    video = tmp_path / "frame-timeout.mp4"
    video.write_bytes(b"x")
    intermediate = tmp_path / "intermediate"
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_MAX_VLM_FRAME_VERIFIES", "1")
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "5.0")
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 10.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())
    task = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        display_time_start=0,
        display_time_end=5,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[1],
        answer_status=AnswerStatus.resolved,
        final_location_text="目标地点",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="单题",
        tasks=[task],
        split_confidence=0.95,
    )

    def route(_prompt: str, schema: Any, **_k: Any) -> Any:
        if schema is audit._LLMFrameVerdict:
            raise TimeoutError("relay timeout")
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit.run_audit_split(str(video), _transcript())

    assert result.tasks[0].status == TaskStatus.needs_review
    checkpoint = (
        intermediate
        / "frame-timeout"
        / "tasks"
        / "frame-timeout__t01"
        / "candidate_assessments.partial.json"
    )
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert any(item["kind"] == "error" for item in saved)
    assert any(
        item.get("reason") == "验收调用失败：TimeoutError" for item in saved
    )


def test_frame_verification_uses_local_narrative_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """全屏实景若是答案后的核验图，也不能误选成原始待定位输入。"""
    video = tmp_path / "narrative-role.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_MAX_VLM_FRAME_VERIFIES", "8")
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 10.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())
    transcript = [
        TranscriptSegment(start=0, end=3, text="展示题目原图，要求定位这里"),
        TranscriptSegment(start=3, end=7, text="开始分析图片线索"),
        TranscriptSegment(start=7, end=10, text="找到答案后查看当地高空全景核验"),
    ]
    task = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        display_time_start=0,
        display_time_end=10,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[1, 9],
        task_summary="定位题目原图，后段为答案核验",
        answer_status=AnswerStatus.resolved,
        final_location_text="目标地点",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="单题",
        tasks=[task],
        split_confidence=0.95,
    )
    seen_prompts: list[str] = []

    def route(prompt: str, schema: Any, **kwargs: Any) -> Any:
        if schema is audit._LLMFrameVerdict:
            seen_prompts.append(prompt)
            if "候选时间：1.0s" in prompt:
                return audit._LLMFrameVerdict(
                    kind=audit.FrameKind.target_photo,
                    quality_score=0.95,
                    clean_source=True,
                    reason="原始题图",
                )
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.teaching_ui,
                quality_score=0.9,
                clean_source=False,
                reason="答案后的高空核验图或无关帧",
            )
        if schema is audit._LLMSourceIdentityResult:
            images = kwargs.get("images") or []
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="mock")
                    for i in range(max(1, len(images)))
                ]
            )
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit.run_audit_split(str(video), transcript)

    assert result.tasks[0].status == TaskStatus.accepted
    assert result.tasks[0].keyframe_timestamps == [1.0]
    assert any("找到答案后查看当地高空全景核验" in p for p in seen_prompts)


def test_numbered_multi_location_answer_only_triggers_model_review() -> None:
    task = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=100,
        target_kind=TargetKind.video_derived,
        keyframe_timestamps=[10, 20, 30],
        multi_target_images=True,
        expected_image_count=3,
        answer_status=AnswerStatus.resolved,
        final_location_text="镜头1：地点甲；镜头2：地点乙；镜头3：地点丙",
    )
    anomalies = audit._objective_split_anomalies(
        [task],
        duration=100,
        boundary_tolerance=20,
    )
    assert len(anomalies) == 1
    assert "需复核" in anomalies[0]


def test_prefers_chain_support_over_clean_broll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """高分干净空镜不得压过低一点分但支撑定位链的题目图。"""
    video = tmp_path / "evidence.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())
    task = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        display_time_start=1.0,
        display_time_end=6.0,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[1.0, 5.0],
        answer_status=AnswerStatus.resolved,
        final_location_text="目标地点",
        task_summary="红瓦屋顶临水",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="证据对齐",
        tasks=[task],
        split_confidence=0.95,
    )

    def route(prompt: str, schema: Any, **kwargs: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMEvidenceBrief":
            return audit._LLMEvidenceBrief(
                visual_evidence_brief="红瓦屋顶与宽阔水面，镜头朝西"
            )
        if schema is audit._LLMFrameVerdict:
            if "候选时间：1.0s" in prompt:
                return audit._LLMFrameVerdict(
                    kind=audit.FrameKind.target_photo,
                    quality_score=0.99,
                    clean_source=True,
                    evidence_role=audit.EvidenceRole.unused_broll,
                    chain_support_score=0.1,
                    reason="干净空镜",
                )
            if "候选时间：5.0s" in prompt:
                return audit._LLMFrameVerdict(
                    kind=audit.FrameKind.target_photo,
                    quality_score=0.8,
                    clean_source=True,
                    tutorial_overlay=False,
                    evidence_role=audit.EvidenceRole.problem_input,
                    chain_support_score=0.92,
                    reason="红瓦屋顶临水原图",
                )
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.teaching_ui,
                evidence_role=audit.EvidenceRole.process_tool,
                chain_support_score=0.05,
                reason="工具界面",
            )
        if schema is audit._LLMSourceIdentityResult:
            images = kwargs.get("images") or []
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="同题")
                    for i in range(max(1, len(images)))
                ]
            )
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit.run_audit_split(str(video), _transcript())
    item = result.tasks[0]
    assert item.status == TaskStatus.accepted
    assert item.visual_evidence_brief.startswith("红瓦屋顶")
    assert item.keyframe_timestamps == [5.0]
    assert "选图质量等级=accepted" in item.image_selection_note
    assert "选中张数=" in item.image_selection_note
    assert any(
        a.timestamp == 5.0 and a.selected and a.evidence_role == "problem_input"
        for a in item.frame_assessments
    )


def test_low_chain_support_marks_needs_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """有 brief 但支撑分过低 → needs_review，不得用空镜充数。"""
    video = tmp_path / "weak.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())
    task = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=8,
        display_time_start=1.0,
        display_time_end=4.0,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[2.0],
        answer_status=AnswerStatus.resolved,
        final_location_text="某地",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="弱支撑",
        tasks=[task],
        split_confidence=0.9,
    )

    def route(prompt: str, schema: Any, **kwargs: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMEvidenceBrief":
            return audit._LLMEvidenceBrief(
                visual_evidence_brief="哥特尖顶与尖拱门廊"
            )
        if schema is audit._LLMFrameVerdict:
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.target_photo,
                quality_score=0.95,
                clean_source=True,
                evidence_role=audit.EvidenceRole.problem_input,
                chain_support_score=0.2,
                reason="画面不像尖顶建筑",
            )
        if schema is audit._LLMSourceIdentityResult:
            images = kwargs.get("images") or []
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="mock")
                    for i in range(max(1, len(images)))
                ]
            )
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit.run_audit_split(str(video), _transcript())
    item = result.tasks[0]
    assert item.status == TaskStatus.needs_review
    assert "定位链视觉证据不对齐" in item.status_reason
    assert item.image_paths  # 仍记录实选；编排器不再因 needs_review 跳过下游
    assert "选图质量等级=needs_review" in item.image_selection_note
    assert "定位链视觉证据不对齐" in item.image_selection_note
    assert all(
        a.evidence_role != "unused_broll" or not a.selected
        for a in item.frame_assessments
    )


def test_same_source_prefers_clean_repr_within_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同源连续段：思维链选源，干净度选帧（有 overlay 的线索帧让位给干净再展示）。"""
    video = tmp_path / "clean_repr.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())
    task = audit._LLMGeoTaskDraft(
        time_start=0,
        time_end=10,
        display_time_start=1.0,
        display_time_end=6.0,
        target_kind=TargetKind.still_image,
        keyframe_timestamps=[2.0, 5.0],
        answer_status=AnswerStatus.resolved,
        final_location_text="目标地点",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="同源择优",
        tasks=[task],
        split_confidence=0.95,
    )

    def route(prompt: str, schema: Any, **kwargs: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMEvidenceBrief":
            return audit._LLMEvidenceBrief(
                visual_evidence_brief="临江木栈道与远山轮廓"
            )
        if schema is audit._LLMFrameVerdict:
            if "候选时间：2.0s" in prompt:
                return audit._LLMFrameVerdict(
                    kind=audit.FrameKind.target_photo,
                    quality_score=0.7,
                    tutorial_overlay=True,
                    clean_source=False,
                    evidence_role=audit.EvidenceRole.problem_input,
                    chain_support_score=0.9,
                    reason="红线圈画线索帧",
                )
            if "候选时间：5.0s" in prompt:
                return audit._LLMFrameVerdict(
                    kind=audit.FrameKind.target_photo,
                    quality_score=0.95,
                    tutorial_overlay=False,
                    clean_source=True,
                    evidence_role=audit.EvidenceRole.problem_input,
                    chain_support_score=0.88,
                    reason="同图干净再展示",
                )
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.other,
                evidence_role=audit.EvidenceRole.other,
                chain_support_score=0.0,
                reason="无关",
            )
        if schema is audit._LLMSourceIdentityResult:
            images = kwargs.get("images") or []
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="同图")
                    for i in range(max(1, len(images)))
                ]
            )
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit.run_audit_split(str(video), _transcript())
    item = result.tasks[0]
    assert item.status == TaskStatus.accepted
    assert item.keyframe_timestamps == [5.0]


def test_process_timeline_samples_disjoint_show_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """两段不相邻 show_source 都应进入采样；中间 tool 段不采样。"""
    video = tmp_path / "timeline_multi.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "50.0")
    monkeypatch.setenv("AUDIT_MAX_SAMPLED_FRAMES", "20")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 500.0)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory(captured))

    task = audit._LLMGeoTaskDraft(
        time_start=0.0,
        time_end=500.0,
        display_time_start=0.0,
        display_time_end=30.0,
        target_kind=TargetKind.still_image,
        answer_status=AnswerStatus.resolved,
        final_location_text="同一地点",
        task_summary="前后两段出示原图",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="多段出示",
        tasks=[task],
        split_confidence=0.95,
    )
    intervals = [
        audit._LLMProcessInterval(
            start=0.0, end=40.0, role=ProcessRole.show_source, confidence=0.9
        ),
        audit._LLMProcessInterval(
            start=40.0, end=400.0, role=ProcessRole.tool, confidence=0.9
        ),
        audit._LLMProcessInterval(
            start=420.0, end=450.0, role=ProcessRole.show_source, confidence=0.9
        ),
    ]

    def route(prompt: str, schema: Any, **kwargs: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMProcessTimeline":
            return audit._LLMProcessTimeline(intervals=intervals)
        if name == "_LLMEvidenceBrief":
            return audit._LLMEvidenceBrief(visual_evidence_brief="路中站立的人")
        if name == "_LLMFrameVerdict":
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.target_photo,
                quality_score=0.9,
                clean_source=True,
                evidence_role=audit.EvidenceRole.problem_input,
                chain_support_score=0.9,
                reason="原图",
            )
        if name == "_LLMContainmentVerdict":
            return audit._LLMContainmentVerdict(
                containment=audit.ContainmentKind.none, confidence=0.0
            )
        if name == "_LLMPhotoRelationVerdict":
            return audit._LLMPhotoRelationVerdict(
                relation=audit.PhotoRelation.different_photo,
                confidence=0.9,
                reason="不同原图",
            )
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit.run_audit_split(str(video), _transcript())
    stamps = [
        float(c["stamps"][0])
        for c in captured.get("calls", [])
        if c.get("stamps") and "audit_candidates" in str(c.get("out_dir", ""))
    ]
    assert any(t <= 40.0 for t in stamps)
    assert any(t >= 420.0 for t in stamps)
    assert not any(100.0 <= t <= 390.0 for t in stamps)
    item = result.tasks[0]
    assert len(item.process_intervals) >= 2
    assert item.expected_image_count == 2


def test_process_timeline_short_display_still_samples_late_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """审核单出示粗窗很短时，时间线后段 show_source 仍进入候选。"""
    video = tmp_path / "late_show.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "100.0")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 500.0)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory(captured))

    task = audit._LLMGeoTaskDraft(
        time_start=0.0,
        time_end=500.0,
        display_time_start=0.0,
        display_time_end=36.0,
        target_kind=TargetKind.still_image,
        answer_status=AnswerStatus.resolved,
        final_location_text="核验同地点",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="后段再出示",
        tasks=[task],
        split_confidence=0.95,
    )

    def route(prompt: str, schema: Any, **kwargs: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMProcessTimeline":
            return audit._LLMProcessTimeline(
                intervals=[
                    audit._LLMProcessInterval(
                        start=0.0,
                        end=36.0,
                        role=ProcessRole.show_source,
                        confidence=0.9,
                    ),
                    audit._LLMProcessInterval(
                        start=36.0,
                        end=420.0,
                        role=ProcessRole.tool,
                        confidence=0.8,
                    ),
                    audit._LLMProcessInterval(
                        start=430.0,
                        end=460.0,
                        role=ProcessRole.show_source,
                        confidence=0.85,
                    ),
                ]
            )
        if name == "_LLMEvidenceBrief":
            return audit._LLMEvidenceBrief(visual_evidence_brief="路中父亲与店铺招牌")
        if name == "_LLMFrameVerdict":
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.target_photo,
                quality_score=0.9,
                clean_source=True,
                evidence_role=audit.EvidenceRole.problem_input,
                chain_support_score=0.85,
                reason="出示原图",
            )
        if name == "_LLMContainmentVerdict":
            return audit._LLMContainmentVerdict(
                containment=audit.ContainmentKind.none, confidence=0.0
            )
        if name == "_LLMPhotoRelationVerdict":
            return audit._LLMPhotoRelationVerdict(
                relation=audit.PhotoRelation.different_photo,
                confidence=0.9,
                reason="不同输入",
            )
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    audit.run_audit_split(str(video), _transcript())
    stamps = [
        float(c["stamps"][0])
        for c in captured.get("calls", [])
        if c.get("stamps") and "audit_candidates" in str(c.get("out_dir", ""))
    ]
    assert any(t >= 430.0 for t in stamps)


def test_process_timeline_skips_tool_tail_of_wide_display(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """宽出示粗窗后段若是 tool，则不采样该段。"""
    video = tmp_path / "wide_tool.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "30.0")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 360.0)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory(captured))

    task = audit._LLMGeoTaskDraft(
        time_start=0.0,
        time_end=360.0,
        display_time_start=0.0,
        display_time_end=360.0,
        target_kind=TargetKind.still_image,
        answer_status=AnswerStatus.resolved,
        final_location_text="目标建筑",
    )
    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="宽窗",
        tasks=[task],
        split_confidence=0.95,
    )

    def route(prompt: str, schema: Any, **kwargs: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMProcessTimeline":
            return audit._LLMProcessTimeline(
                intervals=[
                    audit._LLMProcessInterval(
                        start=0.0,
                        end=180.0,
                        role=ProcessRole.show_source,
                        confidence=0.9,
                    ),
                    audit._LLMProcessInterval(
                        start=180.0,
                        end=360.0,
                        role=ProcessRole.tool,
                        confidence=0.9,
                    ),
                ]
            )
        if name == "_LLMEvidenceBrief":
            return audit._LLMEvidenceBrief(visual_evidence_brief="并排旧建筑外观")
        if name == "_LLMFrameVerdict":
            if "过程时间线软先验：此刻区间角色为 tool" in prompt:
                return audit._LLMFrameVerdict(
                    kind=audit.FrameKind.teaching_ui,
                    evidence_role=audit.EvidenceRole.process_tool,
                    chain_support_score=0.05,
                    reason="搜索页工具图",
                )
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.target_photo,
                quality_score=0.9,
                clean_source=True,
                evidence_role=audit.EvidenceRole.problem_input,
                chain_support_score=0.9,
                reason="原图",
            )
        if name == "_LLMSourceIdentityResult":
            images = kwargs.get("images") or []
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="同题")
                    for i in range(max(1, len(images)))
                ]
            )
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit.run_audit_split(str(video), _transcript())
    stamps = [
        float(c["stamps"][0])
        for c in captured.get("calls", [])
        if c.get("stamps") and "audit_candidates" in str(c.get("out_dir", ""))
    ]
    assert stamps
    assert all(t <= 180.0 + 1e-3 for t in stamps)
    assert result.tasks[0].status == TaskStatus.accepted


def test_empty_process_timeline_falls_back_to_display_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """时间线为空时回退现行单出示粗窗。"""
    video = tmp_path / "fallback.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "2.0")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 40.0)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory(captured))

    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="回退",
        tasks=[
            audit._LLMGeoTaskDraft(
                time_start=0.0,
                time_end=40.0,
                display_time_start=2.0,
                display_time_end=8.0,
                target_kind=TargetKind.still_image,
                answer_status=AnswerStatus.resolved,
                final_location_text="地点",
            )
        ],
        split_confidence=0.95,
    )
    monkeypatch.setattr(
        audit,
        "call_structured",
        _route_call(draft, visual_evidence_brief="红瓦屋顶"),
    )
    result = audit.run_audit_split(str(video), _transcript())
    stamps = [
        float(c["stamps"][0])
        for c in captured.get("calls", [])
        if c.get("stamps") and "audit_candidates" in str(c.get("out_dir", ""))
    ]
    assert stamps
    assert all(2.0 - 1e-3 <= t <= 8.0 + 1e-3 for t in stamps)
    assert result.tasks[0].process_intervals == []


def test_verify_keyframe_injects_process_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """process_role=tool 时 prompt 含软先验；show_source 亦可。"""
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"jpg")
    seen: list[str] = []

    def fake(prompt: str, schema: Any, **_k: Any) -> Any:
        seen.append(prompt)
        if "区间角色为 tool" in prompt:
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.teaching_ui,
                evidence_role=audit.EvidenceRole.process_tool,
                reason="工具",
            )
        return audit._LLMFrameVerdict(
            kind=audit.FrameKind.target_photo,
            evidence_role=audit.EvidenceRole.problem_input,
            clean_source=True,
            reason="原图",
        )

    monkeypatch.setattr(audit, "call_structured", fake)
    tool_v = audit.verify_keyframe(str(frame), process_role=ProcessRole.tool)
    assert tool_v.kind == audit.FrameKind.teaching_ui
    assert any("区间角色为 tool" in p for p in seen)
    show_v = audit.verify_keyframe(str(frame), process_role=ProcessRole.show_source)
    assert show_v.kind == audit.FrameKind.target_photo
    assert any("区间角色为 show_source" in p for p in seen)


def test_two_show_source_same_group_keeps_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """两段 show_source 但同源判定同组 → 仍只留 1 张。"""
    video = tmp_path / "reshow.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "50.0")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 200.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="再展示",
        tasks=[
            audit._LLMGeoTaskDraft(
                time_start=0.0,
                time_end=200.0,
                display_time_start=0.0,
                display_time_end=200.0,
                target_kind=TargetKind.still_image,
                answer_status=AnswerStatus.resolved,
                final_location_text="地点",
            )
        ],
        split_confidence=0.95,
    )

    def route(prompt: str, schema: Any, **kwargs: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMProcessTimeline":
            return audit._LLMProcessTimeline(
                intervals=[
                    audit._LLMProcessInterval(
                        start=0.0,
                        end=30.0,
                        role=ProcessRole.show_source,
                        confidence=0.9,
                    ),
                    audit._LLMProcessInterval(
                        start=30.0,
                        end=150.0,
                        role=ProcessRole.tool,
                        confidence=0.8,
                    ),
                    audit._LLMProcessInterval(
                        start=150.0,
                        end=180.0,
                        role=ProcessRole.show_source,
                        confidence=0.9,
                    ),
                ]
            )
        if name == "_LLMEvidenceBrief":
            return audit._LLMEvidenceBrief(visual_evidence_brief="")
        if name == "_LLMFrameVerdict":
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.target_photo,
                quality_score=0.9,
                clean_source=True,
                evidence_role=audit.EvidenceRole.problem_input,
                chain_support_score=0.5,
                reason="同图再展示",
            )
        if name == "_LLMSourceIdentityResult":
            images = kwargs.get("images") or []
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="同图")
                    for i in range(max(1, len(images)))
                ]
            )
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit.run_audit_split(str(video), _transcript())
    item = result.tasks[0]
    assert item.status == TaskStatus.accepted
    assert item.expected_image_count == 1
    assert len(item.image_paths) == 1


def test_multi_show_source_one_selected_needs_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """brief 非空且多段不相邻 show_source，却只选出 1 张 → needs_review。"""
    video = tmp_path / "missed_second.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC", "50.0")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()
    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 500.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    draft = audit._LLMAuditDraft(
        decision=AuditDecision.accept,
        reason="漏第二张",
        tasks=[
            audit._LLMGeoTaskDraft(
                time_start=0.0,
                time_end=500.0,
                display_time_start=0.0,
                display_time_end=40.0,
                target_kind=TargetKind.still_image,
                answer_status=AnswerStatus.resolved,
                final_location_text="地点",
            )
        ],
        split_confidence=0.95,
    )

    def route(prompt: str, schema: Any, **kwargs: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMProcessTimeline":
            return audit._LLMProcessTimeline(
                intervals=[
                    audit._LLMProcessInterval(
                        start=0.0,
                        end=40.0,
                        role=ProcessRole.show_source,
                        confidence=0.9,
                    ),
                    audit._LLMProcessInterval(
                        start=40.0,
                        end=420.0,
                        role=ProcessRole.tool,
                        confidence=0.8,
                    ),
                    audit._LLMProcessInterval(
                        start=430.0,
                        end=460.0,
                        role=ProcessRole.show_source,
                        confidence=0.9,
                    ),
                ]
            )
        if name == "_LLMEvidenceBrief":
            return audit._LLMEvidenceBrief(visual_evidence_brief="路中站立与店铺招牌")
        if name == "_LLMFrameVerdict":
            return audit._LLMFrameVerdict(
                kind=audit.FrameKind.target_photo,
                quality_score=0.9,
                clean_source=True,
                evidence_role=audit.EvidenceRole.problem_input,
                chain_support_score=0.9,
                reason="原图",
            )
        if name == "_LLMSourceIdentityResult":
            images = kwargs.get("images") or []
            return audit._LLMSourceIdentityResult(
                items=[
                    audit._SourceGroupItem(index=i, source_group=0, reason="误合并")
                    for i in range(max(1, len(images)))
                ]
            )
        return draft

    monkeypatch.setattr(audit, "call_structured", route)
    result = audit.run_audit_split(str(video), _transcript())
    item = result.tasks[0]
    assert item.status == TaskStatus.needs_review
    assert "多段不相邻出示窗" in item.status_reason
    assert item.expected_image_count == 1
    assert "选图质量等级=needs_review" in item.image_selection_note
    assert "多段不相邻出示窗" in item.image_selection_note


def test_resolve_sample_windows_helpers() -> None:
    intervals = [
        ProcessInterval(start=0, end=10, role=ProcessRole.show_source),
        ProcessInterval(start=10, end=50, role=ProcessRole.tool),
        ProcessInterval(start=80, end=100, role=ProcessRole.show_source),
    ]
    assert audit._count_nonadjacent_show_source(intervals) == 2
    assert audit._show_source_windows(intervals) == [(0.0, 10.0), (80.0, 100.0)]
    assert audit._process_role_at(intervals, 5.0) == ProcessRole.show_source
    assert audit._process_role_at(intervals, 30.0) == ProcessRole.tool
    assert audit._process_role_at(intervals, 60.0) is None
    stamps = audit._dense_sample_windows(
        [(0.0, 2.0), (90.0, 92.0)], interval=1.0, max_n=20
    )
    assert any(t <= 2.0 for t in stamps)
    assert any(t >= 90.0 for t in stamps)


def _write_solid_jpg(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> None:
    from PIL import Image

    Image.new("RGB", size, color).save(path, format="PNG")


def _write_collage_and_zoom(tmp_path: Path) -> tuple[Path, Path]:
    """左右拼图 + 右半幅全屏放大。"""
    from PIL import Image

    left = Image.new("RGB", (64, 64), (200, 40, 40))
    right = Image.new("RGB", (64, 64), (40, 40, 200))
    for y in range(64):
        for x in range(64):
            if (x + y) % 7 == 0:
                right.putpixel((x, y), (255, 255, 0))
    collage = Image.new("RGB", (128, 64))
    collage.paste(left, (0, 0))
    collage.paste(right, (64, 0))
    collage_path = tmp_path / "collage.png"
    zoom_path = tmp_path / "zoom.png"
    collage.save(collage_path, format="PNG")
    right.resize((128, 64)).save(zoom_path, format="PNG")
    return collage_path, zoom_path


def test_containment_precheck_collage_contains_zoom(tmp_path: Path) -> None:
    from pipeline.stage_audit_split.frame_prefilter import containment_precheck_score

    collage, zoom = _write_collage_and_zoom(tmp_path)
    kind, score = containment_precheck_score(str(collage), str(zoom), min_score=0.75)
    assert kind == "a_contains_b"
    assert score >= 0.75


def test_containment_hard_merge_keeps_collage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """拼图 + 右半幅放大 → 只留拼图。"""
    collage, zoom = _write_collage_and_zoom(tmp_path)
    reps = [
        audit.KeyframeAssessment(
            timestamp=90.0,
            image_path=str(collage),
            kind="target_photo",
            quality_score=0.8,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.9,
        ),
        audit.KeyframeAssessment(
            timestamp=137.0,
            image_path=str(zoom),
            kind="target_photo",
            quality_score=0.95,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.85,
        ),
    ]

    def fake_call(prompt: str, schema: Any, **_k: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMContainmentVerdict":
            return audit._LLMContainmentVerdict(
                containment=audit.ContainmentKind.a_contains_b,
                confidence=0.95,
                reason="右半幅放大",
            )
        if name == "_LLMPhotoRelationVerdict":
            raise AssertionError("硬合并后不应再问语义关系")
        raise AssertionError(name)

    monkeypatch.setattr(audit, "call_structured", fake_call)
    # 强制走 VLM 包含路径：让廉价预检返回 none
    monkeypatch.setattr(
        "pipeline.stage_audit_split.frame_prefilter.containment_precheck_score",
        lambda *_a, **_k: ("none", 0.0),
    )
    selected = audit._resolve_source_identity(
        reps,
        target_kind=TargetKind.still_image,
        visual_evidence_brief="图一街景与图二建筑",
    )
    assert len(selected) == 1
    assert selected[0].timestamp == 90.0


def test_containment_precheck_alone_merges_without_vlm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """廉价包含预检高分时可不依赖 VLM 分组即可合并。"""
    collage, zoom = _write_collage_and_zoom(tmp_path)
    reps = [
        audit.KeyframeAssessment(
            timestamp=1.0,
            image_path=str(collage),
            kind="target_photo",
            quality_score=0.7,
            clean_source=True,
            evidence_role="problem_input",
        ),
        audit.KeyframeAssessment(
            timestamp=2.0,
            image_path=str(zoom),
            kind="target_photo",
            quality_score=0.9,
            clean_source=True,
            evidence_role="problem_input",
        ),
    ]

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("不应调用 VLM")

    monkeypatch.setattr(audit, "call_structured", boom)
    selected = audit._resolve_source_identity(
        reps, target_kind=TargetKind.still_image
    )
    assert len(selected) == 1
    assert selected[0].timestamp == 1.0


def test_different_street_photos_stay_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """两张不同街拍静图即使 brief 写两张也保持两组。"""
    a = tmp_path / "street_a.jpg"
    b = tmp_path / "street_b.jpg"
    _write_solid_jpg(a, (220, 180, 60))
    _write_solid_jpg(b, (60, 120, 200))
    reps = [
        audit.KeyframeAssessment(
            timestamp=59.0,
            image_path=str(a),
            kind="target_photo",
            quality_score=0.9,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.9,
        ),
        audit.KeyframeAssessment(
            timestamp=462.6,
            image_path=str(b),
            kind="target_photo",
            quality_score=0.85,
            clean_source=True,
            evidence_role="problem_input",
            chain_support_score=0.88,
        ),
    ]

    def fake_call(prompt: str, schema: Any, **_k: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMContainmentVerdict":
            return audit._LLMContainmentVerdict(
                containment=audit.ContainmentKind.none, confidence=0.1
            )
        if name == "_LLMPhotoRelationVerdict":
            assert "不得用来决定要几个" in prompt or "不决定张数" in prompt
            return audit._LLMPhotoRelationVerdict(
                relation=audit.PhotoRelation.different_photo,
                confidence=0.95,
                reason="不同店面立面",
            )
        raise AssertionError(name)

    monkeypatch.setattr(audit, "call_structured", fake_call)
    monkeypatch.setattr(
        "pipeline.stage_audit_split.frame_prefilter.containment_precheck_score",
        lambda *_a, **_k: ("none", 0.0),
    )
    selected = audit._resolve_source_identity(
        reps,
        target_kind=TargetKind.still_image,
        visual_evidence_brief="两张图：宽马路；第二张黄店面六扇窗",
    )
    assert len(selected) == 2
    assert [s.timestamp for s in selected] == [59.0, 462.6]


def test_same_photo_reshow_merges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一静图再展示合并为 1 张。"""
    a = tmp_path / "p1.jpg"
    b = tmp_path / "p2.jpg"
    _write_solid_jpg(a, (100, 100, 100))
    _write_solid_jpg(b, (100, 100, 100))
    reps = [
        audit.KeyframeAssessment(
            timestamp=10.0,
            image_path=str(a),
            kind="target_photo",
            quality_score=0.6,
            clean_source=False,
            tutorial_overlay=True,
            evidence_role="problem_input",
        ),
        audit.KeyframeAssessment(
            timestamp=80.0,
            image_path=str(b),
            kind="target_photo",
            quality_score=0.95,
            clean_source=True,
            evidence_role="problem_input",
        ),
    ]

    def fake_call(prompt: str, schema: Any, **_k: Any) -> Any:
        name = getattr(schema, "__name__", "")
        if name == "_LLMContainmentVerdict":
            return audit._LLMContainmentVerdict(
                containment=audit.ContainmentKind.none, confidence=0.0
            )
        if name == "_LLMPhotoRelationVerdict":
            return audit._LLMPhotoRelationVerdict(
                relation=audit.PhotoRelation.same_photo,
                confidence=0.9,
                reason="再展示",
            )
        raise AssertionError(name)

    monkeypatch.setattr(audit, "call_structured", fake_call)
    monkeypatch.setattr(
        "pipeline.stage_audit_split.frame_prefilter.containment_precheck_score",
        lambda *_a, **_k: ("none", 0.0),
    )
    selected = audit._resolve_source_identity(
        reps, target_kind=TargetKind.still_image
    )
    assert len(selected) == 1
    assert selected[0].timestamp == 80.0


def test_anthropic_enum_wrappers_are_unwrapped() -> None:
    """relay 偶发的 {type}/{primary} 包装不得让正确帧验收整体失败。"""

    frame = audit._LLMFrameVerdict.model_validate(
        {
            "kind": {"primary": "target_photo", "confidence": "high"},
            "quality_score": 0.9,
            "evidence_role": {"category": {"type": "problem_input"}},
        }
    )
    assert frame.kind == audit.FrameKind.target_photo
    assert frame.evidence_role == audit.EvidenceRole.problem_input

    probabilistic = audit._LLMFrameVerdict.model_validate(
        {
            "kind": {"target_photo": 0.8, "teaching_ui": 0.1, "other": 0.1},
            "evidence_role": {"problem_input": 0.9, "other": 0.1},
        }
    )
    assert probabilistic.kind == audit.FrameKind.target_photo
    assert probabilistic.evidence_role == audit.EvidenceRole.problem_input

    recovered = audit._LLMFrameVerdict.model_validate(
        {
            "{": (
                '"kind":"teaching_ui","quality_score":0.2,'
                '"evidence_role":"process_tool","reason":"包装 JSON"}'
            )
        }
    )
    assert recovered.kind == audit.FrameKind.teaching_ui
    assert recovered.evidence_role == audit.EvidenceRole.process_tool

    interval = audit._LLMProcessInterval.model_validate(
        {
            "start_time": 1.0,
            "end_time": 2.0,
            "role": {"primary": "show_source"},
        }
    )
    assert interval.role == ProcessRole.show_source

    relation = audit._LLMPhotoRelationVerdict.model_validate(
        {"relation": {"type": "same_photo"}}
    )
    assert relation.relation == audit.PhotoRelation.same_photo


def test_compose_image_selection_note_for_accepted_and_review() -> None:
    """accepted / needs_review 都必须写出非空选图评价。"""
    from pipeline.schemas.audit import KeyframeAssessment

    selected = KeyframeAssessment(
        timestamp=12.5,
        image_path="/tmp/a.jpg",
        kind="target_photo",
        quality_score=0.88,
        tutorial_overlay=False,
        clean_source=True,
        evidence_role="problem_input",
        chain_support_score=0.91,
        selected=True,
        reason="干净原图",
    )
    ok = audit.compose_image_selection_note(
        status=TaskStatus.accepted,
        status_reason="",
        assessments=[selected],
    )
    assert "选图质量等级=accepted" in ok
    assert "选中张数=1" in ok
    assert "t=12.500s" in ok
    assert "quality=0.88" in ok

    review = audit.compose_image_selection_note(
        status=TaskStatus.needs_review,
        status_reason="选中帧仍含讲解覆盖、界面残留或质量低于阈值",
        assessments=[
            selected.model_copy(
                update={
                    "tutorial_overlay": True,
                    "clean_source": False,
                    "quality_score": 0.4,
                    "reason": "字幕条",
                }
            )
        ],
    )
    assert "选图质量等级=needs_review" in review
    assert "选中帧仍含讲解覆盖" in review
    assert "overlay=True" in review
    assert "reason=字幕条" in review
