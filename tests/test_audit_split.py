"""阶段1.5 审核切分测试（mock LLM / 截帧）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.schemas.audit import (
    AnswerStatus,
    AuditDecision,
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
):
    """按 schema 分流：审核草稿 / 帧验收 / 合并复核。"""

    def _fake(prompt: str, schema: Any, **_k: Any) -> Any:  # noqa: ANN401
        name = getattr(schema, "__name__", "")
        if call_log is not None:
            call_log.append(name)
        if name == "_LLMFrameVerdict":
            kind = frame_kind or audit.FrameKind.target_photo

            class _V:
                pass

            v = _V()
            v.kind = kind
            v.quality_score = 0.9
            v.answer_leakage = False
            v.tutorial_overlay = False
            v.clean_source = kind == audit.FrameKind.target_photo
            v.reason = "mock"
            return v
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
    """has_unresolved_target=false 时即使 decision=accept 也强制 reject。"""
    video = tmp_path / "force.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 12.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 0.0
        time_end = 5.0
        target_kind = TargetKind.video_derived
        keyframe_timestamps = [1.0]
        multi_target_images = False
        segment_start_idx = 0
        segment_end_idx = 0
        task_summary = "误标"

    class _Draft:
        decision = AuditDecision.accept
        reason = "误把科普当定位"
        has_unresolved_target = False
        tasks = [_Task()]

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    assert result.decision == AuditDecision.reject
    assert result.tasks == []
    assert result.has_unresolved_target is False
    assert "has_unresolved_target=false" in result.reason


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
    video = tmp_path / "multiimg.mp4"
    video.write_bytes(b"x")
    _isolate_data_dirs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUDIT_MAX_KEYFRAMES_PER_TASK", "4")

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 120.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 0.0
        time_end = 100.0
        display_time_start = 0.0
        display_time_end = 20.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [1.0, 80.0]
        multi_target_images = True
        segment_start_idx = 0
        segment_end_idx = 1
        task_summary = "同题两图"
        expected_image_count = 2

    class _Draft:
        decision = AuditDecision.accept
        reason = "同题多图"
        has_unresolved_target = True
        tasks = [_Task()]

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    assert len(result.tasks) == 1
    assert len(result.tasks[0].image_paths) >= 2
    assert len(result.tasks[0].image_paths) <= 2
    assert result.tasks[0].multi_target_images is True
    assert all(0.0 <= t <= 20.0 for t in result.tasks[0].keyframe_timestamps)


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

    def _fake(prompt: str, schema: Any, **_k: Any) -> Any:  # noqa: ANN401
        name = getattr(schema, "__name__", "")
        if name == "_LLMFrameVerdict":

            class _V:
                kind = next(kinds, audit.FrameKind.target_photo)
                reason = "mock"

            return _V()
        return _Draft()

    monkeypatch.setattr(audit, "call_structured", _fake)

    result = audit.run_audit_split(str(video), _transcript())
    assert len(result.tasks[0].image_paths) == 1


def test_multi_target_marks_review_if_only_one_valid_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    class _Draft:
        decision = AuditDecision.accept
        reason = "同题多图"
        has_unresolved_target = True
        tasks = [_Task()]

    kinds = iter(
        [
            audit.FrameKind.target_photo,
            audit.FrameKind.teaching_ui,
        ]
    )
    call_log: list[str] = []

    def _fake(prompt: str, schema: Any, **_k: Any) -> Any:  # noqa: ANN401
        name = getattr(schema, "__name__", "")
        call_log.append(name)
        if name == "_LLMFrameVerdict":

            class _V:
                kind = next(kinds, audit.FrameKind.teaching_ui)
                reason = "mock"

            return _V()
        return _Draft()

    monkeypatch.setattr(audit, "call_structured", _fake)

    result = audit.run_audit_split(str(video), _transcript())
    assert result.tasks[0].status == TaskStatus.needs_review
    assert "预计 2 个独立输入" in result.tasks[0].status_reason
    assert "_LLMKeyframeRetry" not in call_log


def test_video_derived_allows_multiple_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    class _Draft:
        decision = AuditDecision.accept
        reason = "视频场景定位"
        has_unresolved_target = True
        tasks = [_Task()]

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    assert len(result.tasks[0].image_paths) >= 2


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
    assert len(result.tasks[1].image_paths) >= 2

    sliced = audit.slice_transcript_for_task(_transcript(), result.tasks[1])
    assert len(sliced) == 1
    assert "第二张" in sliced[0].text


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
    assert "精确关键帧秒数" in hint or "弱先验" in hint
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


def test_video_derived_keeps_nearby_distinct_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """video_derived 下视觉非重复的独立源镜头应保留多张。"""
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

    class _Draft:
        decision = AuditDecision.accept
        reason = "同题多输入"
        has_unresolved_target = True
        tasks = [_Task()]

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    stamps = result.tasks[0].keyframe_timestamps
    assert len(stamps) >= 3
    assert all(40.0 <= s <= 80.0 for s in stamps)


def test_visual_duplicate_replaces_time_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"a.jpg": 0b1010, "b.jpg": 0b1011, "c.jpg": 0b11110000}
    monkeypatch.setattr(
        audit, "_image_dhash", lambda path: values[Path(path).name]
    )
    duplicate, value = audit._is_visual_duplicate(
        "b.jpg", [values["a.jpg"]], max_distance=1
    )
    assert duplicate is True
    assert value == values["b.jpg"]
    duplicate, _ = audit._is_visual_duplicate(
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
    assert result.tasks[0].multi_target_images is True
    assert len(result.tasks[0].image_paths) >= 2


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


def test_max_keyframes_still_multi_capped() -> None:
    assert (
        audit._max_keyframes_for_task(TargetKind.still_image, True, 8) == 2
    )
    assert audit._max_keyframes_for_task(
        TargetKind.video_derived,
        False,
        8,
        expected_image_count=3,
    ) == 3
    assert (
        audit._max_keyframes_for_task(TargetKind.still_image, False, 8) == 1
    )


def test_multi_output_count_is_not_hard_capped_to_two() -> None:
    assert audit._max_keyframes_for_task(
        TargetKind.still_image,
        True,
        8,
        expected_image_count=4,
    ) == 4


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

    def route(prompt: str, schema: Any, **_k: Any) -> Any:
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

    def route(prompt: str, schema: Any, **_k: Any) -> Any:
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
