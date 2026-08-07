"""阶段1.5 审核切分测试（mock LLM / 截帧）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.schemas.audit import AuditDecision, TargetKind
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage_audit_split import run as audit


def _transcript() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=0.0, end=5.0, text="第一张图在哪里"),
        TranscriptSegment(start=5.0, end=10.0, text="第二张图又在哪"),
    ]


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


def _route_call(draft: Any, *, frame_kind: audit.FrameKind | None = None):
    """按 schema 分流：审核草稿 / 帧验收 / 重试时间戳。"""

    def _fake(prompt: str, schema: Any, **_k: Any) -> Any:  # noqa: ANN401
        name = getattr(schema, "__name__", "")
        if name == "_LLMFrameVerdict":
            kind = frame_kind or audit.FrameKind.target_photo

            class _V:
                pass

            v = _V()
            v.kind = kind
            v.reason = "mock"
            return v
        if name == "_LLMKeyframeRetry":

            class _R:
                keyframe_timestamps = [4.0, 7.0]

            return _R()
        if name == "_LLMTaskMergeResult":

            class _M:
                tasks = list(getattr(draft, "tasks", []) or [])
                reason = "keep"

            return _M()
        return draft

    return _fake


def test_run_audit_split_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "reject.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

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
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

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
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUDIT_MAX_KEYFRAMES_PER_TASK", "8")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 20.0)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        audit, "extract_keyframes", _fake_extract_factory(captured)
    )

    class _Task:
        time_start = 2.0
        time_end = 8.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [3.0, 6.0, 7.0]
        multi_target_images = False
        segment_start_idx = 0
        segment_end_idx = 0
        task_summary = "第一题"

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


def test_multi_target_images_allows_multiple_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "multiimg.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUDIT_MAX_KEYFRAMES_PER_TASK", "4")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 120.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 0.0
        time_end = 100.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [1.0, 80.0]
        multi_target_images = True
        segment_start_idx = 0
        segment_end_idx = 1
        task_summary = "同题两图"

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


def test_teaching_ui_frame_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """teaching_ui 帧被剔除；仅保留 target_photo。"""
    video = tmp_path / "ui.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

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
        if name == "_LLMKeyframeRetry":

            class _R:
                keyframe_timestamps: list[float] = []

            return _R()
        return _Draft()

    monkeypatch.setattr(audit, "call_structured", _fake)

    result = audit.run_audit_split(str(video), _transcript())
    assert len(result.tasks[0].image_paths) == 1


def test_multi_target_raises_if_only_one_valid_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "badmulti.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 120.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 0.0
        time_end = 100.0
        target_kind = TargetKind.still_image
        keyframe_timestamps = [1.0, 80.0]
        multi_target_images = True
        segment_start_idx = 0
        segment_end_idx = 1
        task_summary = "同题两图"

    class _Draft:
        decision = AuditDecision.accept
        reason = "同题多图"
        has_unresolved_target = True
        tasks = [_Task()]

    kinds = iter(
        [
            audit.FrameKind.target_photo,
            audit.FrameKind.teaching_ui,
            # retries also teaching
            audit.FrameKind.teaching_ui,
            audit.FrameKind.teaching_ui,
            audit.FrameKind.teaching_ui,
            audit.FrameKind.teaching_ui,
        ]
    )

    def _fake(prompt: str, schema: Any, **_k: Any) -> Any:  # noqa: ANN401
        name = getattr(schema, "__name__", "")
        if name == "_LLMFrameVerdict":

            class _V:
                kind = next(kinds, audit.FrameKind.teaching_ui)
                reason = "mock"

            return _V()
        if name == "_LLMKeyframeRetry":

            class _R:
                keyframe_timestamps = [6.0, 8.0]

            return _R()
        return _Draft()

    monkeypatch.setattr(audit, "call_structured", _fake)

    with pytest.raises(RuntimeError, match="有效关键帧不足 2 张"):
        audit.run_audit_split(str(video), _transcript())


def test_video_derived_allows_multiple_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "vder.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 30.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 5.0
        time_end = 10.0
        target_kind = TargetKind.video_derived
        keyframe_timestamps = [6.0, 8.0]
        multi_target_images = False
        segment_start_idx = 1
        segment_end_idx = 1
        task_summary = "场景"

    class _Draft:
        decision = AuditDecision.accept
        reason = "视频场景定位"
        has_unresolved_target = True
        tasks = [_Task()]

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    assert len(result.tasks[0].image_paths) >= 2


def test_no_overview_fallback_on_extract_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "nofallback.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

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

    with pytest.raises(RuntimeError, match="未能截取任何被定位关键帧"):
        audit.run_audit_split(str(video), _transcript())


def test_run_audit_split_multi_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "multi.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

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
        target_kind = TargetKind.video_derived
        keyframe_timestamps = [6.0, 8.0]
        multi_target_images = False
        segment_start_idx = 1
        segment_end_idx = 1
        task_summary = "题2"

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


def test_frame_verify_hint_rejects_streetview_ui() -> None:
    """验收提示须明确街景/地图 UI、片头与评论区截图。"""
    hint = audit.FRAME_VERIFY_HINT
    assert "街景" in hint
    assert "方向箭头" in hint or "小地图" in hint
    assert "片头" in hint or "标题" in hint
    assert "评论区" in hint
    assert "target_photo" in hint
    assert "teaching_ui" in hint


def test_max_keyframes_still_multi_capped() -> None:
    assert (
        audit._max_keyframes_for_task(TargetKind.still_image, True, 8) == 2
    )
    assert (
        audit._max_keyframes_for_task(TargetKind.video_derived, False, 8) == 8
    )
    assert (
        audit._max_keyframes_for_task(TargetKind.still_image, False, 8) == 1
    )


def test_multi_skips_near_duplicate_stamps() -> None:
    assert audit._is_near_duplicate_stamp(18.0, [15.0], 45.0) is True
    assert audit._is_near_duplicate_stamp(100.0, [15.0], 45.0) is False
    assert audit._multi_span_ok([15.0, 18.0], 45.0) is False
    assert audit._multi_span_ok([15.0, 80.0], 45.0) is True
