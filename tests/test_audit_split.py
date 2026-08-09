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

    with pytest.raises(RuntimeError, match="有效关键帧不足 2 张"):
        audit.run_audit_split(str(video), _transcript())
    assert "_LLMKeyframeRetry" not in call_log


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
    assert "每一个" in hint or "列全" in hint
    assert "过程角色" in hint
    assert "工具" in hint
    assert "揭晓" in hint
    assert "品类清单" in hint or "过程角色" in hint
    assert "整条答案链" in hint
    assert "摘要与时间戳对齐" in hint or "摘要" in hint
    assert "建筑外观" in hint
    assert "卫星" not in hint
    assert "谷歌" not in hint


def test_seed_timestamps_include_shot_mentions() -> None:
    """通用「镜头/首先来看」旁白应产生探测种子。"""
    segs = [
        TranscriptSegment(
            start=32.0,
            end=66.0,
            text="首先来看这个镜头，酒店就在主河道旁。",
        ),
        TranscriptSegment(
            start=66.0,
            end=100.0,
            text="接下来镜头二，河道往西北延伸。",
        ),
    ]
    stamps = audit._seed_photo_mention_timestamps(segs)
    assert stamps
    assert any(30.0 <= t <= 70.0 for t in stamps)


def test_video_derived_keeps_nearby_distinct_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """video_derived 下间隔约 15–20s 的独立源镜头不应被近邻去重丢掉。"""
    video = tmp_path / "neardup.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    monkeypatch.setattr(audit, "video_duration_sec", lambda _p: 100.0)
    monkeypatch.setattr(audit, "extract_keyframes", _fake_extract_factory())

    class _Task:
        time_start = 0.0
        time_end = 100.0
        target_kind = TargetKind.video_derived
        keyframe_timestamps = [40.0, 58.0, 75.0]
        multi_target_images = True
        segment_start_idx = 0
        segment_end_idx = 1
        task_summary = "多源镜头"

    class _Draft:
        decision = AuditDecision.accept
        reason = "同题多输入"
        has_unresolved_target = True
        tasks = [_Task()]

    monkeypatch.setattr(audit, "call_structured", _route_call(_Draft()))

    result = audit.run_audit_split(str(video), _transcript())
    stamps = result.tasks[0].keyframe_timestamps
    assert len(stamps) >= 3
    assert any(abs(s - 40.0) < 1e-3 for s in stamps)
    assert any(abs(s - 58.0) < 1e-3 for s in stamps)
    assert any(abs(s - 75.0) < 1e-3 for s in stamps)


def test_video_derived_near_gap_constant() -> None:
    assert audit.VIDEO_DERIVED_NEAR_GAP == 10.0
    assert audit._is_near_duplicate_stamp(50.0, [40.0], 10.0) is False
    assert audit._is_near_duplicate_stamp(48.0, [40.0], 10.0) is True


def test_merge_same_final_location_into_one_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同最终地点的多草稿须合并为 1 个 task。"""
    video = tmp_path / "sameloc.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

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
        target_kind = TargetKind.video_derived
        keyframe_timestamps = [10.0, 40.0, 70.0]
        multi_target_images = True
        segment_start_idx = 0
        segment_end_idx = 1
        task_summary = "同酒店多源镜头"

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


def test_no_keyframe_retry_when_all_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """提案戳全被软验收拒绝时直接失败，不触发凑数重试。"""
    video = tmp_path / "noretry.mp4"
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

    with pytest.raises(RuntimeError, match="未能截取任何被定位关键帧"):
        audit.run_audit_split(str(video), _transcript())
    assert "_LLMKeyframeRetry" not in call_log
    assert not hasattr(audit, "_request_alt_timestamps")


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
