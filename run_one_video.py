"""单视频入口：串联 stage0–stage7，落盘中间产物并维护 manifest checkpoint。"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pipeline.config import Settings, get_settings
from pipeline.schemas import (
    AgentRole,
    DatasetEntry,
    Move,
    NormalizedStep,
    ObservationExecutionResult,
    PreprocessResult,
    StageManifestEntry,
    StageStatus,
    TimedScreenAction,
    Trajectory,
    TrajectoryVerificationReport,
    VideoInput,
    VideoManifest,
)
from pipeline.stage0_preprocess import preprocess
from pipeline.stage1_parse import detect_screen_actions, extract_keyframes
from pipeline.stage2_moves import build_all_agent_moves
from pipeline.stage3_normalize import normalize_to_steps
from pipeline.stage4_observe import generate_observations
from pipeline.stage5_reconstruct import (
    reconstruct_all_trajectories,
    reconstruct_revision_trajectories,
)
from pipeline.stage6_verify import verify_and_score
from pipeline.stage7_format import format_all_and_save

logger = logging.getLogger(__name__)

STAGE_ORDER: list[str] = [
    "stage0",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "stage7",
]

_STAGE_OUTPUT_FILES: dict[str, str] = {
    "stage0": "stage0_preprocess.json",
    "stage1": "stage1_screen_actions.json",
    "stage2": "stage2_moves.json",
    "stage3": "stage3_normalized_steps.json",
    "stage4": "stage4_observations.json",
    "stage5": "stage5_trajectories.json",
    "stage6": "stage6_verification.json",
    "stage7": "stage7_entries.json",
}

# 可注入的阶段钩子（测试 mock）
StageHooks = dict[str, Callable[..., Any]]


def video_id_from_path(video_path: str) -> str:
    """由视频路径推导 video_id（文件名 stem）。"""
    return Path(video_path).stem


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "model_dump"):
        data = obj.model_dump(mode="json")
    else:
        data = obj
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def intermediate_dir(video_id: str, settings: Settings) -> Path:
    return Path(settings.INTERMEDIATE_DIR) / video_id


def load_or_create_manifest(video_id: str, settings: Settings) -> VideoManifest:
    """读取或初始化 VideoManifest。"""
    path = intermediate_dir(video_id, settings) / "manifest.json"
    if path.is_file():
        return VideoManifest.model_validate(_load_json(path))
    return VideoManifest(
        video_id=video_id,
        stages=[
            StageManifestEntry(stage=name, status=StageStatus.PENDING)
            for name in STAGE_ORDER
        ],
    )


def save_manifest(manifest: VideoManifest, settings: Settings) -> None:
    path = intermediate_dir(manifest.video_id, settings) / "manifest.json"
    _dump_json(path, manifest)


def _get_stage_entry(manifest: VideoManifest, stage: str) -> StageManifestEntry:
    for entry in manifest.stages:
        if entry.stage == stage:
            return entry
    entry = StageManifestEntry(stage=stage, status=StageStatus.PENDING)
    manifest.stages.append(entry)
    return entry


def _should_skip_stage(entry: StageManifestEntry, input_hash: str) -> bool:
    """completed 且 input_hash 未变 → 跳过（断点续跑）。"""
    return (
        entry.status == StageStatus.COMPLETED
        and entry.input_hash is not None
        and entry.input_hash == input_hash
        and entry.output_hash is not None
    )


def _mark_running(
    manifest: VideoManifest,
    stage: str,
    input_hash: str,
    settings: Settings,
) -> None:
    entry = _get_stage_entry(manifest, stage)
    entry.status = StageStatus.RUNNING
    entry.input_hash = input_hash
    entry.started_at = _utcnow_iso()
    entry.finished_at = None
    entry.error_message = None
    save_manifest(manifest, settings)


def _mark_completed(
    manifest: VideoManifest,
    stage: str,
    output_hash: str,
    settings: Settings,
) -> None:
    entry = _get_stage_entry(manifest, stage)
    entry.status = StageStatus.COMPLETED
    entry.output_hash = output_hash
    entry.finished_at = _utcnow_iso()
    entry.error_message = None
    save_manifest(manifest, settings)


def _mark_failed(
    manifest: VideoManifest,
    stage: str,
    error: str,
    settings: Settings,
) -> None:
    entry = _get_stage_entry(manifest, stage)
    entry.status = StageStatus.FAILED
    entry.finished_at = _utcnow_iso()
    entry.error_message = error[:2000]
    save_manifest(manifest, settings)


def _invalidate_downstream(
    manifest: VideoManifest,
    from_stage: str,
    settings: Settings,
) -> None:
    """上游重跑时将下游标记为 invalidated。"""
    try:
        idx = STAGE_ORDER.index(from_stage)
    except ValueError:
        return
    for name in STAGE_ORDER[idx + 1 :]:
        entry = _get_stage_entry(manifest, name)
        if entry.status in (StageStatus.COMPLETED, StageStatus.FAILED):
            entry.status = StageStatus.INVALIDATED
            entry.error_message = f"上游 {from_stage} 变更，需重跑"
    save_manifest(manifest, settings)


def is_video_fully_completed(video_id: str, settings: Optional[Settings] = None) -> bool:
    """若 stage0–7 全部 completed 则视为整视频完成（batch 可跳过）。"""
    cfg = settings or get_settings()
    manifest = load_or_create_manifest(video_id, cfg)
    by_name = {e.stage: e for e in manifest.stages}
    return all(
        by_name.get(s) is not None and by_name[s].status == StageStatus.COMPLETED
        for s in STAGE_ORDER
    )


def _narration_for_range(
    transcript: list[Any],
    time_range: tuple[float, float],
) -> str:
    start, end = time_range
    parts: list[str] = []
    for seg in transcript:
        if seg.end > start and seg.start < end:
            parts.append(seg.text)
    return " ".join(parts)


def _role_key(role: AgentRole) -> str:
    return role.value


def _serialize_role_map(data: dict[AgentRole, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, value in data.items():
        if isinstance(value, list):
            result[_role_key(role)] = [
                v.model_dump(mode="json") if hasattr(v, "model_dump") else v
                for v in value
            ]
        elif hasattr(value, "model_dump"):
            result[_role_key(role)] = value.model_dump(mode="json")
        else:
            result[_role_key(role)] = value
    return result


def _load_role_list(
    raw: dict[str, Any],
    model_cls: type,
) -> dict[AgentRole, list[Any]]:
    out: dict[AgentRole, list[Any]] = {}
    for key, items in raw.items():
        role = AgentRole(key)
        out[role] = [model_cls.model_validate(x) for x in items]
    return out


def _pick_image_path(
    keyframes_by_role: dict[AgentRole, list[str]],
    video_path: str,
) -> str:
    """选取代表性帧：优先 COARSE 首帧，否则任意非空，再否则视频路径占位。"""
    for role in (AgentRole.COARSE, AgentRole.FINE, AgentRole.VERIFIER):
        frames = keyframes_by_role.get(role) or []
        if frames:
            return frames[0]
    return video_path


def run_one_video(
    video_input: VideoInput,
    *,
    video_id: Optional[str] = None,
    settings: Optional[Settings] = None,
    hooks: Optional[StageHooks] = None,
    force_rerun_from: Optional[str] = None,
) -> dict[str, Any]:
    """串联 stage0–7，支持 checkpoint 断点续跑。

    Args:
        video_input: 单视频输入（含 transcript / groundtruth）。
        video_id: 可选覆盖；默认取视频文件名 stem。
        settings: 配置覆盖。
        hooks: 可选阶段函数覆盖（测试注入，键为 stage 名或函数名）。
        force_rerun_from: 若指定，自该阶段起强制重跑（含下游）。

    Returns:
        含 video_id、entries 数量、manifest 路径等摘要。
    """
    cfg = settings or get_settings()
    hooks = hooks or {}
    vid = video_id or video_id_from_path(video_input.video_path)
    inter = intermediate_dir(vid, cfg)
    inter.mkdir(parents=True, exist_ok=True)

    manifest = load_or_create_manifest(vid, cfg)
    if force_rerun_from is not None:
        try:
            start_idx = STAGE_ORDER.index(force_rerun_from)
        except ValueError as exc:
            raise ValueError(f"未知阶段: {force_rerun_from}") from exc
        for name in STAGE_ORDER[start_idx:]:
            entry = _get_stage_entry(manifest, name)
            entry.status = StageStatus.INVALIDATED
        save_manifest(manifest, cfg)

    # ---------- 可变状态（跨阶段） ----------
    preprocess_result: Optional[PreprocessResult] = None
    screen_actions_by_role: dict[AgentRole, list[TimedScreenAction]] = {}
    keyframes_by_role: dict[AgentRole, list[str]] = {}
    moves_by_role: dict[AgentRole, list[Move]] = {}
    steps_by_role: dict[AgentRole, list[NormalizedStep]] = {}
    observations_by_role: dict[AgentRole, list[ObservationExecutionResult]] = {}
    trajectories: dict[AgentRole, Trajectory] = {}
    all_trajectories: list[Trajectory] = []
    reports_by_id: dict[str, TrajectoryVerificationReport] = {}
    entries: list[DatasetEntry] = []
    image_path: str = video_input.video_path

    # ---- stage0 ----
    stage = "stage0"
    in_hash = _stable_hash(
        {
            "transcript": [s.model_dump(mode="json") for s in video_input.transcript],
            "source_platform": video_input.source_platform,
        }
    )
    entry0 = _get_stage_entry(manifest, stage)
    out_path0 = inter / _STAGE_OUTPUT_FILES[stage]
    if _should_skip_stage(entry0, in_hash) and out_path0.is_file():
        preprocess_result = PreprocessResult.model_validate(_load_json(out_path0))
        logger.info("[%s] skip %s", vid, stage)
    else:
        _invalidate_downstream(manifest, stage, cfg)
        _mark_running(manifest, stage, in_hash, cfg)
        try:
            fn = hooks.get("preprocess") or hooks.get(stage) or preprocess
            preprocess_result = fn(video_input)
            _dump_json(out_path0, preprocess_result)
            _mark_completed(manifest, stage, _stable_hash(_load_json(out_path0)), cfg)
        except Exception as exc:
            _mark_failed(manifest, stage, str(exc), cfg)
            raise

    assert preprocess_result is not None

    # ---- stage1 ----
    stage = "stage1"
    in_hash = _stable_hash(
        {
            "video_path": video_input.video_path,
            "agent_segments": [
                s.model_dump(mode="json") for s in preprocess_result.agent_segments
            ],
        }
    )
    entry1 = _get_stage_entry(manifest, stage)
    out_path1 = inter / _STAGE_OUTPUT_FILES[stage]
    if _should_skip_stage(entry1, in_hash) and out_path1.is_file():
        raw1 = _load_json(out_path1)
        screen_actions_by_role = _load_role_list(raw1["screen_actions"], TimedScreenAction)
        keyframes_by_role = {
            AgentRole(k): list(v) for k, v in raw1.get("keyframes", {}).items()
        }
        image_path = raw1.get("image_path") or _pick_image_path(
            keyframes_by_role, video_input.video_path
        )
        logger.info("[%s] skip %s", vid, stage)
    else:
        _invalidate_downstream(manifest, stage, cfg)
        _mark_running(manifest, stage, in_hash, cfg)
        try:
            extract_fn = hooks.get("extract_keyframes") or extract_keyframes
            detect_fn = hooks.get("detect_screen_actions") or detect_screen_actions
            screen_actions_by_role = {}
            keyframes_by_role = {}
            for seg in preprocess_result.agent_segments:
                tr = (seg.start_time, seg.end_time)
                frames = extract_fn(video_input.video_path, tr)
                narration = _narration_for_range(video_input.transcript, tr)
                actions = detect_fn(frames, narration, tr)
                screen_actions_by_role[seg.agent_role] = actions
                keyframes_by_role[seg.agent_role] = frames
            image_path = _pick_image_path(keyframes_by_role, video_input.video_path)
            payload = {
                "screen_actions": _serialize_role_map(screen_actions_by_role),
                "keyframes": {r.value: fs for r, fs in keyframes_by_role.items()},
                "image_path": image_path,
            }
            _dump_json(out_path1, payload)
            _mark_completed(manifest, stage, _stable_hash(payload), cfg)
        except Exception as exc:
            _mark_failed(manifest, stage, str(exc), cfg)
            raise

    # ---- stage2 ----
    stage = "stage2"
    in_hash = _stable_hash(
        {
            "preprocess": preprocess_result.model_dump(mode="json"),
            "screen_actions": _serialize_role_map(screen_actions_by_role),
        }
    )
    entry2 = _get_stage_entry(manifest, stage)
    out_path2 = inter / _STAGE_OUTPUT_FILES[stage]
    if _should_skip_stage(entry2, in_hash) and out_path2.is_file():
        moves_by_role = _load_role_list(_load_json(out_path2), Move)
        logger.info("[%s] skip %s", vid, stage)
    else:
        _invalidate_downstream(manifest, stage, cfg)
        _mark_running(manifest, stage, in_hash, cfg)
        try:
            fn = hooks.get("build_all_agent_moves") or hooks.get(stage) or build_all_agent_moves
            moves_by_role = fn(
                video_input, preprocess_result, screen_actions_by_role
            )
            payload = _serialize_role_map(moves_by_role)
            _dump_json(out_path2, payload)
            _mark_completed(manifest, stage, _stable_hash(payload), cfg)
        except Exception as exc:
            _mark_failed(manifest, stage, str(exc), cfg)
            raise

    # ---- stage3 ----
    stage = "stage3"
    in_hash = _stable_hash(_serialize_role_map(moves_by_role))
    entry3 = _get_stage_entry(manifest, stage)
    out_path3 = inter / _STAGE_OUTPUT_FILES[stage]
    if _should_skip_stage(entry3, in_hash) and out_path3.is_file():
        steps_by_role = _load_role_list(_load_json(out_path3), NormalizedStep)
        logger.info("[%s] skip %s", vid, stage)
    else:
        _invalidate_downstream(manifest, stage, cfg)
        _mark_running(manifest, stage, in_hash, cfg)
        try:
            fn = hooks.get("normalize_to_steps") or normalize_to_steps
            steps_by_role = {}
            for role in (AgentRole.COARSE, AgentRole.FINE, AgentRole.VERIFIER):
                moves = moves_by_role.get(role) or []
                steps_by_role[role] = fn(moves, role)
            payload = _serialize_role_map(steps_by_role)
            _dump_json(out_path3, payload)
            _mark_completed(manifest, stage, _stable_hash(payload), cfg)
        except Exception as exc:
            _mark_failed(manifest, stage, str(exc), cfg)
            raise

    # ---- stage4 ----
    stage = "stage4"
    in_hash = _stable_hash(
        {
            "steps": _serialize_role_map(steps_by_role),
            "image_path": image_path,
        }
    )
    entry4 = _get_stage_entry(manifest, stage)
    out_path4 = inter / _STAGE_OUTPUT_FILES[stage]
    if _should_skip_stage(entry4, in_hash) and out_path4.is_file():
        observations_by_role = _load_role_list(
            _load_json(out_path4), ObservationExecutionResult
        )
        logger.info("[%s] skip %s", vid, stage)
    else:
        _invalidate_downstream(manifest, stage, cfg)
        _mark_running(manifest, stage, in_hash, cfg)
        try:
            fn = hooks.get("generate_observations") or generate_observations
            observations_by_role = {}
            for role in (AgentRole.COARSE, AgentRole.FINE, AgentRole.VERIFIER):
                observations_by_role[role] = fn(
                    steps_by_role.get(role) or [],
                    image_path,
                    role,
                )
            payload = _serialize_role_map(observations_by_role)
            _dump_json(out_path4, payload)
            _mark_completed(manifest, stage, _stable_hash(payload), cfg)
        except Exception as exc:
            _mark_failed(manifest, stage, str(exc), cfg)
            raise

    # ---- stage5 ----
    stage = "stage5"
    in_hash = _stable_hash(
        {
            "steps": _serialize_role_map(steps_by_role),
            "observations": _serialize_role_map(observations_by_role),
            "answer_timestamp": preprocess_result.answer_timestamp,
            "image_path": image_path,
        }
    )
    entry5 = _get_stage_entry(manifest, stage)
    out_path5 = inter / _STAGE_OUTPUT_FILES[stage]
    if _should_skip_stage(entry5, in_hash) and out_path5.is_file():
        raw5 = _load_json(out_path5)
        trajectories = {
            AgentRole(k): Trajectory.model_validate(v)
            for k, v in raw5["main"].items()
        }
        all_trajectories = [Trajectory.model_validate(t) for t in raw5.get("all", [])]
        if not all_trajectories:
            all_trajectories = list(trajectories.values())
        logger.info("[%s] skip %s", vid, stage)
    else:
        _invalidate_downstream(manifest, stage, cfg)
        _mark_running(manifest, stage, in_hash, cfg)
        try:
            recon_all = (
                hooks.get("reconstruct_all_trajectories") or reconstruct_all_trajectories
            )
            recon_rev = (
                hooks.get("reconstruct_revision_trajectories")
                or reconstruct_revision_trajectories
            )
            trajectories = recon_all(
                steps_by_role,
                observations_by_role,
                preprocess_result.answer_timestamp,
                image_path,
            )
            all_trajectories = list(trajectories.values())

            # 视频内纠错返工（高价值）；system_feedback 在 stage6 后补充
            if preprocess_result.revision_segments:
                # 占位 VerificationResult：video_observed 不依赖 verdict
                from pipeline.schemas import VerificationResult

                placeholder = VerificationResult(
                    verdict="pass",
                    failed_checks=[],
                    suggested_recheck="",
                    return_to_agent=None,
                )
                try:
                    revs = recon_rev(
                        trajectories,
                        placeholder,
                        steps_by_role,
                        observations_by_role,
                        preprocess_result.answer_timestamp,
                        image_path,
                        revision_round=1,
                        max_revision_rounds=cfg.MAX_REVISION_ROUNDS,
                        video_revision_segments=list(
                            preprocess_result.revision_segments
                        ),
                    )
                    all_trajectories.extend(revs)
                except Exception as rev_exc:  # noqa: BLE001
                    # 主轨迹已产出；返工 LLM 断连不应整阶段失败
                    logger.warning(
                        "[%s] video_observed revision 失败，继续主轨迹: %s",
                        vid,
                        rev_exc,
                    )

            payload = {
                "main": {r.value: t.model_dump(mode="json") for r, t in trajectories.items()},
                "all": [t.model_dump(mode="json") for t in all_trajectories],
            }
            _dump_json(out_path5, payload)
            _mark_completed(manifest, stage, _stable_hash(payload), cfg)
        except Exception as exc:
            _mark_failed(manifest, stage, str(exc), cfg)
            raise

    # ---- stage6 ----
    stage = "stage6"
    in_hash = _stable_hash(
        {
            "trajectories": [t.model_dump(mode="json") for t in all_trajectories],
            "groundtruth": list(video_input.groundtruth),
        }
    )
    entry6 = _get_stage_entry(manifest, stage)
    out_path6 = inter / _STAGE_OUTPUT_FILES[stage]
    if _should_skip_stage(entry6, in_hash) and out_path6.is_file():
        raw6 = _load_json(out_path6)
        reports_by_id = {
            tid: TrajectoryVerificationReport.model_validate(rep)
            for tid, rep in raw6["reports"].items()
        }
        # 若 stage5 被 skip，all_trajectories 已加载；否则保持
        if raw6.get("all_trajectories"):
            all_trajectories = [
                Trajectory.model_validate(t) for t in raw6["all_trajectories"]
            ]
        logger.info("[%s] skip %s", vid, stage)
    else:
        _invalidate_downstream(manifest, stage, cfg)
        _mark_running(manifest, stage, in_hash, cfg)
        try:
            verify_fn = hooks.get("verify_and_score") or verify_and_score
            recon_rev = (
                hooks.get("reconstruct_revision_trajectories")
                or reconstruct_revision_trajectories
            )
            reports_by_id = {}
            for traj in list(all_trajectories):
                report = verify_fn(traj, video_input.groundtruth)
                reports_by_id[traj.id] = report

            # system_feedback 返工：主 VERIFIER fail 时打回
            verifier_main = trajectories.get(AgentRole.VERIFIER)
            if verifier_main is not None and verifier_main.verifier_output is not None:
                v_out = verifier_main.verifier_output
                if v_out.verdict == "fail":
                    revs = recon_rev(
                        trajectories,
                        v_out,
                        steps_by_role,
                        observations_by_role,
                        preprocess_result.answer_timestamp,
                        image_path,
                        revision_round=1,
                        max_revision_rounds=cfg.MAX_REVISION_ROUNDS,
                        video_revision_segments=None,
                    )
                    for rev in revs:
                        if any(t.id == rev.id for t in all_trajectories):
                            continue
                        all_trajectories.append(rev)
                        reports_by_id[rev.id] = verify_fn(
                            rev, video_input.groundtruth
                        )

            payload = {
                "reports": {
                    tid: r.model_dump(mode="json") for tid, r in reports_by_id.items()
                },
                "all_trajectories": [
                    t.model_dump(mode="json") for t in all_trajectories
                ],
            }
            _dump_json(out_path6, payload)
            # 同步更新 stage5 all（含返工）便于审计
            stage5_payload = {
                "main": {
                    r.value: t.model_dump(mode="json") for r, t in trajectories.items()
                },
                "all": [t.model_dump(mode="json") for t in all_trajectories],
            }
            _dump_json(inter / _STAGE_OUTPUT_FILES["stage5"], stage5_payload)
            _mark_completed(manifest, stage, _stable_hash(payload), cfg)
        except Exception as exc:
            _mark_failed(manifest, stage, str(exc), cfg)
            raise

    # ---- stage7 ----
    stage = "stage7"
    in_hash = _stable_hash(
        {
            "trajectories": [t.model_dump(mode="json") for t in all_trajectories],
            "reports": {
                tid: r.model_dump(mode="json") for tid, r in reports_by_id.items()
            },
            "groundtruth": list(video_input.groundtruth),
        }
    )
    entry7 = _get_stage_entry(manifest, stage)
    out_path7 = inter / _STAGE_OUTPUT_FILES[stage]
    if _should_skip_stage(entry7, in_hash) and out_path7.is_file():
        raw7 = _load_json(out_path7)
        entries = [DatasetEntry.model_validate(e) for e in raw7]
        logger.info("[%s] skip %s", vid, stage)
    else:
        _mark_running(manifest, stage, in_hash, cfg)
        try:
            format_fn = hooks.get("format_all_and_save") or format_all_and_save
            meta: dict[str, Any] = {
                "source_video": vid,
                "groundtruth": video_input.groundtruth,
                "reports": {
                    tid: {
                        "quality_score": rep.quality_score,
                        "verified": rep.passed,
                        "distance_error_km": rep.distance_error_km,
                    }
                    for tid, rep in reports_by_id.items()
                },
            }
            entries = format_fn(
                all_trajectories,
                meta,
                cfg.OUTPUT_DIR,
                vid,
            )
            _dump_json(out_path7, [e.model_dump(mode="json") for e in entries])
            _mark_completed(
                manifest,
                stage,
                _stable_hash([e.model_dump(mode="json") for e in entries]),
                cfg,
            )
        except Exception as exc:
            _mark_failed(manifest, stage, str(exc), cfg)
            raise

    verified_count = sum(1 for e in entries if e.verified)
    return {
        "video_id": vid,
        "status": "completed",
        "entries_total": len(entries),
        "entries_verified": verified_count,
        "manifest_path": str(inter / "manifest.json"),
        "intermediate_dir": str(inter),
    }


def main() -> None:
    """CLI：``python run_one_video.py --video ... --transcript ... --gt LAT,LNG``。"""
    import argparse

    parser = argparse.ArgumentParser(description="单视频 stage0–7 流水线")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument(
        "--transcript",
        required=True,
        help="文字稿 JSON（TranscriptSegment 列表）",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="groundtruth 坐标 LAT,LNG",
    )
    parser.add_argument("--platform", default="unknown", help="来源平台")
    parser.add_argument("--video-id", default=None, help="覆盖 video_id")
    parser.add_argument(
        "--force-from",
        default=None,
        help="自该阶段强制重跑（如 stage4）",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    lat_s, lng_s = args.gt.split(",", 1)
    segments = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
    from pipeline.schemas import TranscriptSegment

    video_input = VideoInput(
        video_path=args.video,
        transcript=[TranscriptSegment.model_validate(s) for s in segments],
        groundtruth=(float(lat_s), float(lng_s)),
        source_platform=args.platform,
    )
    result = run_one_video(
        video_input,
        video_id=args.video_id,
        force_rerun_from=args.force_from,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
