"""阶段1.5：字幕 + 稀疏帧 → 拒识或切分为多定位任务并截关键帧。"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from pipeline.config import get_settings
from pipeline.llm import call_structured
from pipeline.media.keyframes import extract_keyframes, video_duration_sec
from pipeline.schemas.audit import (
    AuditDecision,
    AuditSplitResult,
    GeoTaskSpec,
    TargetKind,
)
from pipeline.schemas.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

MULTI_STAMP_MIN_GAP = 45.0
# video_derived：仅去掉同镜头连拍，勿用静图多图的大间隔误并不同源输入。
VIDEO_DERIVED_NEAR_GAP = 10.0

AUDIT_SYSTEM_HINT = (
    "你审核讲解视频是否适合蒸馏为「图片/静帧地理定位」训练样本。"
    "输入为带时间戳字幕与若干稀疏审计帧（仅供审核，不是训练图）；"
    "禁止使用 groundtruth；禁止按固定词表或渠道特判。\n"
    "拒识主问句：若去掉讲解/旁白/答案，是否仍存在需 agent 定位的图或场景？\n"
    "- has_unresolved_target=false → decision=reject："
    "地名科普、历史故事、奇观介绍、地点仅为讲述对象、"
    "旁白「打开地图就能看到某某地」≠ 定位题。\n"
    "- has_unresolved_target=true → decision=accept："
    "存在一个或多个独立的待定位输入。\n"
    "选关键帧前先理解整段定位过程，区分过程角色：\n"
    "- (A) 待定位实拍输入：人眼/相机拍到的、被定位的现场场景或静帧；\n"
    "- (B) 推理工具与核验步骤：用地图或影像底图比对、街景浏览确认等工具画面；\n"
    "- (C) 答案揭晓：钉点指出「就是这里」、揭晓界面等。\n"
    "切分粒度（关键）：一个 task = 一次独立定位题 = 一条最终答案链。\n"
    "- 同题多图（多张待定位原图共同支撑同一最终地点，或后图只是精化前图）"
    "必须合并为 **一个** task，设 multi_target_images=true，"
    "并给出每张待定位原图出现的时间戳。\n"
    "- **同一最终地点 / 同一条答案链必须合并为一个 task**"
    "（即使线索形态不同，如环境特征与建筑外观）。\n"
    "- 仅当不同目标、不同最终地点、彼此独立出题时才拆成多个 task。\n"
    "- 禁止把「同一条推理链里的第二张参考图」拆成第二个 task。\n"
    "target_kind：\n"
    "- still_image：明确静图/待定位原图；\n"
    "- video_derived：对源视频/连续实拍场景定位（仍入库）。\n"
    "keyframe_timestamps **只选过程角色 (A)**："
    "去掉讲解后 agent 仍要据之定位的实拍主画面（定位输入本身）。\n"
    "当 multi_target_images=true 或 video_derived 时："
    "须一次列出**每一个**独立 (A) 实拍输入的代表时刻（同题列全）；"
    "漏报任一输入镜头会导致样本不完整；张数随独立输入走。\n"
    "**摘要与时间戳对齐**：task_summary 若枚举多个独立实拍输入"
    "（镜头/静帧/目标建筑外观对照等），keyframe_timestamps 必须各有代表时刻；"
    "摘要写了 N 个输入却只给少于 N 个戳 = 不完整。\n"
    "比对/核验旁白段里若穿插「目标建筑外观、早期实拍静帧」等被定位画面，仍属 (A) 应选；"
    "同段地图底图、街景浏览、钉点揭晓仍属 (B)(C)，不得选。\n"
    "**禁止选 (B)(C)**：工具步骤与答案揭晓不是待定位图，"
    "即使画面「很地理」也不得写入 keyframe_timestamps。\n"
    "判断靠过程角色，不靠外观品类清单。"
    "若 task_summary 写外观/镜头等实拍，时间戳必须落在对应实拍段，"
    "不得用工具段或揭晓段顶替。\n"
    "旁白刚提到下一输入时若画面仍是工具/讲解辅助界面，该时刻无效。\n"
    "- still_image 且非同题多图：默认 1 个时间戳；\n"
    "- multi_target_images=true：每张待定位实拍原图各 1 个时间戳；\n"
    "- video_derived：每个独立源实拍输入镜头各 1 个时间戳。\n"
    "time_start/time_end 与可选 segment 索引 = **整条答案链旁白窗口**："
    "从问题设定到最终地点结论（含地图比对叙述与揭晓句），"
    "不得裁成「仅关键帧所在中段」。"
    "时间窗管蒸馏材料完整性；keyframe_timestamps 仍只选 (A)。\n"
    "每个 task 给出 time_start/time_end、keyframe_timestamps（至少 1 个；"
    "同题多图至少 2 个）、multi_target_images、可选 segment 索引、task_summary。"
    "不要输出图像路径。"
)

FRAME_VERIFY_HINT = (
    "判断这张视频截帧是否可作为「定位输入」写入训练关键帧。"
    "只输出 kind 与简短 reason。用过程角色判断，不要按控件/App 品类清单执法。\n"
    "先问：这是被定位的实拍，还是解题时调用的工具/核验画面？\n"
    "- target_photo：主画面是待定位的实拍场景"
    "（地面/现场镜头、静帧照片或目标建筑外观实拍占主画面）。"
    "即使叠有讲解字幕、箭头、方位字，仍判 target_photo。"
    "全屏建筑外观实拍即使夹在核验旁白时间线，仍判 target_photo。\n"
    "- teaching_ui：主内容是定位过程中的工具或核验画面"
    "（用地图/影像底图比对、街景浏览、答案钉点揭晓等），"
    "即使画面「很地理」也不是待定位输入；"
    "亦含纯讲解辅助可视化或过程推演板。\n"
    "不得仅因源实拍上有讲解标注就判 teaching_ui；"
    "也不得把工具步骤画面当成 target_photo。\n"
    "- other：仅黑屏、片尾、或与定位目标完全无关的画面；"
    "不得因源实拍上有方位字/箭头/字幕叠字就判 other。\n"
    "须输入画面本身全屏或主画面。"
)



class FrameKind(str, Enum):
    """关键帧视觉验收类别。"""

    target_photo = "target_photo"
    teaching_ui = "teaching_ui"
    other = "other"


class _LLMGeoTaskDraft(BaseModel):
    """LLM 草稿任务（截帧前）。"""

    time_start: float
    time_end: float
    target_kind: TargetKind
    keyframe_timestamps: list[float] = Field(default_factory=list)
    multi_target_images: bool = False
    segment_start_idx: Optional[int] = None
    segment_end_idx: Optional[int] = None
    task_summary: str = ""


class _LLMAuditDraft(BaseModel):
    """LLM 审核草稿。"""

    decision: AuditDecision
    reason: str = ""
    has_unresolved_target: bool = True
    tasks: list[_LLMGeoTaskDraft] = Field(default_factory=list)


class _LLMFrameVerdict(BaseModel):
    """单帧视觉验收。"""

    kind: FrameKind
    reason: str = ""


class _LLMTaskMergeResult(BaseModel):
    """多 task 合并复核。"""

    tasks: list[_LLMGeoTaskDraft] = Field(default_factory=list)
    reason: str = ""


def _seed_photo_mention_timestamps(
    transcript: list[TranscriptSegment],
) -> list[float]:
    """从字幕中照片/镜头提及处取候选时刻（含邻域偏移，非样本特判）。"""
    keys = (
        "照片",
        "这张图",
        "第二张",
        "原图",
        "两张图",
        "放大照片",
        "求助图",
        "镜头",
        "首先来看",
    )
    mids: list[float] = []
    for seg in transcript:
        text = seg.text or ""
        if any(k in text for k in keys):
            mid = (float(seg.start) + float(seg.end)) * 0.5
            if not mids or abs(mids[-1] - mid) > 1.0:
                mids.append(mid)
    stamps: list[float] = []
    for mid in mids:
        for delta in (-3.0, 0.0, 3.0, 8.0):
            t = mid + delta
            if t < 0:
                continue
            if not stamps or abs(stamps[-1] - t) > 0.5:
                stamps.append(t)
    return stamps


def _maybe_merge_same_question_tasks(
    draft_tasks: list[_LLMGeoTaskDraft],
    *,
    video_id: str,
    transcript: list[TranscriptSegment],
    overview_images: list[str] | None,
) -> list[_LLMGeoTaskDraft]:
    """多 task 时复核：同题多图必须合并为一个 multi_target_images task。"""
    if len(draft_tasks) <= 1:
        return draft_tasks
    payload = [
        {
            "time_start": t.time_start,
            "time_end": t.time_end,
            "target_kind": t.target_kind.value,
            "keyframe_timestamps": t.keyframe_timestamps,
            "multi_target_images": t.multi_target_images,
            "segment_start_idx": t.segment_start_idx,
            "segment_end_idx": t.segment_end_idx,
            "task_summary": t.task_summary,
        }
        for t in draft_tasks
    ]
    prompt = (
        "以下是审核切分得到的多个 tasks。请复核切分粒度：\n"
        "一个 task = 一次独立定位题 = 一条最终答案。\n"
        "若多张待定位原图共同支撑同一最终地点（或后图精化前图），"
        "必须合并为 **一个** task，设 multi_target_images=true，"
        "并给出每张待定位原图出现的 keyframe_timestamps。\n"
        "**同一最终地点 / 同一条答案链必须合并**"
        "（即使线索形态不同，如河道环境与建筑外观）。\n"
        "仅当不同目标、不同最终地点时才保留多个 task。\n"
        f"视频 ID: {video_id}\n"
        f"当前 tasks JSON:\n{json.dumps(payload, ensure_ascii=False)}\n"
        "字幕：\n"
        f"{_format_transcript(transcript)}\n"
        "请输出合并后的 tasks 与 reason。"
    )
    merged = call_structured(
        prompt,
        _LLMTaskMergeResult,
        images=overview_images or None,
        lane="vlm",
    )
    if merged.tasks:
        logger.info(
            "task merge review: %s -> %s (%s)",
            len(draft_tasks),
            len(merged.tasks),
            merged.reason,
        )
        return merged.tasks
    return draft_tasks


def _format_transcript(transcript: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(transcript):
        lines.append(
            f"[{i}] [{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}"
        )
    return "\n".join(lines)


def _pick_sparse_timestamps(duration: float, count: int) -> list[float]:
    if duration <= 0:
        return [0.0]
    n = max(1, int(count))
    if n == 1:
        return [duration * 0.5]
    return [duration * i / (n - 1) for i in range(n)]


def _clamp_timestamps(
    stamps: list[float],
    *,
    start: float,
    end: float,
    max_n: int,
) -> list[float]:
    lo = max(0.0, float(start))
    hi = max(lo, float(end))
    cleaned: list[float] = []
    for raw in stamps:
        t = float(raw)
        if t < lo:
            t = lo
        if t > hi:
            t = hi
        if not cleaned or abs(cleaned[-1] - t) > 1e-3:
            cleaned.append(t)
    if not cleaned:
        cleaned = [lo if hi <= lo else (lo + hi) * 0.5]
    return cleaned[: max(1, max_n)]


def _max_keyframes_for_task(
    target_kind: TargetKind,
    multi_target_images: bool,
    configured_max: int,
) -> int:
    """still_image 默认单帧；多原图或 video_derived 才允许多帧。"""
    hard_cap = max(1, int(configured_max))
    if target_kind == TargetKind.video_derived:
        return hard_cap
    if multi_target_images:
        # 同题多静帧：通常每张原图一帧，上限 2 避免同图变体堆叠
        return min(hard_cap, 2)
    return 1


def _prefix_keyframes(paths: list[str], task_id: str) -> list[str]:
    """将抽帧文件重命名为 ``{task_id}_原名``，避免同目录时间戳冲突。"""
    prefixed: list[str] = []
    for raw in paths:
        src = Path(raw)
        if not src.is_file():
            continue
        if src.name.startswith(f"{task_id}_"):
            prefixed.append(str(src.resolve()))
            continue
        dest = src.with_name(f"{task_id}_{src.name}")
        if dest.resolve() != src.resolve():
            if dest.exists():
                dest.unlink()
            src.replace(dest)
        prefixed.append(str(dest.resolve()))
    return prefixed


def _unlink_quiet(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def verify_keyframe_kind(image_path: str) -> FrameKind:
    """VLM 验收单帧是否为待定位目标图。"""
    verdict = call_structured(
        FRAME_VERIFY_HINT,
        _LLMFrameVerdict,
        images=[image_path],
        lane="vlm",
    )
    return verdict.kind


def _prioritize_diverse_probe_order(stamps: list[float]) -> list[float]:
    """探测顺序：最早与最晚优先，便于同题多图拉开时间。"""
    if len(stamps) <= 2:
        return list(stamps)
    ordered = sorted(stamps)
    out = [ordered[0], ordered[-1]]
    for t in ordered[1:-1]:
        out.append(t)
    return out


def _is_near_duplicate_stamp(
    stamp: float, existing: list[float], gap: float
) -> bool:
    return any(abs(stamp - s) < gap for s in existing)


def _multi_span_ok(stamps: list[float], gap: float) -> bool:
    if len(stamps) < 2:
        return False
    return (max(stamps) - min(stamps)) >= gap


def _materialize_task_images(
    *,
    video_path: str,
    video_id: str,
    task_id: str,
    raw: _LLMGeoTaskDraft,
    t0: float,
    t1: float,
    max_kf: int,
    transcript: list[TranscriptSegment],
) -> tuple[list[float], list[str], bool]:
    """截帧 + 原则软验收（单次探测，不凑数重试）；返回 (stamps, paths, multi_flag)。"""
    multi = bool(raw.multi_target_images)
    video_derived = raw.target_kind == TargetKind.video_derived
    min_need = 2 if multi else 1
    # 同题多图常跨整段讲解；钳制窗过窄会把早期求助原图排除
    try:
        duration = float(video_duration_sec(video_path))
    except Exception:  # noqa: BLE001
        duration = max(t1, t0, 1.0)
    # still_image 同题多图：大间隔防同图变体；video_derived：小间隔仅去同镜头连拍
    if video_derived:
        near_gap = VIDEO_DERIVED_NEAR_GAP
        require_span = False
    elif multi:
        near_gap = max(MULTI_STAMP_MIN_GAP, float(duration) * 0.2)
        require_span = True
    else:
        near_gap = MULTI_STAMP_MIN_GAP
        require_span = False
    use_near_dedup = multi or video_derived
    if multi or video_derived:
        clamp_start, clamp_end = 0.0, max(duration, t1, 1.0)
    else:
        clamp_start, clamp_end = t0, (t1 if t1 > t0 else t0 + 0.1)
    pending = _prioritize_diverse_probe_order(
        _clamp_timestamps(
            list(raw.keyframe_timestamps)
            + _seed_photo_mention_timestamps(transcript),
            start=clamp_start,
            end=clamp_end,
            max_n=max(max_kf * 3, min_need + 8, 12),
        )
    )
    accepted_stamps: list[float] = []
    accepted_paths: list[str] = []
    tried: set[str] = set()
    frame_dir = Path(get_settings().INTERMEDIATE_DIR) / video_id

    for stamp in pending:
        if len(accepted_paths) >= max_kf and (
            not require_span or _multi_span_ok(accepted_stamps, near_gap)
        ):
            break
        key = f"{stamp:.3f}"
        if key in tried:
            continue
        tried.add(key)
        try:
            paths = extract_keyframes(
                video_path, [stamp], out_dir=str(frame_dir)
            )
            paths = _prefix_keyframes(paths, task_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "task %s stamp %.3f extract failed: %s",
                task_id,
                stamp,
                exc,
            )
            continue
        if not paths:
            continue
        path = paths[0]
        kind = verify_keyframe_kind(path)
        if kind == FrameKind.target_photo:
            if use_near_dedup and _is_near_duplicate_stamp(
                stamp, accepted_stamps, near_gap
            ):
                logger.info(
                    "skip near-duplicate stamp %.3f task=%s",
                    stamp,
                    task_id,
                )
                _unlink_quiet(path)
                continue
            # still multi 已满但时间不分散：用更晚的帧替换最近邻
            if require_span and len(accepted_paths) >= max_kf:
                if not _multi_span_ok(accepted_stamps, near_gap):
                    nearest_i = min(
                        range(len(accepted_stamps)),
                        key=lambda i: abs(accepted_stamps[i] - stamp),
                    )
                    if abs(accepted_stamps[nearest_i] - stamp) < near_gap:
                        _unlink_quiet(path)
                        continue
                    _unlink_quiet(accepted_paths[nearest_i])
                    accepted_stamps[nearest_i] = stamp
                    accepted_paths[nearest_i] = path
                continue
            if len(accepted_paths) >= max_kf:
                _unlink_quiet(path)
                continue
            accepted_stamps.append(stamp)
            accepted_paths.append(path)
        else:
            logger.info(
                "drop frame %s kind=%s task=%s",
                path,
                kind.value,
                task_id,
            )
            _unlink_quiet(path)

    if not accepted_paths:
        raise RuntimeError(f"task {task_id} 未能截取任何被定位关键帧")
    if multi and len(accepted_paths) < 2:
        raise RuntimeError(
            f"task {task_id} multi_target_images 验收后有效关键帧不足 2 张"
        )

    # 按时间排序，优先保留较早出现的待定位原图
    paired = sorted(
        zip(accepted_stamps, accepted_paths, strict=True),
        key=lambda x: x[0],
    )
    accepted_stamps = [p[0] for p in paired]
    accepted_paths = [p[1] for p in paired]

    # still_image 单图题钳制 1 帧；video_derived / multi 保留多帧
    single_still = (
        raw.target_kind == TargetKind.still_image and not multi
    )
    if single_still:
        for extra in accepted_paths[1:]:
            _unlink_quiet(extra)
        return accepted_stamps[:1], accepted_paths[:1], False

    for extra in accepted_paths[max_kf:]:
        _unlink_quiet(extra)
    return accepted_stamps[:max_kf], accepted_paths[:max_kf], multi


def slice_transcript_for_task(
    transcript: list[TranscriptSegment],
    task: GeoTaskSpec,
) -> list[TranscriptSegment]:
    """按 task 的字幕索引或时间窗切片。"""
    if not transcript:
        return []
    if (
        task.segment_start_idx is not None
        and task.segment_end_idx is not None
        and 0 <= task.segment_start_idx <= task.segment_end_idx < len(transcript)
    ):
        return transcript[task.segment_start_idx : task.segment_end_idx + 1]

    sliced = [
        seg
        for seg in transcript
        if seg.end >= task.time_start - 1e-6 and seg.start <= task.time_end + 1e-6
    ]
    return sliced if sliced else list(transcript)


def run_audit_split(
    video_path: str,
    transcript: list[TranscriptSegment],
    *,
    out_path: str | None = None,
) -> AuditSplitResult:
    """审核视频是否可蒸馏，并切分为带关键帧的定位任务。

    Args:
        video_path: 视频路径。
        transcript: 阶段1 字幕。
        out_path: 审核结果落盘路径；默认 intermediate/{id}/stage_audit_split.json。

    Returns:
        AuditSplitResult；reject 时 tasks 为空。
    """
    settings = get_settings()
    video_id = Path(video_path).stem
    duration = 0.0
    try:
        duration = float(video_duration_sec(video_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit duration probe failed: %s", exc)

    sparse_n = max(1, int(settings.AUDIT_SPARSE_FRAME_COUNT))
    sparse_stamps = _pick_sparse_timestamps(duration, sparse_n)
    sparse_dir = Path(settings.CACHE_DIR) / "audit_sparse" / video_id
    overview_images: list[str] = []
    try:
        overview_images = extract_keyframes(
            video_path, sparse_stamps, out_dir=str(sparse_dir)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit sparse keyframes failed: %s", exc)

    prompt = (
        f"{AUDIT_SYSTEM_HINT}\n\n"
        f"视频 ID: {video_id}\n"
        f"时长约: {duration:.1f}s\n"
        f"字幕段数: {len(transcript)}\n"
        "字幕（含段索引）：\n"
        f"{_format_transcript(transcript)}\n\n"
        "请输出 has_unresolved_target / decision / reason / tasks。"
    )
    draft = call_structured(
        prompt,
        _LLMAuditDraft,
        images=overview_images or None,
        lane="vlm",
    )

    force_reject = not bool(draft.has_unresolved_target)
    if draft.decision == AuditDecision.reject or force_reject:
        reason = draft.reason.strip() or "非地理定位任务"
        if force_reject and draft.decision != AuditDecision.reject:
            reason = (
                f"{reason}（强制拒识：has_unresolved_target=false）"
                if reason
                else "强制拒识：has_unresolved_target=false"
            )
        result = AuditSplitResult(
            video_id=video_id,
            decision=AuditDecision.reject,
            reason=reason,
            has_unresolved_target=False,
            tasks=[],
        )
    else:
        max_kf_cfg = max(1, int(settings.AUDIT_MAX_KEYFRAMES_PER_TASK))
        draft_tasks = _maybe_merge_same_question_tasks(
            list(draft.tasks),
            video_id=video_id,
            transcript=transcript,
            overview_images=overview_images or None,
        )
        tasks: list[GeoTaskSpec] = []
        for i, raw in enumerate(draft_tasks, start=1):
            task_id = f"{video_id}__t{i:02d}"
            t0 = float(raw.time_start)
            t1 = float(raw.time_end)
            if t1 < t0:
                t0, t1 = t1, t0
            if duration > 0:
                t0 = min(max(0.0, t0), duration)
                t1 = min(max(t0, t1), duration)
            max_kf = _max_keyframes_for_task(
                raw.target_kind, bool(raw.multi_target_images), max_kf_cfg
            )
            stamps, paths, multi = _materialize_task_images(
                video_path=video_path,
                video_id=video_id,
                task_id=task_id,
                raw=raw,
                t0=t0,
                t1=t1,
                max_kf=max_kf,
                transcript=transcript,
            )
            tasks.append(
                GeoTaskSpec(
                    task_id=task_id,
                    time_start=t0,
                    time_end=t1,
                    target_kind=raw.target_kind,
                    keyframe_timestamps=stamps,
                    image_paths=paths,
                    multi_target_images=multi,
                    segment_start_idx=raw.segment_start_idx,
                    segment_end_idx=raw.segment_end_idx,
                    task_summary=(raw.task_summary or "").strip(),
                )
            )
        if not tasks:
            raise ValueError("模型返回 accept 但未给出任何 task")
        result = AuditSplitResult(
            video_id=video_id,
            decision=AuditDecision.accept,
            reason=draft.reason.strip(),
            has_unresolved_target=True,
            tasks=tasks,
        )

    dest = Path(out_path) if out_path else (
        Path(settings.INTERMEDIATE_DIR) / video_id / "stage_audit_split.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def load_audit_split(path: str | Path) -> AuditSplitResult:
    """从落盘 JSON 加载审核结果。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AuditSplitResult.model_validate(data)
