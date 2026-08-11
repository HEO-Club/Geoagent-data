"""阶段1.5：字幕 + 稀疏帧 → 拒识或切分为多定位任务并截关键帧。"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.config import get_settings
from pipeline.llm import call_structured
from pipeline.media.keyframes import extract_keyframes, video_duration_sec
from pipeline.schemas.audit import (
    AnswerStatus,
    AuditDecision,
    AuditSplitResult,
    GeoTaskSpec,
    KeyframeAssessment,
    TargetKind,
    TaskStatus,
)
from pipeline.schemas.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

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
    "同题多图至少 2 个）、multi_target_images、expected_image_count、"
    "可选 segment 索引、task_summary。\n"
    "答案质量门禁：每个 task 必须给 answer_status 与 final_location_text：\n"
    "- resolved：旁白明确给出唯一最终地点；final_location_text 原样概括该地点；\n"
    "- ambiguous：存在冲突答案、只有不确定猜测或只能缩小到模糊范围；\n"
    "- unsolved：讲解者明确无解或没有给出最终地点。\n"
    "ambiguous/unsolved 的 final_location_text 可为空，后续不会进入 Stage 2。\n"
    "同时输出 split_confidence（0~1）与 needs_split_review。只有确实无法判断"
    "同题多图/独立新题边界时才设 needs_split_review=true；不要因为普通措辞差异"
    "把清晰、正确的切分标成异常。"
    "不要输出图像路径。"
)

FRAME_VERIFY_HINT = (
    "判断这张视频截帧是否可作为「定位输入」写入训练关键帧。"
    "逐帧输出 kind、quality_score、answer_leakage、tutorial_overlay、"
    "clean_source 与简短 reason。用过程角色判断，不要按控件/App 品类清单执法。\n"
    "先问：这是被定位的实拍，还是解题时调用的工具/核验画面？\n"
    "- target_photo：主画面是待定位的实拍场景"
    "（地面/现场镜头、静帧照片或目标建筑外观实拍占主画面）。"
    "即使叠有讲解字幕、箭头、方位字，主体仍可判 target_photo；但如果这些内容"
    "是讲解过程后来添加的推理标注，则 tutorial_overlay=true、clean_source=false。"
    "全屏建筑外观实拍即使夹在核验旁白时间线，仍判 target_photo。\n"
    "- teaching_ui：主内容是定位过程中的工具或核验画面"
    "（用地图/影像底图比对、街景浏览、答案钉点揭晓等），"
    "即使画面「很地理」也不是待定位输入；"
    "亦含纯讲解辅助可视化或过程推演板。\n"
    "不得仅因源实拍上有讲解标注就判 teaching_ui；"
    "也不得把工具步骤画面当成 target_photo。\n"
    "- other：仅黑屏、片尾、或与定位目标完全无关的画面；"
    "不得因源实拍上有方位字/箭头/字幕叠字就判 other。\n"
    "须输入画面本身全屏或主画面。\n"
    "quality_score 衡量作为 SFT 原始输入的质量：干净完整原图接近 1；带侧栏、"
    "大面积字幕、后加红线/箭头、裁切遮挡或只是拼贴中的小区域应降低。"
    "若画面直接出现最终答案地名、答案钉点或足以泄露结论的标注，"
    "answer_leakage=true；原题自带且不泄露答案的文字线索不算泄露。"
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
    segment_start_idx: int | None = None
    segment_end_idx: int | None = None
    task_summary: str = ""
    answer_status: AnswerStatus = AnswerStatus.resolved
    final_location_text: str = ""
    expected_image_count: int = Field(default=1, ge=1)


class _LLMAuditDraft(BaseModel):
    """LLM 审核草稿。"""

    decision: AuditDecision
    reason: str = ""
    has_unresolved_target: bool = True
    tasks: list[_LLMGeoTaskDraft] = Field(default_factory=list)
    split_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_split_review: bool = False


class _LLMFrameVerdict(BaseModel):
    """单帧视觉验收。"""

    kind: FrameKind
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    answer_leakage: bool = False
    tutorial_overlay: bool = False
    clean_source: bool = False
    reason: str = ""


class _LLMTaskMergeResult(BaseModel):
    """条件触发的双向切分复核（可合并，也可补拆）。"""

    tasks: list[_LLMGeoTaskDraft] = Field(default_factory=list)
    reason: str = ""


def _normalize_location_key(value: str) -> str:
    """仅用于发现明确的重复答案，不尝试做地名语义推断。"""
    return re.sub(r"[\s，,。.;；:：()（）\-—_/]+", "", value).casefold()


def _enumerates_parallel_inputs(value: str) -> bool:
    """只识别明确编号的并列输入；用于触发复核，不直接判定切分错误。"""
    markers = re.findall(
        r"(?:镜头|图|camera|image)\s*([0-9一二三四五六七八九十]+)",
        value,
        flags=re.IGNORECASE,
    )
    return len(set(markers)) >= 2


def _objective_split_anomalies(
    draft_tasks: list[_LLMGeoTaskDraft],
    *,
    duration: float,
    boundary_tolerance: float,
) -> list[str]:
    """返回高精度结构异常；宁可少报，不用脆弱词表猜语义。"""
    anomalies: list[str] = []
    if not draft_tasks:
        return ["accept 但 tasks 为空"]

    resolved_locations: dict[str, list[int]] = {}
    for i, task in enumerate(draft_tasks, start=1):
        t0 = float(task.time_start)
        t1 = float(task.time_end)
        if t1 < t0:
            anomalies.append(f"task {i} 的 time_end 小于 time_start")
        if duration > 0 and (t0 < -1e-6 or t1 > duration + 1e-6):
            anomalies.append(f"task {i} 的时间窗超出视频物理范围")

        far_stamps = [
            float(stamp)
            for stamp in task.keyframe_timestamps
            if float(stamp) < min(t0, t1) - boundary_tolerance
            or float(stamp) > max(t0, t1) + boundary_tolerance
        ]
        if far_stamps:
            anomalies.append(f"task {i} 的候选时间与题目时间窗明显矛盾")

        answer_status = getattr(task, "answer_status", AnswerStatus.resolved)
        if answer_status == AnswerStatus.resolved:
            final_location = str(getattr(task, "final_location_text", "") or "")
            key = _normalize_location_key(final_location)
            if not key:
                anomalies.append(f"task {i} 标记 resolved 但缺少明确最终地点")
            else:
                resolved_locations.setdefault(key, []).append(i)
            if (
                bool(getattr(task, "multi_target_images", False))
                and int(getattr(task, "expected_image_count", 1) or 1) >= 2
                and _enumerates_parallel_inputs(final_location)
            ):
                anomalies.append(
                    f"task {i} 的多输入最终答案明确枚举多个编号镜头，需复核它们"
                    "是共同支撑一个地点还是多个独立地点"
                )

    for indexes in resolved_locations.values():
        if len(indexes) > 1:
            anomalies.append(
                "多个 task 给出完全相同的明确最终地点："
                + ", ".join(str(i) for i in indexes)
            )
    return anomalies


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


def _maybe_review_task_split(
    draft_tasks: list[_LLMGeoTaskDraft],
    *,
    video_id: str,
    transcript: list[TranscriptSegment],
    overview_images: list[str] | None,
    duration: float,
    split_confidence: float,
    model_requests_review: bool,
    boundary_tolerance: float,
) -> list[_LLMGeoTaskDraft]:
    """只在客观异常或模型低置信时复核，避免重复改坏正确切分。"""
    anomalies = _objective_split_anomalies(
        draft_tasks,
        duration=duration,
        boundary_tolerance=boundary_tolerance,
    )
    should_review = (
        bool(model_requests_review) or float(split_confidence) < 0.65 or bool(anomalies)
    )
    if not should_review:
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
            "answer_status": getattr(t, "answer_status", AnswerStatus.resolved).value,
            "final_location_text": str(getattr(t, "final_location_text", "") or ""),
            "expected_image_count": int(getattr(t, "expected_image_count", 1) or 1),
        }
        for t in draft_tasks
    ]
    prompt = (
        "以下是首次审核得到的 tasks。仅因模型低置信或客观结构矛盾触发了复核。\n"
        f"触发原因：{json.dumps(anomalies, ensure_ascii=False)}；"
        f"模型主动请求复核={model_requests_review}；首次置信度={split_confidence:.2f}。\n"
        "请保守复核：如果现有切分正确，必须原样保留，不要为了体现复核而修改。\n"
        "一个 task = 一次独立定位题 = 一条最终答案。\n"
        "若多张待定位原图共同支撑同一最终地点（或后图精化前图），"
        "必须合并为 **一个** task，设 multi_target_images=true，"
        "并给出每张待定位原图出现的 keyframe_timestamps。\n"
        "**同一最终地点 / 同一条答案链必须合并**"
        "（即使线索形态不同，如河道环境与建筑外观）。\n"
        "仅当不同目标、不同最终地点时才保留多个 task。"
        "如果一个 task 内实际含多条独立最终答案链，必须补拆；"
        "如果多个 task 属于同一最终答案链，必须合并。\n"
        "每个结果 task 继续填写 answer_status、final_location_text、"
        "expected_image_count 与全部原始输入时间戳。\n"
        f"视频 ID: {video_id}\n"
        f"当前 tasks JSON:\n{json.dumps(payload, ensure_ascii=False)}\n"
        "字幕：\n"
        f"{_format_transcript(transcript)}\n"
        "请输出复核后的 tasks 与 reason。"
    )
    merged = call_structured(
        prompt,
        _LLMTaskMergeResult,
        images=overview_images or None,
        lane="vlm",
    )
    if merged.tasks:
        logger.info(
            "conditional task split review: %s -> %s (%s)",
            len(draft_tasks),
            len(merged.tasks),
            merged.reason,
        )
        return merged.tasks
    return draft_tasks


def _maybe_merge_same_question_tasks(
    draft_tasks: list[_LLMGeoTaskDraft],
    *,
    video_id: str,
    transcript: list[TranscriptSegment],
    overview_images: list[str] | None,
) -> list[_LLMGeoTaskDraft]:
    """兼容旧调用：显式多 task 复核；新流水线使用条件式双向复核。"""
    return _maybe_review_task_split(
        draft_tasks,
        video_id=video_id,
        transcript=transcript,
        overview_images=overview_images,
        duration=0.0,
        split_confidence=0.0,
        model_requests_review=True,
        boundary_tolerance=20.0,
    )


def _format_transcript(transcript: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(transcript):
        lines.append(f"[{i}] [{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}")
    return "\n".join(lines)


def _pick_sparse_timestamps(duration: float, count: int) -> list[float]:
    if duration <= 0:
        return [0.0]
    n = max(1, int(count))
    if n == 1:
        return [duration * 0.5]
    # 精确 duration 已位于最后一帧之后；保留极小余量，避免配置 N 张只解出 N-1 张。
    last = max(0.0, duration - min(0.05, duration * 0.001))
    return [last * i / (n - 1) for i in range(n)]


def _filter_timestamps(
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
        # 越界候选直接忽略，不能把多个错误时间全部挤到边界同一帧。
        if t < lo - 1e-6 or t > hi + 1e-6:
            continue
        if not any(abs(old - t) <= 1e-3 for old in cleaned):
            cleaned.append(t)
    if not cleaned:
        cleaned = [lo if hi <= lo else (lo + hi) * 0.5]
    return cleaned[: max(1, max_n)]


def _progressive_probe_timestamps(
    start: float,
    end: float,
    *,
    count: int,
) -> list[float]:
    """题目范围内由开头向全段渐进探测，不跨到相邻题。"""
    lo = max(0.0, float(start))
    hi = max(lo, float(end))
    n = max(1, int(count))
    if hi <= lo + 1e-6:
        return [lo]
    early = [lo + delta for delta in (1.0, 3.0, 6.0, 10.0, 15.0)]
    even = [lo + (hi - lo) * (i + 1) / (n + 1) for i in range(n)]
    return _filter_timestamps(
        early + even,
        start=lo,
        end=hi,
        max_n=n,
    )


def _normalize_task_window(
    raw: _LLMGeoTaskDraft,
    *,
    duration: float,
    transcript: list[TranscriptSegment],
    boundary_tolerance: float,
) -> tuple[float, float]:
    """以字幕索引和近邻候选修复轻微边界偏差，再限制在视频物理范围。"""
    t0 = float(raw.time_start)
    t1 = float(raw.time_end)
    if t1 < t0:
        t0, t1 = t1, t0

    seg0 = getattr(raw, "segment_start_idx", None)
    seg1 = getattr(raw, "segment_end_idx", None)
    if (
        seg0 is not None
        and seg1 is not None
        and 0 <= int(seg0) <= int(seg1) < len(transcript)
    ):
        t0 = min(t0, float(transcript[int(seg0)].start))
        t1 = max(t1, float(transcript[int(seg1)].end))

    tolerance = max(0.0, float(boundary_tolerance))
    for stamp in getattr(raw, "keyframe_timestamps", []) or []:
        value = float(stamp)
        if t0 - tolerance <= value <= t1 + tolerance:
            t0 = min(t0, value)
            t1 = max(t1, value)

    max_time = max(0.0, float(duration) - 0.001) if duration > 0 else max(t1, 0.0)
    t0 = min(max(0.0, t0), max_time)
    t1 = min(max(t0, t1), max_time)
    return t0, t1


def _max_keyframes_for_task(
    target_kind: TargetKind,
    multi_target_images: bool,
    configured_max: int,
    expected_image_count: int | None = None,
    proposed_count: int = 0,
) -> int:
    """输出张数由独立输入数决定；普通静图始终只输出最佳一帧。"""
    hard_cap = max(1, int(configured_max))
    expected = max(1, int(expected_image_count or 1), int(proposed_count or 0))
    if target_kind == TargetKind.video_derived:
        return min(hard_cap, expected)
    if multi_target_images:
        return min(hard_cap, max(2, expected))
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
    return verify_keyframe(image_path).kind


def verify_keyframe(
    image_path: str,
    *,
    narrative_context: str = "",
) -> _LLMFrameVerdict:
    """返回带质量与泄露标记的逐帧验收结果。"""
    prompt = FRAME_VERIFY_HINT
    if narrative_context.strip():
        prompt += (
            "\n\n以下是该帧在讲解时间线附近的叙事上下文，只用于判断画面角色：\n"
            f"{narrative_context.strip()}\n"
            "若上下文表明此时正在展示搜索结果、地图/遥感/街景核验、答案揭晓、"
            "找到后的航拍/全景或候选比对，则即使画面本身是全屏实景，也不是原始"
            "待定位输入，应判 teaching_ui。若上下文明确是展示题目原图、另一张原图"
            "或放大原图，则可判 target_photo。不要仅因上下文写出了最终地点就把"
            "画面标为 answer_leakage；answer_leakage 仍要求画面本身可见答案。"
        )
    return call_structured(
        prompt,
        _LLMFrameVerdict,
        images=[image_path],
        lane="vlm",
        max_attempts=1,
    )


def _frame_narrative_context(
    transcript: list[TranscriptSegment],
    *,
    stamp: float,
    task_summary: str,
) -> str:
    """给单帧验收提供局部叙事角色，不把整段答案材料重复发送。"""
    if not transcript:
        return f"候选时间：{stamp:.1f}s\n任务摘要：{task_summary.strip()}"
    center = min(
        range(len(transcript)),
        key=lambda i: (
            0.0
            if transcript[i].start <= stamp <= transcript[i].end
            else min(abs(stamp - transcript[i].start), abs(stamp - transcript[i].end))
        ),
    )
    lo = max(0, center - 1)
    hi = min(len(transcript), center + 2)
    nearby = "\n".join(
        f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}"
        for seg in transcript[lo:hi]
    )
    return (
        f"候选时间：{stamp:.1f}s\n"
        f"任务摘要：{task_summary.strip()}\n"
        f"邻近字幕：\n{nearby}"
    )


def _image_dhash(image_path: str) -> int | None:
    """计算轻量视觉哈希；读取失败时不做去重，避免误删。"""
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            pixels = list(image.convert("L").resize((9, 8)).getdata())
    except Exception:  # noqa: BLE001
        return None
    value = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value


def _is_visual_duplicate(
    image_path: str,
    selected_hashes: list[int],
    *,
    max_distance: int,
) -> tuple[bool, int | None]:
    current = _image_dhash(image_path)
    if current is None:
        return False, None
    duplicate = any(
        (current ^ old).bit_count() <= max(0, int(max_distance))
        for old in selected_hashes
    )
    return duplicate, current


def _even_probe_timestamps(start: float, end: float, count: int) -> list[float]:
    lo = max(0.0, float(start))
    hi = max(lo, float(end))
    n = max(1, int(count))
    if hi <= lo + 1e-6:
        return [lo]
    return [lo + (hi - lo) * (i + 1) / (n + 1) for i in range(n)]


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
    task_dir: Path,
) -> tuple[
    list[float],
    list[str],
    bool,
    list[KeyframeAssessment],
    str,
]:
    """在 task 范围内逐帧验收、质量排序并用视觉内容去重。"""
    settings = get_settings()
    candidate_budget = max(1, int(settings.AUDIT_MAX_CANDIDATE_PROBES))
    fallback_budget = max(0, int(settings.AUDIT_FALLBACK_PROBE_COUNT))
    quality_floor = min(1.0, max(0.0, float(settings.AUDIT_MIN_FRAME_QUALITY)))
    hash_distance = max(0, int(settings.AUDIT_VISUAL_HASH_DISTANCE))
    initial = _filter_timestamps(
        list(getattr(raw, "keyframe_timestamps", []) or [])
        + _seed_photo_mention_timestamps(transcript)
        + _progressive_probe_timestamps(t0, t1, count=5),
        start=t0,
        end=t1,
        max_n=candidate_budget,
    )
    fallback = (
        _filter_timestamps(
            _even_probe_timestamps(t0, t1, fallback_budget),
            start=t0,
            end=t1,
            max_n=max(1, fallback_budget),
        )
        if fallback_budget > 0
        else []
    )

    assessment_checkpoint = task_dir / "candidate_assessments.partial.json"
    assessments: list[KeyframeAssessment] = []
    if assessment_checkpoint.is_file():
        try:
            saved = json.loads(assessment_checkpoint.read_text(encoding="utf-8"))
            assessments = [KeyframeAssessment.model_validate(item) for item in saved]
            logger.info(
                "resume %s candidate assessments for task %s",
                len(assessments),
                task_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ignore invalid candidate checkpoint task=%s: %s",
                task_id,
                exc,
            )
            assessments = []
    tried: set[str] = {f"{item.timestamp:.3f}" for item in assessments}
    frame_dir = task_dir / "candidates"

    def assess(stamps: list[float]) -> None:
        for stamp in stamps:
            key = f"{stamp:.3f}"
            if key in tried:
                continue
            tried.add(key)
            try:
                paths = extract_keyframes(video_path, [stamp], out_dir=str(frame_dir))
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
            try:
                verdict = verify_keyframe(
                    path,
                    narrative_context=_frame_narrative_context(
                        transcript,
                        stamp=stamp,
                        task_summary=str(getattr(raw, "task_summary", "") or ""),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "task %s frame verify failed %.3f: %s",
                    task_id,
                    stamp,
                    exc,
                )
                assessments.append(
                    KeyframeAssessment(
                        timestamp=float(stamp),
                        image_path=str(Path(path).resolve()),
                        kind="error",
                        reason=f"验收调用失败：{type(exc).__name__}",
                    )
                )
                _write_json(
                    assessment_checkpoint,
                    [item.model_dump(mode="json") for item in assessments],
                )
                continue
            kind = verdict.kind
            # 测试/旧适配器可能只返回 kind；此时保持向后兼容的中性质量。
            quality = float(getattr(verdict, "quality_score", 0.8))
            leakage = bool(getattr(verdict, "answer_leakage", False))
            overlay = bool(getattr(verdict, "tutorial_overlay", False))
            clean = bool(
                getattr(verdict, "clean_source", kind == FrameKind.target_photo)
            )
            assessments.append(
                KeyframeAssessment(
                    timestamp=float(stamp),
                    image_path=str(Path(path).resolve()),
                    kind=kind.value,
                    quality_score=quality,
                    answer_leakage=leakage,
                    tutorial_overlay=overlay,
                    clean_source=clean,
                    reason=str(getattr(verdict, "reason", "") or ""),
                )
            )
            _write_json(
                assessment_checkpoint,
                [item.model_dump(mode="json") for item in assessments],
            )

    assess(initial)
    has_clean_candidate = any(
        item.kind == FrameKind.target_photo.value
        and not item.answer_leakage
        and not item.tutorial_overlay
        and item.clean_source
        and item.quality_score >= quality_floor
        for item in assessments
    )
    if not has_clean_candidate and fallback:
        assess(fallback)

    eligible = [
        item
        for item in assessments
        if item.kind == FrameKind.target_photo.value and not item.answer_leakage
    ]
    for item in assessments:
        item.selected = False
    eligible.sort(
        key=lambda item: (
            item.clean_source,
            not item.tutorial_overlay,
            item.quality_score,
        ),
        reverse=True,
    )

    selected: list[KeyframeAssessment] = []
    selected_hashes: list[int] = []
    for item in eligible:
        duplicate, image_hash = _is_visual_duplicate(
            item.image_path,
            selected_hashes,
            max_distance=hash_distance,
        )
        if duplicate:
            logger.info(
                "skip visually duplicate frame %.3f task=%s",
                item.timestamp,
                task_id,
            )
            continue
        item.selected = True
        selected.append(item)
        if image_hash is not None:
            selected_hashes.append(image_hash)
        if len(selected) >= max_kf:
            break

    selected.sort(key=lambda item: item.timestamp)
    selected_stamps = [item.timestamp for item in selected]
    selected_paths = [item.image_path for item in selected]
    multi = max_kf > 1
    _write_json(
        assessment_checkpoint,
        [item.model_dump(mode="json") for item in assessments],
    )

    if not selected:
        return [], [], multi, assessments, "未找到任何无答案泄露的待定位原图"
    if len(selected) < max_kf:
        return (
            selected_stamps,
            selected_paths,
            multi,
            assessments,
            f"预计 {max_kf} 个独立输入，仅选到 {len(selected)} 个",
        )
    low_quality = [
        item
        for item in selected
        if item.quality_score < quality_floor
        or item.tutorial_overlay
        or not item.clean_source
    ]
    if low_quality:
        return (
            selected_stamps,
            selected_paths,
            multi,
            assessments,
            "选中帧仍含讲解覆盖、界面残留或质量低于阈值",
        )
    return selected_stamps, selected_paths, multi, assessments, ""


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


def _raw_task_payload(raw: object) -> dict[str, object]:
    """将模型草稿写入可审计 checkpoint。"""
    if hasattr(raw, "model_dump"):
        return raw.model_dump(mode="json")  # type: ignore[no-any-return,union-attr]
    answer_status = getattr(raw, "answer_status", AnswerStatus.resolved)
    if isinstance(answer_status, Enum):
        answer_status = answer_status.value
    target_kind = getattr(raw, "target_kind", TargetKind.still_image)
    if isinstance(target_kind, Enum):
        target_kind = target_kind.value
    return {
        "time_start": float(getattr(raw, "time_start", 0.0)),
        "time_end": float(getattr(raw, "time_end", 0.0)),
        "target_kind": target_kind,
        "keyframe_timestamps": list(getattr(raw, "keyframe_timestamps", []) or []),
        "multi_target_images": bool(getattr(raw, "multi_target_images", False)),
        "segment_start_idx": getattr(raw, "segment_start_idx", None),
        "segment_end_idx": getattr(raw, "segment_end_idx", None),
        "task_summary": str(getattr(raw, "task_summary", "") or ""),
        "answer_status": answer_status,
        "final_location_text": str(getattr(raw, "final_location_text", "") or ""),
        "expected_image_count": int(getattr(raw, "expected_image_count", 1) or 1),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        text = payload.model_dump_json(indent=2)
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


def run_audit_split(
    video_path: str,
    transcript: list[TranscriptSegment],
    *,
    out_path: str | None = None,
    resume_tasks: bool = True,
) -> AuditSplitResult:
    """审核视频是否可蒸馏，并切分为带关键帧的定位任务。

    Args:
        video_path: 视频路径。
        transcript: 阶段1 字幕。
        out_path: 审核结果落盘路径；默认 intermediate/{id}/stage_audit_split.json。
        resume_tasks: 复用同目录已完成的审核草稿和 task checkpoint。

    Returns:
        AuditSplitResult；reject 时 tasks 为空。
    """
    settings = get_settings()
    video_id = Path(video_path).stem
    dest = (
        Path(out_path)
        if out_path
        else (Path(settings.INTERMEDIATE_DIR) / video_id / "stage_audit_split.json")
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tasks_root = dest.parent / "tasks"
    draft_checkpoint = dest.parent / "stage_audit_split_draft.json"
    if resume_tasks and dest.is_file():
        return load_audit_split(dest)
    duration = 0.0
    try:
        duration = float(video_duration_sec(video_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit duration probe failed: %s", exc)

    overview_images: list[str] = []
    resumed_draft = False
    if resume_tasks and draft_checkpoint.is_file():
        checkpoint_data = json.loads(draft_checkpoint.read_text(encoding="utf-8"))
        draft = _LLMAuditDraft(
            decision=AuditDecision.accept,
            reason=str(checkpoint_data.get("reason", "")),
            has_unresolved_target=True,
            tasks=[
                _LLMGeoTaskDraft.model_validate(item)
                for item in checkpoint_data.get("tasks", [])
            ],
            split_confidence=float(checkpoint_data.get("split_confidence", 1.0)),
            needs_split_review=False,
        )
        resumed_draft = True
        logger.info("resume audit draft for %s", video_id)
    else:
        sparse_n = max(1, int(settings.AUDIT_SPARSE_FRAME_COUNT))
        sparse_stamps = _pick_sparse_timestamps(duration, sparse_n)
        sparse_dir = Path(settings.CACHE_DIR) / "audit_sparse" / video_id
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
            "请输出 has_unresolved_target / decision / reason / split_confidence / "
            "needs_split_review / tasks。"
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
        boundary_tolerance = max(0.0, float(settings.AUDIT_TASK_BOUNDARY_TOLERANCE_SEC))
        draft_tasks = list(draft.tasks)
        if not resumed_draft:
            draft_tasks = _maybe_review_task_split(
                draft_tasks,
                video_id=video_id,
                transcript=transcript,
                overview_images=overview_images or None,
                duration=duration,
                split_confidence=float(getattr(draft, "split_confidence", 1.0)),
                model_requests_review=bool(getattr(draft, "needs_split_review", False)),
                boundary_tolerance=boundary_tolerance,
            )
        _write_json(
            draft_checkpoint,
            {
                "video_id": video_id,
                "reason": str(getattr(draft, "reason", "") or ""),
                "split_confidence": float(getattr(draft, "split_confidence", 1.0)),
                "needs_split_review": bool(getattr(draft, "needs_split_review", False)),
                "tasks": [_raw_task_payload(item) for item in draft_tasks],
            },
        )
        tasks: list[GeoTaskSpec] = []
        for i, raw in enumerate(draft_tasks, start=1):
            task_id = f"{video_id}__t{i:02d}"
            task_dir = tasks_root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            task_checkpoint = task_dir / "task_audit.json"
            if resume_tasks and task_checkpoint.is_file():
                task = GeoTaskSpec.model_validate_json(
                    task_checkpoint.read_text(encoding="utf-8")
                )
                if task.task_id == task_id:
                    tasks.append(task)
                    logger.info(
                        "resume completed task %s status=%s",
                        task_id,
                        task.status.value,
                    )
                    continue
            t0, t1 = _normalize_task_window(
                raw,
                duration=duration,
                transcript=transcript,
                boundary_tolerance=boundary_tolerance,
            )
            raw_multi = bool(getattr(raw, "multi_target_images", False))
            answer_status = getattr(raw, "answer_status", AnswerStatus.resolved)
            if not isinstance(answer_status, AnswerStatus):
                answer_status = AnswerStatus(str(answer_status))
            final_location = str(getattr(raw, "final_location_text", "") or "").strip()
            expected_count = max(1, int(getattr(raw, "expected_image_count", 1) or 1))
            max_kf = _max_keyframes_for_task(
                raw.target_kind,
                raw_multi,
                max_kf_cfg,
                expected_image_count=expected_count,
                proposed_count=len(getattr(raw, "keyframe_timestamps", []) or [])
                if (raw_multi or raw.target_kind == TargetKind.video_derived)
                else 0,
            )
            status = TaskStatus.accepted
            status_reason = ""
            stamps: list[float] = []
            paths: list[str] = []
            assessments: list[KeyframeAssessment] = []
            multi = max_kf > 1

            if answer_status != AnswerStatus.resolved:
                status = TaskStatus.rejected
                status_reason = (
                    "字幕最终答案存在歧义"
                    if answer_status == AnswerStatus.ambiguous
                    else "讲解未给出可用最终答案"
                )
            elif isinstance(raw, _LLMGeoTaskDraft) and not final_location:
                status = TaskStatus.needs_review
                status_reason = "模型标记答案已解决，但未提供明确最终地点"
            else:
                try:
                    (
                        stamps,
                        paths,
                        multi,
                        assessments,
                        quality_reason,
                    ) = _materialize_task_images(
                        video_path=video_path,
                        video_id=video_id,
                        task_id=task_id,
                        raw=raw,
                        t0=t0,
                        t1=t1,
                        max_kf=max_kf,
                        transcript=transcript,
                        task_dir=task_dir,
                    )
                    if quality_reason:
                        status = TaskStatus.needs_review
                        status_reason = quality_reason
                except Exception as exc:
                    logger.exception("task %s materialize failed", task_id)
                    status = TaskStatus.needs_review
                    status_reason = f"关键帧处理失败：{exc}"

            task = GeoTaskSpec(
                task_id=task_id,
                time_start=t0,
                time_end=t1,
                target_kind=raw.target_kind,
                keyframe_timestamps=stamps,
                image_paths=paths,
                multi_target_images=multi,
                segment_start_idx=getattr(raw, "segment_start_idx", None),
                segment_end_idx=getattr(raw, "segment_end_idx", None),
                task_summary=str(getattr(raw, "task_summary", "") or "").strip(),
                status=status,
                status_reason=status_reason,
                answer_status=answer_status,
                final_location_text=final_location,
                expected_image_count=max_kf,
                frame_assessments=assessments,
            )
            tasks.append(task)
            _write_json(task_checkpoint, task)
            _write_json(
                dest.parent / "stage_audit_split.partial.json",
                AuditSplitResult(
                    video_id=video_id,
                    decision=AuditDecision.accept,
                    reason=str(getattr(draft, "reason", "") or "").strip(),
                    has_unresolved_target=True,
                    tasks=tasks,
                ),
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

    _write_json(dest, result)
    return result


def load_audit_split(path: str | Path) -> AuditSplitResult:
    """从落盘 JSON 加载审核结果。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AuditSplitResult.model_validate(data)
