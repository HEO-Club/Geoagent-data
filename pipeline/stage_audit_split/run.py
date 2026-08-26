"""阶段1.5：字幕 + 稀疏帧 → 拒识或切分为多定位任务并截关键帧。"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from pipeline.config import get_settings
from pipeline.llm import call_structured
from pipeline.media.keyframes import extract_keyframes, video_duration_sec
from pipeline.schemas.audit import (
    AnswerStatus,
    AuditDecision,
    AuditSplitResult,
    GeoTaskSpec,
    KeyframeAssessment,
    ProcessInterval,
    ProcessRole,
    TargetKind,
    TaskStatus,
)
from pipeline.schemas.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


def _unwrap_structured_enum(value: object) -> object:
    """兼容部分 Anthropic relay 把枚举值包装成 type/primary 对象。"""

    if not isinstance(value, dict):
        return value
    for key in (
        "value",
        "type",
        "primary",
        "role",
        "kind",
        "name",
        "relation",
        "containment",
        "status",
        "decision",
        "category",
        "label",
        "classification",
        "result",
        "relationship",
    ):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, dict):
            unwrapped = _unwrap_structured_enum(candidate)
            if isinstance(unwrapped, str) and unwrapped.strip():
                return unwrapped.strip()
    scored = [
        (str(key), float(score))
        for key, score in value.items()
        if isinstance(score, (int, float)) and not isinstance(score, bool)
    ]
    if scored:
        return max(scored, key=lambda item: item[1])[0]
    return value

AUDIT_SYSTEM_HINT = (
    "你审核讲解视频是否适合蒸馏为「图片/静帧地理定位」训练样本。"
    "输入为带时间戳字幕与若干稀疏审计帧（仅供审核，不是训练图）；"
    "禁止使用 groundtruth；禁止按固定词表或渠道特判。\n"
    "拒识主问句：若去掉讲解/旁白/答案，是否仍存在需 agent 定位的图或场景？\n"
    "本项目的原料本来就是已经给出完整解法和最终答案的讲解视频。判断时必须把时间"
    "截在每道题的待定位原图首次出示处：假设 Agent 只能看到该原图和题面已知条件，"
    "是否仍需要推理定位。后续人工解出、AI 已回答、答案已揭晓、视频以对战/复盘形式"
    "呈现，都绝不能作为 reject 理由；应把后续内容当作要蒸馏的监督链。\n"
    "定位输入不只限于静态照片：一段连续实拍现场同样是"
    "有效的 video_derived 输入；只要讲解者正通过这些画面推断拍摄地点，就应 accept，"
    "不能把被分析的现场画面说成普通素材。任务也不要求仅凭像素孤立求解：题面明确"
    "给出的时间、属地、活动范围、文字提示等可作为 Agent 已知线索；存在目标实拍"
    "且这些线索参与定位时仍是有效样本。只有完全不存在待定位视觉目标、地点纯粹是"
    "文章/科普的讲述对象时才 reject。\n"
    "- has_unresolved_target=true：去掉讲解/旁白/答案后，仍存在需 Agent 定位的视觉目标"
    "（与视频后面是否已揭晓无关；本项目原料通常已揭晓）。\n"
    "- has_unresolved_target=false：完全没有待定位视觉目标（地名科普、历史故事等）。\n"
    "- decision 必须与 has_unresolved_target 一致："
    "true→accept，false→reject。\n"
    "地名科普、历史故事、奇观介绍、地点仅为讲述对象、"
    "旁白「打开地图就能看到某某地」≠ 定位题。\n"
    "存在一个或多个独立的待定位输入时必须 accept。\n"
    "先理解整段定位过程，区分过程角色：\n"
    "- (A) 待定位实拍输入：人眼/相机拍到的、被定位的现场场景或静帧；\n"
    "- (B) 推理工具与核验步骤：用地图或影像底图比对、街景浏览确认等工具画面；\n"
    "- (C) 答案揭晓：钉点指出「就是这里」、揭晓界面等。\n"
    "切分粒度（关键）：一个 task = 一次独立定位题 = 一条最终答案链。\n"
    "- 同题多图（多张待定位原图共同支撑同一最终地点，或后图只是精化前图）"
    "必须合并为 **一个** task；下游会按画面判定源输入，"
    "**不要**用旁白里的「第一张/再看/两张照片」去预定选图张数。\n"
    "- **同一最终地点 / 同一条答案链必须合并为一个 task**"
    "（即使线索形态不同，如环境特征与建筑外观）。\n"
    "- 仅当不同目标、不同最终地点、彼此独立出题时才拆成多个 task。\n"
    "- 禁止把「同一条推理链里的第二张参考图」拆成第二个 task。\n"
    "target_kind：\n"
    "- still_image：明确静图/待定位原图；\n"
    "- video_derived：对源视频/连续实拍场景定位（仍入库）。\n"
    "两个时间窗（必须区分）：\n"
    "- time_start/time_end 与可选 segment 索引 = **蒸馏窗**："
    "整条答案链旁白（问题设定 → 比对叙述 → 最终地点结论），"
    "不得裁成仅原图中段；供下游字幕切片，不是选图密采样区间。\n"
    "- display_time_start/display_time_end = **出示粗窗**："
    "主画面实际展示待定位实拍 (A) 的大约区间（秒级粗估即可）；"
    "须落在蒸馏窗内；不要把地图比对、街景核验、钉点揭晓段放进出示窗。\n"
    "**不要**预定 expected_image_count 或精确关键帧秒数；"
    "keyframe_timestamps 可选，仅作弱先验。"
    "下游会在出示窗内密采样，并构造最小源输入集。\n"
    "每个 task 给出 time_start/time_end、display_time_start/display_time_end、"
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
    "clean_source、evidence_role、chain_support_score 与简短 reason。"
    "用过程角色判断，不要按控件/App 品类清单执法。\n"
    "先问：这是被定位的实拍，还是解题时调用的工具/核验画面？\n"
    "- target_photo：主画面是待定位的实拍场景"
    "（地面/现场镜头、静帧照片或目标建筑外观实拍占主画面）。"
    "即使叠有讲解字幕、箭头、方位字，主体仍可判 target_photo；但如果这些内容"
    "是讲解过程后来添加的推理标注，则 tutorial_overlay=true、clean_source=false。\n"
    "- teaching_ui：主内容是定位过程中的工具或核验画面"
    "（用地图/影像底图比对、街景浏览、搜索结果页、答案钉点揭晓等），"
    "即使画面「很地理」或主体是全屏建筑外观，也不是待定位输入；"
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
    "不要仅因邻近旁白写出了最终地点就把画面标为 answer_leakage；"
    "answer_leakage 仍要求画面本身可见答案。\n"
    "evidence_role（与 kind 配合，禁止 App 词表）：\n"
    "- problem_input：本条定位链正在观察/比较的待定位实拍；\n"
    "- unused_broll：看起来像实拍，但视觉证据简报与邻近叙事都未把它当定位输入"
    "（片头空镜、过场风景、未引用镜头）；\n"
    "- process_tool：工具/核验步骤画面；\n"
    "- reveal：答案揭晓；\n"
    "- other / unknown：其余或无法判断。\n"
    "chain_support_score：画面能支撑证据简报中视觉事实的程度（0~1）。"
    "无简报时对 target_photo 可给中性分（约 0.5），对 unused_broll/工具/揭晓给低分。"
)

PROCESS_TIMELINE_HINT = (
    "根据本定位题蒸馏窗旁白，按时间顺序列出过程区间 process_intervals。\n"
    "角色仅四种：\n"
    "- show_source：主画面正在出示待定位原图/现场（可有多段；同一原图稍后再次出示"
    "仍可再标一段 show_source，下游会做同源合并）；\n"
    "- tool：搜索/地图/街景/卫星/核验等外部动作或工具画面；\n"
    "- reveal：钉点揭晓、明确指出最终地点的界面；\n"
    "- other：开场、过场、空镜、闲聊等。\n"
    "区间须落在给出的蒸馏窗内；粗估秒级即可；尽量互不重叠。"
    "后段核验同地点仍是 tool/reveal，不是新的 show_source。"
    "不确定时宁可标 other，不要把整段蒸馏窗标成一段 show_source。"
    "不要预定选图张数；禁止使用 groundtruth。"
)

EVIDENCE_BRIEF_HINT = (
    "从本定位题的蒸馏窗旁白中，抽取解题者**从待定位图/现场读出并用于缩小范围**"
    "的视觉事实，写成简短 visual_evidence_brief（一两段即可）。\n"
    "只保留画面上可观察的线索：建筑形态、植被、朝向/光影、招牌文字形态、"
    "水面走向、地形轮廓等。\n"
    "明确排除：地图/街景/卫星核验结果、搜索命中、钉点揭晓、开场空镜描述、"
    "与看图无关的社交闲聊。\n"
    "不要预定选图张数；不确定是否第二张独立原图时不要写「必须两张」。\n"
    "旁白不足以概括视觉证据时 brief 可为空字符串，不要发明事实。"
)

SOURCE_IDENTITY_HINT = (
    "比较两张候选截帧，判断它们是否为同一张待定位照片。\n"
    "只问「是不是同一张照片」，不要问「brief 需要几张图」。\n"
    "关系仅四类：\n"
    "- same_photo：同一静图的再展示、改字幕、虚化边、放大/裁切、"
    "或拼图中已经出现的同一格后来全屏再出示；\n"
    "- same_scene：仅当 target_kind=video_derived 时，同一连续现场的换机位；\n"
    "- different_photo：另一张静图，即使同一人、同一天、同一条定位题、"
    "用来核验同地点；画面主体场景结构明显不同"
    "（不同建筑立面/店面/路幅，不是同一取景的远近）必须选此项；\n"
    "- not_input：其中任一实际是工具/核验/揭晓（a_not_input / b_not_input / "
    "both_not_input）。\n"
    "禁止：用「图一/图二」去凑两组；用「都是路中站人」合并；"
    "把拼图算成第三张源；用时间间隔作依据。\n"
    "视觉证据简报只用于核验画面有没有那些视觉事实，不得用来决定要几个 group。\n"
    "不确定是 same_photo 还是 different_photo 时选 same_photo（宁可少一张）。"
)

CONTAINMENT_HINT = (
    "判断两张截帧的包含/放大关系（不看时间戳）。\n"
    "- a_contains_b：A 是左右/上下拼图或更完整帧，B 是 A 中某一格的放大/裁切"
    "或虚化衬底上的同一原图；\n"
    "- b_contains_a：反之；\n"
    "- none：互不包含。\n"
    "仅在高置信时输出非 none；构图仅相似但不是同一照片区域则 none。"
)


class FrameKind(str, Enum):
    """关键帧视觉验收类别。"""

    target_photo = "target_photo"
    teaching_ui = "teaching_ui"
    other = "other"


class EvidenceRole(str, Enum):
    """候选帧在定位链中的证据角色。"""

    problem_input = "problem_input"
    unused_broll = "unused_broll"
    process_tool = "process_tool"
    reveal = "reveal"
    other = "other"
    unknown = "unknown"


class _LLMGeoTaskDraft(BaseModel):
    """LLM 草稿任务（截帧前）。"""

    time_start: float
    time_end: float
    target_kind: TargetKind
    display_time_start: float | None = None
    display_time_end: float | None = None
    keyframe_timestamps: list[float] = Field(default_factory=list)
    multi_target_images: bool = False
    segment_start_idx: int | None = None
    segment_end_idx: int | None = None
    task_summary: str = ""
    answer_status: AnswerStatus = AnswerStatus.resolved
    final_location_text: str = ""
    expected_image_count: int = Field(default=1, ge=1)

    @field_validator("target_kind", "answer_status", mode="before")
    @classmethod
    def _unwrap_enum_fields(cls, value: object) -> object:
        return _unwrap_structured_enum(value)


class _LLMAuditDraft(BaseModel):
    """LLM 审核草稿。"""

    decision: AuditDecision
    reason: str = ""
    has_unresolved_target: bool = True
    tasks: list[_LLMGeoTaskDraft] = Field(default_factory=list)
    split_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_split_review: bool = False

    @field_validator("decision", mode="before")
    @classmethod
    def _unwrap_decision(cls, value: object) -> object:
        return _unwrap_structured_enum(value)


class _LLMFrameVerdict(BaseModel):
    """单帧视觉验收。"""

    kind: FrameKind
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    answer_leakage: bool = False
    tutorial_overlay: bool = False
    clean_source: bool = False
    evidence_role: EvidenceRole = EvidenceRole.unknown
    chain_support_score: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _recover_wrapped_json(cls, value: object) -> object:
        if not isinstance(value, dict) or "kind" in value or len(value) != 1:
            return value
        key, payload = next(iter(value.items()))
        if key == "{" and isinstance(payload, str):
            try:
                recovered = json.loads("{" + payload)
            except json.JSONDecodeError:
                return value
            if isinstance(recovered, dict):
                return recovered
        return value

    @field_validator("kind", "evidence_role", mode="before")
    @classmethod
    def _unwrap_enum_fields(cls, value: object) -> object:
        return _unwrap_structured_enum(value)


class _LLMEvidenceBrief(BaseModel):
    """题级视觉证据简报（纯文本，不看图）。"""

    visual_evidence_brief: str = ""


class _LLMProcessInterval(BaseModel):
    """过程时间线单段（LLM 草稿）。"""

    start: float
    end: float
    role: ProcessRole
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_time_aliases(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        copied = dict(value)
        if "start" not in copied:
            for alias in (
                "start_s",
                "start_sec",
                "start_time",
                "begin",
                "begin_s",
            ):
                if alias in copied:
                    copied["start"] = copied[alias]
                    break
        if "end" not in copied:
            for alias in ("end_s", "end_sec", "end_time", "stop", "stop_s"):
                if alias in copied:
                    copied["end"] = copied[alias]
                    break
        return copied

    @field_validator("role", mode="before")
    @classmethod
    def _unwrap_role(cls, value: object) -> object:
        return _unwrap_structured_enum(value)


class _LLMProcessTimeline(BaseModel):
    """题级过程时间线抽取结果。"""

    intervals: list[_LLMProcessInterval] = Field(default_factory=list)


class _SourceGroupItem(BaseModel):
    """源输入归并中的单帧归属（旧契约，测试兼容）。"""

    index: int = Field(ge=0)
    source_group: int = Field(
        description="-1=not_input；非负整数表示源输入组 id"
    )
    reason: str = ""


class _LLMSourceIdentityResult(BaseModel):
    """出示段代表帧的源输入归并结果（旧契约，测试兼容）。"""

    items: list[_SourceGroupItem] = Field(default_factory=list)


class ContainmentKind(str, Enum):
    """两帧包含/放大关系。"""

    none = "none"
    a_contains_b = "a_contains_b"
    b_contains_a = "b_contains_a"


class _LLMContainmentVerdict(BaseModel):
    """两帧包含关系判定。"""

    containment: ContainmentKind = ContainmentKind.none
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("containment", mode="before")
    @classmethod
    def _unwrap_containment(cls, value: object) -> object:
        return _unwrap_structured_enum(value)


class PhotoRelation(str, Enum):
    """两帧是否同一张照片。"""

    same_photo = "same_photo"
    same_scene = "same_scene"
    different_photo = "different_photo"
    a_not_input = "a_not_input"
    b_not_input = "b_not_input"
    both_not_input = "both_not_input"


class _LLMPhotoRelationVerdict(BaseModel):
    """两帧照片关系判定。"""

    relation: PhotoRelation = PhotoRelation.same_photo
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("relation", mode="before")
    @classmethod
    def _unwrap_relation(cls, value: object) -> object:
        return _unwrap_structured_enum(value)


class _LLMTaskMergeResult(BaseModel):
    """条件触发的双向切分复核（可合并，也可补拆）。"""

    tasks: list[_LLMGeoTaskDraft] = Field(default_factory=list)
    reason: str = ""


class _SamePlacePair(BaseModel):
    """两个 resolved task 的最终地点是否同地/同链。"""

    task_i: int = Field(ge=1, description="1-based task 序号")
    task_j: int = Field(ge=1, description="1-based task 序号")
    same_place_or_chain: bool = False
    reason: str = ""


class _LLMSamePlaceGate(BaseModel):
    """多 task 措辞不同时的廉价同地门禁（只触发复核，不直接改切分）。"""

    pairs: list[_SamePlacePair] = Field(default_factory=list)
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


def _resolved_location_entries(
    draft_tasks: list[_LLMGeoTaskDraft],
) -> list[tuple[int, str, str]]:
    """返回 (1-based index, raw location, normalized key) 列表。"""
    entries: list[tuple[int, str, str]] = []
    for i, task in enumerate(draft_tasks, start=1):
        answer_status = getattr(task, "answer_status", AnswerStatus.resolved)
        if answer_status != AnswerStatus.resolved:
            continue
        raw = str(getattr(task, "final_location_text", "") or "").strip()
        if not raw:
            continue
        entries.append((i, raw, _normalize_location_key(raw)))
    return entries


def _has_exact_duplicate_location_keys(entries: list[tuple[int, str, str]]) -> bool:
    keys = [key for _, _, key in entries if key]
    return len(keys) != len(set(keys))


def _llm_same_place_anomalies(
    draft_tasks: list[_LLMGeoTaskDraft],
    *,
    transcript: list[TranscriptSegment],
    overview_images: list[str] | None,
) -> list[str]:
    """措辞不同的多 resolved 地点：用 LLM 判定是否同地/同链，命中则触发复核。"""
    entries = _resolved_location_entries(draft_tasks)
    if len(entries) < 2:
        return []
    if _has_exact_duplicate_location_keys(entries):
        # 字符串全等已由客观异常覆盖，无需再花一次同地门禁。
        return []

    payload = []
    for i, task in enumerate(draft_tasks, start=1):
        payload.append(
            {
                "task_index": i,
                "time_start": float(task.time_start),
                "time_end": float(task.time_end),
                "task_summary": str(getattr(task, "task_summary", "") or ""),
                "answer_status": (
                    getattr(task, "answer_status", AnswerStatus.resolved).value
                    if isinstance(
                        getattr(task, "answer_status", AnswerStatus.resolved), Enum
                    )
                    else str(getattr(task, "answer_status", AnswerStatus.resolved))
                ),
                "final_location_text": str(
                    getattr(task, "final_location_text", "") or ""
                ),
            }
        )
    prompt = (
        "以下是首次审核得到的多个 resolved task。"
        "它们的最终地点字符串并不完全相同。"
        "请两两判断：是否其实是同一最终地点，或同一条答案链"
        "（同题多图共同支撑一地 / 后图精化前图）。\n"
        "- same_place_or_chain=true：应合并为同一个 task，属过拆嫌疑；\n"
        "- same_place_or_chain=false：确实是不同目标、不同最终地点。\n"
        "只做同地/同链判定，不要输出新的切分结果。"
        "不确定时宁可判 true（交给后续保守双向复核）。\n"
        f"tasks JSON:\n{json.dumps(payload, ensure_ascii=False)}\n"
        "字幕（供判断是否同一条讲解链）：\n"
        f"{_format_transcript(transcript)}\n"
        "请输出所有需要比较的 task 对（至少覆盖全部 resolved 两两组合）与简短 reason。"
    )
    try:
        gate = call_structured(
            prompt,
            _LLMSamePlaceGate,
            images=overview_images or None,
            lane="llm",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("same-place gate failed: %s", exc)
        return []

    hits: list[str] = []
    for pair in gate.pairs or []:
        if not bool(pair.same_place_or_chain):
            continue
        i = int(pair.task_i)
        j = int(pair.task_j)
        detail = str(pair.reason or gate.reason or "").strip()
        suffix = f"（{detail}）" if detail else ""
        hits.append(
            f"task {i} 与 task {j} 经模型判定为同一最终地点或同一答案链，"
            f"疑似同题过拆{suffix}"
        )
    return hits


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

        d0 = getattr(task, "display_time_start", None)
        d1 = getattr(task, "display_time_end", None)
        if d0 is not None and d1 is not None:
            ds = float(d0)
            de = float(d1)
            if de < ds:
                anomalies.append(f"task {i} 的出示窗终点早于起点")
            elif (
                de < min(t0, t1) - boundary_tolerance
                or ds > max(t0, t1) + boundary_tolerance
            ):
                anomalies.append(f"task {i} 的出示窗与蒸馏窗明显矛盾")

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
    """只在客观异常、同地嫌疑或模型低置信时复核，避免重复改坏正确切分。"""
    anomalies = _objective_split_anomalies(
        draft_tasks,
        duration=duration,
        boundary_tolerance=boundary_tolerance,
    )
    # 措辞不同但可能同地/同链：先廉价 LLM 门禁，命中再进入双向复核。
    if not any("完全相同的明确最终地点" in a for a in anomalies):
        anomalies.extend(
            _llm_same_place_anomalies(
                draft_tasks,
                transcript=transcript,
                overview_images=overview_images,
            )
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
            "target_kind": (
                t.target_kind.value
                if isinstance(t.target_kind, Enum)
                else t.target_kind
            ),
            "display_time_start": getattr(t, "display_time_start", None),
            "display_time_end": getattr(t, "display_time_end", None),
            "keyframe_timestamps": list(getattr(t, "keyframe_timestamps", []) or []),
            "multi_target_images": bool(getattr(t, "multi_target_images", False)),
            "segment_start_idx": getattr(t, "segment_start_idx", None),
            "segment_end_idx": getattr(t, "segment_end_idx", None),
            "task_summary": str(getattr(t, "task_summary", "") or ""),
            "answer_status": (
                getattr(t, "answer_status", AnswerStatus.resolved).value
                if isinstance(
                    getattr(t, "answer_status", AnswerStatus.resolved), Enum
                )
                else str(getattr(t, "answer_status", AnswerStatus.resolved))
            ),
            "final_location_text": str(getattr(t, "final_location_text", "") or ""),
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
        "必须合并为 **一个** task，并给出 display 出示粗窗；"
        "**不要**预定选图张数。\n"
        "**同一最终地点 / 同一条答案链必须合并**"
        "（即使线索形态不同，如河道环境与建筑外观）。\n"
        "仅当不同目标、不同最终地点时才保留多个 task。"
        "如果一个 task 内实际含多条独立最终答案链，必须补拆；"
        "如果多个 task 属于同一最终答案链，必须合并。\n"
        "每个结果 task 继续填写 answer_status、final_location_text、"
        "display_time_start/display_time_end。\n"
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


def _normalize_task_window(
    raw: _LLMGeoTaskDraft,
    *,
    duration: float,
    transcript: list[TranscriptSegment],
    boundary_tolerance: float,
) -> tuple[float, float]:
    """以字幕索引和近邻出示窗修复轻微边界偏差，再限制在视频物理范围。"""
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
    for value in (
        getattr(raw, "display_time_start", None),
        getattr(raw, "display_time_end", None),
    ):
        if value is None:
            continue
        stamp = float(value)
        if t0 - tolerance <= stamp <= t1 + tolerance:
            t0 = min(t0, stamp)
            t1 = max(t1, stamp)

    max_time = max(0.0, float(duration) - 0.001) if duration > 0 else max(t1, 0.0)
    t0 = min(max(0.0, t0), max_time)
    t1 = min(max(t0, t1), max_time)
    return t0, t1


def _resolve_display_window(
    raw: _LLMGeoTaskDraft,
    *,
    distill_start: float,
    distill_end: float,
) -> tuple[float, float]:
    """解析出示粗窗：优先模型给出，否则蒸馏窗前段封顶。"""
    settings = get_settings()
    max_fallback = max(1.0, float(settings.AUDIT_DISPLAY_WINDOW_MAX_SEC))
    lo = max(0.0, float(distill_start))
    hi = max(lo, float(distill_end))
    d0 = getattr(raw, "display_time_start", None)
    d1 = getattr(raw, "display_time_end", None)
    if d0 is not None and d1 is not None:
        start = float(d0)
        end = float(d1)
        if end < start:
            start, end = end, start
        start = min(max(start, lo), hi)
        end = min(max(end, start), hi)
        if end > start + 1e-3:
            return start, end
        logger.warning(
            "invalid display window [%.2f, %.2f]; fallback to distill prefix",
            float(d0),
            float(d1),
        )
    end = min(hi, lo + max_fallback)
    if end <= lo + 1e-6:
        end = hi
    logger.info(
        "display window fallback distill_prefix [%.2f, %.2f] (cap=%.1fs)",
        lo,
        end,
        max_fallback,
    )
    return lo, end


def _clip_process_intervals(
    intervals: list[ProcessInterval],
    *,
    distill_start: float,
    distill_end: float,
) -> list[ProcessInterval]:
    """裁剪到蒸馏窗内并丢弃无效/过短区间。"""
    lo = max(0.0, float(distill_start))
    hi = max(lo, float(distill_end))
    cleaned: list[ProcessInterval] = []
    for item in intervals:
        start = min(max(float(item.start), lo), hi)
        end = min(max(float(item.end), lo), hi)
        if end < start:
            start, end = end, start
        if end <= start + 1e-3:
            continue
        cleaned.append(
            ProcessInterval(
                start=start,
                end=end,
                role=item.role,
                confidence=float(item.confidence),
            )
        )
    cleaned.sort(key=lambda x: (x.start, x.end))
    return cleaned


def _show_source_windows(
    intervals: list[ProcessInterval],
) -> list[tuple[float, float]]:
    """合并相邻/重叠的 show_source 段，返回采样窗列表。"""
    shows = [
        (float(item.start), float(item.end))
        for item in intervals
        if item.role == ProcessRole.show_source
    ]
    if not shows:
        return []
    shows.sort(key=lambda pair: pair[0])
    merged: list[list[float]] = [[shows[0][0], shows[0][1]]]
    for start, end in shows[1:]:
        if start <= merged[-1][1] + 1e-3:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(a), float(b)) for a, b in merged]


def _count_nonadjacent_show_source(intervals: list[ProcessInterval]) -> int:
    """不相邻 show_source 段数（相邻/重叠先合并）。"""
    return len(_show_source_windows(intervals))


def _process_role_at(
    intervals: list[ProcessInterval],
    stamp: float,
) -> ProcessRole | None:
    """返回覆盖 stamp 的区间角色；无覆盖则 None。"""
    t = float(stamp)
    for item in intervals:
        if item.start - 1e-6 <= t <= item.end + 1e-6:
            return item.role
    return None


def _resolve_sample_windows(
    raw: _LLMGeoTaskDraft,
    *,
    distill_start: float,
    distill_end: float,
    intervals: list[ProcessInterval],
) -> tuple[list[tuple[float, float]], list[ProcessInterval]]:
    """决定密采样窗：优先 show_source 并集，否则回退单出示粗窗。"""
    clipped = _clip_process_intervals(
        intervals, distill_start=distill_start, distill_end=distill_end
    )
    show_windows = _show_source_windows(clipped)
    if show_windows:
        return show_windows, clipped
    fallback = _resolve_display_window(
        raw, distill_start=distill_start, distill_end=distill_end
    )
    return [fallback], clipped


def _dense_sample_windows(
    windows: list[tuple[float, float]],
    *,
    interval: float,
    max_n: int,
) -> list[float]:
    """对多个出示段密采样后合并去重，再受总上限约束。"""
    from pipeline.stage_audit_split.frame_prefilter import subsample_timestamps

    stamps: list[float] = []
    seen: set[str] = set()
    per_cap = max(1, int(max_n))
    for start, end in windows:
        for stamp in _dense_sample_timestamps(
            start, end, interval=interval, max_n=per_cap
        ):
            key = f"{stamp:.3f}"
            if key in seen:
                continue
            seen.add(key)
            stamps.append(float(stamp))
    stamps.sort()
    return subsample_timestamps(stamps, max(1, int(max_n)))


def _dense_sample_timestamps(
    start: float,
    end: float,
    *,
    interval: float,
    max_n: int,
) -> list[float]:
    """出示窗内按间隔密采样，受硬上限约束。"""
    lo = max(0.0, float(start))
    hi = max(lo, float(end))
    if hi <= lo + 1e-9:
        return [lo]
    step = max(0.05, float(interval))
    stamps: list[float] = []
    t = lo
    while t <= hi + 1e-9:
        stamps.append(round(t, 3))
        t += step
    if not stamps or stamps[-1] < hi - 1e-3:
        stamps.append(round(hi, 3))
    # 超上限时均匀抽稀
    from pipeline.stage_audit_split.frame_prefilter import subsample_timestamps

    return subsample_timestamps(stamps, max(1, int(max_n)))


def _assessment_rank_key(item: KeyframeAssessment) -> tuple[bool, bool, float]:
    """段内/组内择优：干净原图 > 无讲解覆盖 > 质量分。"""
    return (item.clean_source, not item.tutorial_overlay, item.quality_score)


def _group_rank_key(
    item: KeyframeAssessment,
    *,
    use_evidence: bool,
) -> tuple[float, bool, bool, float]:
    """组间排序：有证据简报时支撑分优先，再走干净度。"""
    support = float(item.chain_support_score) if use_evidence else 0.0
    clean, no_overlay, quality = _assessment_rank_key(item)
    return (support, clean, no_overlay, quality)


def _is_problem_input_frame(
    item: KeyframeAssessment,
    *,
    require_evidence_role: bool,
) -> bool:
    """是否可作为出示连续段的合格帧。"""
    if item.kind != FrameKind.target_photo.value or item.answer_leakage:
        return False
    if not require_evidence_role:
        return True
    role = (item.evidence_role or EvidenceRole.unknown.value).strip()
    return role == EvidenceRole.problem_input.value


def _fold_presentation_episodes(
    assessments: list[KeyframeAssessment],
    *,
    require_evidence_role: bool = False,
) -> list[KeyframeAssessment]:
    """把连续合格出示帧折成段，每段只留质量最高代表。

    中间出现已验收的非目标帧（工具/揭晓/黑场/错误/unused_broll）则打断出示段。
    """
    ordered = sorted(assessments, key=lambda item: item.timestamp)
    episodes: list[list[KeyframeAssessment]] = []
    current: list[KeyframeAssessment] = []
    for item in ordered:
        eligible = _is_problem_input_frame(
            item, require_evidence_role=require_evidence_role
        )
        if eligible:
            current.append(item)
            continue
        if current:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return [max(ep, key=_assessment_rank_key) for ep in episodes if ep]


class _UnionFind:
    """简单并查集，供源输入归并聚合。"""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _completeness_rank_key(
    item: KeyframeAssessment,
    *,
    preferred: set[int],
    index: int,
) -> tuple[int, bool, bool, float]:
    """组内择优：包含关系中的更完整帧优先，再走干净度。"""
    boost = 1 if index in preferred else 0
    clean, no_overlay, quality = _assessment_rank_key(item)
    return (boost, clean, no_overlay, quality)


def _pair_containment(
    path_a: str,
    path_b: str,
    *,
    min_confidence: float = 0.8,
) -> ContainmentKind:
    """先廉价预检，再 VLM；高置信才返回非 none。"""
    from pipeline.stage_audit_split.frame_prefilter import containment_precheck_score

    kind, score = containment_precheck_score(path_a, path_b)
    if kind != "none" and score >= 0.82:
        try:
            return ContainmentKind(kind)
        except ValueError:
            pass
    prompt = (
        f"{CONTAINMENT_HINT}\n\n"
        "已附上两张图：第一张为 A，第二张为 B。"
        "请输出 containment、confidence、reason。"
    )
    try:
        result = call_structured(
            prompt,
            _LLMContainmentVerdict,
            images=[path_a, path_b],
            lane="vlm",
            max_attempts=1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("containment call failed: %s", exc)
        return ContainmentKind.none
    if float(getattr(result, "confidence", 0.0)) < float(min_confidence):
        return ContainmentKind.none
    raw = getattr(result, "containment", ContainmentKind.none)
    if isinstance(raw, ContainmentKind):
        return raw
    try:
        return ContainmentKind(str(raw))
    except ValueError:
        return ContainmentKind.none


def _pair_photo_relation(
    path_a: str,
    path_b: str,
    *,
    target_kind: TargetKind,
    visual_evidence_brief: str = "",
) -> PhotoRelation:
    """两两判定是否同一张照片。"""
    kind_label = (
        target_kind.value if isinstance(target_kind, Enum) else str(target_kind)
    )
    brief = visual_evidence_brief.strip()
    brief_block = (
        f"视觉证据简报（只核验画面事实，不决定张数）：\n{brief}\n"
        if brief
        else "视觉证据简报：（空）\n"
    )
    prompt = (
        f"{SOURCE_IDENTITY_HINT}\n\n"
        f"target_kind={kind_label}\n"
        f"{brief_block}"
        "已附上两张图：第一张为 A，第二张为 B。"
        "请输出 relation、confidence、reason。"
    )
    try:
        result = call_structured(
            prompt,
            _LLMPhotoRelationVerdict,
            images=[path_a, path_b],
            lane="vlm",
            max_attempts=1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("photo relation call failed: %s; default same_photo", exc)
        return PhotoRelation.same_photo
    raw = getattr(result, "relation", PhotoRelation.same_photo)
    if isinstance(raw, PhotoRelation):
        return raw
    try:
        return PhotoRelation(str(raw))
    except ValueError:
        return PhotoRelation.same_photo


def _resolve_source_identity(
    representatives: list[KeyframeAssessment],
    *,
    target_kind: TargetKind,
    visual_evidence_brief: str = "",
) -> list[KeyframeAssessment]:
    """对出示段代表做源输入归并：包含硬合并 + 是否同一张照片。

    不确定时合并，默认宁可少一张；包含关系命中时优先留更完整帧。
    """
    if len(representatives) <= 1:
        return list(representatives)

    n = len(representatives)
    uf = _UnionFind(n)
    preferred: set[int] = set()
    discarded: set[int] = set()
    hard_merged: set[tuple[int, int]] = set()

    # 1) 包含/放大硬合并
    for i in range(n):
        for j in range(i + 1, n):
            kind = _pair_containment(
                representatives[i].image_path,
                representatives[j].image_path,
            )
            if kind == ContainmentKind.a_contains_b:
                uf.union(i, j)
                preferred.add(i)
                hard_merged.add((i, j))
            elif kind == ContainmentKind.b_contains_a:
                uf.union(i, j)
                preferred.add(j)
                hard_merged.add((i, j))

    # 2) 语义两两关系（已硬合并的不再重开）
    for i in range(n):
        if i in discarded:
            continue
        for j in range(i + 1, n):
            if j in discarded:
                continue
            if (i, j) in hard_merged or uf.find(i) == uf.find(j):
                continue
            relation = _pair_photo_relation(
                representatives[i].image_path,
                representatives[j].image_path,
                target_kind=target_kind,
                visual_evidence_brief=visual_evidence_brief,
            )
            if relation == PhotoRelation.same_photo:
                uf.union(i, j)
            elif relation == PhotoRelation.same_scene:
                if target_kind == TargetKind.video_derived:
                    uf.union(i, j)
                # still_image 上的 same_scene 不当合并信号
            elif relation == PhotoRelation.different_photo:
                continue
            elif relation == PhotoRelation.a_not_input:
                discarded.add(i)
            elif relation == PhotoRelation.b_not_input:
                discarded.add(j)
            elif relation == PhotoRelation.both_not_input:
                discarded.add(i)
                discarded.add(j)

    buckets: dict[int, list[tuple[int, KeyframeAssessment]]] = {}
    for idx, frame in enumerate(representatives):
        if idx in discarded:
            continue
        root = uf.find(idx)
        buckets.setdefault(root, []).append((idx, frame))

    if not buckets:
        return [max(representatives, key=_assessment_rank_key)]

    selected: list[KeyframeAssessment] = []
    for members in buckets.values():
        best_idx, best = max(
            members,
            key=lambda pair: _completeness_rank_key(
                pair[1], preferred=preferred, index=pair[0]
            ),
        )
        _ = best_idx
        selected.append(best)

    brief = visual_evidence_brief.strip()
    use_evidence = bool(brief)
    selected.sort(
        key=lambda item: _group_rank_key(item, use_evidence=use_evidence),
        reverse=True,
    )
    return selected


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


def _candidate_frame_dir(video_id: str, task_id: str) -> Path:
    """密采样探测帧目录（缓存，可随时清空）。"""
    settings = get_settings()
    path = Path(settings.CACHE_DIR) / "audit_candidates" / video_id / task_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _promote_selected_images(
    selected: list[KeyframeAssessment],
    *,
    video_id: str,
    task_id: str,
) -> list[str]:
    """把入选帧复制到 ``SELECTED_DIR``，并回写 assessment.image_path。"""
    settings = get_settings()
    dest_dir = Path(settings.SELECTED_DIR) / video_id / task_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for item in selected:
        src = Path(item.image_path)
        name = (
            src.name
            if src.name.startswith(f"{task_id}_")
            else f"{task_id}_{src.name}"
        )
        dest = dest_dir / name
        if src.is_file() and (
            dest.resolve() != src.resolve() or not dest.is_file()
        ):
            dest.write_bytes(src.read_bytes())
        final = str(dest.resolve()) if dest.is_file() else str(src.resolve())
        item.image_path = final
        paths.append(final)
    return paths


def _unlink_quiet(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def verify_keyframe_kind(image_path: str) -> FrameKind:
    """VLM 验收单帧是否为待定位目标图。"""
    return verify_keyframe(image_path).kind


def _default_evidence_role(kind: FrameKind) -> EvidenceRole:
    """无模型字段时的证据角色回退。"""
    if kind == FrameKind.target_photo:
        return EvidenceRole.problem_input
    if kind == FrameKind.teaching_ui:
        return EvidenceRole.process_tool
    return EvidenceRole.other


def verify_keyframe(
    image_path: str,
    *,
    narrative_context: str = "",
    visual_evidence_brief: str = "",
    process_role: ProcessRole | str | None = None,
) -> _LLMFrameVerdict:
    """返回带质量、泄露与证据角色标记的逐帧验收结果。"""
    prompt = FRAME_VERIFY_HINT
    brief = visual_evidence_brief.strip()
    if brief:
        prompt += (
            "\n\n本题视觉证据简报（只用于判断本帧是否为定位链依赖的输入，"
            "以及 chain_support_score；不得据此把旁白地名当成 answer_leakage）：\n"
            f"{brief}\n"
        )
    else:
        prompt += (
            "\n\n本题无视觉证据简报：对明确 target_photo 可给 evidence_role=problem_input、"
            "chain_support_score≈0.5；看起来像未参与推理的空镜/过场则 unused_broll。\n"
        )
    role_value = (
        process_role.value
        if isinstance(process_role, ProcessRole)
        else (str(process_role).strip() if process_role else "")
    )
    if role_value == ProcessRole.tool.value or role_value == ProcessRole.reveal.value:
        prompt += (
            f"\n\n过程时间线软先验：此刻区间角色为 {role_value}。"
            "搜索页、侧栏、结果缩略图、街景浏览、地图/遥感核验、钉点揭晓，"
            "即使主体是全屏建筑外观，也应判 teaching_ui，"
            f"evidence_role={'reveal' if role_value == ProcessRole.reveal.value else 'process_tool'}。"
            "画面仍是最终裁判；仅当画面明确是题目原图本身且无工具 UI 时才可判 target_photo。\n"
        )
    elif role_value == ProcessRole.show_source.value:
        prompt += (
            "\n\n过程时间线软先验：此刻区间角色为 show_source。"
            "主画面若是待定位实拍/原图，可判 target_photo 且 evidence_role=problem_input；"
            "若画面其实是工具界面或揭晓，仍按画面判 teaching_ui。\n"
        )
    elif role_value == ProcessRole.other.value:
        prompt += (
            "\n\n过程时间线软先验：此刻区间角色为 other。"
            "更可能是空镜/过场；若画面像实拍但未作定位输入，优先 unused_broll。\n"
        )
    if narrative_context.strip():
        prompt += (
            "\n\n以下是该帧在讲解时间线附近的叙事上下文，只用于判断画面角色：\n"
            f"{narrative_context.strip()}\n"
            "若上下文表明此时正在展示搜索结果、地图/遥感/街景核验、答案揭晓、"
            "找到后的航拍/全景或候选比对，则即使画面本身是全屏实景，也不是原始"
            "待定位输入，应判 teaching_ui，evidence_role=process_tool 或 reveal。"
            "若上下文明确是展示题目原图、另一张原图或放大原图，则可判 target_photo"
            "且 evidence_role=problem_input。"
            "若画面像实拍但上下文与简报都未把它当定位输入，判 unused_broll。"
            "不要仅因上下文写出了最终地点就把画面标为 answer_leakage；"
            "answer_leakage 仍要求画面本身可见答案。"
        )
    return call_structured(
        prompt,
        _LLMFrameVerdict,
        images=[image_path],
        lane="vlm",
        max_attempts=1,
    )


def extract_visual_evidence_brief(
    transcript: list[TranscriptSegment],
    *,
    time_start: float,
    time_end: float,
    task_summary: str = "",
) -> str:
    """从蒸馏窗字幕抽取题级视觉证据简报（纯文本，不看图、不读 GT）。"""
    window = [
        seg
        for seg in transcript
        if seg.end >= float(time_start) - 1e-6 and seg.start <= float(time_end) + 1e-6
    ]
    if not window:
        return ""
    lines = "\n".join(
        f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}" for seg in window
    )
    prompt = (
        f"{EVIDENCE_BRIEF_HINT}\n\n"
        f"任务摘要：{task_summary.strip() or '（无）'}\n"
        f"蒸馏窗旁白：\n{lines}\n"
        "请输出 visual_evidence_brief。"
    )
    try:
        result = call_structured(
            prompt,
            _LLMEvidenceBrief,
            lane="llm",
            max_attempts=1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("visual evidence brief failed: %s", exc)
        return ""
    return str(getattr(result, "visual_evidence_brief", "") or "").strip()


def extract_process_timeline(
    transcript: list[TranscriptSegment],
    *,
    time_start: float,
    time_end: float,
    task_summary: str = "",
    overview_images: list[str] | None = None,
) -> list[ProcessInterval]:
    """从蒸馏窗字幕抽取过程时间线（内部材料，不进阶段2 prompt）。"""
    window = [
        seg
        for seg in transcript
        if seg.end >= float(time_start) - 1e-6 and seg.start <= float(time_end) + 1e-6
    ]
    if not window:
        return []
    lines = "\n".join(
        f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}" for seg in window
    )
    prompt = (
        f"{PROCESS_TIMELINE_HINT}\n\n"
        f"任务摘要：{task_summary.strip() or '（无）'}\n"
        f"蒸馏窗：{float(time_start):.1f}s – {float(time_end):.1f}s\n"
        f"蒸馏窗旁白：\n{lines}\n"
        "请输出 intervals。"
    )
    images = [p for p in (overview_images or []) if str(p).strip() and Path(p).is_file()]
    try:
        result = call_structured(
            prompt,
            _LLMProcessTimeline,
            images=images or None,
            lane="vlm" if images else "llm",
            max_attempts=1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("process timeline failed: %s", exc)
        return []
    raw_items = getattr(result, "intervals", []) or []
    out: list[ProcessInterval] = []
    for item in raw_items:
        try:
            role = item.role
            if not isinstance(role, ProcessRole):
                role = ProcessRole(str(role))
            start = float(item.start)
            end = float(item.end)
            if end < start:
                start, end = end, start
            out.append(
                ProcessInterval(
                    start=start,
                    end=end,
                    role=role,
                    confidence=float(getattr(item, "confidence", 0.5) or 0.5),
                )
            )
        except (TypeError, ValueError) as exc:
            logger.debug("skip invalid process interval: %s", exc)
            continue
    return _clip_process_intervals(
        out, distill_start=time_start, distill_end=time_end
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


def _materialize_task_images(
    *,
    video_path: str,
    video_id: str,
    task_id: str,
    raw: _LLMGeoTaskDraft,
    t0: float,
    t1: float,
    hard_cap: int,
    transcript: list[TranscriptSegment],
    task_dir: Path,
    resume_tasks: bool = True,
) -> tuple[
    list[float],
    list[str],
    bool,
    list[KeyframeAssessment],
    str,
    str,
    list[ProcessInterval],
]:
    """在 show_source 并集内密采样、验收，再折叠出示段并归并为最小源输入集。

    Returns:
        stamps, paths, multi_target, assessments, quality_reason,
        visual_evidence_brief, process_intervals
    """
    from pipeline.stage_audit_split.frame_prefilter import (
        prefilter_frame,
        subsample_timestamps,
    )

    settings = get_settings()
    quality_floor = min(1.0, max(0.0, float(settings.AUDIT_MIN_FRAME_QUALITY)))
    support_floor = min(1.0, max(0.0, float(settings.AUDIT_MIN_CHAIN_SUPPORT)))
    interval = max(0.05, float(settings.AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC))
    max_sampled = max(1, int(settings.AUDIT_MAX_SAMPLED_FRAMES))
    max_vlm = max(1, int(settings.AUDIT_MAX_VLM_FRAME_VERIFIES))
    cap = max(1, int(hard_cap))

    brief = extract_visual_evidence_brief(
        transcript,
        time_start=t0,
        time_end=t1,
        task_summary=str(getattr(raw, "task_summary", "") or ""),
    )
    use_evidence = bool(brief.strip())

    process_intervals = extract_process_timeline(
        transcript,
        time_start=t0,
        time_end=t1,
        task_summary=str(getattr(raw, "task_summary", "") or ""),
    )
    sample_windows, process_intervals = _resolve_sample_windows(
        raw,
        distill_start=t0,
        distill_end=t1,
        intervals=process_intervals,
    )
    sample_stamps = _dense_sample_windows(
        sample_windows,
        interval=interval,
        max_n=max_sampled,
    )
    # 可选弱先验：模型戳仅当落在某段采样窗内才提前
    weak_priors = {
        round(float(s), 3)
        for s in (getattr(raw, "keyframe_timestamps", []) or [])
        if any(
            float(w0) - 1e-6 <= float(s) <= float(w1) + 1e-6
            for w0, w1 in sample_windows
        )
    }
    sample_stamps.sort(
        key=lambda s: (0 if round(s, 3) in weak_priors else 1, s),
    )

    assessment_checkpoint = task_dir / "candidate_assessments.partial.json"
    assessments: list[KeyframeAssessment] = []
    if not resume_tasks and assessment_checkpoint.is_file():
        _unlink_quiet(str(assessment_checkpoint))
    elif resume_tasks and assessment_checkpoint.is_file():
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
    frame_dir = _candidate_frame_dir(video_id, task_id)
    pending_for_vlm: list[tuple[float, str, bool]] = []

    for stamp in sample_stamps:
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
        # 近重复不再抽帧时贪心删除；留给出示段折叠与源输入归并择优
        verdict_pre = prefilter_frame(path)
        if not verdict_pre.keep:
            assessments.append(
                KeyframeAssessment(
                    timestamp=float(stamp),
                    image_path=str(Path(path).resolve()),
                    kind="other",
                    quality_score=0.0,
                    evidence_role=EvidenceRole.other.value,
                    chain_support_score=0.0,
                    reason=f"prefilter:{verdict_pre.skip_reason}",
                )
            )
            continue
        pending_for_vlm.append(
            (float(stamp), str(Path(path).resolve()), verdict_pre.ui_or_map_penalty)
        )

    # 优先非 UI/地图惩罚帧；超出 VLM 预算再均匀抽稀
    pending_for_vlm.sort(key=lambda item: (item[2], item[0]))
    if len(pending_for_vlm) > max_vlm:
        keep_stamps = set(
            subsample_timestamps([p[0] for p in pending_for_vlm], max_vlm)
        )
        pending_for_vlm = [p for p in pending_for_vlm if p[0] in keep_stamps][:max_vlm]

    for stamp, path, ui_penalty in pending_for_vlm:
        try:
            verdict = verify_keyframe(
                path,
                narrative_context=_frame_narrative_context(
                    transcript,
                    stamp=stamp,
                    task_summary=str(getattr(raw, "task_summary", "") or ""),
                ),
                visual_evidence_brief=brief,
                process_role=_process_role_at(process_intervals, stamp),
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
                    image_path=path,
                    kind="error",
                    evidence_role=EvidenceRole.unknown.value,
                    chain_support_score=0.0,
                    reason=f"验收调用失败：{type(exc).__name__}",
                )
            )
            _write_json(
                assessment_checkpoint,
                [item.model_dump(mode="json") for item in assessments],
            )
            continue
        kind = verdict.kind
        quality = float(getattr(verdict, "quality_score", 0.8))
        if ui_penalty:
            quality = max(0.0, quality - 0.15)
        leakage = bool(getattr(verdict, "answer_leakage", False))
        overlay = bool(getattr(verdict, "tutorial_overlay", False))
        clean = bool(
            getattr(verdict, "clean_source", kind == FrameKind.target_photo)
        )
        role_raw = getattr(verdict, "evidence_role", None)
        if isinstance(role_raw, EvidenceRole):
            evidence_role = role_raw
        elif role_raw:
            try:
                evidence_role = EvidenceRole(str(role_raw))
            except ValueError:
                evidence_role = _default_evidence_role(kind)
        else:
            evidence_role = _default_evidence_role(kind)
        support = float(getattr(verdict, "chain_support_score", 0.5))
        if evidence_role == EvidenceRole.unused_broll:
            support = min(support, 0.2)
        elif evidence_role in {
            EvidenceRole.process_tool,
            EvidenceRole.reveal,
            EvidenceRole.other,
        }:
            support = min(support, 0.15)
        reason = str(getattr(verdict, "reason", "") or "")
        if ui_penalty:
            reason = (reason + " | prefilter:ui_or_map_penalty").strip(" |")
        assessments.append(
            KeyframeAssessment(
                timestamp=float(stamp),
                image_path=path,
                kind=kind.value,
                quality_score=quality,
                answer_leakage=leakage,
                tutorial_overlay=overlay,
                clean_source=clean,
                evidence_role=evidence_role.value,
                chain_support_score=support,
                reason=reason,
            )
        )
        _write_json(
            assessment_checkpoint,
            [item.model_dump(mode="json") for item in assessments],
        )

    for item in assessments:
        item.selected = False

    # 多段 show_source 时按窗分别折叠，避免未采样的中间 tool 段无法打断连续段
    if len(sample_windows) > 1:
        episode_reps: list[KeyframeAssessment] = []
        for w0, w1 in sample_windows:
            subset = [
                item
                for item in assessments
                if float(w0) - 1e-6 <= float(item.timestamp) <= float(w1) + 1e-6
            ]
            episode_reps.extend(
                _fold_presentation_episodes(
                    subset,
                    require_evidence_role=use_evidence,
                )
            )
    else:
        episode_reps = _fold_presentation_episodes(
            assessments,
            require_evidence_role=use_evidence,
        )
    if not episode_reps:
        _write_json(
            assessment_checkpoint,
            [item.model_dump(mode="json") for item in assessments],
        )
        return (
            [],
            [],
            False,
            assessments,
            "出示窗内未找到干净待定位原图",
            brief,
            process_intervals,
        )

    # 组间：有 brief 时按证据支撑优先；组内折叠已按干净度取代表
    episode_reps.sort(
        key=lambda item: _group_rank_key(item, use_evidence=use_evidence),
        reverse=True,
    )
    merged = _resolve_source_identity(
        episode_reps,
        target_kind=raw.target_kind,
        visual_evidence_brief=brief,
    )
    selected = merged[:cap]
    selected.sort(key=lambda item: item.timestamp)
    for item in selected:
        item.selected = True

    selected_stamps = [item.timestamp for item in selected]
    selected_paths = _promote_selected_images(
        selected, video_id=video_id, task_id=task_id
    )
    multi = len(selected) > 1
    _write_json(
        assessment_checkpoint,
        [item.model_dump(mode="json") for item in assessments],
    )

    if not selected:
        return (
            [],
            [],
            False,
            assessments,
            "出示窗内未找到干净待定位原图",
            brief,
            process_intervals,
        )

    # 多段不相邻出示却只留下 1 张：漏图信号（不采 tool 段凑张）
    if (
        use_evidence
        and len(selected) == 1
        and _count_nonadjacent_show_source(process_intervals) >= 2
    ):
        return (
            selected_stamps,
            selected_paths,
            multi,
            assessments,
            "过程时间线含多段不相邻出示窗，但最小源输入集仅选出 1 张",
            brief,
            process_intervals,
        )

    if use_evidence:
        weak_support = [
            item for item in selected if item.chain_support_score < support_floor
        ]
        if weak_support:
            return (
                selected_stamps,
                selected_paths,
                multi,
                assessments,
                "选中帧与定位链视觉证据不对齐",
                brief,
                process_intervals,
            )
    low_quality = [
        item
        for item in selected
        if item.quality_score < quality_floor
        or item.tutorial_overlay
        or not item.clean_source
    ]
    if low_quality:
        reason = "选中帧仍含讲解覆盖、界面残留或质量低于阈值"
        if not use_evidence:
            reason = f"{reason}（无视觉证据简报，已回退质量择优）"
        return (
            selected_stamps,
            selected_paths,
            multi,
            assessments,
            reason,
            brief,
            process_intervals,
        )
    return (
        selected_stamps,
        selected_paths,
        multi,
        assessments,
        "",
        brief,
        process_intervals,
    )


def compose_image_selection_note(
    *,
    status: TaskStatus,
    status_reason: str = "",
    assessments: list[KeyframeAssessment] | None = None,
) -> str:
    """程序化组装选图评价 note（质量等级 + 选中帧明细）。

    ``accepted`` / ``needs_review`` 都必须有非空 note；不另调 VLM。
    """
    grade = status.value
    selected = [item for item in (assessments or []) if item.selected]
    parts: list[str] = [f"选图质量等级={grade}", f"选中张数={len(selected)}"]
    reason = (status_reason or "").strip()
    if reason:
        parts.append(f"选图原因: {reason}")
    if not selected:
        parts.append("选中帧: 无")
    else:
        parts.append("选中帧明细:")
        for idx, item in enumerate(selected, start=1):
            frame_reason = (item.reason or "").strip() or "（无逐帧理由）"
            parts.append(
                f"- [{idx}] t={item.timestamp:.3f}s quality={item.quality_score:.2f} "
                f"overlay={item.tutorial_overlay} clean={item.clean_source} "
                f"support={item.chain_support_score:.2f} reason={frame_reason}"
            )
    return "\n".join(parts)


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
        "display_time_start": getattr(raw, "display_time_start", None),
        "display_time_end": getattr(raw, "display_time_end", None),
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

    # 字段名易被模型理解成「视频未揭晓」；若 decision=accept 且已给 tasks，
    # 以 decision/tasks 为准，不因 has_unresolved_target=false 整片强制拒识。
    force_reject = not bool(draft.has_unresolved_target)
    accept_with_tasks = (
        draft.decision == AuditDecision.accept and bool(getattr(draft, "tasks", None))
    )
    if force_reject and accept_with_tasks:
        logger.warning(
            "ignore has_unresolved_target=false for %s: decision=accept with %d tasks",
            video_id,
            len(draft.tasks),
        )
        force_reject = False
        draft.has_unresolved_target = True

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
            elif not resume_tasks and task_checkpoint.is_file():
                _unlink_quiet(str(task_checkpoint))
            t0, t1 = _normalize_task_window(
                raw,
                duration=duration,
                transcript=transcript,
                boundary_tolerance=boundary_tolerance,
            )
            answer_status = getattr(raw, "answer_status", AnswerStatus.resolved)
            if not isinstance(answer_status, AnswerStatus):
                answer_status = AnswerStatus(str(answer_status))
            final_location = str(getattr(raw, "final_location_text", "") or "").strip()
            hard_cap = max_kf_cfg
            status = TaskStatus.accepted
            status_reason = ""
            stamps: list[float] = []
            paths: list[str] = []
            assessments: list[KeyframeAssessment] = []
            multi = False
            visual_brief = ""
            process_intervals: list[ProcessInterval] = []

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
                        visual_brief,
                        process_intervals,
                    ) = _materialize_task_images(
                        video_path=video_path,
                        video_id=video_id,
                        task_id=task_id,
                        raw=raw,
                        t0=t0,
                        t1=t1,
                        hard_cap=hard_cap,
                        transcript=transcript,
                        task_dir=task_dir,
                        resume_tasks=resume_tasks,
                    )
                    if quality_reason:
                        status = TaskStatus.needs_review
                        status_reason = quality_reason
                except Exception as exc:
                    logger.exception("task %s materialize failed", task_id)
                    status = TaskStatus.needs_review
                    status_reason = f"关键帧处理失败：{exc}"

            # multi_target / expected_image_count 记录实选结果，不作预定配额
            multi = len(paths) > 1
            # rejected（答案歧义/无解）仍可不写选图明细；其余均写评价 note
            selection_note = ""
            if status != TaskStatus.rejected:
                selection_note = compose_image_selection_note(
                    status=status,
                    status_reason=status_reason,
                    assessments=assessments,
                )
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
                visual_evidence_brief=visual_brief,
                process_intervals=process_intervals,
                status=status,
                status_reason=status_reason,
                answer_status=answer_status,
                final_location_text=final_location,
                expected_image_count=max(1, len(paths)) if paths else 1,
                frame_assessments=assessments,
                image_selection_note=selection_note,
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
