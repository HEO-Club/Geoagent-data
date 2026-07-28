"""Agent1 证据路由：语义角色、视频事实闭包、工作范围与候选更新（内部契约）。

不改公共 Trajectory / NormalizedStep Schema；契约经 thought_draft 嵌入传递。
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field

# thought_draft 嵌入标记（stage3 → stage4/5，不改 NormalizedStep Schema）
EVIDENCE_INTENT_PREFIX = "<<<EVIDENCE_INTENT:"
EVIDENCE_INTENT_SUFFIX = ">>>"
VIDEO_CONTEXT_PREFIX = "<<<VIDEO_CONTEXT:"
VIDEO_CONTEXT_SUFFIX = ">>>"


class SemanticRoute(str, Enum):
    """答案前 Move 的训练角色路由。"""

    COARSE = "coarse"
    FINE = "fine"
    NON_TRAINING = "non_training"


class ContentType(str, Enum):
    """帧/内容区类型。"""

    PRIMARY_SCENE = "primary_scene"
    SUPPORTING_GEO_VISUAL = "supporting_geo_visual"
    INTERFACE_ONLY = "interface_only"
    UNKNOWN = "unknown"


class InfoSourceKind(str, Enum):
    """信息来源类别。"""

    RAW_GIVEN_CLUE = "raw_given_clue"
    WORKING_SCOPE = "working_scope"
    GIVEN_CLUE = "given_clue"  # 兼容旧名
    CANDIDATE_HYPOTHESIS = "candidate_hypothesis"
    VISUAL_FACT = "visual_fact"
    SPATIAL_RELATION = "spatial_relation"
    DISPLAY_CONTEXT = "display_context"


class RangeUpdateKind(str, Enum):
    """候选范围更新类型。"""

    NARROW = "narrow"
    EXPAND = "expand"
    SHIFT = "shift"
    CORRECT = "correct"
    EXCLUDE = "exclude"


class CoarseStepKind(str, Enum):
    """Agent1 步类型：观察积累 vs 候选更新。"""

    OBSERVE = "observe"
    UPDATE = "update"


class RawClueRole(str, Enum):
    """外部给定线索语义角色（抽取内部契约）。"""

    PHOTO_LOCATION_CONSTRAINT = "photo_location_constraint"
    PERSON_OR_SOCIAL_ATTRIBUTE = "person_or_social_attribute"
    OTHER_NON_LOCATION = "other_non_location"


class ScopeBoundKind(str, Enum):
    """工作范围边界强度：软先验 vs 硬行政区界。"""

    INSIDE = "inside"  # 拍摄地在 X 内 / 未出 X
    NEAR = "near"  # 拍摄地在 X 附近（含籍贯+离家不远的软先验）
    UNSPECIFIED = "unspecified"


class SubjectScope(str, Enum):
    """事实结论作用的空间/语义对象（防跨对象错误互斥）。"""

    CAMERA_POSITION = "camera_position"
    SCENE_REGION = "scene_region"
    LOCATION_CANDIDATE = "location_candidate"
    UNKNOWN = "unknown"


class EvidenceIntent(BaseModel):
    """单步原子证据意图（嵌入 thought_draft，供 stage4/5 消费）。"""

    target_object: str = Field(description="由本视频来源声明指定的观察对象")
    content_type: ContentType = ContentType.UNKNOWN
    target_features: list[str] = Field(default_factory=list)
    expected_spatial_relation: Optional[str] = None
    suggested_bbox: list[float] = Field(
        default_factory=lambda: [0.25, 0.25, 0.5, 0.5]
    )
    source_kind: InfoSourceKind = InfoSourceKind.VISUAL_FACT
    screen_action_untrusted: bool = False
    route: SemanticRoute = SemanticRoute.COARSE
    # 兼容字段：由当前视频事实动态抽取的概念，不是全局词表
    source_concepts: list[str] = Field(default_factory=list)
    video_fact_ids: list[str] = Field(default_factory=list)
    source_claims: list[str] = Field(default_factory=list)
    step_kind: CoarseStepKind = CoarseStepKind.OBSERVE
    subject_scope: SubjectScope = SubjectScope.UNKNOWN
    spatial_anchor: Optional[str] = None


class ContentRegion(BaseModel):
    """内容区识别结果。"""

    content_type: ContentType = ContentType.UNKNOWN
    content_bbox: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 1.0, 1.0]
    )
    target_visible: bool = True
    notes: str = ""


class RawGivenClue(BaseModel):
    """问题设置阶段外部给定的原始软线索（逐字语义，非规范化）。"""

    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    quote: str = ""
    clue_role: RawClueRole = RawClueRole.OTHER_NON_LOCATION


class WorkingScope(BaseModel):
    """可直接使用的工作范围（展示短语须与原文边界强度一致）。"""

    region: str  # 如「河南许昌附近」或「河南省内」；禁止误写成过强边界
    bound_kind: ScopeBoundKind = ScopeBoundKind.UNSPECIFIED
    raw_clue_texts: list[str] = Field(default_factory=list)
    rationale: str = ""


class VideoFactClaim(BaseModel):
    """带时间引用的原子视频事实（仅接受旁白/可见线索明确提出者）。"""

    fact_id: str
    start_time: float
    end_time: float
    quote: str
    tokens: list[str] = Field(default_factory=list)
    relation: Optional[str] = None
    kind: str = "observe"  # observe|correct|exclude|candidate|stall
    consumed_by_update: bool = False
    source_move_index: Optional[int] = None
    supporting_move_indices: list[int] = Field(default_factory=list)
    excluded_candidates: list[str] = Field(default_factory=list)
    subject_scope: SubjectScope = SubjectScope.UNKNOWN
    spatial_anchor: Optional[str] = None
    corrected_from: Optional[str] = None
    corrected_to: Optional[str] = None


class ExtractedRawClue(BaseModel):
    """LLM 从单个视频问题设置段抽出的原始线索。"""

    move_index: int = Field(ge=0)
    text: str
    clue_role: RawClueRole = RawClueRole.OTHER_NON_LOCATION


class ExtractedWorkingScope(BaseModel):
    """由该视频拍摄地硬边界或软先验规范化出的工作范围。"""

    region: str
    supporting_move_indices: list[int] = Field(min_length=1)
    rationale: str = ""
    bound_kind: ScopeBoundKind = ScopeBoundKind.UNSPECIFIED


class ExtractedVideoFact(BaseModel):
    """逐视频抽取的原子事实；概念完全来自该视频，不依赖全局词表。"""

    move_index: int = Field(ge=0)
    claim: str
    concepts: list[str] = Field(default_factory=list)
    relation: Optional[str] = None
    kind: str = "observe"
    supporting_move_indices: list[int] = Field(default_factory=list)
    excluded_candidates: list[str] = Field(default_factory=list)
    proposed_candidates: list[str] = Field(default_factory=list)
    subject_scope: SubjectScope = SubjectScope.UNKNOWN
    spatial_anchor: Optional[str] = None
    corrected_from: Optional[str] = None
    corrected_to: Optional[str] = None


class VideoContextExtraction(BaseModel):
    """stage3 的逐视频动态来源抽取结果。"""

    raw_clues: list[ExtractedRawClue] = Field(default_factory=list)
    working_scope: Optional[ExtractedWorkingScope] = None
    facts: list[ExtractedVideoFact] = Field(default_factory=list)


class VisualFactEntry(BaseModel):
    """账本中的视觉事实条目。"""

    step_index: int
    summary: str
    source_tool: str = ""
    video_fact_ids: list[str] = Field(default_factory=list)


class SpatialRelationEntry(BaseModel):
    """账本中的空间关系条目。"""

    description: str
    supporting_fact_steps: list[int] = Field(default_factory=list)
    subject_scope: SubjectScope = SubjectScope.UNKNOWN
    spatial_anchor: Optional[str] = None


class CandidateUpdateEntry(BaseModel):
    """候选范围更新记录。"""

    kind: RangeUpdateKind
    old_candidates: list[str] = Field(default_factory=list)
    new_candidates: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    evidence_steps: list[int] = Field(default_factory=list)
    video_fact_ids: list[str] = Field(default_factory=list)
    exclusion_reason: str = ""
    subject_scope: SubjectScope = SubjectScope.UNKNOWN
    spatial_anchor: Optional[str] = None


class CandidateState(BaseModel):
    """按作用域分区的候选状态（内部；可重放）。"""

    working_scope: Optional[str] = None
    active_by_scope: dict[str, list[str]] = Field(default_factory=dict)
    excluded_by_scope: dict[str, list[str]] = Field(default_factory=dict)
    emitted_terms: list[str] = Field(default_factory=list)
    consumed_fact_ids: list[str] = Field(default_factory=list)
    updates: list[CandidateUpdateEntry] = Field(default_factory=list)


class CoarseEvidenceLedger(BaseModel):
    """Agent1 内部证据账本（不进入公共 Trajectory Schema）。"""

    raw_given_clues: list[str] = Field(default_factory=list)
    working_scope: Optional[str] = None
    given_clues: list[str] = Field(default_factory=list)
    candidate_hypotheses: list[str] = Field(default_factory=list)
    visual_facts: list[VisualFactEntry] = Field(default_factory=list)
    spatial_relations: list[SpatialRelationEntry] = Field(default_factory=list)
    candidate_updates: list[CandidateUpdateEntry] = Field(default_factory=list)
    collapsed_evidence: list[str] = Field(default_factory=list)
    unusable_ui_steps: list[int] = Field(default_factory=list)
    source_concepts: list[str] = Field(default_factory=list)
    video_fact_claims: dict[str, str] = Field(default_factory=dict)


class VideoChainContext(BaseModel):
    """stage3 → stage5 传递的视频链上下文（嵌入首步 thought_draft）。"""

    raw_given_clues: list[RawGivenClue] = Field(default_factory=list)
    working_scope: Optional[WorkingScope] = None
    video_facts: list[VideoFactClaim] = Field(default_factory=list)
    candidate_hypotheses: list[str] = Field(default_factory=list)


class MoveRouteDecision(BaseModel):
    """单 Move 路由判定（可 LLM 或启发式）。"""

    route: SemanticRoute
    reason: str = ""
    intent: Optional[EvidenceIntent] = None


_UI_RE = re.compile(
    r"聊天|消息|置顶|弹幕|播放器|进度条|标题卡|片头|点赞|评论区|"
    r"微信|界面|ui\b|字幕条|难度\s*\d|粉丝",
    re.I,
)
_GEO_NARR_RE = re.compile(
    r"高地|平原|山脉?|桥|河|江|湖|海岸|峡谷|丘陵|盆地|地形|地貌|"
    r"地理|空间关系|位置关系|卫星|地图|排除|候选|附近|收窄|"
    r"俯视|远景|背景|河岸|纠正|误认",
    re.I,
)
_FINE_RE = re.compile(
    r"公园|扶手|建筑|楼阁|亭|坐标|经纬|打卡|景点|门牌|店招|"
    r"submit|精确|精定位|匹配花纹",
    re.I,
)
_COARSE_RE = re.compile(
    r"高地|平原|桥|宽河|地貌|地形|空间关系|排除|候选|"
    r"应该不是|重新分析|地理环境|自然区域|山脉|俯视|河岸|纠正|收窄范围",
    re.I,
)
_MAP_RE = re.compile(r"卫星|地图|遥感|排查|对比", re.I)
_PHOTO_RE = re.compile(r"照片|画面|主场景", re.I)
_SETUP_RE = re.compile(
    r"沟通|求助|网友|评论|弹幕|粉丝|聊天|家乡|籍贯|离家|不远|应该不会太远",
    re.I,
)
_SOFT_NEAR_SCOPE_RE = re.compile(
    r"附近|周围|周边|离家|不远|应该不会太远|不会太远",
    re.I,
)
_HARD_INSIDE_SCOPE_RE = re.compile(
    r"未出|之内|以内|(?<!附)内|就在.{0,12}(省|市|县|区)(里|内)?",
    re.I,
)
_UNEXITED_SCOPE_RE = re.compile(r"未出\s*([^\s，。；、,.]+)")
_BOUND_SUFFIX_RE = re.compile(r"(附近|之内|以内|内)$")
_META_SETUP_NARR_RE = re.compile(
    r"花了|半年|求助|粉丝向我|想让我帮忙|找一下|忘记照片|故地重游|"
    r"勾起|回忆|沟通|网友|评论区",
    re.I,
)
_STALL_RE = re.compile(r"置顶|想不到|先放着|以后再|等有空|暂时", re.I)
_CORRECT_RE = re.compile(
    r"误认|其实是|应该是|重新分析|纠正|视觉误差|正确的(?:地理|关系|判断)",
    re.I,
)
_EXCLUDE_RE = re.compile(
    r"排除|应该不是|并没有|不符|没发现|根本就没|不是在",
    re.I,
)
_EXCLUDED_PHRASE_RE = re.compile(
    r"(?:排除|应该不是|不是|不符合|不符于)\s*([^，。；,;]{1,40})",
    re.I,
)
_CANDIDATE_RE = re.compile(r"很可能|应该位于|候选|符合.{0,6}设想", re.I)
# 方位/视角短语：用于空间作用域，不是地貌词表
_CAMERA_ANCHOR_RE = re.compile(
    r"拍摄地|拍摄点|拍摄位置|此处|这里|下方|俯视|近处|近景|脚下|身下|本处",
    re.I,
)
_SCENE_ANCHOR_RE = re.compile(
    r"对面|对岸|远处|远景|背景|后方|前方|左侧|右侧|画面中|斜向上|边缘",
    re.I,
)
_LOCATION_ANCHOR_RE = re.compile(
    r"位于|应该位于|很可能在|候选地点|工作范围|行政区",
    re.I,
)


def infer_subject_scope(
    *,
    claim: str,
    kind: str,
    spatial_anchor: Optional[str] = None,
    proposed_candidates: Optional[list[str]] = None,
) -> SubjectScope:
    """由原文短语启发式推断作用域；不做地名/地貌黑白名单。"""
    text = f"{claim} {spatial_anchor or ''}"
    if kind == "candidate" or (proposed_candidates and _LOCATION_ANCHOR_RE.search(text)):
        return SubjectScope.LOCATION_CANDIDATE
    if _CAMERA_ANCHOR_RE.search(text):
        return SubjectScope.CAMERA_POSITION
    if _SCENE_ANCHOR_RE.search(text):
        return SubjectScope.SCENE_REGION
    if kind in {"correct", "exclude"} and _SCENE_ANCHOR_RE.search(claim):
        return SubjectScope.SCENE_REGION
    if kind == "candidate":
        return SubjectScope.LOCATION_CANDIDATE
    return SubjectScope.UNKNOWN


def extract_spatial_anchor(text: str) -> Optional[str]:
    """从原文提取首个方位/视角短语作为 spatial_anchor。"""
    blob = text or ""
    for pattern in (_CAMERA_ANCHOR_RE, _SCENE_ANCHOR_RE, _LOCATION_ANCHOR_RE):
        match = pattern.search(blob)
        if match is not None:
            return match.group(0)
    return None


def is_meta_setup_narration(text: str) -> bool:
    """是否为问题设置/耗时元叙事（非地理观察事实）。

    若同时含明确地理观察/纠正/排除信号，则不算纯元叙事。
    """
    blob = (text or "").strip()
    if not blob:
        return False
    if (
        _GEO_NARR_RE.search(blob)
        or _COARSE_RE.search(blob)
        or _CORRECT_RE.search(blob)
        or _EXCLUDE_RE.search(blob)
        or _CANDIDATE_RE.search(blob)
    ):
        return False
    return bool(_META_SETUP_NARR_RE.search(blob) or _SETUP_RE.search(blob))


def move_has_geo_fact_signal(
    *,
    narration: str,
    visible_clues: Optional[list[str]] = None,
) -> bool:
    """Move 是否含应进入来源契约的地理/空间信号（非地名词表）。"""
    narr = (narration or "").strip()
    clues = _dedupe_nonempty(list(visible_clues or []))
    blob = f"{narr} {' '.join(clues)}"
    if not blob.strip():
        return False
    if is_meta_setup_narration(narr) and not (
        _GEO_NARR_RE.search(blob) or _COARSE_RE.search(blob)
    ):
        return False
    if (
        _GEO_NARR_RE.search(blob)
        or _COARSE_RE.search(blob)
        or _CORRECT_RE.search(narr)
        or _EXCLUDE_RE.search(narr)
        or _CANDIDATE_RE.search(narr)
    ):
        return True
    # 方位/视角 + 可见线索：如「下方建筑屋顶」
    if (_CAMERA_ANCHOR_RE.search(narr) or _SCENE_ANCHOR_RE.search(narr)) and (
        clues or len(narr) >= 8
    ):
        return True
    return False


def gap_fill_missing_geo_facts(
    moves: list[Any],
    extraction: VideoContextExtraction,
) -> VideoContextExtraction:
    """对 LLM 漏抽的地理 Move 做保守补全；不发明原文没有的 claim。"""
    covered = {item.move_index for item in extraction.facts}
    extras: list[ExtractedVideoFact] = []
    for index, move in enumerate(moves):
        if index in covered:
            continue
        narration = (getattr(move, "narration", None) or "").strip()
        clues = list(getattr(move, "visible_clues", None) or [])
        if not move_has_geo_fact_signal(narration=narration, visible_clues=clues):
            continue
        if is_meta_setup_narration(narration):
            continue
        kind = "observe"
        if _CORRECT_RE.search(narration):
            kind = "correct"
        elif _EXCLUDE_RE.search(narration):
            kind = "exclude"
        elif _CANDIDATE_RE.search(narration):
            kind = "candidate"
        non_ui = [c for c in _dedupe_nonempty(clues) if not _UI_RE.search(c)]
        anchor = extract_spatial_anchor(narration)
        scope = infer_subject_scope(
            claim=narration,
            kind=kind,
            spatial_anchor=anchor,
        )
        extras.append(
            ExtractedVideoFact(
                move_index=index,
                claim=narration,
                concepts=non_ui[:8],
                kind=kind,
                subject_scope=scope,
                spatial_anchor=anchor,
                excluded_candidates=(
                    [m.group(1).strip()]
                    if (m := _EXCLUDED_PHRASE_RE.search(narration)) is not None
                    else []
                ),
            )
        )
    if not extras:
        return extraction
    return extraction.model_copy(
        update={"facts": list(extraction.facts) + extras}
    )


def drop_meta_setup_facts(
    extraction: VideoContextExtraction,
) -> VideoContextExtraction:
    """去掉纯元叙事 observe，避免蒸馏链被开场耗时话术带偏。"""
    kept: list[ExtractedVideoFact] = []
    for item in extraction.facts:
        if item.kind == "observe" and is_meta_setup_narration(item.claim):
            continue
        kept.append(item)
    if len(kept) == len(extraction.facts):
        return extraction
    return extraction.model_copy(update={"facts": kept})


def embed_evidence_intent(draft: str, intent: EvidenceIntent) -> str:
    """将 EvidenceIntent 嵌入 thought_draft。"""
    payload = intent.model_dump_json()
    body = (draft or "").strip()
    return f"{EVIDENCE_INTENT_PREFIX}{payload}{EVIDENCE_INTENT_SUFFIX}\n{body}"


def parse_evidence_intent(thought_draft: str) -> Optional[EvidenceIntent]:
    """从 thought_draft 解析 EvidenceIntent；失败返回 None。"""
    text = thought_draft or ""
    start = text.find(EVIDENCE_INTENT_PREFIX)
    if start < 0:
        return None
    start += len(EVIDENCE_INTENT_PREFIX)
    end = text.find(EVIDENCE_INTENT_SUFFIX, start)
    if end < 0:
        return None
    raw = text[start:end].strip()
    try:
        return EvidenceIntent.model_validate_json(raw)
    except Exception:  # noqa: BLE001
        try:
            return EvidenceIntent.model_validate(json.loads(raw))
        except Exception:  # noqa: BLE001
            return None


def strip_evidence_intent(thought_draft: str) -> str:
    """去掉 EvidenceIntent 与 VideoContext 嵌入，返回可读草稿。"""
    text = thought_draft or ""
    for prefix, suffix in (
        (EVIDENCE_INTENT_PREFIX, EVIDENCE_INTENT_SUFFIX),
        (VIDEO_CONTEXT_PREFIX, VIDEO_CONTEXT_SUFFIX),
    ):
        start = text.find(prefix)
        if start < 0:
            continue
        end = text.find(suffix, start)
        if end < 0:
            continue
        end += len(suffix)
        text = (text[:start] + text[end:]).lstrip("\n")
    return text.strip()


def embed_video_context(draft: str, ctx: VideoChainContext) -> str:
    """将 VideoChainContext 嵌入 thought_draft（通常挂在首步）。"""
    payload = ctx.model_dump_json()
    body = (draft or "").strip()
    return f"{VIDEO_CONTEXT_PREFIX}{payload}{VIDEO_CONTEXT_SUFFIX}\n{body}"


def parse_video_context(thought_draft: str) -> Optional[VideoChainContext]:
    """从 thought_draft 解析 VideoChainContext。"""
    text = thought_draft or ""
    start = text.find(VIDEO_CONTEXT_PREFIX)
    if start < 0:
        return None
    start += len(VIDEO_CONTEXT_PREFIX)
    end = text.find(VIDEO_CONTEXT_SUFFIX, start)
    if end < 0:
        return None
    raw = text[start:end].strip()
    try:
        return VideoChainContext.model_validate_json(raw)
    except Exception:  # noqa: BLE001
        try:
            return VideoChainContext.model_validate(json.loads(raw))
        except Exception:  # noqa: BLE001
            return None


def context_from_extraction(
    moves: list[Any],
    extraction: VideoContextExtraction,
) -> VideoChainContext:
    """把逐视频结构化抽取结果绑定回原 Move 时间窗。

    所有 quote 都取原 Move，模型只能选择 move_index，不能伪造来源文本。
    working_scope 须有合法支撑：硬边界仅拍摄地约束；软先验可为拍摄地附近
    或籍贯+软距离组合，且展示短语不得过强。
    """
    raw: list[RawGivenClue] = []
    photo_constraint_by_move: dict[int, list[ExtractedRawClue]] = {}
    person_attr_by_move: dict[int, list[ExtractedRawClue]] = {}
    for item in extraction.raw_clues:
        if item.move_index >= len(moves):
            continue
        move = moves[item.move_index]
        quote = (getattr(move, "narration", None) or "").strip()
        role = item.clue_role
        raw.append(
            RawGivenClue(
                text=item.text.strip(),
                start_time=float(getattr(move, "start_time", 0.0) or 0.0),
                end_time=float(getattr(move, "end_time", 0.0) or 0.0),
                quote=quote[:200],
                clue_role=role,
            )
        )
        if role is RawClueRole.PHOTO_LOCATION_CONSTRAINT:
            photo_constraint_by_move.setdefault(item.move_index, []).append(item)
        elif role is RawClueRole.PERSON_OR_SOCIAL_ATTRIBUTE:
            person_attr_by_move.setdefault(item.move_index, []).append(item)

    scope: Optional[WorkingScope] = None
    if extraction.working_scope is not None:
        region = extraction.working_scope.region.strip()
        requested = [
            i
            for i in extraction.working_scope.supporting_move_indices
            if 0 <= i < len(moves)
        ]
        photo_ok = all(i in photo_constraint_by_move for i in requested)
        soft_ok = all(
            i in photo_constraint_by_move or i in person_attr_by_move
            for i in requested
        )
        support_texts = [
            clue.text.strip()
            for idx in requested
            for clue in (
                photo_constraint_by_move.get(idx, [])
                + person_attr_by_move.get(idx, [])
            )
            if clue.text.strip()
        ]
        # 先规范化边界强度，再按强度校验支撑角色
        phrase, bound_kind = normalize_working_scope_phrase(
            region,
            clue_texts=support_texts,
            bound_kind=extraction.working_scope.bound_kind,
        )
        support_allowed = photo_ok or (
            bound_kind is ScopeBoundKind.NEAR and soft_ok and bool(requested)
        )
        if phrase and requested and support_allowed:
            # 硬边界仍要求全部支撑为拍摄地约束
            if bound_kind is ScopeBoundKind.INSIDE and not photo_ok:
                scope = None
            else:
                scope = WorkingScope(
                    region=phrase,
                    bound_kind=bound_kind,
                    raw_clue_texts=list(dict.fromkeys(support_texts)),
                    rationale=extraction.working_scope.rationale.strip(),
                )

    facts: list[VideoFactClaim] = []
    candidates: list[str] = []
    valid_kinds = {"observe", "correct", "exclude", "candidate", "stall"}
    for seq, item in enumerate(extraction.facts):
        if item.move_index >= len(moves):
            continue
        move = moves[item.move_index]
        narration = (getattr(move, "narration", None) or "").strip()
        visible_clues = list(getattr(move, "visible_clues", None) or [])
        # 仅旁白作为可引用声明正文；visible_clues 只补充非 UI 概念，避免聊天/置顶污染 Obs
        non_ui_clues = [
            c
            for c in _dedupe_nonempty(visible_clues)
            if not _UI_RE.search(c)
        ]
        claim_blob = narration + " " + " ".join(non_ui_clues)
        # 排除对象可出现在旁白或任意 visible_clues（含被过滤的 UI 词，因旁白可能在排除界面噪声）
        exclusion_blob = narration + " " + " ".join(_dedupe_nonempty(visible_clues))
        if (
            not item.claim.strip()
            or re.sub(r"\s+", "", item.claim)
            not in re.sub(r"\s+", "", claim_blob)
        ):
            continue
        quote = narration or "；".join(non_ui_clues)
        source_concepts = [
            concept
            for concept in _dedupe_nonempty(item.concepts)
            if re.sub(r"\s+", "", concept) in re.sub(r"\s+", "", claim_blob)
        ]
        kind = item.kind if item.kind in valid_kinds else "observe"
        anchor = (item.spatial_anchor or "").strip() or extract_spatial_anchor(
            item.claim + " " + narration
        )
        if anchor and re.sub(r"\s+", "", anchor) not in re.sub(
            r"\s+", "", claim_blob
        ):
            anchor = extract_spatial_anchor(claim_blob)
        subject_scope = item.subject_scope
        if subject_scope is SubjectScope.UNKNOWN:
            subject_scope = infer_subject_scope(
                claim=item.claim,
                kind=kind,
                spatial_anchor=anchor,
                proposed_candidates=item.proposed_candidates,
            )
        corrected_from = (item.corrected_from or "").strip() or None
        corrected_to = (item.corrected_to or "").strip() or None
        if corrected_from and re.sub(r"\s+", "", corrected_from) not in re.sub(
            r"\s+", "", claim_blob
        ):
            corrected_from = None
        if corrected_to and re.sub(r"\s+", "", corrected_to) not in re.sub(
            r"\s+", "", claim_blob
        ):
            corrected_to = None
        facts.append(
            VideoFactClaim(
                fact_id=f"vf{item.move_index}_{seq}_{kind}",
                start_time=float(getattr(move, "start_time", 0.0) or 0.0),
                end_time=float(getattr(move, "end_time", 0.0) or 0.0),
                quote=quote[:200],
                tokens=source_concepts,
                relation=(
                    item.relation.strip()
                    if item.relation
                    and re.sub(r"\s+", "", item.relation)
                    in re.sub(r"\s+", "", claim_blob)
                    else None
                ),
                kind=kind,
                consumed_by_update=kind in {"correct", "exclude", "candidate"},
                source_move_index=item.move_index,
                supporting_move_indices=[
                    i for i in item.supporting_move_indices if 0 <= i < len(moves)
                ],
                excluded_candidates=[
                    text
                    for text in _dedupe_nonempty(item.excluded_candidates)
                    if text in exclusion_blob
                ],
                subject_scope=subject_scope,
                spatial_anchor=anchor,
                corrected_from=corrected_from,
                corrected_to=corrected_to,
            )
        )
        # 仅 location_candidate 作用域的提议进入拍摄点/地点候选池
        if subject_scope is SubjectScope.LOCATION_CANDIDATE or (
            kind == "candidate"
            and subject_scope
            in {SubjectScope.LOCATION_CANDIDATE, SubjectScope.CAMERA_POSITION}
        ):
            for candidate in item.proposed_candidates:
                text = candidate.strip()
                if text and text in claim_blob and text not in candidates:
                    candidates.append(text)
        elif kind == "candidate" and subject_scope is SubjectScope.UNKNOWN:
            for candidate in item.proposed_candidates:
                text = candidate.strip()
                if text and text in claim_blob and text not in candidates:
                    candidates.append(text)
    return VideoChainContext(
        raw_given_clues=raw,
        working_scope=scope,
        video_facts=facts,
        candidate_hypotheses=candidates,
    )


def fallback_video_context(moves: list[Any]) -> VideoChainContext:
    """无 LLM 时的保守通用回退。

    不猜工作范围、地名或地貌类别；只保留 Move 已提供的 visible_clues
    和原始旁白作为来源声明。没有 visible_clues 的 Move 不构造事实。
    """
    facts: list[VideoFactClaim] = []
    prior_fact_indices: list[int] = []
    for index, move in enumerate(moves):
        clues = _dedupe_nonempty(list(getattr(move, "visible_clues", None) or []))
        if not clues:
            continue
        quote = (getattr(move, "narration", None) or "").strip()
        kind = "observe"
        if _CORRECT_RE.search(quote):
            kind = "correct"
        elif _EXCLUDE_RE.search(quote):
            kind = "exclude"
        elif _CANDIDATE_RE.search(quote):
            kind = "candidate"
        excluded_match = _EXCLUDED_PHRASE_RE.search(quote)
        excluded = (
            [excluded_match.group(1).strip()] if excluded_match is not None else []
        )
        anchor = extract_spatial_anchor(quote + " " + " ".join(clues))
        subject_scope = infer_subject_scope(
            claim=quote or "；".join(clues),
            kind=kind,
            spatial_anchor=anchor,
        )
        facts.append(
            VideoFactClaim(
                fact_id=f"vf{index}_fallback",
                start_time=float(getattr(move, "start_time", 0.0) or 0.0),
                end_time=float(getattr(move, "end_time", 0.0) or 0.0),
                quote=(quote or "；".join(clues))[:200],
                tokens=clues,
                kind=kind,
                source_move_index=index,
                supporting_move_indices=list(prior_fact_indices[-2:]),
                excluded_candidates=excluded,
                subject_scope=subject_scope,
                spatial_anchor=anchor,
            )
        )
        prior_fact_indices.append(index)
    return VideoChainContext(video_facts=facts)


def _dedupe_nonempty(items: list[str]) -> list[str]:
    """保序去重。"""
    out: list[str] = []
    for item in items:
        text = item.strip()
        if text and text not in out:
            out.append(text)
    return out


def filter_geo_reasoning_moves(
    routed_moves: list[Any],
    pre_answer: list[Any],
    video_context: VideoChainContext,
) -> list[Any]:
    """保留完整地理推理链（试错+成功），剔除 stall / 无地理训练价值段。

    与「最短成功链」不同：不因未被 supporting 引用而删除试错 exclude/candidate。
    若过滤后为空，回退保留全部非 stall 事实源 Move，禁止空 COARSE 链。
    """
    if not routed_moves:
        return []

    def _move_key(m: Any) -> tuple[float, float, str, str]:
        return (
            float(getattr(m, "start_time", 0.0)),
            float(getattr(m, "end_time", 0.0)),
            (getattr(m, "narration", None) or "").strip(),
            (getattr(m, "screen_action", None) or "").strip(),
        )

    index_by_key: dict[tuple[float, float, str, str], int] = {}
    for i, m in enumerate(pre_answer):
        index_by_key[_move_key(m)] = i

    stall_only: set[int] = set()
    geo_sources: set[int] = set()
    for fact in video_context.video_facts:
        if fact.source_move_index is None:
            continue
        if fact.kind == "stall":
            stall_only.add(fact.source_move_index)
        elif fact.kind in {"observe", "correct", "exclude", "candidate"}:
            geo_sources.add(fact.source_move_index)
    # 同时有 geo 与 stall 标注时，以 geo 为准
    stall_only -= geo_sources

    from pipeline.stage2_moves import is_non_trainable_move

    kept: list[Any] = []
    for move in routed_moves:
        idx = index_by_key.get(_move_key(move))
        if idx is not None and idx in stall_only:
            continue
        if is_non_trainable_move(
            getattr(move, "narration", "") or "",
            getattr(move, "screen_action", None),
            list(getattr(move, "visible_clues", None) or []),
        ):
            continue
        kept.append(move)

    if kept:
        return kept

    # 回退：全部非 stall 事实源，避免空链
    fallback_idx = set(geo_sources)
    if not fallback_idx:
        for fact in video_context.video_facts:
            if fact.kind == "stall" or fact.source_move_index is None:
                continue
            fallback_idx.add(fact.source_move_index)
    return [pre_answer[i] for i in sorted(fallback_idx) if 0 <= i < len(pre_answer)]


def distill_shortest_success_moves(
    moves: list[Any],
    video_context: VideoChainContext,
) -> list[Any]:
    """兼容旧名：现委托完整地理链过滤（不再删除试错支线）。"""
    return filter_geo_reasoning_moves(moves, moves, video_context)


def source_concepts_from_facts(
    facts: list[VideoFactClaim],
    *,
    working_scope: Optional[WorkingScope] = None,
    raw_clues: Optional[list[RawGivenClue]] = None,
) -> list[str]:
    """合并该视频动态抽取的来源概念。"""
    concepts: list[str] = []
    for f in facts:
        if f.kind == "stall":
            continue
        for t in f.tokens:
            if t and t not in concepts:
                concepts.append(t)
        if f.relation and f.relation not in concepts:
            concepts.append(f.relation)
    if working_scope and working_scope.region:
        if working_scope.region not in concepts:
            concepts.append(working_scope.region)
    for r in raw_clues or []:
        if r.text and r.text not in concepts:
            concepts.append(r.text)
    return concepts


def heuristic_route_move(
    narration: str,
    screen_action: Optional[str],
    visible_clues: list[str],
) -> MoveRouteDecision:
    """无 LLM 时的启发式路由（测试与降级路径）。"""
    narr = (narration or "").strip()
    screen = (screen_action or "").strip()
    clues = " ".join(visible_clues or [])
    blob = f"{narr} {screen} {clues}"

    ui_heavy = bool(_UI_RE.search(screen) or _UI_RE.search(clues))
    geo_narr = bool(_GEO_NARR_RE.search(narr))
    fine_hit = bool(_FINE_RE.search(blob))
    coarse_hit = bool(_COARSE_RE.search(narr) or _COARSE_RE.search(clues))

    conflict = geo_narr and ui_heavy and not _MAP_RE.search(screen)

    if fine_hit and not coarse_hit and not conflict:
        intent = build_evidence_intent(
            narr, screen, visible_clues, SemanticRoute.FINE, conflict=False
        )
        return MoveRouteDecision(
            route=SemanticRoute.FINE, reason="精确验证语义", intent=intent
        )

    if coarse_hit or (geo_narr and not fine_hit) or conflict:
        intent = build_evidence_intent(
            narr,
            screen,
            visible_clues,
            SemanticRoute.COARSE,
            conflict=conflict,
        )
        return MoveRouteDecision(
            route=SemanticRoute.COARSE,
            reason="广域地貌/排除或旁白-UI 冲突采信旁白",
            intent=intent,
        )

    if ui_heavy and not geo_narr:
        return MoveRouteDecision(
            route=SemanticRoute.NON_TRAINING,
            reason="纯 UI/社交界面",
            intent=None,
        )

    if not narr and ui_heavy:
        return MoveRouteDecision(
            route=SemanticRoute.NON_TRAINING, reason="无旁白 UI", intent=None
        )

    if _MAP_RE.search(screen) or _PHOTO_RE.search(screen) or geo_narr:
        intent = build_evidence_intent(
            narr, screen, visible_clues, SemanticRoute.COARSE, conflict=False
        )
        return MoveRouteDecision(
            route=SemanticRoute.COARSE, reason="默认地理观察", intent=intent
        )

    return MoveRouteDecision(
        route=SemanticRoute.NON_TRAINING, reason="无地理训练价值", intent=None
    )


def build_evidence_intent(
    narration: str,
    screen_action: Optional[str],
    visible_clues: list[str],
    route: SemanticRoute,
    *,
    conflict: bool,
    source_concepts: Optional[list[str]] = None,
    video_fact_ids: Optional[list[str]] = None,
    step_kind: Optional[CoarseStepKind] = None,
) -> EvidenceIntent:
    """构造 EvidenceIntent；特征只来自本视频动态概念或 visible_clues。"""
    narr = narration or ""
    screen = screen_action or ""
    features = _dedupe_nonempty(
        list(source_concepts)
        if source_concepts is not None
        else list(visible_clues or [])
    )

    text = f"{narr} {screen} {' '.join(visible_clues or [])}"
    if _MAP_RE.search(text):
        content = ContentType.SUPPORTING_GEO_VISUAL
        bbox = _bbox_for_features(features, default=[0.05, 0.1, 0.9, 0.8])
    elif conflict or _PHOTO_RE.search(narr) or "老照片" in text:
        content = ContentType.PRIMARY_SCENE
        bbox = _bbox_for_features(features, default=[0.1, 0.15, 0.8, 0.7])
    elif _UI_RE.search(screen) and not features:
        content = ContentType.INTERFACE_ONLY
        bbox = [0.0, 0.0, 1.0, 1.0]
    else:
        content = ContentType.PRIMARY_SCENE
        bbox = _bbox_for_features(features, default=[0.2, 0.2, 0.6, 0.6])
    target = features[0] if features else "来源声明指定目标"

    relation = None

    kind = step_kind
    if kind is None:
        # COARSE 训练步默认 UPDATE：每步须排除/纠正；纯 observe 仅当无排除语义时
        if _CORRECT_RE.search(narr) or _EXCLUDE_RE.search(narr) or _CANDIDATE_RE.search(
            narr
        ):
            kind = CoarseStepKind.UPDATE
        elif route is SemanticRoute.COARSE and features:
            kind = CoarseStepKind.UPDATE
        else:
            kind = CoarseStepKind.OBSERVE

    fact_kind = "observe"
    if _CORRECT_RE.search(narr):
        fact_kind = "correct"
    elif _EXCLUDE_RE.search(narr):
        fact_kind = "exclude"
    elif _CANDIDATE_RE.search(narr):
        fact_kind = "candidate"
    anchor = extract_spatial_anchor(text)
    subject_scope = infer_subject_scope(
        claim=text,
        kind=fact_kind,
        spatial_anchor=anchor,
    )

    return EvidenceIntent(
        target_object=target,
        content_type=content,
        target_features=features[:6],
        expected_spatial_relation=relation,
        suggested_bbox=bbox,
        source_kind=(
            InfoSourceKind.DISPLAY_CONTEXT
            if conflict
            else InfoSourceKind.VISUAL_FACT
        ),
        screen_action_untrusted=conflict,
        route=route,
        source_concepts=list(source_concepts or features),
        video_fact_ids=list(video_fact_ids or []),
        step_kind=kind,
        subject_scope=subject_scope,
        spatial_anchor=anchor,
    )


def _bbox_for_features(
    features: list[str], *, default: list[float]
) -> list[float]:
    """返回内容类型默认 bbox；不按任何特定地貌词分支。"""
    return list(default)


def combine_bboxes(
    content_bbox: list[float], relative_bbox: list[float]
) -> list[float]:
    """将内容区 bbox 与相对 Action bbox 合成全图归一化 xywh。"""
    cx, cy, cw, ch = _as_xywh(content_bbox)
    rx, ry, rw, rh = _as_xywh(relative_bbox)
    x = cx + rx * cw
    y = cy + ry * ch
    w = rw * cw
    h = rh * ch
    return [
        max(0.0, min(1.0, x)),
        max(0.0, min(1.0, y)),
        max(0.02, min(1.0 - x, w)),
        max(0.02, min(1.0 - y, h)),
    ]


def _as_xywh(bbox: list[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        return 0.0, 0.0, 1.0, 1.0
    a, b, c, d = (float(x) for x in bbox)
    if a + c <= 1.05 and b + d <= 1.05 and c > 0 and d > 0:
        return a, b, c, d
    if c >= a and d >= b:
        return a, b, max(0.02, c - a), max(0.02, d - b)
    return a, b, max(0.02, c), max(0.02, d)


def heuristic_content_region(
    *,
    intent: Optional[EvidenceIntent],
    screen_action: Optional[str] = None,
) -> ContentRegion:
    """无 VLM 时的内容区启发式。"""
    if intent is not None:
        if intent.content_type is ContentType.INTERFACE_ONLY:
            return ContentRegion(
                content_type=ContentType.INTERFACE_ONLY,
                content_bbox=[0.0, 0.0, 1.0, 1.0],
                target_visible=False,
                notes="interface_only",
            )
        if intent.content_type is ContentType.SUPPORTING_GEO_VISUAL:
            return ContentRegion(
                content_type=ContentType.SUPPORTING_GEO_VISUAL,
                content_bbox=[0.05, 0.1, 0.9, 0.8],
                target_visible=True,
                notes="map_canvas",
            )
        return ContentRegion(
            content_type=intent.content_type,
            # 略放宽：过紧裁剪易切掉地貌边缘，导致合成误报 empty
            content_bbox=[0.05, 0.08, 0.9, 0.84],
            target_visible=True,
            notes="primary_scene_canvas",
        )
    screen = screen_action or ""
    if _UI_RE.search(screen) and not _MAP_RE.search(screen):
        return ContentRegion(
            content_type=ContentType.INTERFACE_ONLY,
            target_visible=False,
            notes="ui_screen_action",
        )
    if _MAP_RE.search(screen):
        return ContentRegion(
            content_type=ContentType.SUPPORTING_GEO_VISUAL,
            content_bbox=[0.05, 0.1, 0.9, 0.8],
            target_visible=True,
        )
    return ContentRegion(
        content_type=ContentType.PRIMARY_SCENE,
        content_bbox=[0.05, 0.08, 0.9, 0.84],
        target_visible=True,
    )


def obs_fingerprint(observation: Optional[dict[str, Any]]) -> str:
    """Observation 语义指纹（用于折叠判定）。"""
    if not observation:
        return ""
    return json.dumps(observation, ensure_ascii=False, sort_keys=True, default=str)


def _strip_bound_suffix(region: str) -> str:
    """去掉末尾「附近/内」等边界后缀，便于按强度重拼展示短语。"""
    text = region.strip()
    while True:
        updated = _BOUND_SUFFIX_RE.sub("", text).strip()
        if updated == text:
            return text
        text = updated


def infer_scope_bound_kind(
    region: str,
    clue_texts: Sequence[str] = (),
    *,
    bound_kind: ScopeBoundKind = ScopeBoundKind.UNSPECIFIED,
) -> ScopeBoundKind:
    """从原文与已有标注推断工作范围边界强度。

    软先验（附近/离家不远）在无「未出/拍摄地…内」硬证据时优先，
    避免把误标的「X内」当作硬边界。
    """
    clue_blob = " ".join(clue_texts)
    blob = " ".join([region, clue_blob])
    soft_hit = bool(_SOFT_NEAR_SCOPE_RE.search(blob))
    hard_in_clues = bool(
        _UNEXITED_SCOPE_RE.search(clue_blob)
        or re.search(
            r"(拍摄地|照片地点|照片).{0,16}(之内|以内|(?<!附)内)",
            clue_blob,
        )
        or ("未出" in clue_blob)
    )
    hard_in_region = bool(
        _UNEXITED_SCOPE_RE.search(region)
        or region.endswith("内")
        or ("之内" in region)
        or ("以内" in region)
    )
    if soft_hit and not hard_in_clues:
        return ScopeBoundKind.NEAR
    if hard_in_clues or (hard_in_region and not soft_hit):
        return ScopeBoundKind.INSIDE
    if bound_kind is not ScopeBoundKind.UNSPECIFIED:
        return bound_kind
    if "附近" in region:
        return ScopeBoundKind.NEAR
    if hard_in_region:
        return ScopeBoundKind.INSIDE
    return ScopeBoundKind.UNSPECIFIED


def normalize_working_scope_phrase(
    region: str,
    *,
    clue_texts: Sequence[str] = (),
    bound_kind: ScopeBoundKind = ScopeBoundKind.UNSPECIFIED,
) -> tuple[str, ScopeBoundKind]:
    """将工作范围规范为准确展示短语，禁止把软先验写成「X内」。"""
    raw = (region or "").strip()
    if not raw:
        return "", ScopeBoundKind.UNSPECIFIED
    kind = infer_scope_bound_kind(raw, clue_texts, bound_kind=bound_kind)
    unexited = _UNEXITED_SCOPE_RE.search(" ".join([raw, *clue_texts]))
    core = _strip_bound_suffix(raw)
    if not core and unexited:
        core = unexited.group(1).strip()
    if kind is ScopeBoundKind.NEAR:
        if not core:
            return raw, kind
        phrase = raw if "附近" in raw and not raw.endswith("内") else f"{core}附近"
        # 杜绝「附近内」或残留硬后缀
        phrase = phrase.replace("附近内", "附近")
        if phrase.endswith("内") and "附近" in phrase:
            phrase = _strip_bound_suffix(phrase) + "附近"
        return phrase, ScopeBoundKind.NEAR
    if kind is ScopeBoundKind.INSIDE:
        if "未出" in raw and "内" not in raw:
            # 统一展示为「X内」，便于 user_query 稳定
            place = unexited.group(1).strip() if unexited else core
            return (f"{place}内" if place else raw), ScopeBoundKind.INSIDE
        if raw.endswith("内") or "之内" in raw or "以内" in raw:
            return raw, ScopeBoundKind.INSIDE
        return (f"{core}内" if core else raw), ScopeBoundKind.INSIDE
    return raw, kind


def format_working_scope_user_query(scope: Optional[WorkingScope]) -> str:
    """构造 Agent1 user_query 工作范围句。

    仅写入已规范化的展示短语；不得强行追加「内」。
    """
    base = "请根据图像进行粗定位，缩小到可能的国家/地区。"
    if scope and scope.region:
        phrase, _kind = normalize_working_scope_phrase(
            scope.region,
            clue_texts=scope.raw_clue_texts,
            bound_kind=scope.bound_kind,
        )
        if phrase:
            return f"{base}\n工作范围：{phrase}"
    return base


def scope_partition_key(
    subject_scope: SubjectScope,
    spatial_anchor: Optional[str] = None,
) -> str:
    """按作用域 + 方位短语生成稳定分区键。"""
    anchor = (spatial_anchor or "").strip() or "_"
    return f"{subject_scope.value}::{anchor}"


def same_scope_partition(
    a_scope: SubjectScope,
    a_anchor: Optional[str],
    b_scope: SubjectScope,
    b_anchor: Optional[str],
) -> bool:
    """判断两个事实是否落在同一空间更新域。"""
    return scope_partition_key(a_scope, a_anchor) == scope_partition_key(
        b_scope, b_anchor
    )


def init_candidate_state(
    *,
    working_scope: Optional[str] = None,
    candidate_hypotheses: Optional[list[str]] = None,
) -> CandidateState:
    """初始化按作用域分区的候选状态。"""
    state = CandidateState(working_scope=(working_scope or None))
    region = (working_scope or "").strip()
    if region:
        state.emitted_terms.append(region)
    hyps = [
        text.strip()
        for text in (candidate_hypotheses or [])
        if text and text.strip()
    ]
    if hyps:
        key = scope_partition_key(SubjectScope.LOCATION_CANDIDATE)
        state.active_by_scope[key] = list(dict.fromkeys(hyps))
        for term in state.active_by_scope[key]:
            if term not in state.emitted_terms:
                state.emitted_terms.append(term)
    return state


def apply_candidate_update(
    state: CandidateState,
    update: CandidateUpdateEntry,
) -> CandidateState:
    """在同一作用域分区内应用候选增量；不跨分区互斥撤销。"""
    new_state = state.model_copy(deep=True)
    key = scope_partition_key(update.subject_scope, update.spatial_anchor)
    active = list(new_state.active_by_scope.get(key, []))
    excluded = list(new_state.excluded_by_scope.get(key, []))

    def _emit(term: str) -> None:
        text = term.strip()
        if text and text not in new_state.emitted_terms:
            new_state.emitted_terms.append(text)

    for term in update.new_candidates:
        text = term.strip()
        if text and text not in active:
            active.append(text)
        _emit(text)
    for term in update.old_candidates:
        _emit(term)
    for term in update.excluded:
        text = term.strip()
        if text and text not in excluded:
            excluded.append(text)
        active = [item for item in active if item != text]
        _emit(text)

    if update.kind is RangeUpdateKind.CORRECT:
        for term in update.old_candidates:
            text = term.strip()
            active = [item for item in active if item != text]
            if text and text not in excluded:
                excluded.append(text)

    new_state.active_by_scope[key] = active
    new_state.excluded_by_scope[key] = excluded
    for fact_id in update.video_fact_ids:
        if fact_id and fact_id not in new_state.consumed_fact_ids:
            new_state.consumed_fact_ids.append(fact_id)
    new_state.updates.append(update)
    return new_state


def replay_candidate_state(
    *,
    working_scope: Optional[str],
    candidate_hypotheses: list[str],
    updates: list[CandidateUpdateEntry],
) -> CandidateState:
    """从初始假设与逐步更新重放最终候选状态。"""
    state = init_candidate_state(
        working_scope=working_scope,
        candidate_hypotheses=candidate_hypotheses,
    )
    for update in updates:
        state = apply_candidate_update(state, update)
    return state


def build_candidate_updates_from_facts(
    facts: list[VideoFactClaim],
    *,
    evidence_steps: Optional[dict[str, list[int]]] = None,
) -> list[CandidateUpdateEntry]:
    """由视频事实构造候选状态增量（按作用域分区）。"""
    updates: list[CandidateUpdateEntry] = []
    step_map = evidence_steps or {}
    for fact in facts:
        if fact.kind not in {"correct", "exclude", "candidate"}:
            continue
        kind = {
            "correct": RangeUpdateKind.CORRECT,
            "exclude": RangeUpdateKind.EXCLUDE,
            "candidate": RangeUpdateKind.NARROW,
        }[fact.kind]
        new_candidates: list[str] = []
        old_candidates: list[str] = []
        excluded = list(fact.excluded_candidates)
        if fact.corrected_from:
            old_candidates.append(fact.corrected_from)
            if fact.corrected_from not in excluded:
                excluded.append(fact.corrected_from)
        if fact.corrected_to:
            new_candidates.append(fact.corrected_to)
        if (
            fact.kind == "candidate"
            and fact.subject_scope is SubjectScope.LOCATION_CANDIDATE
        ):
            for tok in fact.tokens:
                if tok and tok not in new_candidates:
                    new_candidates.append(tok)
        updates.append(
            CandidateUpdateEntry(
                kind=kind,
                old_candidates=old_candidates,
                new_candidates=new_candidates,
                excluded=excluded,
                evidence_steps=list(step_map.get(fact.fact_id, [])),
                video_fact_ids=[fact.fact_id],
                exclusion_reason=fact.quote[:120],
                subject_scope=fact.subject_scope,
                spatial_anchor=fact.spatial_anchor,
            )
        )
    return updates


def _term_covered_by_allowed(term: str, allowed: set[str]) -> bool:
    """术语是否已被已建立候选/范围覆盖（双向子串，非词表）。"""
    text = term.strip()
    if not text:
        return True
    if text in allowed:
        return True
    compact = re.sub(r"\s+", "", text)
    for item in allowed:
        other = re.sub(r"\s+", "", item)
        if not other:
            continue
        if compact in other or other in compact:
            return True
    return False


def assert_coarse_output_closed(
    hyp: Any,
    state: CandidateState,
) -> list[str]:
    """LocationHypothesis 只能选用已建立候选状态中的地点。

    对含 CJK 的地点短语做严格闭包；纯拉丁行政区在无对齐翻译时交由
    覆盖检查处理，避免单测罗马化误杀。
    """
    issues: list[str] = []
    allowed: set[str] = set(state.emitted_terms)
    for values in state.active_by_scope.values():
        allowed.update(values)
    if state.working_scope:
        allowed.add(state.working_scope)
    countries = list(getattr(hyp, "possible_countries", None) or [])
    regions = list(getattr(hyp, "possible_regions", None) or [])
    for country in countries:
        text = str(country)
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        if len(chars) >= 5 and not _term_covered_by_allowed(text, allowed):
            issues.append(f"candidate_provenance_gap: country={country!r}")
    for region in regions:
        text = str(region)
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        if len(chars) >= 5 and not _term_covered_by_allowed(text, allowed):
            issues.append(f"candidate_provenance_gap: region={region!r}")
    return issues


def fact_update_fingerprint(
    video_fact_ids: list[str],
    subject_scope: SubjectScope,
    update_kind: Optional[str] = None,
) -> str:
    """事实簇指纹：video_fact_ids + 作用域 + 更新类型。"""
    ids = ",".join(sorted({fid for fid in video_fact_ids if fid}))
    return f"{ids}|{subject_scope.value}|{update_kind or '-'}"


def sanitize_verification_for_coarse_prompt(
    result: Any,
) -> dict[str, Any]:
    """COARSE revision prompt 仅保留抽象失败码与目标 Agent。"""
    verdict = getattr(result, "verdict", None)
    return_to = getattr(result, "return_to_agent", None)
    failed = list(getattr(result, "failed_checks", None) or [])
    codes: list[str] = []
    for item in failed:
        text = str(item).strip()
        match = re.match(r"^([a-z][a-z0-9_]{2,})", text)
        if match is not None and "_" in match.group(1):
            codes.append(match.group(1))
        else:
            codes.append("verification_failed")
    if not codes:
        codes = ["verification_failed"]
    return {
        "verdict": verdict,
        "return_to_agent": return_to,
        "failure_codes": list(dict.fromkeys(codes)),
    }


def sanitize_revision_input_for_coarse_shard(
    result: Any,
) -> Any:
    """Agent1 shard 的 revision_input：去掉地名自然语言，只留审计码。"""
    from pipeline.schemas import VerificationResult

    if result is None:
        return None
    if not isinstance(result, VerificationResult):
        return result
    sanitized = sanitize_verification_for_coarse_prompt(result)
    return VerificationResult(
        verdict=result.verdict,
        failed_checks=list(sanitized["failure_codes"]),
        suggested_recheck="",
        return_to_agent=result.return_to_agent,
    )
