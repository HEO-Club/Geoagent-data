"""stage3：Move → NormalizedStep（匹配 / 组合 / 注册新 Tool / fallback / thought_only）。

本阶段不生成 Observation（Observation 属于 stage4）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from pipeline.coarse_tool_policy import (
    COARSE_CORE_TOOLS,
    is_coarse_allowed_tool,
    is_coarse_forbidden_tool,
)
from pipeline.evidence_routing import (
    ContentType,
    EvidenceIntent,
    ExtractedWorkingScope,
    RawClueRole,
    ScopeBoundKind,
    SemanticRoute,
    VideoChainContext,
    VideoContextExtraction,
    build_evidence_intent,
    context_from_extraction,
    embed_evidence_intent,
    embed_video_context,
    fallback_video_context,
    filter_geo_reasoning_moves,
    heuristic_route_move,
    normalize_working_scope_phrase,
    parse_evidence_intent,
)
from pipeline.llm import RealAPIDisabledError, call_structured
from pipeline.schemas import (
    ALLOWED_VERBS_HINT,
    Action,
    AgentRole,
    Move,
    NormalizationMode,
    NormalizedStep,
    ObservationField,
    ParamField,
    ToolDefinition,
)
from pipeline.tools.registry import load_registry, register_tool
from pipeline.tools.validation import validate_action_params

logger = logging.getLogger(__name__)

# 纯 UI 操作关键词（G6）：不得注册新 Tool，也不得无标记硬套为 Tool
_PURE_UI_RE = re.compile(
    r"滚动(?:页面|条)?|scroll(?:ing)?|"
    r"移动鼠标|mouse\s*move|mousemove|"
    r"切换(?:浏览器)?标签|switch\s*(?:browser\s*)?tab|"
    r"拖拽窗口|resize\s*window|"
    r"点击空白|hover(?:ing)?|"
    r"置顶|消息列表|聊天记录界面",
    re.IGNORECASE,
)
_TOOLISH_RE = re.compile(
    r"搜索|search|ocr|缩放|zoom|地图|map|街景|太阳|sun|"
    r"提交|submit|识别|detect|查询|query|lookup|放大",
    re.IGNORECASE,
)

_WEB_SEARCH_DEFAULT_PURPOSE: dict[AgentRole, str] = {
    AgentRole.COARSE: "broad_discovery",
    AgentRole.FINE: "precise_lookup",
    AgentRole.VERIFIER: "verification",
}
_VIDEO_CONTEXT_EXTRACT_ATTEMPTS = 3
_VIDEO_CONTEXT_BATCH_SIZE = 16


class _ProposedAction(BaseModel):
    """LLM 提出的单次 Tool 调用。"""

    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class _GRuleFlags(BaseModel):
    """G 规则八项评估；注册新 Tool 时必须全部为 True。"""

    cannot_match_existing: bool
    cannot_compose: bool
    io_semantics_clear: bool
    reusable_in_geolocation: bool
    observation_schema_complete: bool
    not_pure_ui: bool
    not_one_off_for_video: bool
    not_similar_to_existing: bool

    def all_passed(self) -> bool:
        """是否同时满足全部 G 条件。"""
        return all(
            (
                self.cannot_match_existing,
                self.cannot_compose,
                self.io_semantics_clear,
                self.reusable_in_geolocation,
                self.observation_schema_complete,
                self.not_pure_ui,
                self.not_one_off_for_video,
                self.not_similar_to_existing,
            )
        )


class _NewToolProposal(BaseModel):
    """LLM 提议的新 Tool schema（注册前再经 ToolDefinition 校验）。"""

    name: str
    description: str
    params: list[ParamField]
    observation_fields: list[ObservationField]
    allowed_agents: list[AgentRole]
    derived_from_existing_tools: list[str] = Field(default_factory=list)


class _MatchLLMResponse(BaseModel):
    """match_or_register_tool 的结构化 LLM 决策。"""

    decision: Literal["matched", "composed", "tool_registered", "fallback"]
    actions: list[_ProposedAction] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fallback_reason: Optional[str] = None
    g_flags: Optional[_GRuleFlags] = None
    new_tool: Optional[_NewToolProposal] = None


class _VideoContextGroundingReview(BaseModel):
    """动态来源抽取的独立语义复核。"""

    unsupported_raw_clue_positions: list[int] = Field(default_factory=list)
    working_scope_supported: bool
    unsupported_fact_positions: list[int] = Field(default_factory=list)
    reason: str = ""


class _WorkingScopeDerivation(BaseModel):
    """由拍摄地硬边界或软先验推导工作范围；不足则 region 为空。"""

    region: str = ""
    supporting_raw_clue_positions: list[int] = Field(default_factory=list)
    rationale: str = ""
    bound_kind: ScopeBoundKind = ScopeBoundKind.UNSPECIFIED


def _is_pure_ui(screen_action: str) -> bool:
    """判断是否为纯 UI 噪声操作（G6）。"""
    text = screen_action.strip()
    if not text:
        return False
    if not _PURE_UI_RE.search(text):
        return False
    # 同时含明显 tool 语义词时，不视为纯 UI
    return _TOOLISH_RE.search(text) is None


def _looks_toolish(screen_action: str) -> bool:
    """screen_action 是否含明显 Tool 语义（非纯 UI）。"""
    text = (screen_action or "").strip()
    if not text or _is_pure_ui(text):
        return False
    return _TOOLISH_RE.search(text) is not None


def _tools_catalog_text(tools: list[ToolDefinition], agent_role: AgentRole) -> str:
    """供 prompt 使用的 tool 摘要。"""
    lines: list[str] = []
    for t in tools:
        allowed = agent_role in t.allowed_agents
        param_desc = ", ".join(
            f"{p.name}:{p.type}{'*' if p.required else ''}" for p in t.params
        )
        lines.append(
            f"- {t.name} | allowed_for_role={allowed} | "
            f"agents={[a.value for a in t.allowed_agents]} | "
            f"params=[{param_desc}] | desc={t.description}"
        )
    return "\n".join(lines) if lines else "(empty)"


def _count_similar_screen_actions(screen_action: str, all_moves: list[Move]) -> int:
    """统计 all_moves 中与当前 screen_action 有词重叠的条目数（G7 参考）。"""
    tokens = {w for w in re.split(r"[\s，,。；;、/_-]+", screen_action.lower()) if len(w) >= 2}
    if not tokens:
        return 0
    count = 0
    for m in all_moves:
        other = (m.screen_action or "").strip().lower()
        if not other:
            continue
        other_tokens = {w for w in re.split(r"[\s，,。；;、/_-]+", other) if len(w) >= 2}
        if tokens & other_tokens:
            count += 1
    return count


def _inject_web_search_purpose(
    tool_name: str,
    params: dict[str, Any],
    agent_role: AgentRole,
) -> dict[str, Any]:
    """web_search 缺 purpose 时按角色注入默认值。"""
    if tool_name != "web_search":
        return params
    out = dict(params)
    if "purpose" not in out or out["purpose"] is None:
        out["purpose"] = _WEB_SEARCH_DEFAULT_PURPOSE[agent_role]
    return out


_HEURISTIC_CONFIDENCE = 0.55
_MATCH_LLM_MAX_ATTEMPTS = 2

_DEFAULT_ZOOM_BBOX: list[float] = [0.25, 0.25, 0.5, 0.5]

# Agent1 训练轨迹允许 Tool（核心三工具 ∪ 视觉地图/卫星；与 stage5 投影对齐）
_COARSE_TRAINING_TOOLS: frozenset[str] = COARSE_CORE_TOOLS
# 仅禁止类（外部检索/地名 API）须分解；地图/卫星允许原样进链
_COARSE_FORBIDDEN_DECOMPOSE_RE = re.compile(
    r"^(?:web_search|map_query|reverse_image_search)$",
    re.IGNORECASE,
)
_TEXT_EVIDENCE_RE = re.compile(
    r"ocr|文字|路牌|店招|招牌|字幕文字|标识|牌匾|读字|recognize\s*text",
    re.I,
)
_SUN_EVIDENCE_RE = re.compile(
    r"太阳|阴影|日照|sun\s*position|shadow|azimuth|altitude",
    re.I,
)
_VISUAL_EVIDENCE_RE = re.compile(
    r"放大|缩放|zoom|局部|立面|细节|对比|比对|卫星|画面|观察|inspect|compare",
    re.I,
)
# COARSE：应优先匹配视觉地图/卫星/地形 Tool，而非 zoom 兜底
_COARSE_GEO_SEMANTIC_RE = re.compile(
    r"卫星|遥感|历史地图|历史遥感|卫星地图|"
    r"比对|对比\s*(?:图|照片|卫星|影像)|"
    r"标注.*(?:地理|地形|平原|桥梁|环境|朝向|位置)|"
    r"(?:地理|地形).*(?:标注|特征)|地形|地貌视觉|视觉误差|视觉错觉",
    re.I,
)


def _is_coarse_evidence_only_tool(tool_name: str) -> bool:
    """COARSE 禁止原样进链、须分解或丢弃的 Tool（web_search/map_query/RIS）。"""
    return is_coarse_forbidden_tool(tool_name) or bool(
        _COARSE_FORBIDDEN_DECOMPOSE_RE.search(tool_name)
    )


def _is_coarse_keepable_tool(tool_name: str) -> bool:
    """COARSE 可原样保留的训练 Tool（核心 + 视觉地图/卫星）。"""
    return is_coarse_allowed_tool(tool_name)


def _coarse_decompose_to_training_actions(
    screen_action: str,
    narration: str,
    tool_by_name: dict[str, ToolDefinition],
    *,
    intent: Optional[EvidenceIntent] = None,
) -> Optional[list[Action]]:
    """将检索/双图等证据侧语义分解为有图像依据的训练 Tool 组合。

    依据 EvidenceIntent 给出差异化 bbox；仅在语义前置条件满足时添加 OCR/sun。
    """
    text = f"{screen_action} {narration}".strip()
    if not text:
        return None
    if intent is None:
        intent = build_evidence_intent(
            narration,
            screen_action,
            [],
            SemanticRoute.COARSE,
            conflict=False,
        )
    if intent.content_type is ContentType.INTERFACE_ONLY:
        return None

    proposed: list[_ProposedAction] = []
    needs_visual = bool(
        _VISUAL_EVIDENCE_RE.search(text)
        or re.search(r"搜索|search|检索|比对|对比|卫星|地图", text, re.I)
        or intent.target_features
    )
    if needs_visual and "zoom_inspect" in tool_by_name:
        bbox = list(intent.suggested_bbox) if intent.suggested_bbox else list(
            _DEFAULT_ZOOM_BBOX
        )
        proposed.append(
            _ProposedAction(tool="zoom_inspect", params={"bbox": bbox})
        )
    # 仅文本区域语义才加 OCR（聊天 UI 冲突时不加）
    if (
        _TEXT_EVIDENCE_RE.search(text)
        and "ocr" in tool_by_name
        and not intent.screen_action_untrusted
    ):
        proposed.append(
            _ProposedAction(
                tool="ocr",
                params={"bbox": [0.15, 0.15, 0.7, 0.35]},
            )
        )
    if _SUN_EVIDENCE_RE.search(text) and "sun_position_calc" in tool_by_name:
        proposed.append(_ProposedAction(tool="sun_position_calc", params={}))

    if not proposed:
        return None
    seen: set[str] = set()
    unique: list[_ProposedAction] = []
    for item in proposed:
        if item.tool in seen:
            continue
        seen.add(item.tool)
        unique.append(item)
    try:
        return _validate_actions(unique, tool_by_name, AgentRole.COARSE)
    except (ValueError, PermissionError, ValidationError):
        return None


def _apply_coarse_training_guard(
    actions: list[Action],
    mode: NormalizationMode,
    screen_action: str,
    narration: str,
    tool_by_name: dict[str, ToolDefinition],
    *,
    intent: Optional[EvidenceIntent] = None,
) -> tuple[list[Action], NormalizationMode, Optional[str]]:
    """COARSE：允许集原样保留；禁止类分解为 zoom/ocr/sun。"""
    if not actions:
        return actions, mode, None
    if all(_is_coarse_keepable_tool(a.tool) for a in actions):
        # 仍可用 intent 纠正 zoom 的同质 bbox
        if intent is not None and intent.suggested_bbox:
            fixed: list[Action] = []
            for a in actions:
                if a.tool == "zoom_inspect":
                    fixed.append(
                        Action(
                            tool="zoom_inspect",
                            params={"bbox": list(intent.suggested_bbox)},
                        )
                    )
                else:
                    fixed.append(a)
            return fixed, mode, None
        return actions, mode, None

    # 部分禁止：保留允许的，禁止类尝试分解替换
    kept: list[Action] = [a for a in actions if _is_coarse_keepable_tool(a.tool)]
    if any(_is_coarse_evidence_only_tool(a.tool) for a in actions) or not kept:
        decomposed = _coarse_decompose_to_training_actions(
            screen_action, narration, tool_by_name, intent=intent
        )
        if decomposed:
            merged = kept + [
                a for a in decomposed if a.tool not in {x.tool for x in kept}
            ]
            new_mode = (
                NormalizationMode.COMPOSED
                if len(merged) >= 2
                else NormalizationMode.MATCHED
            )
            return merged, new_mode, None
        if kept:
            return kept, mode, None
        # 无法分解：强制 zoom，避免空链
        if "zoom_inspect" in tool_by_name:
            bbox = (
                list(intent.suggested_bbox)
                if intent is not None and intent.suggested_bbox
                else list(_DEFAULT_ZOOM_BBOX)
            )
            try:
                forced = _validate_actions(
                    [_ProposedAction(tool="zoom_inspect", params={"bbox": bbox})],
                    tool_by_name,
                    AgentRole.COARSE,
                )
                return forced, NormalizationMode.MATCHED, None
            except (ValueError, PermissionError, ValidationError):
                pass
        return [], mode, "COARSE 禁止 Tool 无法分解为训练 Tool"
    # 未知动态 Tool（非允许非禁止）：尝试分解，否则丢弃
    decomposed = _coarse_decompose_to_training_actions(
        screen_action, narration, tool_by_name, intent=intent
    )
    if decomposed:
        new_mode = (
            NormalizationMode.COMPOSED
            if len(decomposed) >= 2
            else NormalizationMode.MATCHED
        )
        return decomposed, new_mode, None
    if "zoom_inspect" in tool_by_name:
        bbox = (
            list(intent.suggested_bbox)
            if intent is not None and intent.suggested_bbox
            else list(_DEFAULT_ZOOM_BBOX)
        )
        try:
            forced = _validate_actions(
                [_ProposedAction(tool="zoom_inspect", params={"bbox": bbox})],
                tool_by_name,
                AgentRole.COARSE,
            )
            return forced, NormalizationMode.MATCHED, None
        except (ValueError, PermissionError, ValidationError):
            pass
    return [], mode, "COARSE 非允许 Tool 已拒绝，待分解为 zoom/ocr/sun"


def _intent_for_move(
    move: Move,
    route: SemanticRoute,
    *,
    source_concepts: Optional[list[str]] = None,
    video_fact_ids: Optional[list[str]] = None,
    step_kind: Optional[Any] = None,
) -> EvidenceIntent:
    """为 Move 构造 EvidenceIntent（含 narration/screen 冲突检测）。"""
    from pipeline.evidence_routing import CoarseStepKind

    decision = heuristic_route_move(
        move.narration, move.screen_action, list(move.visible_clues)
    )
    if decision.intent is not None:
        intent = decision.intent
        intent.route = route
        if source_concepts is not None:
            intent.source_concepts = list(source_concepts)
            # 特征与闭包取交，避免 UI/旁白外扩
            if intent.target_features:
                inter = [t for t in intent.target_features if t in source_concepts]
                if inter:
                    intent.target_features = inter
        if video_fact_ids is not None:
            intent.video_fact_ids = list(video_fact_ids)
        if step_kind is not None:
            intent.step_kind = step_kind
        elif route is SemanticRoute.COARSE:
            intent.step_kind = CoarseStepKind.UPDATE
        return intent
    intent = build_evidence_intent(
        move.narration,
        move.screen_action,
        list(move.visible_clues),
        route,
        conflict=False,
        source_concepts=source_concepts,
        video_fact_ids=video_fact_ids,
        step_kind=step_kind,
    )
    if step_kind is not None:
        intent.step_kind = step_kind
    elif route is SemanticRoute.COARSE:
        intent.step_kind = CoarseStepKind.UPDATE
    return intent


def _build_thought_draft(
    move: Move, *, intent: Optional[EvidenceIntent] = None
) -> str:
    """生成极短视觉/操作线索；可嵌入 EvidenceIntent。"""
    from pipeline.evidence_routing import CoarseStepKind

    exclude_hint = ""
    if intent is not None and intent.step_kind is CoarseStepKind.UPDATE:
        exclude_hint = "；本步须基于视频事实明确排除至少一个候选地点/类别"

    if intent is not None and intent.screen_action_untrusted:
        feats = "、".join(intent.target_features) or intent.target_object
        draft = f"观察目标：{feats}"
        if intent.expected_spatial_relation:
            draft += f"；关注：{intent.expected_spatial_relation}"
        draft += exclude_hint
        return embed_evidence_intent(draft, intent)

    if move.screen_action and move.screen_action.strip():
        sa = move.screen_action.strip()
        if len(sa) > 40:
            sa = sa[:37].rstrip() + "…"
        draft = sa + exclude_hint
        if intent is not None:
            return embed_evidence_intent(draft, intent)
        return draft

    narr = (move.narration or "").strip()
    if narr:
        draft = (narr[:60] + ("…" if len(narr) > 60 else "")) + exclude_hint
        if intent is not None:
            return embed_evidence_intent(draft, intent)
        return draft

    draft = "观察画面地理要素" + exclude_hint
    if intent is not None:
        return embed_evidence_intent(draft, intent)
    return draft


def _heuristic_query(screen_action: str, narration: str) -> str:
    """从 screen_action / narration 抽短 query，失败则给通用占位。"""
    raw = f"{screen_action} {narration}".strip()
    cleaned = re.sub(
        r"(?:在|打开|点击|输入|搜索框|放大|缩小|查看|识别|OCR|ocr|"
        r"地图|街景|提交|查询|搜索|search|zoom|map)+",
        " ",
        raw,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。.;；")
    if len(cleaned) >= 2:
        return cleaned[:80]
    fallback = (screen_action or narration or "visual landmark clues").strip()
    return fallback[:80] or "visual landmark clues"


def _coarse_geo_feature_list(screen_action: str, narration: str) -> list[str]:
    """从旁白/操作文本抽短特征列表，供 annotate/terrain 启发式参数。"""
    text = f"{screen_action} {narration}"
    feats = [
        m.group(0)
        for m in re.finditer(
            r"平原|河流|桥梁|山体|山脉|河岸|高地|依山亭|铁路|道路|公园|建筑",
            text,
        )
    ]
    return list(dict.fromkeys(feats))[:6] or ["地理特征"]


def _propose_coarse_geo_action(
    screen_action: str,
    narration: str,
    tool_by_name: dict[str, ToolDefinition],
) -> Optional[list[Action]]:
    """COARSE：卫星/标注/比对/地形语义 → 允许集内 geo Tool（非 zoom）。"""
    text = f"{screen_action} {narration}"
    if not _COARSE_GEO_SEMANTIC_RE.search(text):
        return None
    query = _heuristic_query(screen_action, narration)
    feats = _coarse_geo_feature_list(screen_action, narration)
    proposed: Optional[_ProposedAction] = None

    if re.search(r"标注|叠加标注", text) and (
        "annotate_geographic_environment_on_image" in tool_by_name
    ):
        proposed = _ProposedAction(
            tool="annotate_geographic_environment_on_image",
            params={
                "target_image": "primary_frame",
                "geographic_features": feats,
            },
        )
    elif re.search(r"比对|对比", text) and (
        "compare_images_for_geolocation" in tool_by_name
    ):
        proposed = _ProposedAction(
            tool="compare_images_for_geolocation",
            params={
                "image_a": "primary_photo",
                "image_b": "map_or_satellite_view",
                "focus_features": feats,
            },
        )
    elif re.search(r"视觉误差|视觉错觉|误认|模糊性", text) and (
        "analyze_terrain_ambiguity" in tool_by_name
    ):
        proposed = _ProposedAction(
            tool="analyze_terrain_ambiguity",
            params={
                "target_image": "primary_frame",
                "suspected_features": feats,
            },
        )
    elif re.search(r"地形|地貌|平原|河岸|山脉", text) and (
        "detect_terrain_features" in tool_by_name
    ):
        proposed = _ProposedAction(
            tool="detect_terrain_features",
            params={
                "target_image": "primary_frame",
                "focus_features": feats,
            },
        )
    elif re.search(r"历史|布局|最早", text) and (
        "lookup_historical_map_layout" in tool_by_name
    ):
        proposed = _ProposedAction(
            tool="lookup_historical_map_layout",
            params={
                "year": 2002,
                "query": query,
                "purpose": "broad_discovery",
            },
        )
    elif "lookup_historical_satellite_map" in tool_by_name:
        proposed = _ProposedAction(
            tool="lookup_historical_satellite_map",
            params={"year": 2002, "query": query},
        )

    if proposed is None:
        return None
    try:
        return _validate_actions([proposed], tool_by_name, AgentRole.COARSE)
    except (ValueError, PermissionError, ValidationError):
        return None


def _prefer_coarse_geo_actions(
    screen_action: str,
    narration: str,
    actions: list[Action],
    tool_by_name: dict[str, ToolDefinition],
) -> list[Action]:
    """卫星/标注等语义下，禁止停留在同质 zoom；优先替换为 geo Tool。"""
    text = f"{screen_action} {narration}"
    if not _COARSE_GEO_SEMANTIC_RE.search(text):
        return actions
    if actions and all(
        is_coarse_allowed_tool(a.tool) and a.tool not in COARSE_CORE_TOOLS
        for a in actions
    ):
        return actions
    only_core = (not actions) or all(a.tool in COARSE_CORE_TOOLS for a in actions)
    if not only_core:
        return actions
    geo = _propose_coarse_geo_action(screen_action, narration, tool_by_name)
    return geo if geo else actions


def _try_heuristic_actions(
    screen_action: str,
    narration: str,
    agent_role: AgentRole,
    tool_by_name: dict[str, ToolDefinition],
) -> Optional[list[Action]]:
    """将常见 screen_action 启发式映射到现有种子 Tool；无法映射则 None。"""
    if not _looks_toolish(screen_action):
        return None

    text = f"{screen_action} {narration}".lower()
    query = _heuristic_query(screen_action, narration)
    proposed: Optional[_ProposedAction] = None

    # 更具体的语义优先
    if re.search(r"提交|submit\s*answer|交卷|确认答案", text):
        if agent_role == AgentRole.FINE and "submit_answer" in tool_by_name:
            # terminal 需要完整 params；启发式无法知坐标 → 不硬套
            return None
    if agent_role == AgentRole.COARSE:
        geo = _propose_coarse_geo_action(screen_action, narration, tool_by_name)
        if geo is not None:
            return geo
    if re.search(r"太阳|阴影|sun\s*position|shadow", text):
        if agent_role == AgentRole.COARSE and "sun_position_calc" in tool_by_name:
            proposed = _ProposedAction(tool="sun_position_calc", params={})
    elif re.search(r"地图|街景|map\s*query|打开地图", text):
        if agent_role in (AgentRole.FINE, AgentRole.VERIFIER) and "map_query" in tool_by_name:
            proposed = _ProposedAction(tool="map_query", params={"query": query})
        elif agent_role == AgentRole.COARSE:
            geo = _propose_coarse_geo_action(screen_action, narration, tool_by_name)
            if geo is not None:
                return geo
            if "zoom_inspect" in tool_by_name:
                proposed = _ProposedAction(
                    tool="zoom_inspect",
                    params={"bbox": list(_DEFAULT_ZOOM_BBOX)},
                )
    elif re.search(r"ocr|识别文字|路牌|店招|字幕文字", text):
        if "ocr" in tool_by_name and agent_role in tool_by_name["ocr"].allowed_agents:
            proposed = _ProposedAction(tool="ocr", params={})
    elif re.search(r"放大|缩放|zoom|局部|尖顶|立面细节", text):
        if (
            "zoom_inspect" in tool_by_name
            and agent_role in tool_by_name["zoom_inspect"].allowed_agents
        ):
            proposed = _ProposedAction(
                tool="zoom_inspect",
                params={"bbox": list(_DEFAULT_ZOOM_BBOX)},
            )
    elif re.search(r"以图搜|reverse\s*image|相似图", text):
        if (
            agent_role in (AgentRole.FINE, AgentRole.VERIFIER)
            and "reverse_image_search" in tool_by_name
            and agent_role in tool_by_name["reverse_image_search"].allowed_agents
        ):
            proposed = _ProposedAction(tool="reverse_image_search", params={})
        elif "web_search" in tool_by_name and agent_role in tool_by_name["web_search"].allowed_agents:
            proposed = _ProposedAction(
                tool="web_search",
                params={
                    "query": query,
                    "purpose": _WEB_SEARCH_DEFAULT_PURPOSE[agent_role],
                },
            )
    elif re.search(r"搜索|search|检索|查询|lookup|google|百度", text):
        if "web_search" in tool_by_name and agent_role in tool_by_name["web_search"].allowed_agents:
            proposed = _ProposedAction(
                tool="web_search",
                params={
                    "query": query,
                    "purpose": _WEB_SEARCH_DEFAULT_PURPOSE[agent_role],
                },
            )

    if proposed is None:
        return None
    try:
        return _validate_actions([proposed], tool_by_name, agent_role)
    except (ValueError, PermissionError, ValidationError):
        return None


def _heuristic_recovery(
    screen_action: str,
    narration: str,
    agent_role: AgentRole,
    tool_by_name: dict[str, ToolDefinition],
) -> Optional[tuple[list[Action], NormalizationMode, Optional[str], Optional[float]]]:
    """尝试启发式恢复；成功返回 matched 四元组。"""
    actions = _try_heuristic_actions(
        screen_action, narration, agent_role, tool_by_name
    )
    if not actions:
        return None
    mode = NormalizationMode.MATCHED
    if agent_role == AgentRole.COARSE:
        actions, mode, _ = _apply_coarse_training_guard(
            actions, mode, screen_action, narration, tool_by_name
        )
    return actions, mode, None, _HEURISTIC_CONFIDENCE


def _call_match_llm_with_retry(
    screen_action: str,
    narration: str,
    agent_role: AgentRole,
    existing_tools: list[ToolDefinition],
    all_moves: list[Move],
) -> _MatchLLMResponse:
    """LLM 匹配决策；短暂失败时重试一次。"""
    last_exc: Exception | None = None
    for _ in range(_MATCH_LLM_MAX_ATTEMPTS):
        try:
            return _call_match_llm(
                screen_action, narration, agent_role, existing_tools, all_moves
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def _validate_actions(
    proposed: list[_ProposedAction],
    tool_by_name: dict[str, ToolDefinition],
    agent_role: AgentRole,
) -> list[Action]:
    """校验权限与 params，返回规范化 Action 列表；失败抛 ValueError。"""
    if not proposed:
        raise ValueError("匹配/组合决策必须至少包含一个 Action")
    actions: list[Action] = []
    for item in proposed:
        if item.tool not in tool_by_name:
            raise ValueError(f"未知 tool: {item.tool}")
        tool = tool_by_name[item.tool]
        if agent_role not in tool.allowed_agents:
            raise PermissionError(
                f"角色 {agent_role.value} 无权使用 tool {tool.name}；"
                f"允许: {[r.value for r in tool.allowed_agents]}"
            )
        raw_params = _inject_web_search_purpose(item.tool, item.params, agent_role)
        validated = validate_action_params(tool, raw_params, agent_role=agent_role)
        actions.append(Action(tool=item.tool, params=validated))
    return actions


def _proposal_to_tool_definition(
    proposal: _NewToolProposal,
    narration: str,
    agent_role: AgentRole,
) -> ToolDefinition:
    """将 LLM 新 Tool 提案转为可注册的 ToolDefinition。"""
    allowed = list(proposal.allowed_agents)
    if agent_role not in allowed:
        allowed.append(agent_role)
    payload: dict[str, Any] = {
        "name": proposal.name,
        "description": proposal.description,
        "params": [p.model_dump(mode="json") for p in proposal.params],
        "observation_fields": [o.model_dump(mode="json") for o in proposal.observation_fields],
        "allowed_agents": allowed,
        "is_terminal": False,
        "created_at": "1970-01-01T00:00:00Z",  # register_tool 会覆盖
        "source_narration": narration or None,
        "derived_from_existing_tools": list(proposal.derived_from_existing_tools),
    }
    return ToolDefinition.model_validate(payload)


def _call_match_llm(
    screen_action: str,
    narration: str,
    agent_role: AgentRole,
    existing_tools: list[ToolDefinition],
    all_moves: list[Move],
) -> _MatchLLMResponse:
    """调用结构化 LLM 做匹配决策。"""
    similar_n = _count_similar_screen_actions(screen_action, all_moves)
    verbs = ", ".join(sorted(ALLOWED_VERBS_HINT))
    prompt = (
        "你是地理定位 Agent 轨迹的 Tool 规范化器。"
        "根据 screen_action 与 narration，在现有 Tool 中匹配、组合，"
        "或在满足 G 规则时注册新 Tool（仅 schema，Observation 由 LLM 合成）；否则 fallback。"
        "禁止无标记硬套；禁止把纯 UI 操作（滚动/移鼠标/切标签）建成 Tool。"
        "优先匹配现有种子 Tool（web_search/zoom_inspect/ocr/map_query 等），"
        "仅在确实无法表达时才 tool_registered 或 fallback。"
        f"\n当前 AgentRole: {agent_role.value}"
        f"\nscreen_action: {screen_action}"
        f"\nnarration: {narration}"
        f"\nall_moves 中相似 screen_action 条数（含自身参考）: {similar_n}"
        f"\n现有 Tool 列表:\n{_tools_catalog_text(existing_tools, agent_role)}"
        "\n决策规则："
        "\n- matched: 单一现有 Tool，填写 actions（1 个）与 confidence"
        "\n- composed: 多个现有 Tool 组合，actions 长度≥2"
        "\n- tool_registered: 仅当无法匹配且无法组合，且 g_flags 八项全 True；"
        "同时给出 new_tool（名称须小写 snake_case、至少两 token；"
        "observation_fields 须完整可合成，含 status/error_message；"
        f"命名可参考动词提示: {verbs}）"
        "\n- fallback: 无法安全映射时给出 fallback_reason"
        "\nweb_search.purpose 角色约束："
        "COARSE→broad_discovery；FINE→broad_discovery|precise_lookup；"
        "VERIFIER→verification。"
        "\nAgent1 不得使用 reverse_image_search / map_query / web_search / submit_answer；"
        "Agent2 不得使用 sun_position_calc。"
        "\nCOARSE 训练轨迹允许：zoom_inspect/ocr/sun_position_calc，以及"
        "视觉地图/卫星/地形类（compare_images_for_geolocation、"
        "lookup_historical_*、find_specific_features_in_satellite_map、"
        "annotate_geographic_*、detect_terrain_*、analyze_terrain_*）；"
        "禁止类才分解为 zoom/ocr/sun。"
        "\n若 screen_action/narration 含卫星/遥感/历史地图/标注地理特征/"
        "双图比对/地形分析语义：必须匹配对应 geo Tool，禁止用 zoom_inspect 兜底。"
        "仅有文字区域才加 ocr，仅有阴影/日照才加 sun。"
        "\n只输出结构化字段。"
    )
    result = call_structured(prompt, _MatchLLMResponse)
    if not isinstance(result, _MatchLLMResponse):
        result = _MatchLLMResponse.model_validate(result)
    return result


def _match_or_register_tool_impl(
    screen_action: str,
    narration: str,
    agent_role: AgentRole,
    existing_tools: list[ToolDefinition],
    all_moves: list[Move],
) -> tuple[list[Action], NormalizationMode, Optional[str], Optional[float]]:
    """内部实现：额外返回 matched_tool_confidence。"""
    text = (screen_action or "").strip()
    if not text:
        return [], NormalizationMode.THOUGHT_ONLY, "screen_action 为空", None

    if _is_pure_ui(text):
        return (
            [],
            NormalizationMode.FALLBACK,
            "纯 UI 操作（滚动/移鼠标/切标签等），禁止硬套或注册新 Tool",
            None,
        )

    tool_by_name = {t.name: t for t in existing_tools}

    try:
        decision = _call_match_llm_with_retry(
            text, narration, agent_role, existing_tools, all_moves
        )
    except Exception as exc:  # noqa: BLE001 — LLM 失败 → 启发式 → fallback
        recovered = _heuristic_recovery(text, narration, agent_role, tool_by_name)
        if recovered is not None:
            actions, mode, reason, conf = recovered
            if agent_role == AgentRole.COARSE:
                actions, mode, _ = _apply_coarse_training_guard(
                    actions, mode, text, narration, tool_by_name
                )
            return actions, mode, reason, conf
        return [], NormalizationMode.FALLBACK, f"LLM 决策失败: {exc}", None

    mode_map = {
        "matched": NormalizationMode.MATCHED,
        "composed": NormalizationMode.COMPOSED,
        "tool_registered": NormalizationMode.TOOL_REGISTERED,
        "fallback": NormalizationMode.FALLBACK,
    }
    mode = mode_map[decision.decision]
    confidence = decision.confidence

    if mode is NormalizationMode.FALLBACK:
        reason = decision.fallback_reason or "模型判定无法安全映射"
        recovered = _heuristic_recovery(text, narration, agent_role, tool_by_name)
        if recovered is not None:
            return recovered
        return [], mode, reason, None

    if mode in (NormalizationMode.MATCHED, NormalizationMode.COMPOSED):
        try:
            if mode is NormalizationMode.COMPOSED and len(decision.actions) < 2:
                recovered = _heuristic_recovery(
                    text, narration, agent_role, tool_by_name
                )
                if recovered is not None:
                    return recovered
                return (
                    [],
                    NormalizationMode.FALLBACK,
                    "composed 决策但 actions 少于 2，禁止硬套",
                    None,
                )
            if mode is NormalizationMode.MATCHED and len(decision.actions) != 1:
                recovered = _heuristic_recovery(
                    text, narration, agent_role, tool_by_name
                )
                if recovered is not None:
                    return recovered
                return (
                    [],
                    NormalizationMode.FALLBACK,
                    "matched 决策必须恰好 1 个 Action",
                    None,
                )
            actions = _validate_actions(decision.actions, tool_by_name, agent_role)
            if agent_role == AgentRole.COARSE:
                actions, mode, _guard_reason = _apply_coarse_training_guard(
                    actions, mode, text, narration, tool_by_name
                )
                actions = _prefer_coarse_geo_actions(
                    text, narration, actions, tool_by_name
                )
            if confidence is None:
                confidence = 0.8 if mode is NormalizationMode.MATCHED else 0.7
            return actions, mode, None, confidence
        except (ValueError, PermissionError, ValidationError) as exc:
            recovered = _heuristic_recovery(text, narration, agent_role, tool_by_name)
            if recovered is not None:
                return recovered
            return [], NormalizationMode.FALLBACK, f"Action 校验失败: {exc}", None

    # tool_registered
    if decision.g_flags is None or not decision.g_flags.all_passed():
        recovered = _heuristic_recovery(text, narration, agent_role, tool_by_name)
        if recovered is not None:
            return recovered
        return (
            [],
            NormalizationMode.FALLBACK,
            "不满足 G 规则全部条件，禁止注册新 Tool",
            None,
        )
    if decision.new_tool is None:
        recovered = _heuristic_recovery(text, narration, agent_role, tool_by_name)
        if recovered is not None:
            return recovered
        return [], NormalizationMode.FALLBACK, "tool_registered 但缺少 new_tool", None
    if _is_pure_ui(text) or not decision.g_flags.not_pure_ui:
        return [], NormalizationMode.FALLBACK, "纯 UI 操作禁止注册新 Tool", None

    # COARSE：禁止注册 web_search/map_query/RIS；改为核心 Tool 分解
    if agent_role == AgentRole.COARSE and _is_coarse_evidence_only_tool(
        decision.new_tool.name
    ):
        decomposed = _coarse_decompose_to_training_actions(
            text, narration, tool_by_name
        )
        if decomposed:
            new_mode = (
                NormalizationMode.COMPOSED
                if len(decomposed) >= 2
                else NormalizationMode.MATCHED
            )
            return decomposed, new_mode, None, confidence or 0.7
        recovered = _heuristic_recovery(text, narration, agent_role, tool_by_name)
        if recovered is not None:
            return recovered
        return (
            [],
            NormalizationMode.FALLBACK,
            "COARSE 禁止注册检索/map_query 类新 Tool，且无法分解为训练 Tool",
            None,
        )

    try:
        new_def = _proposal_to_tool_definition(
            decision.new_tool, narration, agent_role
        )
        register_tool(new_def)
        refreshed = load_registry()
        tool_by_name = dict(refreshed)

        if decision.actions and decision.actions[0].tool == new_def.name:
            actions = _validate_actions(decision.actions[:1], tool_by_name, agent_role)
        else:
            example_params: dict[str, Any] = {
                p.name: p.example for p in new_def.params if p.required
            }
            for p in new_def.params:
                if not p.required and p.default is not None:
                    example_params.setdefault(p.name, p.default)
            actions = _validate_actions(
                [_ProposedAction(tool=new_def.name, params=example_params)],
                tool_by_name,
                agent_role,
            )
        return actions, NormalizationMode.TOOL_REGISTERED, None, confidence
    except (ValueError, PermissionError, ValidationError) as exc:
        recovered = _heuristic_recovery(text, narration, agent_role, tool_by_name)
        if recovered is not None:
            return recovered
        return (
            [],
            NormalizationMode.FALLBACK,
            f"新 Tool 注册或参数校验失败: {exc}",
            None,
        )


def match_or_register_tool(
    screen_action: str,
    narration: str,
    agent_role: AgentRole,
    existing_tools: list[ToolDefinition],
    all_moves: list[Move],
) -> tuple[list[Action], NormalizationMode, Optional[str]]:
    """按 G 规则决定：匹配 / 组合 / 注册新 Tool / fallback。

    同时检查 allowed_agents 与 web_search.purpose 角色约束。
    LLM 失败或空 fallback 时，对含 Tool 语义的操作尝试启发式映射现有 Tool。
    返回 actions（组合可为多个）、mode、fallback_reason。
    """
    actions, mode, reason, _confidence = _match_or_register_tool_impl(
        screen_action, narration, agent_role, existing_tools, all_moves
    )
    return actions, mode, reason


def normalize_to_steps(
    moves: list[Move],
    agent_role: AgentRole,
) -> list[NormalizedStep]:
    """每个 Move → NormalizedStep。

    screen_action 为空 → thought_only，actions=[]。
    一个 Move 可对应多个 Action（composed）。
    COARSE 写入 EvidenceIntent；旁白与 UI screen_action 冲突时采信旁白目标。
    """
    steps: list[NormalizedStep] = []
    route = (
        SemanticRoute.COARSE
        if agent_role == AgentRole.COARSE
        else (
            SemanticRoute.FINE
            if agent_role == AgentRole.FINE
            else SemanticRoute.NON_TRAINING
        )
    )

    for move in moves:
        intent: Optional[EvidenceIntent] = None
        if agent_role in (AgentRole.COARSE, AgentRole.FINE):
            intent = _intent_for_move(move, route)
            # 纯 UI 且无地理旁白 → thought_only
            if (
                agent_role == AgentRole.COARSE
                and intent.content_type is ContentType.INTERFACE_ONLY
                and not intent.target_features
            ):
                steps.append(
                    NormalizedStep(
                        move=move,
                        thought_draft=_build_thought_draft(move, intent=intent),
                        actions=[],
                        normalization_mode=NormalizationMode.THOUGHT_ONLY,
                        matched_tool_confidence=None,
                        fallback_reason="interface_only：不生成训练 Action",
                    )
                )
                continue

        thought = _build_thought_draft(move, intent=intent)
        screen = move.screen_action
        # 冲突时用旁白驱动分解，不依赖不可信 screen_action 文本
        effective_screen = str(screen) if screen else ""
        if intent is not None and intent.screen_action_untrusted:
            effective_screen = (
                f"观察{intent.target_object}："
                + "、".join(intent.target_features)
            )

        if not effective_screen.strip() and (
            screen is None or not str(screen).strip()
        ):
            steps.append(
                NormalizedStep(
                    move=move,
                    thought_draft=thought,
                    actions=[],
                    normalization_mode=NormalizationMode.THOUGHT_ONLY,
                    matched_tool_confidence=None,
                    fallback_reason="screen_action 为空，不得伪造 Tool Action",
                )
            )
            continue

        existing_tools = list(load_registry().values())
        tool_by_name = {t.name: t for t in existing_tools}
        actions, mode, reason, confidence = _match_or_register_tool_impl(
            screen_action=effective_screen or str(screen),
            narration=move.narration,
            agent_role=agent_role,
            existing_tools=existing_tools,
            all_moves=moves,
        )
        if agent_role == AgentRole.COARSE and intent is not None:
            actions, mode, _ = _apply_coarse_training_guard(
                actions,
                mode,
                effective_screen or str(screen),
                move.narration,
                tool_by_name,
                intent=intent,
            )
            actions = _prefer_coarse_geo_actions(
                effective_screen or str(screen),
                move.narration,
                actions,
                tool_by_name,
            )
            # 匹配失败或证据侧 Tool 被拒后：优先 geo，再分解为 zoom/ocr/sun
            if not actions:
                geo = _propose_coarse_geo_action(
                    effective_screen or str(screen) or move.narration,
                    move.narration,
                    tool_by_name,
                )
                if geo:
                    actions = geo
                    mode = NormalizationMode.MATCHED
                    reason = None
                    confidence = confidence or 0.65
                else:
                    decomposed = _coarse_decompose_to_training_actions(
                        effective_screen or str(screen) or move.narration,
                        move.narration,
                        tool_by_name,
                        intent=intent,
                    )
                    if decomposed:
                        actions = decomposed
                        mode = (
                            NormalizationMode.COMPOSED
                            if len(decomposed) >= 2
                            else NormalizationMode.MATCHED
                        )
                        reason = None
                        confidence = confidence or 0.6

        steps.append(
            NormalizedStep(
                move=move,
                thought_draft=thought,
                actions=actions,
                normalization_mode=mode,
                matched_tool_confidence=confidence,
                fallback_reason=reason,
            )
        )

    if agent_role == AgentRole.COARSE:
        steps = _diversify_zoom_bboxes(steps)
        steps = _dedupe_homogeneous_zoom_steps(steps)
    return steps


def _diversify_zoom_bboxes(steps: list[NormalizedStep]) -> list[NormalizedStep]:
    """避免重复 zoom 落成完全相同 bbox（横向+纵向+尺度变化，避免触顶同框）。"""
    seen: dict[tuple[float, ...], int] = {}
    out: list[NormalizedStep] = []
    for step in steps:
        new_actions: list[Action] = []
        changed = False
        for action in step.actions:
            if action.tool != "zoom_inspect":
                new_actions.append(action)
                continue
            bbox = action.params.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                new_actions.append(action)
                continue
            key = tuple(round(float(x), 3) for x in bbox)
            count = seen.get(key, 0)
            seen[key] = count + 1
            if count == 0:
                new_actions.append(action)
                continue
            x, y, w, h = (float(v) for v in bbox)
            x2 = min(0.55, max(0.0, x + 0.06 * (count % 5)))
            y2 = min(0.60, max(0.0, y + 0.07 * ((count // 2) % 4)))
            w2 = max(0.25, min(0.85, w - 0.04 * (count % 3)))
            h2 = max(0.25, min(0.70, h - 0.03 * ((count + 1) % 3)))
            if x2 + w2 > 1.0:
                w2 = max(0.25, 1.0 - x2)
            if y2 + h2 > 1.0:
                h2 = max(0.25, 1.0 - y2)
            new_bbox = [x2, y2, w2, h2]
            new_key = tuple(round(float(v), 3) for v in new_bbox)
            # 仍撞车则再横向挤一次
            if new_key in seen:
                x2 = min(0.7, x2 + 0.05)
                if x2 + w2 > 1.0:
                    w2 = max(0.25, 1.0 - x2)
                new_bbox = [x2, y2, w2, h2]
            seen[tuple(round(float(v), 3) for v in new_bbox)] = (
                seen.get(tuple(round(float(v), 3) for v in new_bbox), 0) + 1
            )
            new_actions.append(Action(tool="zoom_inspect", params={"bbox": new_bbox}))
            changed = True
        if changed:
            out.append(step.model_copy(update={"actions": new_actions}))
        else:
            out.append(step)
    return out


def _zoom_bbox_key(step: NormalizedStep) -> Optional[tuple[float, ...]]:
    """单 Action zoom 步的 bbox 指纹；非纯 zoom 返回 None。"""
    if len(step.actions) != 1 or step.actions[0].tool != "zoom_inspect":
        return None
    bbox = step.actions[0].params.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return tuple(round(float(x), 3) for x in bbox)


def _dedupe_homogeneous_zoom_steps(
    steps: list[NormalizedStep],
) -> list[NormalizedStep]:
    """连续同 bbox 的纯 zoom 且无新 target_features 时丢弃后步。"""
    if not steps:
        return steps
    out: list[NormalizedStep] = []
    prev_key: Optional[tuple[float, ...]] = None
    prev_feats: Optional[tuple[str, ...]] = None
    for step in steps:
        key = _zoom_bbox_key(step)
        intent = parse_evidence_intent(step.thought_draft)
        feats = tuple(intent.target_features) if intent is not None else ()
        if (
            key is not None
            and prev_key is not None
            and key == prev_key
            and feats == prev_feats
        ):
            continue
        out.append(step)
        if key is not None:
            prev_key = key
            prev_feats = feats
        else:
            prev_key = None
            prev_feats = None
    return out if out else steps[:1]


def normalize_all_agent_steps(
    moves_by_role: dict[AgentRole, list[Move]],
    answer_timestamp: float,
) -> dict[AgentRole, list[NormalizedStep]]:
    """对答案前 Move 做语义重路由、地理链去噪后再 normalize。

    广域地貌/排除/自然区域 → COARSE；精确 POI/建筑/坐标 → FINE；
    纯 UI/故事 → 丢弃（NON_TRAINING）。抽取 working_scope 与视频事实闭包。
    COARSE 保留试错+区域成功，剔除 stall/无地理段；不影响 FINE。
    不读 groundtruth。
    """
    pre_answer: list[Move] = []
    for role in (AgentRole.COARSE, AgentRole.FINE, AgentRole.VERIFIER):
        for move in moves_by_role.get(role) or []:
            if float(move.start_time) >= float(answer_timestamp) - 1e-9:
                continue
            pre_answer.append(move)
    pre_answer.sort(key=lambda m: (m.start_time, m.end_time))

    video_context = _extract_video_chain_context(pre_answer)

    coarse_moves: list[Move] = []
    fine_moves: list[Move] = []
    coarse_indices = {
        fact.source_move_index
        for fact in video_context.video_facts
        if fact.source_move_index is not None
        and fact.kind in {"observe", "correct", "exclude", "candidate"}
    }
    for index, move in enumerate(pre_answer):
        decision = heuristic_route_move(
            move.narration, move.screen_action, list(move.visible_clues)
        )
        if decision.route is SemanticRoute.NON_TRAINING:
            continue
        # 精定位粒度优先：禁止「凡有 video_fact 就归 COARSE」把 FINE 抽空
        if decision.route is SemanticRoute.FINE:
            fine_moves.append(
                move.model_copy(update={"agent_role": AgentRole.FINE})
            )
            continue
        if index in coarse_indices or decision.route is SemanticRoute.COARSE:
            coarse_moves.append(
                move.model_copy(update={"agent_role": AgentRole.COARSE})
            )

    # 完整地理链去噪：作用于路由后的 COARSE，不得用未路由 pre_answer 覆盖
    coarse_moves = filter_geo_reasoning_moves(
        coarse_moves, pre_answer, video_context
    )
    coarse_steps = _normalize_coarse_with_closure(
        coarse_moves,
        video_context=video_context,
    )
    result: dict[AgentRole, list[NormalizedStep]] = {
        AgentRole.COARSE: coarse_steps,
        AgentRole.FINE: normalize_to_steps(fine_moves, AgentRole.FINE),
        AgentRole.VERIFIER: normalize_to_steps(
            list(moves_by_role.get(AgentRole.VERIFIER) or []),
            AgentRole.VERIFIER,
        ),
    }
    return result


def _extract_video_chain_context(moves: list[Move]) -> VideoChainContext:
    """逐视频动态抽取来源事实；不使用固定地名、地貌或候选词表。

    真实 API 路径下抽取失败必须抛错，禁止静默降级到低质量 fallback
   （否则后续 Observation 全空、stage5 硬失败）。仅在 ALLOW_REAL_API=false
    （单元测试无 mock）时允许 fallback。
    """
    if not moves:
        return VideoChainContext()

    last_error: Optional[BaseException] = None
    for attempt in range(1, _VIDEO_CONTEXT_EXTRACT_ATTEMPTS + 1):
        try:
            return _extract_video_chain_context_once(moves)
        except RealAPIDisabledError:
            logger.info("ALLOW_REAL_API=false；使用 fallback_video_context")
            return fallback_video_context(moves)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "VideoContextExtraction attempt %s/%s failed: %s",
                attempt,
                _VIDEO_CONTEXT_EXTRACT_ATTEMPTS,
                exc,
            )
            if attempt >= _VIDEO_CONTEXT_EXTRACT_ATTEMPTS:
                break
    assert last_error is not None
    raise RuntimeError(
        "逐视频来源抽取失败（已重试），拒绝降级到低质量 fallback，"
        f"请检查 LLM 连通性后 --force-from stage3 重跑。根因: {last_error}"
    ) from last_error


def _extract_video_chain_context_once(moves: list[Move]) -> VideoChainContext:
    """单次完整抽取：分批以避免超大 prompt，再合并复核。"""
    batch_extractions: list[VideoContextExtraction] = []
    for start in range(0, len(moves), _VIDEO_CONTEXT_BATCH_SIZE):
        end = min(start + _VIDEO_CONTEXT_BATCH_SIZE, len(moves))
        batch = moves[start:end]
        move_lines = [
            json.dumps(
                {
                    "move_index": start + offset,
                    "start_time": move.start_time,
                    "end_time": move.end_time,
                    "narration": move.narration,
                    "screen_action": move.screen_action,
                    "visible_clues": list(move.visible_clues),
                },
                ensure_ascii=False,
            )
            for offset, move in enumerate(batch)
        ]
        prompt = (
            "从本视频答案前 Moves 中抽取 Agent1 的逐视频来源契约。不得使用任何预设"
            "地名/地貌词表，也不得补充常识或从图片自由发现事实。\n"
            "要求：\n"
            "1. raw_clues 仅取问题设置中外部直接给出的信息；每条必须填写 clue_role：\n"
            "   - photo_location_constraint：原文明确约束「拍摄地/照片地点」"
            "（含「未出X」「在X内」「拍摄地为X附近」等）；text 须为原文可核对短语；\n"
            "   - person_or_social_attribute：籍贯、居住地、身份、关系等——关于人，"
            "不是关于照片地点；\n"
            "   - other_non_location：其它非地点外部信息。\n"
            "2. 禁止把人物属性改写成硬边界「拍摄地在籍贯市内」；"
            "籍贯+「离家不远/附近」只可作为软先验「籍贯地附近」，不得写成「X内」。\n"
            "3. working_scope 须标注 bound_kind 并写准展示短语：\n"
            "   - inside：仅原文直接说拍摄地在X内/未出X → region 写「X内」；"
            "支撑 move 必须是 photo_location_constraint；\n"
            "   - near：聊天「拍摄地为X附近」，或籍贯地名+离家不远等软距离 → "
            "region 写「X附近」；支撑可为拍摄地附近约束和/或人物属性；\n"
            "   - 禁止仅有附近/软距离证据时写成「X内」；无可用约束则 working_scope=null。\n"
            "4. facts 必须是原 Move 明确说出的原子事实。claim、concepts、relation、"
            "excluded_candidates、proposed_candidates、spatial_anchor、"
            "corrected_from、corrected_to 均须复制该 Move 原文中的连续短语，"
            "不得概括、改写或补全；\n"
            "5. 每条 fact 必须标注 subject_scope（结论作用对象）：\n"
            "   camera_position=拍摄点/近处/俯视脚下等；\n"
            "   scene_region=远处/对岸/背景/画面局部等；\n"
            "   location_candidate=地点/行政区候选；\n"
            "   unknown=无法判断。不使用任何地名或地貌词表，只根据原文方位/对象短语；\n"
            "   作用域不确定时标 unknown，**不得**因 scope 犹豫而漏抽事实。\n"
            "   地点排除/候选推进必须标 location_candidate；近景/远景观察只作 "
            "supporting_move_indices，禁止把「联合收窄」标成对 camera/scene 的 correct。\n"
            "6. 召回优先：凡旁白明确说出的画面地理/空间关系（俯视、下方屋顶、高地、"
            "远山、河岸、平原、纠正误认等）必须各建一条 fact；禁止只抽开场耗时/"
            "求助元叙事而漏掉中段观察。\n"
            "7. kind 只能 observe/correct/exclude/candidate/stall；开场「花了多久」"
            "类话术不要做成 observe 地理事实（可忽略或 stall）；"
            "置顶等待、纯聊天记录、静待/消息列表等无地理操作标 stall。\n"
            "8. correct/exclude/candidate 必须列 supporting_move_indices、"
            "excluded_candidates/proposed_candidates；correct 须填 "
            "corrected_from/corrected_to（均来自原文）；\n"
            "9. proposed_candidates 仅当 subject_scope=location_candidate（或原文明确"
            "提出地点候选）时填写；背景/对岸地貌不得写入地点候选。\n"
            "10. 仅当 claim 无法从该 Move 原文逐字复制时才跳过；禁止猜测补全。\n"
            "11. move_index 必须使用输入中的全局索引。\n"
            f"本批 Moves（全局 index {start}..{end - 1}）:\n"
            + "\n".join(move_lines)
        )
        batch_extractions.append(call_structured(prompt, VideoContextExtraction))

    merged = _merge_video_context_extractions(batch_extractions)
    all_move_lines = [
        json.dumps(
            {
                "move_index": index,
                "narration": move.narration,
                "visible_clues": list(move.visible_clues),
            },
            ensure_ascii=False,
        )
        for index, move in enumerate(moves)
    ]
    review_prompt = (
        "独立复核逐视频来源抽取，不使用任何地名/地貌词表或外部常识。\n"
        "规则：\n"
        "1. raw clue：视觉观察与候选假设不得进 raw_clues；"
        "籍贯/身份等必须为 person_or_social_attribute；"
        "拍摄地硬边界或「拍摄地为X附近」可为 photo_location_constraint；"
        "禁止把人物属性误标为硬拍摄地边界。\n"
        "2. fact：仅当 claim 不被该 move_index 原文直接蕴含时才判 unsupported；"
        "subject_scope/spatial_anchor 标错**不足以**丢弃整条事实"
        "（作用域可后处理修正）。\n"
        "3. working_scope：region/bound_kind 不得比支撑线索更强或更细；"
        "inside 支撑须为 photo_location_constraint；near 可为拍摄地附近或"
        "籍贯+软距离；若把附近/籍贯软话写成「X内」则 working_scope_supported=false。\n"
        "4. 不同空间对象（如拍摄点 vs 远处/对岸）的观察结论可并存；"
        "背景地貌纠正不得被标成对拍摄点结论的撤销——此类应改 scope，而非删除事实。"
        "多条不同作用域观察共同支撑同一 location_candidate 排除/收窄为合法，"
        "不要判 unsupported。\n"
        "列出所有不受支持的数组位置。对 claim 蕴含存疑才删除；禁止因 scope 犹豫误杀。\n"
        f"Moves:\n{chr(10).join(all_move_lines)}\n"
        f"Extraction:\n{merged.model_dump_json()}"
    )
    review = call_structured(review_prompt, _VideoContextGroundingReview)
    unsupported_raw = set(review.unsupported_raw_clue_positions)
    unsupported_facts = set(review.unsupported_fact_positions)
    filtered = merged.model_copy(
        update={
            "raw_clues": [
                item
                for index, item in enumerate(merged.raw_clues)
                if index not in unsupported_raw
            ],
            "working_scope": (
                merged.working_scope if review.working_scope_supported else None
            ),
            "facts": [
                item
                for index, item in enumerate(merged.facts)
                if index not in unsupported_facts
            ],
        }
    )
    filtered = _filter_working_scope_to_photo_constraints(filtered)
    if filtered.working_scope is None:
        filtered = _derive_working_scope_from_raw_clues(filtered)
    filtered = _sanitize_extracted_working_scope(filtered)
    # 召回修复：去掉开场元叙事事实，并补全 LLM/复核漏掉的地理 Move
    from pipeline.evidence_routing import (
        drop_meta_setup_facts,
        gap_fill_missing_geo_facts,
    )

    filtered = drop_meta_setup_facts(filtered)
    filtered = gap_fill_missing_geo_facts(moves, filtered)
    ctx = context_from_extraction(moves, filtered)
    if not ctx.video_facts:
        raise RuntimeError(
            "逐视频来源抽取结果为空（复核后无有效 facts）；"
            "拒绝继续生成无证据 COARSE 链"
        )
    return ctx


def _filter_working_scope_to_photo_constraints(
    extraction: VideoContextExtraction,
) -> VideoContextExtraction:
    """按边界强度校验 working_scope 支撑；过强硬边界无拍摄地约束则丢弃。"""
    if extraction.working_scope is None:
        return extraction
    if not extraction.working_scope.region.strip():
        return extraction.model_copy(update={"working_scope": None})
    photo_moves = {
        item.move_index
        for item in extraction.raw_clues
        if item.clue_role is RawClueRole.PHOTO_LOCATION_CONSTRAINT
    }
    soft_moves = photo_moves | {
        item.move_index
        for item in extraction.raw_clues
        if item.clue_role is RawClueRole.PERSON_OR_SOCIAL_ATTRIBUTE
    }
    support = list(extraction.working_scope.supporting_move_indices)
    if not support:
        return extraction.model_copy(update={"working_scope": None})
    support_texts = [
        item.text
        for item in extraction.raw_clues
        if item.move_index in set(support)
    ]
    _phrase, bound_kind = normalize_working_scope_phrase(
        extraction.working_scope.region,
        clue_texts=support_texts,
        bound_kind=extraction.working_scope.bound_kind,
    )
    if bound_kind is ScopeBoundKind.INSIDE:
        if any(i not in photo_moves for i in support):
            return extraction.model_copy(update={"working_scope": None})
    elif any(i not in soft_moves for i in support):
        return extraction.model_copy(update={"working_scope": None})
    return extraction


def _sanitize_extracted_working_scope(
    extraction: VideoContextExtraction,
) -> VideoContextExtraction:
    """把抽取/推导的 working_scope 规范成准确展示短语。"""
    if extraction.working_scope is None:
        return extraction
    support = set(extraction.working_scope.supporting_move_indices)
    clue_texts = [
        item.text for item in extraction.raw_clues if item.move_index in support
    ]
    phrase, bound_kind = normalize_working_scope_phrase(
        extraction.working_scope.region,
        clue_texts=clue_texts,
        bound_kind=extraction.working_scope.bound_kind,
    )
    if not phrase:
        return extraction.model_copy(update={"working_scope": None})
    return extraction.model_copy(
        update={
            "working_scope": extraction.working_scope.model_copy(
                update={"region": phrase, "bound_kind": bound_kind}
            )
        }
    )


def _derive_working_scope_from_raw_clues(
    extraction: VideoContextExtraction,
) -> VideoContextExtraction:
    """当抽取未给出 working_scope 时，从拍摄地约束或软先验再推导一次。"""
    photo_clues = [
        (index, item)
        for index, item in enumerate(extraction.raw_clues)
        if item.clue_role is RawClueRole.PHOTO_LOCATION_CONSTRAINT
    ]
    person_clues = [
        (index, item)
        for index, item in enumerate(extraction.raw_clues)
        if item.clue_role is RawClueRole.PERSON_OR_SOCIAL_ATTRIBUTE
    ]
    if not photo_clues and not person_clues:
        return extraction
    # 仅人物属性时，须含软距离语义才允许推导 near
    if not photo_clues:
        soft_blob = " ".join(item.text for _, item in person_clues)
        if not re.search(r"附近|周围|周边|离家|不远|不会太远", soft_blob):
            return extraction
        usable = person_clues
    else:
        usable = []
        seen_pos: set[int] = set()
        for pair in photo_clues + person_clues:
            if pair[0] in seen_pos:
                continue
            seen_pos.add(pair[0])
            usable.append(pair)
    clue_lines = [
        json.dumps(
            {
                "position": index,
                "text": item.text,
                "move_index": item.move_index,
                "clue_role": item.clue_role.value,
            },
            ensure_ascii=False,
        )
        for index, item in usable
    ]
    prompt = (
        "根据下列 raw_clues 规范化拍摄地工作范围。"
        "不得使用常识补全未出现的地名。"
        "bound_kind=inside 仅当原文直接说拍摄地在X内/未出X，region 写「X内」。"
        "bound_kind=near 用于「拍摄地为X附近」或籍贯地名+离家不远等软先验，"
        "region 必须写「X附近」，禁止写成「X内」或「就在X市」。"
        "若不足以形成工作范围，region 留空。\n"
        + "\n".join(clue_lines)
    )
    derived = call_structured(prompt, _WorkingScopeDerivation)
    region = derived.region.strip()
    if not region:
        return extraction
    usable_index_set = {index for index, _ in usable}
    support = [
        extraction.raw_clues[i].move_index
        for i in derived.supporting_raw_clue_positions
        if i in usable_index_set
    ]
    if not support:
        support = [usable[0][1].move_index]
    phrase, bound_kind = normalize_working_scope_phrase(
        region,
        clue_texts=[item.text for _, item in usable],
        bound_kind=derived.bound_kind,
    )
    if not phrase:
        return extraction
    # 无拍摄地约束时只允许 soft near
    if not photo_clues and bound_kind is not ScopeBoundKind.NEAR:
        return extraction
    return extraction.model_copy(
        update={
            "working_scope": ExtractedWorkingScope(
                region=phrase,
                supporting_move_indices=support,
                rationale=derived.rationale.strip(),
                bound_kind=bound_kind,
            )
        }
    )

def _merge_video_context_extractions(
    parts: list[VideoContextExtraction],
) -> VideoContextExtraction:
    """合并分批抽取结果；working_scope 取首个非空。"""
    raw_clues = []
    facts = []
    scope = None
    for part in parts:
        raw_clues.extend(part.raw_clues)
        facts.extend(part.facts)
        if scope is None and part.working_scope is not None:
            scope = part.working_scope
    return VideoContextExtraction(
        raw_clues=raw_clues,
        working_scope=scope,
        facts=facts,
    )


def _normalize_coarse_with_closure(
    moves: list[Move],
    *,
    video_context: VideoChainContext,
) -> list[NormalizedStep]:
    """COARSE normalize：按作用域组织闭包，并在首步嵌入 VideoChainContext。"""
    from pipeline.evidence_routing import (
        CoarseStepKind,
        SubjectScope,
        scope_partition_key,
    )

    steps = normalize_to_steps(moves, AgentRole.COARSE)
    if not steps:
        return steps

    rewritten: list[NormalizedStep] = []
    for i, step in enumerate(steps):
        current_facts = [
            fact
            for fact in video_context.video_facts
            if abs(fact.start_time - float(step.move.start_time)) < 1e-6
        ]
        primary = current_facts[0] if current_facts else None
        primary_key = (
            scope_partition_key(primary.subject_scope, primary.spatial_anchor)
            if primary is not None
            else None
        )
        source_indices = {
            index
            for fact in current_facts
            for index in fact.supporting_move_indices
        }
        cited_facts = []
        for fact in video_context.video_facts:
            if fact in current_facts:
                cited_facts.append(fact)
                continue
            if fact.source_move_index not in source_indices:
                continue
            # 支撑事实仅同作用域分区可并入本步更新域；避免远处纠正混入拍摄点
            if primary_key is None:
                cited_facts.append(fact)
                continue
            if (
                scope_partition_key(fact.subject_scope, fact.spatial_anchor)
                == primary_key
            ):
                cited_facts.append(fact)
                continue
            # location_candidate 可引用同候选域支撑；其余跨域只作旁证不并 token
            if (
                primary is not None
                and primary.subject_scope is SubjectScope.LOCATION_CANDIDATE
                and fact.subject_scope is SubjectScope.LOCATION_CANDIDATE
            ):
                cited_facts.append(fact)
        step_concepts = [
            concept
            for fact in cited_facts
            for concept in fact.tokens
            if concept
        ]
        step_claims = [
            fact.quote.strip()
            for fact in cited_facts
            if fact.quote.strip()
        ]
        kinds = {fact.kind for fact in current_facts}
        step_kind = (
            CoarseStepKind.UPDATE
            if kinds & {"correct", "exclude", "candidate"}
            else CoarseStepKind.OBSERVE
        )
        intent = _intent_for_move(
            step.move,
            SemanticRoute.COARSE,
            source_concepts=list(dict.fromkeys(step_concepts)),
            video_fact_ids=[fact.fact_id for fact in cited_facts],
            step_kind=step_kind,
        )
        intent.target_features = list(dict.fromkeys(step_concepts))
        intent.source_claims = list(dict.fromkeys(step_claims))
        if primary is not None:
            intent.subject_scope = primary.subject_scope
            intent.spatial_anchor = primary.spatial_anchor
            intent.expected_spatial_relation = next(
                (fact.relation for fact in current_facts if fact.relation),
                intent.expected_spatial_relation,
            )
        draft = _build_thought_draft(step.move, intent=intent)
        if i == 0:
            draft = embed_video_context(draft, video_context)
        rewritten.append(step.model_copy(update={"thought_draft": draft}))
    return rewritten
