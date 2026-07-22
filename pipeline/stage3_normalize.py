"""stage3：Move → NormalizedStep（匹配 / 组合 / 注册新 Tool / fallback / thought_only）。

本阶段不生成 Observation（Observation 属于 stage4）。
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from pipeline.llm import call_structured
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

# 纯 UI 操作关键词（G6）：不得注册新 Tool，也不得无标记硬套为 Tool
_PURE_UI_RE = re.compile(
    r"滚动(?:页面|条)?|scroll(?:ing)?|"
    r"移动鼠标|mouse\s*move|mousemove|"
    r"切换(?:浏览器)?标签|switch\s*(?:browser\s*)?tab|"
    r"拖拽窗口|resize\s*window|"
    r"点击空白|hover(?:ing)?",
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

# 旁白中的第一人称/叙事套话，生成 thought_draft 时剥离
_NARRATION_FILLER_RE = re.compile(
    r"为了找到这张照片的拍摄地[,，]?我足足花了半年的时间[。.]?|"
    r"当我知道答案的那一刻起[,，]?|"
    r"我才发现[,，]?|"
    r"半年前一位粉丝向我求助[,，]?|"
    r"想让我帮忙找一下[,，]?|"
    r"希望我能找出来[,，]?|"
    r"勾起年少时的回忆[。.]?",
    re.IGNORECASE,
)


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


def _is_pure_ui(screen_action: str) -> bool:
    """判断是否为纯 UI 噪声操作（G6）。"""
    text = screen_action.strip()
    if not text:
        return False
    if not _PURE_UI_RE.search(text):
        return False
    # 同时含明显 tool 语义词时，不视为纯 UI
    return _TOOLISH_RE.search(text) is None


def _build_thought_draft(move: Move) -> str:
    """生成极短视觉/操作线索；禁止整段旁白叙事进入草稿。

    优先 screen_action；旁白仅抽取短名词短语（去人称/故事套话后截断）。
    """
    if move.screen_action and move.screen_action.strip():
        sa = move.screen_action.strip()
        if len(sa) > 40:
            sa = sa[:37].rstrip() + "…"
        return f"操作线索：{sa}"

    narration = (move.narration or "").strip()
    if narration:
        cleaned = _NARRATION_FILLER_RE.sub("", narration)
        # 去掉明显人称/求助叙事残留
        cleaned = re.sub(
            r"(?:求助者|粉丝|博主|父亲|我|我们|咱们)[^。．.!！？?]{0,40}[。．.!！？?]?",
            "",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。.;；")
        # 只保留较短片段，避免故事骨架
        if len(cleaned) > 48:
            cleaned = cleaned[:45].rstrip() + "…"
        if cleaned and len(cleaned) >= 4:
            return f"视觉线索：{cleaned}"
    return "继续基于画面推理。"


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

# Agent1 训练轨迹最终保留的固定 Tool（与 stage5 投影对齐）
_COARSE_TRAINING_TOOLS: frozenset[str] = frozenset(
    {"zoom_inspect", "ocr", "sun_position_calc"}
)
_COARSE_EVIDENCE_ONLY_NAME_RE = re.compile(
    r"^(?:web_search|map_query|reverse_image_search)$|compare_images|"
    r"satellite|map_compare|image_pair",
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


def _is_coarse_evidence_only_tool(tool_name: str) -> bool:
    """COARSE 在 stage3 可匹配、但 stage5 必剔除的工具。"""
    return bool(_COARSE_EVIDENCE_ONLY_NAME_RE.search(tool_name))


def _coarse_decompose_to_training_actions(
    screen_action: str,
    narration: str,
    tool_by_name: dict[str, ToolDefinition],
) -> Optional[list[Action]]:
    """将检索/双图等证据侧语义分解为有图像依据的训练 Tool 组合。

    仅在语义前置条件满足时添加 OCR/sun；禁止无依据凑步。
    """
    text = f"{screen_action} {narration}".strip()
    if not text:
        return None
    proposed: list[_ProposedAction] = []
    needs_visual = bool(
        _VISUAL_EVIDENCE_RE.search(text)
        or re.search(r"搜索|search|检索|比对|对比|卫星|地图", text, re.I)
    )
    # 视觉/比对/检索 → zoom（不同默认关注区，避免重复全图）
    if needs_visual and "zoom_inspect" in tool_by_name:
        bbox = list(_DEFAULT_ZOOM_BBOX)
        if re.search(r"远|山|背景|horizon|bridge|桥", text, re.I):
            bbox = [0.0, 0.0, 1.0, 0.45]
        elif re.search(r"近|栏杆|人物|前景|foreground", text, re.I):
            bbox = [0.2, 0.35, 0.6, 0.55]
        proposed.append(
            _ProposedAction(tool="zoom_inspect", params={"bbox": bbox})
        )
    # 仅文本区域语义才加 OCR
    if _TEXT_EVIDENCE_RE.search(text) and "ocr" in tool_by_name:
        proposed.append(
            _ProposedAction(
                tool="ocr",
                params={"bbox": [0.15, 0.15, 0.7, 0.35]},
            )
        )
    # 仅阴影/日照语义才加 sun
    if _SUN_EVIDENCE_RE.search(text) and "sun_position_calc" in tool_by_name:
        proposed.append(_ProposedAction(tool="sun_position_calc", params={}))

    if not proposed:
        return None
    # 去重同 tool
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
) -> tuple[list[Action], NormalizationMode, Optional[str]]:
    """COARSE：证据侧禁止 Tool 优先分解为训练 Tool；失败则保留原证据步。"""
    if not actions:
        return actions, mode, None
    if all(a.tool in _COARSE_TRAINING_TOOLS for a in actions):
        return actions, mode, None
    if not any(a.tool not in _COARSE_TRAINING_TOOLS for a in actions):
        return actions, mode, None

    decomposed = _coarse_decompose_to_training_actions(
        screen_action, narration, tool_by_name
    )
    if decomposed:
        new_mode = (
            NormalizationMode.COMPOSED
            if len(decomposed) >= 2
            else NormalizationMode.MATCHED
        )
        return decomposed, new_mode, None
    # 无法可靠分解：保留证据侧原始步骤（stage5 投影剔除，不伪装训练步）
    return actions, mode, None


def _looks_toolish(screen_action: str) -> bool:
    """screen_action 是否含明显 Tool 语义（非纯 UI）。"""
    text = (screen_action or "").strip()
    if not text or _is_pure_ui(text):
        return False
    return _TOOLISH_RE.search(text) is not None


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
    if re.search(r"太阳|阴影|sun\s*position|shadow", text):
        if agent_role == AgentRole.COARSE and "sun_position_calc" in tool_by_name:
            proposed = _ProposedAction(tool="sun_position_calc", params={})
    elif re.search(r"地图|街景|map\s*query|打开地图", text):
        if agent_role in (AgentRole.FINE, AgentRole.VERIFIER) and "map_query" in tool_by_name:
            proposed = _ProposedAction(tool="map_query", params={"query": query})
        elif agent_role == AgentRole.COARSE and "web_search" in tool_by_name:
            # COARSE 无权 map_query：降级为检索
            proposed = _ProposedAction(
                tool="web_search",
                params={
                    "query": query,
                    "purpose": _WEB_SEARCH_DEFAULT_PURPOSE[agent_role],
                },
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
        "\nAgent1 不得使用 reverse_image_search / map_query；"
        "Agent2 不得使用 sun_position_calc。"
        "\nCOARSE 训练轨迹最终仅保留 zoom_inspect/ocr/sun_position_calc："
        "优先 composed 这三者表达画面观察；勿为 COARSE 注册 compare_images/"
        "双图比对类 Tool；仅有文字区域才加 ocr，仅有阴影/日照才加 sun。"
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

    # COARSE：禁止注册双图比对类；改为训练 Tool 分解
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
            "COARSE 禁止注册 compare_images/检索类新 Tool，且无法分解为训练 Tool",
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
    """
    steps: list[NormalizedStep] = []

    for move in moves:
        thought = _build_thought_draft(move)
        screen = move.screen_action

        if screen is None or not str(screen).strip():
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

        # 每次决策前重载 registry（新 Tool 可能刚写入）
        existing_tools = list(load_registry().values())
        actions, mode, reason, confidence = _match_or_register_tool_impl(
            screen_action=str(screen),
            narration=move.narration,
            agent_role=agent_role,
            existing_tools=existing_tools,
            all_moves=moves,
        )

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

    return steps
