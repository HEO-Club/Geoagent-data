"""stage3：Move → NormalizedStep（匹配 / 组合 / 建 Draft / fallback / thought_only）。

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
    ToolTier,
)
from pipeline.tools.registry import load_registry, register_tool
from pipeline.tools.validation import validate_action_params

# 纯 UI 操作关键词（G6）：不得建 Draft，也不得无标记硬套为 Tool
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


class _ProposedAction(BaseModel):
    """LLM 提出的单次 Tool 调用。"""

    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class _GRuleFlags(BaseModel):
    """G 规则八项评估；建 Draft 时必须全部为 True。"""

    cannot_match_existing: bool
    cannot_compose: bool
    io_semantics_clear: bool
    reusable_in_geolocation: bool
    future_executor_possible: bool
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
                self.future_executor_possible,
                self.not_pure_ui,
                self.not_one_off_for_video,
                self.not_similar_to_existing,
            )
        )


class _DraftToolProposal(BaseModel):
    """LLM 提议的 Draft Tool schema（注册前再经 ToolDefinition 校验）。"""

    name: str
    description: str
    params: list[ParamField]
    observation_fields: list[ObservationField]
    allowed_agents: list[AgentRole]
    derived_from_existing_tools: list[str] = Field(default_factory=list)


class _MatchLLMResponse(BaseModel):
    """match_or_register_tool 的结构化 LLM 决策。"""

    decision: Literal["matched", "composed", "draft_created", "fallback"]
    actions: list[_ProposedAction] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fallback_reason: Optional[str] = None
    g_flags: Optional[_GRuleFlags] = None
    draft_tool: Optional[_DraftToolProposal] = None


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
    """由旁白生成 thought_draft；旁白为空时回退到 screen_action。"""
    narration = (move.narration or "").strip()
    if narration:
        return narration
    if move.screen_action and move.screen_action.strip():
        return f"执行屏幕操作：{move.screen_action.strip()}"
    return "继续推理。"


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
    proposal: _DraftToolProposal,
    narration: str,
    agent_role: AgentRole,
) -> ToolDefinition:
    """将 LLM Draft 提案转为可注册的 ToolDefinition。"""
    allowed = list(proposal.allowed_agents)
    if agent_role not in allowed:
        allowed.append(agent_role)
    payload: dict[str, Any] = {
        "name": proposal.name,
        "description": proposal.description,
        "tier": ToolTier.DRAFT,
        "params": [p.model_dump(mode="json") for p in proposal.params],
        "observation_fields": [o.model_dump(mode="json") for o in proposal.observation_fields],
        "allowed_agents": allowed,
        "is_terminal": False,
        "executor_ref": None,
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
        "或在满足 G 规则时创建 Draft Tool；否则 fallback。"
        "禁止无标记硬套；禁止把纯 UI 操作（滚动/移鼠标/切标签）建成 Tool。"
        f"\n当前 AgentRole: {agent_role.value}"
        f"\nscreen_action: {screen_action}"
        f"\nnarration: {narration}"
        f"\nall_moves 中相似 screen_action 条数（含自身参考）: {similar_n}"
        f"\n现有 Tool 列表:\n{_tools_catalog_text(existing_tools, agent_role)}"
        "\n决策规则："
        "\n- matched: 单一现有 Tool，填写 actions（1 个）与 confidence"
        "\n- composed: 多个现有 Tool 组合，actions 长度≥2"
        "\n- draft_created: 仅当无法匹配且无法组合，且 g_flags 八项全 True；"
        "同时给出 draft_tool（名称须小写 snake_case、至少两 token；"
        f"命名可参考动词提示: {verbs}）"
        "\n- fallback: 无法安全映射时给出 fallback_reason"
        "\nweb_search.purpose 角色约束："
        "COARSE→broad_discovery；FINE→broad_discovery|precise_lookup；"
        "VERIFIER→verification。"
        "\nAgent1 不得使用 reverse_image_search / map_query；"
        "Agent2 不得使用 sun_position_calc。"
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
            "纯 UI 操作（滚动/移鼠标/切标签等），禁止硬套或建 Draft",
            None,
        )

    tool_by_name = {t.name: t for t in existing_tools}

    try:
        decision = _call_match_llm(
            text, narration, agent_role, existing_tools, all_moves
        )
    except Exception as exc:  # noqa: BLE001 — LLM/校验失败降级为 fallback
        return [], NormalizationMode.FALLBACK, f"LLM 决策失败: {exc}", None

    mode_map = {
        "matched": NormalizationMode.MATCHED,
        "composed": NormalizationMode.COMPOSED,
        "draft_created": NormalizationMode.DRAFT_CREATED,
        "fallback": NormalizationMode.FALLBACK,
    }
    mode = mode_map[decision.decision]
    confidence = decision.confidence

    if mode is NormalizationMode.FALLBACK:
        reason = decision.fallback_reason or "模型判定无法安全映射"
        return [], mode, reason, None

    if mode in (NormalizationMode.MATCHED, NormalizationMode.COMPOSED):
        try:
            if mode is NormalizationMode.COMPOSED and len(decision.actions) < 2:
                return (
                    [],
                    NormalizationMode.FALLBACK,
                    "composed 决策但 actions 少于 2，禁止硬套",
                    None,
                )
            if mode is NormalizationMode.MATCHED and len(decision.actions) != 1:
                return (
                    [],
                    NormalizationMode.FALLBACK,
                    "matched 决策必须恰好 1 个 Action",
                    None,
                )
            actions = _validate_actions(decision.actions, tool_by_name, agent_role)
            if confidence is None:
                confidence = 0.8 if mode is NormalizationMode.MATCHED else 0.7
            return actions, mode, None, confidence
        except (ValueError, PermissionError, ValidationError) as exc:
            return [], NormalizationMode.FALLBACK, f"Action 校验失败: {exc}", None

    # draft_created
    if decision.g_flags is None or not decision.g_flags.all_passed():
        return (
            [],
            NormalizationMode.FALLBACK,
            "不满足 G 规则全部条件，禁止创建 Draft",
            None,
        )
    if decision.draft_tool is None:
        return [], NormalizationMode.FALLBACK, "draft_created 但缺少 draft_tool", None
    if _is_pure_ui(text) or not decision.g_flags.not_pure_ui:
        return [], NormalizationMode.FALLBACK, "纯 UI 操作禁止创建 Draft", None

    try:
        draft_def = _proposal_to_tool_definition(
            decision.draft_tool, narration, agent_role
        )
        register_tool(draft_def)
        refreshed = load_registry()
        tool_by_name = dict(refreshed)

        if decision.actions and decision.actions[0].tool == draft_def.name:
            actions = _validate_actions(decision.actions[:1], tool_by_name, agent_role)
        else:
            example_params: dict[str, Any] = {
                p.name: p.example for p in draft_def.params if p.required
            }
            for p in draft_def.params:
                if not p.required and p.default is not None:
                    example_params.setdefault(p.name, p.default)
            actions = _validate_actions(
                [_ProposedAction(tool=draft_def.name, params=example_params)],
                tool_by_name,
                agent_role,
            )
        return actions, NormalizationMode.DRAFT_CREATED, None, confidence
    except (ValueError, PermissionError, ValidationError) as exc:
        return (
            [],
            NormalizationMode.FALLBACK,
            f"Draft 注册或参数校验失败: {exc}",
            None,
        )


def match_or_register_tool(
    screen_action: str,
    narration: str,
    agent_role: AgentRole,
    existing_tools: list[ToolDefinition],
    all_moves: list[Move],
) -> tuple[list[Action], NormalizationMode, Optional[str]]:
    """按 G 规则决定：匹配 / 组合 / 建 Draft / fallback。

    同时检查 allowed_agents 与 web_search.purpose 角色约束。
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

        # 每次决策前重载 registry（Draft 可能刚写入）
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
