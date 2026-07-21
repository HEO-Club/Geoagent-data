"""stage5：三 Agent 主轨迹与 revision 轨迹重构。

本阶段禁止访问 groundtruth；函数签名不得包含 groundtruth。
LLM 改写前向推理 Thought，并产出角色结构化输出；
若 FINE 脚手架缺少 terminal submit_answer，则基于证据合成该步（仍禁止 GT）。
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from pipeline.llm import call_structured
from pipeline.schemas import (
    Action,
    AgentRole,
    LocationHypothesis,
    Move,
    NormalizationMode,
    NormalizedStep,
    ObservationExecutionResult,
    RevisionContext,
    RevisionSource,
    SEED_TOOL_NAMES,
    SubmitAnswerResult,
    Trajectory,
    TrajectoryStep,
    VerificationResult,
)
from pipeline.tao_style_examples import fewshot_block_for_role
from pipeline.tools.base import execute_action
from pipeline.tools.registry import load_registry
from pipeline.tools.validation import validate_action_params

# ---------------------------------------------------------------------------
# LLM 结构化输出（仅本模块内部使用）
# ---------------------------------------------------------------------------

# Agent1 训练轨迹 Tool 投影
_COARSE_FIXED_TOOLS: frozenset[str] = frozenset(
    {"zoom_inspect", "ocr", "sun_position_calc"}
)
_COARSE_FORBIDDEN_SEED_TOOLS: frozenset[str] = frozenset(
    {"web_search", "map_query", "reverse_image_search", "submit_answer"}
)


class _RewrittenTrajectory(BaseModel):
    """LLM 改写后的逐步 Thought（与可展开 Action 步一一对应）。"""

    thoughts: list[str] = Field(min_length=1)


class _CoarseOutputBundle(BaseModel):
    """Agent1：改写 Thought + LocationHypothesis。"""

    thoughts: list[str] = Field(min_length=1)
    coarse_output: LocationHypothesis


class _VerifierOutputBundle(BaseModel):
    """Agent3：改写 Thought + VerificationResult（把 fine_handoff 当候选）。"""

    thoughts: list[str] = Field(min_length=1)
    verifier_output: VerificationResult


class _TaoStyleCheck(BaseModel):
    """stage5 改写后的 TAO 形态自检。"""

    is_standard_tao: bool
    issues: list[str] = Field(default_factory=list)


class _ExternalHints(BaseModel):
    """从答案前旁白抽取的外部给定地名线索（非推理、非真值）。"""

    hints: list[str] = Field(default_factory=list)


class _CoarseToolSuitability(BaseModel):
    """动态 Tool 是否适合进入 Agent1 训练轨迹。"""

    suitable_for_coarse_reasoning: bool
    reason: str = ""


class _CoarseReasoningCheck(BaseModel):
    """stage5 COARSE 递进推理链自检。"""

    identifies_geo_human_features: bool
    narrows_scope_progressively: bool
    has_reasoning_gap: bool
    thought_action_aligned: bool
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 角色提示（不含任何真值坐标 / 地名）
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS: dict[AgentRole, str] = {
    AgentRole.COARSE: (
        "你是粗定位 Agent（COARSE）。识别地理/人文特征，用地理常识演绎与排除，"
        "按「特征识别 → 候选排除/范围收窄 → 下一步验证」逐步缩小到国家/地区。"
        "Thought 必须指出具体特征并说明支持/排除哪些范围；禁止跳步；"
        "禁止视频旁白叙事；禁止最终精确坐标/POI；不得使用 web_search。"
    ),
    AgentRole.FINE: (
        "你是精定位 Agent（FINE）。在粗定位假设基础上收窄到具体地点；"
        "缩小范围无上限——若画面、Observation 或 user_query 线索已足够，"
        "可尽早提出较精确地点/坐标假设并核实。"
        "Thought 必须是标准图片地理定位推理；最后一步必须 submit_answer。"
        "禁止旁白叙事、后见之明与无依据粘贴真值；不要在 Thought 中解释线索来源。"
    ),
    AgentRole.VERIFIER: (
        "你是验证 Agent（VERIFIER）。把候选 SubmitAnswerResult 与图像特征交叉验证。"
        "Thought 必须是标准验证推理；可复述候选，不得把真值当作已知答案。"
    ),
}


def _new_traj_id(agent_role: AgentRole, *, is_revision: bool = False) -> str:
    """生成轨迹 id。"""
    prefix = "rev" if is_revision else "main"
    return f"{prefix}-{agent_role.value}-{uuid.uuid4().hex[:10]}"


def _expand_action_units(
    steps: list[NormalizedStep],
    observations: list[ObservationExecutionResult],
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """将 NormalizedStep 与 Observation 按 Action 展开对齐。

    thought_only（actions=[]）不产生 TrajectoryStep，其 thought_draft 并入
    后续可展开步的上下文（由调用方拼入 prompt）。
    """
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]] = []
    obs_idx = 0
    pending_thoughts: list[str] = []

    for step in steps:
        if not step.actions:
            if step.thought_draft.strip():
                pending_thoughts.append(step.thought_draft.strip())
            continue
        for action in step.actions:
            if obs_idx >= len(observations):
                raise ValueError(
                    f"Observation 数量不足：需要至少 {obs_idx + 1} 条，实际 {len(observations)}"
                )
            obs = observations[obs_idx]
            obs_idx += 1
            if obs.action.tool != action.tool:
                raise ValueError(
                    f"Action/Observation 不对齐：期望 tool={action.tool!r}，"
                    f"得到 {obs.action.tool!r}（index={obs_idx - 1}）"
                )
            thought_parts = pending_thoughts + [step.thought_draft]
            pending_thoughts = []
            merged_thought = "\n".join(p for p in thought_parts if p.strip())
            units.append((merged_thought, action, obs, step))

    if obs_idx != len(observations):
        raise ValueError(
            f"Observation 未完全消费：已用 {obs_idx}，共 {len(observations)}"
        )
    if not units:
        raise ValueError("无可重构的 Action 步（全为 thought_only 或空列表）")
    return units


def _judge_dynamic_tool_for_coarse(tool_name: str) -> bool:
    """判定非种子动态 Tool 是否适合 Agent1 演绎推理；失败则 fail-closed。"""
    registry = load_registry()
    tool = registry.get(tool_name)
    if tool is None:
        return False
    param_desc = ", ".join(
        f"{p.name}:{p.type}" for p in tool.params
    ) or "(none)"
    obs_desc = ", ".join(
        f"{o.name}:{o.type}" for o in tool.observation_fields
    ) or "(none)"
    prompt = (
        "判断该 Tool 是否适合进入 Agent1（粗定位）训练轨迹。\n"
        "适合条件：直接服务于地理/人文特征观察或演绎推断"
        "（如视觉细节、文字、阴影/日照、植被/建筑比对）。\n"
        "不适合：检索网页、地图查询、提交答案、依赖外部答案库等。\n"
        f"name: {tool.name}\n"
        f"description: {tool.description}\n"
        f"params: [{param_desc}]\n"
        f"observation_fields: [{obs_desc}]\n"
        f"derived_from_existing_tools: {tool.derived_from_existing_tools}\n"
    )
    try:
        result = call_structured(prompt, _CoarseToolSuitability)
    except Exception:  # noqa: BLE001
        return False
    return bool(result.suitable_for_coarse_reasoning)


def _project_coarse_units(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """将全量 COARSE Action/Obs 投影为 Agent1 训练子集。

    固定保留 zoom_inspect/ocr/sun_position_calc；硬排除 web_search 等种子禁止 Tool；
    动态 Tool 经结构化适配判定后可选保留。保持原序与 A/O 对齐。
    """
    projected: list[
        tuple[str, Action, ObservationExecutionResult, NormalizedStep]
    ] = []
    suitability_cache: dict[str, bool] = {}
    for unit in units:
        _draft, action, _obs, _step = unit
        name = action.tool
        if name in _COARSE_FIXED_TOOLS:
            projected.append(unit)
            continue
        if name in _COARSE_FORBIDDEN_SEED_TOOLS or name in SEED_TOOL_NAMES:
            # 其他种子 Tool（含禁止列表）一律不进训练轨迹
            continue
        if name not in suitability_cache:
            suitability_cache[name] = _judge_dynamic_tool_for_coarse(name)
        if suitability_cache[name]:
            projected.append(unit)
    if not projected:
        raise ValueError(
            "COARSE Tool 投影后无可重构 Action 步"
            "（仅剩 web_search 等禁止 Tool 或无适配动态 Tool）"
        )
    return projected


def _format_unit_for_prompt(
    index: int,
    thought_draft: str,
    action: Action,
    obs: ObservationExecutionResult,
) -> str:
    """将单步脚手架写成 prompt 片段（不含 groundtruth）。"""
    obs_repr: Any
    if obs.status == "skipped" or action.tool == "submit_answer":
        obs_repr = None
    else:
        obs_repr = obs.observation
    return (
        f"### Step {index}\n"
        f"thought_draft: {thought_draft}\n"
        f"action: tool={action.tool} params={action.params!r}\n"
        f"observation_status: {obs.status}\n"
        f"observation: {obs_repr!r}\n"
    )


def _build_scaffold_prompt(
    agent_role: AgentRole,
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    answer_timestamp: float,
    *,
    coarse_handoff: Optional[LocationHypothesis],
    fine_handoff: Optional[SubmitAnswerResult],
    revision_context: Optional[RevisionContext],
    user_query: Optional[str] = None,
) -> str:
    """构造轨迹改写 prompt；强制标准地理定位 TAO，禁止旁白叙事。"""
    lines = [
        "请将下列脚手架改写为标准图片地理定位 ReAct 推理链。",
        "输出的 thoughts 必须且只能是标准地理定位 TAO Thought（见下方风格规范）。",
        "要求：",
        "1. thoughts 列表长度必须与 Step 数量完全一致；",
        "2. Thought 可使用该步及之前 Observation，以及 user_query 中的已知线索；",
        "3. 禁止后见之明；禁止无图像/Obs/user_query 依据地粘贴真值；",
        "4. Thought 主语是画面/线索，禁止照抄旁白或博主/求助者/粉丝叙事；",
        "5. thought_draft 仅为可选短线索，禁止复述其中的故事或人称；",
        "6. 本步 Thought 不得把本步 Observation 当作已知；",
        "7. 可使用 user_query 中的地名线索，但不要在 Thought 里解释线索来源"
        "（禁止「网友说/评论说」等）。",
        f"agent_role: {agent_role.value}",
        f"answer_timestamp: {answer_timestamp}",
    ]
    if agent_role == AgentRole.FINE:
        lines.append(
            "8. FINE：缩小范围无上限；证据足够时可尽早写出较精确地点/坐标假设。"
        )
    if agent_role == AgentRole.COARSE:
        lines.append(
            "8. COARSE：每步须「特征识别 → 排除/收窄 → 为何调用本工具」；"
            "不得跳步；不得用 web_search；结论仅国家/地区级；"
            "coarse_output.reasoning_summary 须概括特征→排除/收窄→候选范围。"
        )
    if user_query:
        lines.append(f"user_query（任务与已知线索）: {user_query}")
    lines.extend(
        [
            "\n## 风格规范与示例",
            fewshot_block_for_role(agent_role),
        ]
    )
    if agent_role in (AgentRole.COARSE, AgentRole.FINE):
        lines.append(
            "时间规则：COARSE/FINE 默认只使用 answer_timestamp 之前的证据。"
        )
    else:
        lines.append(
            "时间规则：VERIFIER 可使用答案宣布后的验证片段，"
            "但博主直接宣布答案的语句不能作为验证证据。"
        )
        lines.append(
            "验证深度：须交叉核对 fine_handoff 与地图/检索 Observation 及图像特征，"
            "再给出 VerificationResult。"
        )

    if coarse_handoff is not None:
        lines.append(f"coarse_handoff: {coarse_handoff.model_dump_json()}")
    if fine_handoff is not None:
        lines.append(
            "fine_handoff（候选答案，非真值）: "
            f"{fine_handoff.model_dump_json()}"
        )
    if revision_context is not None:
        lines.append(
            "revision_context: "
            f"source={revision_context.source.value} "
            f"round={revision_context.revision_round} "
            f"target={revision_context.target_agent.value}"
        )
        if revision_context.verification_result is not None:
            lines.append(
                "previous_verification: "
                f"{revision_context.verification_result.model_dump_json()}"
            )
        if revision_context.video_segment is not None:
            lines.append(f"video_segment: {revision_context.video_segment!r}")

    lines.append("\n## Scaffold")
    for i, (thought, action, obs, _step) in enumerate(units, start=1):
        lines.append(_format_unit_for_prompt(i, thought, action, obs))
    return "\n".join(lines)


def _narration_overlap_ratio(thought: str, narration: str) -> float:
    """粗略字面重叠率（字符 bigram Jaccard）；用于旁白照抄检测。"""
    a = re.sub(r"\s+", "", thought)
    b = re.sub(r"\s+", "", narration)
    if len(a) < 8 or len(b) < 8:
        return 0.0
    def bigrams(s: str) -> set[str]:
        return {s[i : i + 2] for i in range(len(s) - 1)}
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def _thoughts_too_similar_to_narration(
    thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    *,
    threshold: float = 0.55,
) -> bool:
    """任一改写 Thought 与对应步旁白高度重叠则 True。"""
    for thought, (_draft, _action, _obs, step) in zip(thoughts, units, strict=True):
        narr = (step.move.narration or "").strip()
        if narr and _narration_overlap_ratio(thought, narr) >= threshold:
            return True
    return False


def _check_tao_style(
    thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    agent_role: AgentRole,
) -> _TaoStyleCheck:
    """LLM 判定改写 Thought 是否为标准地理定位 TAO。"""
    lines = [
        "判定下列改写后的 thoughts 是否为标准图片地理定位 TAO。",
        fewshot_block_for_role(agent_role),
        f"agent_role: {agent_role.value}",
        "\n## Rewritten thoughts",
    ]
    for i, (thought, (_d, action, _o, _s)) in enumerate(
        zip(thoughts, units, strict=True), start=1
    ):
        lines.append(f"Step {i} tool={action.tool}: {thought}")
    return call_structured("\n".join(lines), _TaoStyleCheck)


def _check_coarse_reasoning(
    thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    coarse_output: Optional[LocationHypothesis],
) -> _CoarseReasoningCheck:
    """LLM 判定 COARSE 是否为严密递进推理链（不含 GT）。"""
    lines = [
        "判定下列 COARSE 轨迹是否为严密的「特征识别 → 缩小范围」递进链。",
        fewshot_block_for_role(AgentRole.COARSE),
        "判定要点：",
        "1. identifies_geo_human_features：Thought 是否指出画面或此前 Obs 的具体地理/人文特征；",
        "2. narrows_scope_progressively：是否逐步排除/收窄范围，而非跳步到国家/地区；",
        "3. has_reasoning_gap：是否存在无依据跳步、本步 Obs 时序倒置、单一弱特征直接结论；",
        "4. thought_action_aligned：每步 Thought 是否解释为何调用该 Action；",
        "5. coarse_output.reasoning_summary 须概括特征→排除/收窄→候选范围。",
        "\n## Steps",
    ]
    for i, (thought, (_d, action, obs, _s)) in enumerate(
        zip(thoughts, units, strict=True), start=1
    ):
        obs_brief = None if obs.status == "skipped" else obs.observation
        lines.append(
            f"Step {i}: tool={action.tool}; thought={thought!r}; "
            f"observation={obs_brief!r}"
        )
    if coarse_output is not None:
        lines.append(f"\ncoarse_output: {coarse_output.model_dump_json()}")
    return call_structured("\n".join(lines), _CoarseReasoningCheck)


def _needs_tao_rewrite(
    thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    agent_role: AgentRole,
    *,
    coarse_output: Optional[LocationHypothesis] = None,
) -> tuple[bool, list[str]]:
    """字面旁白重叠、TAO 形态失败或 COARSE 递进链失败 → 需要重写。"""
    issues: list[str] = []
    need = False
    if _thoughts_too_similar_to_narration(thoughts, units):
        issues.append("Thought 与旁白字面高度重叠")
        need = True
    try:
        check = _check_tao_style(thoughts, units, agent_role)
        if not check.is_standard_tao:
            issues.extend(check.issues or ["非标准地理定位 TAO"])
            need = True
    except Exception as exc:  # noqa: BLE001
        # 形态检查失败不强制重写；交给 stage6
        issues.append(f"TAO 形态自检调用失败: {exc}")

    if agent_role == AgentRole.COARSE:
        try:
            chain = _check_coarse_reasoning(thoughts, units, coarse_output)
            if (
                not chain.identifies_geo_human_features
                or not chain.narrows_scope_progressively
                or chain.has_reasoning_gap
                or not chain.thought_action_aligned
            ):
                issues.extend(chain.issues or ["COARSE 递进推理链不合格"])
                if chain.has_reasoning_gap:
                    issues.append("存在推理跳步")
                need = True
        except Exception as exc:  # noqa: BLE001
            issues.append(f"COARSE 递进链自检调用失败: {exc}")
    return need, issues


def _to_trajectory_steps(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    thoughts: list[str],
) -> list[TrajectoryStep]:
    """脚手架 + 改写 Thought → TrajectoryStep；terminal 步 observation 均为 None。"""
    if len(thoughts) != len(units):
        raise ValueError(
            f"thoughts 长度 {len(thoughts)} 与 Action 步数 {len(units)} 不一致"
        )
    out: list[TrajectoryStep] = []
    for thought, ( _draft, action, obs, _step) in zip(thoughts, units, strict=True):
        is_terminal = action.tool == "submit_answer" or obs.status == "skipped"
        out.append(
            TrajectoryStep(
                thought=thought.strip(),
                action=action,
                observation=None if is_terminal else obs.observation,
                observation_source=None if is_terminal else obs.source,
            )
        )
    return out


def _synthesize_submit_answer(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    *,
    coarse_handoff: LocationHypothesis,
    image_path: str,
    answer_timestamp: float,
) -> SubmitAnswerResult:
    """根据脚手架证据结构化生成 SubmitAnswerResult（禁止使用 groundtruth）。"""
    evidence_lines = [
        "FINE 轨迹缺少 terminal submit_answer。请仅根据下列脚手架证据"
        "生成 SubmitAnswerResult，作为最后一步 submit_answer 的 params。",
        "约束：",
        "1. 不得使用 groundtruth、真值地名或由真值反推的地址；",
        "2. 坐标优先采用成功 map_query Observation 中的 resolved_latlng；",
        "3. location_name / reasoning 只能来自 thought_draft 与 Observation 已出现信息；",
        "4. 禁止编造 Observation 中完全不支持的精确坐标。",
        f"answer_timestamp: {answer_timestamp}",
        f"coarse_handoff: {coarse_handoff.model_dump_json()}",
        "\n## Scaffold evidence",
    ]
    for i, (thought, action, obs, _step) in enumerate(units, start=1):
        evidence_lines.append(_format_unit_for_prompt(i, thought, action, obs))
    return call_structured(
        "\n".join(evidence_lines),
        SubmitAnswerResult,
        images=[image_path],
    )


def _ensure_fine_terminal_submit(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    *,
    coarse_handoff: LocationHypothesis,
    image_path: str,
    answer_timestamp: float,
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """若 FINE 末步不是 submit_answer，则基于证据合成并追加 terminal 步。

    Action/Observation 骨架仍由程序构造；LLM 只产出 SubmitAnswerResult params。
    """
    if not units:
        raise ValueError("FINE 无可重构的 Action 步，无法合成 submit_answer")
    if units[-1][1].tool == "submit_answer":
        return units

    submit_result = _synthesize_submit_answer(
        units,
        coarse_handoff=coarse_handoff,
        image_path=image_path,
        answer_timestamp=answer_timestamp,
    )
    registry = load_registry()
    tool = registry["submit_answer"]
    params = validate_action_params(
        tool,
        submit_result.model_dump(),
        agent_role=AgentRole.FINE,
    )
    action = Action(tool="submit_answer", params=params)
    obs = ObservationExecutionResult(
        action=action,
        observation=None,
        source=None,
        status="skipped",
        error_message=None,
        cache_hit=False,
    )
    last_step = units[-1][3]
    t_end = float(last_step.move.end_time)
    syn_step = NormalizedStep(
        move=Move(
            start_time=t_end,
            end_time=t_end,
            narration="提交最终定位答案。",
            screen_action="submit_answer",
            visible_clues=[],
            agent_role=AgentRole.FINE,
        ),
        thought_draft="综合已有 Observation，提交最终定位答案。",
        actions=[action],
        normalization_mode=NormalizationMode.FALLBACK,
        matched_tool_confidence=None,
        fallback_reason="stage5 合成 FINE terminal submit_answer",
    )
    draft = syn_step.thought_draft
    return [*units, (draft, action, obs, syn_step)]


def _verifier_syn_step(
    *,
    action: Action,
    draft: str,
    narration: str,
    reason: str,
) -> NormalizedStep:
    """构造 VERIFIER 合成步的 NormalizedStep 外壳。"""
    return NormalizedStep(
        move=Move(
            start_time=0.0,
            end_time=0.0,
            narration=narration,
            screen_action=action.tool,
            visible_clues=[],
            agent_role=AgentRole.VERIFIER,
        ),
        thought_draft=draft,
        actions=[action],
        normalization_mode=NormalizationMode.FALLBACK,
        matched_tool_confidence=None,
        fallback_reason=reason,
    )


def _make_verifier_map_query_unit(
    fine_handoff: SubmitAnswerResult,
    image_path: str,
) -> tuple[str, Action, ObservationExecutionResult, NormalizedStep]:
    """合成 map_query 验证步（核对 fine_handoff 候选坐标）。"""
    registry = load_registry()
    tool = registry["map_query"]
    raw_params = {
        "latlng": [fine_handoff.latitude, fine_handoff.longitude],
        "query": fine_handoff.location_name,
    }
    params = validate_action_params(
        tool, raw_params, agent_role=AgentRole.VERIFIER
    )
    action = Action(tool="map_query", params=params)
    obs = execute_action(action, image_path, AgentRole.VERIFIER)
    draft = (
        "先用地图工具核对 Agent2 候选坐标与地点名，"
        f"检查 {fine_handoff.location_name} 解析结果是否合理。"
    )
    syn_step = _verifier_syn_step(
        action=action,
        draft=draft,
        narration="地图核对候选定位。",
        reason="stage5 合成 VERIFIER map_query 验证步",
    )
    return (draft, action, obs, syn_step)


def _make_verifier_web_search_unit(
    fine_handoff: SubmitAnswerResult,
    image_path: str,
) -> tuple[str, Action, ObservationExecutionResult, NormalizedStep]:
    """合成 web_search(verification) 佐证步。"""
    registry = load_registry()
    tool = registry["web_search"]
    query = (
        f"verify landmark visual features near {fine_handoff.location_name} "
        f"{fine_handoff.latitude:.4f},{fine_handoff.longitude:.4f}"
    )
    raw_params = {
        "query": query,
        "top_k": 3,
        "purpose": "verification",
    }
    params = validate_action_params(
        tool, raw_params, agent_role=AgentRole.VERIFIER
    )
    action = Action(tool="web_search", params=params)
    obs = execute_action(action, image_path, AgentRole.VERIFIER)
    draft = (
        "再用检索工具核对外界描述是否与图像可见特征一致，"
        "作为对候选定位的第二重验证。"
    )
    syn_step = _verifier_syn_step(
        action=action,
        draft=draft,
        narration="检索佐证候选定位。",
        reason="stage5 合成 VERIFIER web_search 验证步",
    )
    return (draft, action, obs, syn_step)


def _synthesize_verifier_units(
    fine_handoff: SubmitAnswerResult,
    image_path: str,
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """基于 fine_handoff 合成至少两步验证脚手架：map_query + web_search。

    禁止使用 groundtruth；候选坐标仅来自 Agent2 交接物。
    """
    return [
        _make_verifier_map_query_unit(fine_handoff, image_path),
        _make_verifier_web_search_unit(fine_handoff, image_path),
    ]


def _augment_thin_verifier_units(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    fine_handoff: SubmitAnswerResult,
    image_path: str,
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """视频侧可展开步过薄时补齐 map_query / web_search 验证深度。"""
    tools_used = {u[1].tool for u in units}
    out = list(units)
    if "map_query" not in tools_used:
        out.insert(0, _make_verifier_map_query_unit(fine_handoff, image_path))
        tools_used.add("map_query")
    if len(out) < 2 or "web_search" not in tools_used:
        if "web_search" not in tools_used:
            out.append(_make_verifier_web_search_unit(fine_handoff, image_path))
    return out


def _expand_or_synthesize_verifier_units(
    steps: list[NormalizedStep],
    observations: list[ObservationExecutionResult],
    fine_handoff: SubmitAnswerResult,
    image_path: str,
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """优先展开视频侧 Action；若无可展开步则合成验证脚手架；过薄则补齐。"""
    try:
        units = _expand_action_units(steps, observations)
    except ValueError:
        return _synthesize_verifier_units(fine_handoff, image_path)
    return _augment_thin_verifier_units(units, fine_handoff, image_path)


def _extract_submit_answer(steps: list[TrajectoryStep]) -> SubmitAnswerResult:
    """FINE 最后一步必须为 submit_answer，params 解析为 SubmitAnswerResult。"""
    if not steps:
        raise ValueError("FINE 轨迹 steps 为空")
    last = steps[-1]
    if last.action.tool != "submit_answer":
        raise ValueError(
            f"FINE 最后一步必须为 submit_answer，实际为 {last.action.tool!r}"
        )
    if last.observation is not None or last.observation_source is not None:
        raise ValueError("submit_answer 步的 observation / observation_source 必须为 None")
    try:
        return SubmitAnswerResult.model_validate(last.action.params)
    except ValidationError as exc:
        raise ValueError(f"submit_answer params 无法解析为 SubmitAnswerResult: {exc}") from exc


def _collect_pre_answer_narrations(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    answer_timestamp: float,
) -> list[str]:
    """收集 answer_timestamp 之前的旁白文本（去重保序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for _draft, _action, _obs, step in units:
        move = step.move
        if move.end_time > answer_timestamp and move.start_time >= answer_timestamp:
            continue
        narr = (move.narration or "").strip()
        if not narr or narr in seen:
            continue
        seen.add(narr)
        out.append(narr)
    return out


def _extract_external_hints(
    narrations: list[str],
    agent_role: AgentRole,
) -> list[str]:
    """从旁白抽取外部给定地名线索；失败则返回空列表。不含 groundtruth。"""
    if not narrations:
        return []
    # VERIFIER 主任务是验证候选，一般不需要再注入外部线索
    if agent_role == AgentRole.VERIFIER:
        return []
    joined = "\n".join(f"- {n}" for n in narrations[:40])
    prompt = (
        "从下列视频旁白片段中，抽取「外部给定、非推理得出」的地名/地区线索"
        "（例如网友、评论、弹幕、求助者直接给出的河南信阳等）。\n"
        "规则：\n"
        "1. 只保留地名短语本身，不要保留「网友说/评论说」等来源话术；\n"
        "2. 排除博主宣布最终答案、揭晓坐标的句子；\n"
        "3. 排除纯视觉推理描述（无明确给定地名）；\n"
        "4. 若无可抽线索，hints 为空列表；\n"
        "5. 不得编造旁白中未出现的地名。\n"
        f"agent_role: {agent_role.value}\n"
        f"narrations:\n{joined}\n"
    )
    try:
        result = call_structured(prompt, _ExternalHints)
    except Exception:  # noqa: BLE001
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for h in result.hints:
        text = h.strip()
        if not text or text in seen:
            continue
        # 去掉常见来源套话前缀
        text = re.sub(
            r"^(?:网友|粉丝|评论|弹幕|有人)(?:说|提到|给出|告诉)[：:，,\s]*",
            "",
            text,
        ).strip()
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned[:8]


def _user_query_for_role(
    agent_role: AgentRole,
    *,
    coarse_handoff: Optional[LocationHypothesis],
    fine_handoff: Optional[SubmitAnswerResult],
    external_hints: Optional[list[str]] = None,
) -> str:
    """构造 user_query（不含 groundtruth；可含外部给定线索）。"""
    hint_suffix = ""
    if external_hints:
        hint_suffix = "\n已知线索：" + "；".join(external_hints)

    if agent_role == AgentRole.COARSE:
        return "请根据图像进行粗定位，缩小到可能的国家/地区。" + hint_suffix
    if agent_role == AgentRole.FINE:
        hyp = coarse_handoff.model_dump_json() if coarse_handoff else "{}"
        return (
            "请在粗定位假设基础上精确定位并 submit_answer。"
            f"假设：{hyp}" + hint_suffix
        )
    cand = fine_handoff.model_dump_json() if fine_handoff else "{}"
    return f"请交叉验证以下候选定位结果是否与图像自洽：{cand}" + hint_suffix


def reconstruct_single_trajectory(
    steps: list[NormalizedStep],
    observations: list[ObservationExecutionResult],
    agent_role: AgentRole,
    answer_timestamp: float,
    image_path: str,
    coarse_handoff: Optional[LocationHypothesis] = None,
    fine_handoff: Optional[SubmitAnswerResult] = None,
    is_revision: bool = False,
    revision_context: Optional[RevisionContext] = None,
) -> Trajectory:
    """组装 T→A→O，用 LLM 改写为前向推理链。

    禁止将 groundtruth / 真值地名 / 反向地理编码地址写入 prompt。
    Agent1 → coarse_output=LocationHypothesis
    Agent2 → 最后一步 submit_answer，fine_output=SubmitAnswerResult
    Agent3 → verifier_output=VerificationResult；把 fine_handoff 当候选验证
    terminal 步 observation 与 observation_source 均为 None
    """
    if agent_role == AgentRole.FINE and coarse_handoff is None:
        raise ValueError("FINE 轨迹要求 coarse_handoff 必填")
    if agent_role == AgentRole.VERIFIER and fine_handoff is None:
        raise ValueError("VERIFIER 轨迹要求 fine_handoff 必填")
    if agent_role == AgentRole.COARSE and (
        coarse_handoff is not None or fine_handoff is not None
    ):
        raise ValueError("COARSE 轨迹不得携带 coarse_handoff/fine_handoff")

    if is_revision and revision_context is None:
        raise ValueError("is_revision=True 时 revision_context 必填")
    if revision_context is not None and not is_revision:
        raise ValueError("提供 revision_context 时 is_revision 必须为 True")

    if agent_role == AgentRole.VERIFIER:
        assert fine_handoff is not None  # 上文已校验
        units = _expand_or_synthesize_verifier_units(
            steps, observations, fine_handoff, image_path
        )
    else:
        units = _expand_action_units(steps, observations)
    if agent_role == AgentRole.FINE:
        assert coarse_handoff is not None  # 上文已校验
        units = _ensure_fine_terminal_submit(
            units,
            coarse_handoff=coarse_handoff,
            image_path=image_path,
            answer_timestamp=answer_timestamp,
        )

    # 外部线索从投影前完整旁白抽取；Agent1 scaffold 使用投影后子集
    narrations = _collect_pre_answer_narrations(units, answer_timestamp)
    if agent_role == AgentRole.COARSE:
        units = _project_coarse_units(units)

    external_hints = _extract_external_hints(narrations, agent_role)
    user_query = _user_query_for_role(
        agent_role,
        coarse_handoff=coarse_handoff,
        fine_handoff=fine_handoff,
        external_hints=external_hints,
    )
    prompt = _build_scaffold_prompt(
        agent_role,
        units,
        answer_timestamp,
        coarse_handoff=coarse_handoff,
        fine_handoff=fine_handoff,
        revision_context=revision_context,
        user_query=user_query,
    )

    # 角色结构化输出
    coarse_output: Optional[LocationHypothesis] = None
    fine_output: Optional[SubmitAnswerResult] = None
    verifier_output: Optional[VerificationResult] = None

    if agent_role == AgentRole.COARSE:
        bundle = call_structured(
            prompt
            + "\n\n请输出改写后的 thoughts，以及最终 LocationHypothesis（coarse_output）。"
            "reasoning_summary 须概括特征→排除/收窄→候选范围。",
            _CoarseOutputBundle,
            images=[image_path],
        )
        thoughts = bundle.thoughts
        coarse_output = bundle.coarse_output
        need, issues = _needs_tao_rewrite(
            thoughts, units, agent_role, coarse_output=coarse_output
        )
        if need:
            issue_txt = "；".join(issues[:5])
            bundle = call_structured(
                prompt
                + f"\n\n上次输出不符合标准地理定位 TAO / 递进推理：{issue_txt}\n"
                "请重新改写 thoughts（必须符合风格规范与递进链），并给出 LocationHypothesis。",
                _CoarseOutputBundle,
                images=[image_path],
            )
            thoughts = bundle.thoughts
            coarse_output = bundle.coarse_output
    elif agent_role == AgentRole.FINE:
        rewritten = call_structured(
            prompt + "\n\n请仅输出与各 Step 一一对应的改写 thoughts。",
            _RewrittenTrajectory,
            images=[image_path],
        )
        thoughts = rewritten.thoughts
        need, issues = _needs_tao_rewrite(thoughts, units, agent_role)
        if need:
            issue_txt = "；".join(issues[:5])
            rewritten = call_structured(
                prompt
                + f"\n\n上次输出不符合标准地理定位 TAO：{issue_txt}\n"
                "请重新改写 thoughts，必须符合风格规范。",
                _RewrittenTrajectory,
                images=[image_path],
            )
            thoughts = rewritten.thoughts
    else:
        bundle_v = call_structured(
            prompt
            + "\n\n请输出改写后的 thoughts，以及 VerificationResult（把 fine_handoff 当候选）。",
            _VerifierOutputBundle,
            images=[image_path],
        )
        thoughts = bundle_v.thoughts
        verifier_output = bundle_v.verifier_output
        need, issues = _needs_tao_rewrite(thoughts, units, agent_role)
        if need:
            issue_txt = "；".join(issues[:5])
            bundle_v = call_structured(
                prompt
                + f"\n\n上次输出不符合标准地理定位 TAO：{issue_txt}\n"
                "请重新改写 thoughts（必须符合风格规范），并给出 VerificationResult。",
                _VerifierOutputBundle,
                images=[image_path],
            )
            thoughts = bundle_v.thoughts
            verifier_output = bundle_v.verifier_output

    traj_steps = _to_trajectory_steps(units, thoughts)
    if agent_role == AgentRole.FINE:
        fine_output = _extract_submit_answer(traj_steps)

    traj_id = _new_traj_id(agent_role, is_revision=is_revision)
    parent_id: Optional[str] = None
    revision_round = 0
    revision_source: Optional[RevisionSource] = None
    revision_input: Optional[VerificationResult] = None

    if revision_context is not None:
        parent_id = revision_context.parent_trajectory_id
        revision_round = revision_context.revision_round
        revision_source = revision_context.source
        if revision_context.source == RevisionSource.SYSTEM_FEEDBACK:
            revision_input = revision_context.verification_result

    return Trajectory(
        id=traj_id,
        agent_role=agent_role,
        system_prompt=_SYSTEM_PROMPTS[agent_role],
        user_query=user_query,
        image_path=image_path,
        steps=traj_steps,
        coarse_handoff=coarse_handoff if agent_role != AgentRole.COARSE else None,
        fine_handoff=fine_handoff if agent_role == AgentRole.VERIFIER else None,
        coarse_output=coarse_output,
        fine_output=fine_output,
        verifier_output=verifier_output,
        is_revision=is_revision,
        parent_trajectory_id=parent_id,
        revision_round=revision_round,
        revision_source=revision_source,
        revision_input=revision_input,
    )


def reconstruct_all_trajectories(
    all_steps: dict[AgentRole, list[NormalizedStep]],
    all_observations: dict[AgentRole, list[ObservationExecutionResult]],
    answer_timestamp: float,
    image_path: str,
) -> dict[AgentRole, Trajectory]:
    """为三 Agent 重构主轨迹并传递交接物。

    Agent1.coarse_output → Agent2.coarse_handoff
    Agent2.fine_output → Agent3.fine_handoff
    Agent3：若视频侧无任何可展开 Action，则基于 fine_handoff 合成验证脚手架。
    """
    required = (AgentRole.COARSE, AgentRole.FINE, AgentRole.VERIFIER)
    for role in required:
        if role not in all_steps:
            raise ValueError(f"all_steps 缺少 {role}")
        if role not in all_observations:
            raise ValueError(f"all_observations 缺少 {role}")

    coarse = reconstruct_single_trajectory(
        all_steps[AgentRole.COARSE],
        all_observations[AgentRole.COARSE],
        AgentRole.COARSE,
        answer_timestamp,
        image_path,
    )
    if coarse.coarse_output is None:
        raise ValueError("Agent1 未产出 coarse_output")

    fine = reconstruct_single_trajectory(
        all_steps[AgentRole.FINE],
        all_observations[AgentRole.FINE],
        AgentRole.FINE,
        answer_timestamp,
        image_path,
        coarse_handoff=coarse.coarse_output,
    )
    if fine.fine_output is None:
        raise ValueError("Agent2 未产出 fine_output")

    verifier = reconstruct_single_trajectory(
        all_steps[AgentRole.VERIFIER],
        all_observations[AgentRole.VERIFIER],
        AgentRole.VERIFIER,
        answer_timestamp,
        image_path,
        coarse_handoff=coarse.coarse_output,
        fine_handoff=fine.fine_output,
    )
    if verifier.verifier_output is None:
        raise ValueError("Agent3 未产出 verifier_output")

    return {
        AgentRole.COARSE: coarse,
        AgentRole.FINE: fine,
        AgentRole.VERIFIER: verifier,
    }


def _steps_overlapping_segment(
    steps: list[NormalizedStep],
    segment: tuple[float, float],
) -> list[NormalizedStep]:
    """筛选与 video_segment 时间重叠的 NormalizedStep。"""
    seg_start, seg_end = segment
    return [
        s
        for s in steps
        if s.move.end_time > seg_start and s.move.start_time < seg_end
    ]


def _observations_for_steps(
    steps: list[NormalizedStep],
    all_steps: list[NormalizedStep],
    all_observations: list[ObservationExecutionResult],
) -> list[ObservationExecutionResult]:
    """按完整步骤列表中的 Action 下标，提取子集步骤对应的 Observation。"""
    # 建立全局 Action 下标
    action_index: dict[int, list[int]] = {}
    cursor = 0
    for i, step in enumerate(all_steps):
        n = len(step.actions)
        action_index[i] = list(range(cursor, cursor + n))
        cursor += n

    selected: list[ObservationExecutionResult] = []
    step_id_map = {id(s): i for i, s in enumerate(all_steps)}
    for step in steps:
        idx = step_id_map.get(id(step))
        if idx is None:
            # 回退：按对象相等查找
            try:
                idx = all_steps.index(step)
            except ValueError as exc:
                raise ValueError("子集 step 不在 all_steps 中") from exc
        for ai in action_index[idx]:
            selected.append(all_observations[ai])
    return selected


def reconstruct_revision_trajectories(
    parent_trajectories: dict[AgentRole, Trajectory],
    verification: VerificationResult,
    all_steps: dict[AgentRole, list[NormalizedStep]],
    all_observations: dict[AgentRole, list[ObservationExecutionResult]],
    answer_timestamp: float,
    image_path: str,
    revision_round: int,
    max_revision_rounds: int,
    video_revision_segments: Optional[list[tuple[float, float]]] = None,
) -> list[Trajectory]:
    """闭合返工路径。

    - system_feedback：return_to_agent=1→COARSE；=2→FINE；构造 RevisionContext
    - video_observed：使用 video_revision_segments 生成高价值返工轨迹
    - revision_round > max_revision_rounds → 不再生成 system_feedback 返工（rejected）
    """
    results: list[Trajectory] = []

    # --- system_feedback ---
    if verification.verdict == "fail":
        if revision_round > max_revision_rounds:
            # 超过上限：不再生成系统打回返工，交由上层记入 rejected
            pass
        elif verification.return_to_agent in (1, 2):
            target = (
                AgentRole.COARSE
                if verification.return_to_agent == 1
                else AgentRole.FINE
            )
            parent = parent_trajectories.get(target)
            if parent is None:
                raise ValueError(f"parent_trajectories 缺少目标角色 {target}")
            ctx = RevisionContext(
                source=RevisionSource.SYSTEM_FEEDBACK,
                parent_trajectory_id=parent.id,
                target_agent=target,
                revision_round=revision_round,
                verification_result=verification,
                video_segment=None,
            )
            coarse_h = parent.coarse_handoff
            fine_h = parent.fine_handoff
            if target == AgentRole.FINE:
                # FINE 返工仍需要 coarse_handoff：优先用父轨迹，否则用 COARSE 主轨迹输出
                coarse_h = parent.coarse_handoff
                if coarse_h is None:
                    coarse_parent = parent_trajectories.get(AgentRole.COARSE)
                    if coarse_parent is None or coarse_parent.coarse_output is None:
                        raise ValueError("FINE 返工缺少 coarse_handoff")
                    coarse_h = coarse_parent.coarse_output
            rev = reconstruct_single_trajectory(
                all_steps[target],
                all_observations[target],
                target,
                answer_timestamp,
                image_path,
                coarse_handoff=coarse_h,
                fine_handoff=fine_h,
                is_revision=True,
                revision_context=ctx,
            )
            results.append(rev)

    # --- video_observed ---
    if video_revision_segments:
        for segment in video_revision_segments:
            # 视频内纠错通常发生在精定位阶段；优先 FINE，其次 COARSE
            produced = False
            for target in (AgentRole.FINE, AgentRole.COARSE):
                subset = _steps_overlapping_segment(all_steps[target], segment)
                actionable = [s for s in subset if s.actions]
                # 无时间重叠时回退为该角色全量可展开步，避免 revision 空跑
                if not actionable:
                    actionable = [s for s in all_steps[target] if s.actions]
                if not actionable:
                    continue
                obs_subset = _observations_for_steps(
                    actionable,
                    all_steps[target],
                    all_observations[target],
                )
                parent = parent_trajectories[target]
                ctx = RevisionContext(
                    source=RevisionSource.VIDEO_OBSERVED,
                    parent_trajectory_id=parent.id,
                    target_agent=target,
                    revision_round=revision_round,
                    verification_result=None,
                    video_segment=segment,
                )
                coarse_h = None
                if target == AgentRole.FINE:
                    coarse_h = parent.coarse_handoff
                    if coarse_h is None:
                        coarse_parent = parent_trajectories.get(AgentRole.COARSE)
                        if coarse_parent is None or coarse_parent.coarse_output is None:
                            raise ValueError("FINE video_observed 返工缺少 coarse_handoff")
                        coarse_h = coarse_parent.coarse_output
                rev = reconstruct_single_trajectory(
                    actionable,
                    obs_subset,
                    target,
                    answer_timestamp,
                    image_path,
                    coarse_handoff=coarse_h,
                    fine_handoff=None,
                    is_revision=True,
                    revision_context=ctx,
                )
                results.append(rev)
                produced = True
                break  # 每个 segment 只生成一条优先命中的返工轨迹
            if not produced:
                continue

    return results
