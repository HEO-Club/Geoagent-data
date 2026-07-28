"""stage5：三 Agent 主轨迹与 revision 轨迹重构。

生成范式：逐步因果生成（teacher-forced rollout）→ polish 润色 →
轻量硬校验 → 固定 rubric judge best-of-k 拒绝采样。
第 t 步 Thought 的上下文只包含前 t-1 步完整 T/A/O 与本步 Action，
不含本步 Observation 与任何后续信息，从构造上保证不预知、不跳步。

本阶段禁止访问 groundtruth；函数签名不得包含 groundtruth。
若 FINE 脚手架缺少 terminal submit_answer，则基于证据合成该步（仍禁止 GT）。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from pipeline.coarse_tool_policy import (
    COARSE_CORE_TOOLS,
    is_coarse_allowed_tool,
    is_coarse_forbidden_tool,
)
from pipeline.config import get_settings
from pipeline.evidence_routing import (
    CoarseEvidenceLedger,
    CoarseStepKind,
    CandidateUpdateEntry,
    RangeUpdateKind,
    VideoChainContext,
    build_candidate_updates_from_facts,
    embed_video_context,
    fact_update_fingerprint,
    format_working_scope_user_query,
    obs_fingerprint,
    parse_evidence_intent,
    parse_video_context,
    sanitize_revision_input_for_coarse_shard,
    sanitize_verification_for_coarse_prompt,
    strip_evidence_intent,
)
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

# Agent1 训练轨迹 Tool 投影（与 coarse_tool_policy 对齐）
_COARSE_FIXED_TOOLS: frozenset[str] = COARSE_CORE_TOOLS
# Thought 中出现、但非本步 Action 的异工具话术
_FOREIGN_TOOL_RHETORIC: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("web_search", re.compile(r"\bweb_search\b|网页搜索|打开搜索引擎", re.I)),
    ("map_query", re.compile(r"\bmap_query\b|地理编码|解析坐标查询", re.I)),
    (
        "reverse_image_search",
        re.compile(r"\breverse_image_search\b|以图搜图", re.I),
    ),
)
_CORE_TOOL_SATELLITE_MISMATCH_RE = re.compile(
    r"调用\s*(?:web_search|map_query|卫星检索|网页搜索)|"
    r"调取\s*\d{4}\s*年?\s*(?:遥感|卫星)\s*(?:地图|影像)?\s*(?:接口|API|服务)?",
    re.I,
)

logger = logging.getLogger(__name__)


class TrajectoryQualityRejected(RuntimeError):
    """best-of-k 全部候选低于 STAGE5_JUDGE_THRESHOLD：该角色轨迹废弃，不入库。"""


class _StepThought(BaseModel):
    """逐步因果生成：单步 Thought。"""

    thought: str = Field(min_length=1)


class _PolishedThoughts(BaseModel):
    """polish 润色后的整链 thoughts（必须与步数等长）。"""

    thoughts: list[str] = Field(min_length=1)


class _FaithfulnessCheck(BaseModel):
    """polish 前后忠实性对比：列出事实/结论被改动的步序号（1-based）。"""

    unfaithful_steps: list[int] = Field(default_factory=list)


class _TrajectoryJudgement(BaseModel):
    """固定 rubric 的轨迹质量评分（无 groundtruth）。"""

    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)


class _ExternalHints(BaseModel):
    """从答案前旁白抽取的线索分类（不含真值；FINE 仍可用）。"""

    given_clues: list[str] = Field(
        default_factory=list,
        description="问题设置阶段外部给定的软先验地名",
    )
    candidate_hypotheses: list[str] = Field(
        default_factory=list,
        description="推理过程首次出现的待证候选，不得注入 Agent1 user_query",
    )
    hints: list[str] = Field(default_factory=list)


class _CoarseToolSuitability(BaseModel):
    """动态 Tool 是否适合进入 Agent1 训练轨迹。"""

    suitable_for_coarse_reasoning: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# 角色提示：面向推理期的简洁角色指令（训练样本 system prompt；不含内部术语与禁令墙）
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS: dict[AgentRole, str] = {
    AgentRole.COARSE: (
        "你是图片地理定位的粗定位 Agent。根据图像与用户给出的线索逐步收集证据："
        "每一步先写 Thought（基于已有观察说明当前判断与本步要验证什么），"
        "再给出 Action（工具调用）。依据每次 Observation 逐步排除或收窄候选区域，"
        "最终输出 LocationHypothesis（可能的国家与行政区、推理摘要、置信度、"
        "剩余线索）。不要给出精确 POI 或坐标。"
    ),
    AgentRole.FINE: (
        "你是图片地理定位的精定位 Agent。在粗定位假设的基础上，"
        "逐步调用工具验证并收窄到具体地点；证据足够时可尽早提出精确地点假设。"
        "每一步先写 Thought 再给出 Action，"
        "最后一步必须调用 submit_answer 提交坐标、地点名与置信度。"
    ),
    AgentRole.VERIFIER: (
        "你是图片地理定位的验证 Agent。将候选定位结果与图像特征交叉验证："
        "调用地图与检索工具核对候选坐标、地名与画面证据是否自洽，"
        "每一步先写 Thought 再给出 Action，最终输出 VerificationResult"
        "（pass/fail、失败项与建议复查点）。"
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
    """判定未入允许清单的动态 Tool 是否适合 Agent1；失败则 fail-closed。"""
    if is_coarse_forbidden_tool(tool_name):
        return False
    if is_coarse_allowed_tool(tool_name):
        return True
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
        "适合：单图/双图地理特征观察、画面内地图或卫星布局比对、"
        "地形/阴影推断（服务区域级排除，非精确 POI）。\n"
        "明确拒绝：web_search、map_query（解析坐标/标准地址）、"
        "reverse_image_search、submit_answer。\n"
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


def _sanitize_zoom_bbox_in_units(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """纠正 zoom_inspect 非法 bbox（绝对值>1.5 视为非归一化图像框 → 全图框）。"""
    out: list[
        tuple[str, Action, ObservationExecutionResult, NormalizedStep]
    ] = []
    for draft, action, obs, step in units:
        if action.tool != "zoom_inspect":
            out.append((draft, action, obs, step))
            continue
        bbox = action.params.get("bbox")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(x, (int, float)) for x in bbox)
            and any(abs(float(x)) > 1.5 for x in bbox)
        ):
            fixed = Action(
                tool=action.tool,
                params={**action.params, "bbox": [0.0, 0.0, 1.0, 1.0]},
            )
            out.append((draft, fixed, obs, step))
        else:
            out.append((draft, action, obs, step))
    return out


def _project_coarse_units(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """将全量 COARSE Action/Obs 投影为 Agent1 训练子集。

    保留核心三工具与视觉地图/卫星允许集；硬排除 web_search/map_query/RIS/submit；
    其余动态 Tool 经结构化适配判定后可选保留。
    """
    projected: list[
        tuple[str, Action, ObservationExecutionResult, NormalizedStep]
    ] = []
    suitability_cache: dict[str, bool] = {}
    for unit in units:
        _draft, action, _obs, _step = unit
        name = action.tool
        if is_coarse_allowed_tool(name):
            projected.append(unit)
            continue
        if is_coarse_forbidden_tool(name) or name in SEED_TOOL_NAMES:
            continue
        if name not in suitability_cache:
            suitability_cache[name] = _judge_dynamic_tool_for_coarse(name)
        if suitability_cache[name]:
            projected.append(unit)
    if not projected:
        raise ValueError(
            "COARSE Tool 投影后无可重构 Action 步"
            "（仅剩禁止 Tool 或无适配动态 Tool）"
        )
    sanitized = _sanitize_zoom_bbox_in_units(projected)
    usable, ui_removed = _filter_unusable_ui_units(sanitized)
    usable = _drop_noninformative_empty_units(usable)
    if not usable:
        raise ValueError(
            "COARSE 投影后无可重构 Action 步"
            "（UI/overlay/empty Observation 剔除后证据不足）"
        )
    collapsed = _collapse_consecutive_duplicate_actions(usable)
    collapsed = _collapse_semantic_fact_clusters(collapsed)
    _validate_coarse_projection_richness(collapsed)
    # 将 UI 剔除信息挂到首步草稿旁注，供账本消费（不改公共 schema）
    if ui_removed and collapsed:
        draft0, a0, o0, s0 = collapsed[0]
        collapsed[0] = (
            f"{draft0}\n<<<UI_REMOVED_STEPS:{ui_removed}>>>",
            a0,
            o0,
            s0,
        )
    return collapsed


def _action_fingerprint(action: Action) -> str:
    """tool + 规范化 params 指纹，用于重复步检测。"""
    return f"{action.tool}|{json.dumps(action.params, sort_keys=True, ensure_ascii=False, default=str)}"


_UI_OBS_RE = re.compile(
    r"聊天|消息|置顶|弹幕|播放器|进度条|标题卡|片头|点赞|评论区|"
    r"微信|界面|字幕|overlay|粉丝|难度\s*\d",
    re.I,
)


def _is_ui_overlay_observation(obs: ObservationExecutionResult) -> bool:
    """判定 Observation 是否主要为 UI/overlay（不可用作地理证据）。"""
    if obs.status in ("skipped", "error"):
        return False
    text = _obs_brief_text(obs)
    if not text:
        return False
    # empty 且明确无场景地理 → 不算 UI 污染，视为证据不足占位
    if obs.status == "empty":
        return False
    return bool(_UI_OBS_RE.search(text))


def _filter_unusable_ui_units(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> tuple[
    list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    list[int],
]:
    """剔除 UI/overlay Observation 步；返回保留列表与被剔除的 1-based 原序号。"""
    kept: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]] = []
    removed: list[int] = []
    for i, unit in enumerate(units, start=1):
        if _is_ui_overlay_observation(unit[2]):
            removed.append(i)
            continue
        kept.append(unit)
    return kept, removed


def _is_geo_evidence_unit(
    unit: tuple[str, Action, ObservationExecutionResult, NormalizedStep],
) -> bool:
    """empty 步是否仍有地理训练增益（应保留在投影链中）。"""
    from pipeline.evidence_routing import ContentType
    from pipeline.stage2_moves import is_non_trainable_move

    draft, _action, _obs, step = unit
    intent = parse_evidence_intent(draft)
    if intent is not None:
        if (
            intent.content_type is ContentType.INTERFACE_ONLY
            and not intent.target_features
            and not intent.source_concepts
            and not intent.source_claims
        ):
            return False
        if intent.source_claims or intent.target_features or intent.source_concepts:
            return True
        if intent.step_kind in (CoarseStepKind.OBSERVE, CoarseStepKind.UPDATE):
            return True
    move = step.move
    if is_non_trainable_move(
        move.narration,
        move.screen_action,
        list(move.visible_clues or []),
    ):
        return False
    return bool((move.screen_action or "").strip() or (move.narration or "").strip())


def _drop_noninformative_empty_units(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """去掉无地理增益的 empty；保留 success、error 与有地理增益的 empty。

    不得因存在 success 就掏空失败/未命中步（避免 TAO 中间环断链）。
    若过滤后为空，则保留最多 2 步供 stage5 明确拒样。
    """
    out: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]] = []
    for unit in units:
        _draft, _action, obs, _step = unit
        if obs.status == "success":
            out.append(unit)
            continue
        if obs.status == "error":
            out.append(unit)
            continue
        if obs.status == "empty":
            if _is_geo_evidence_unit(unit):
                out.append(unit)
            continue
        # skipped 等其它状态：保留以维持链完整
        out.append(unit)
    if out:
        return out
    fixed_empty = [u for u in units if u[1].tool in _COARSE_FIXED_TOOLS][:2]
    return fixed_empty or units[:2]


def _intent_candidate_update_signature(draft: str) -> str:
    """候选状态增量签名：仅 UPDATE 步计入；纯 observe 视为无增量。"""
    intent = parse_evidence_intent(draft)
    if intent is None:
        return ""
    if intent.step_kind is not CoarseStepKind.UPDATE:
        return ""
    kind = intent.step_kind.value
    return fact_update_fingerprint(
        list(intent.video_fact_ids),
        intent.subject_scope,
        kind,
    )


def _has_new_candidate_delta(prev_draft: str, curr_draft: str) -> bool:
    """当前步相对前步是否有新的 exclude/narrow/shift/correct 类增量。"""
    curr_sig = _intent_candidate_update_signature(curr_draft)
    if not curr_sig:
        return False
    prev_sig = _intent_candidate_update_signature(prev_draft)
    return curr_sig != prev_sig


def _collapse_consecutive_duplicate_actions(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """折叠连续同 tool+params 且无新候选增量的步骤（即使 Obs 字面不同）。

    保留首步 Obs；被折叠步写入旁注。有 exclude/narrow 等状态增量则保留。
    """
    if not units:
        return units
    out: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]] = [
        units[0]
    ]
    collapsed_notes: list[str] = []
    for unit in units[1:]:
        prev = out[-1]
        same_action = _action_fingerprint(unit[1]) == _action_fingerprint(prev[1])
        same_obs = obs_fingerprint(
            None if unit[2].status == "skipped" else unit[2].observation
        ) == obs_fingerprint(
            None if prev[2].status == "skipped" else prev[2].observation
        )
        if same_action and same_obs:
            continue
        if same_action and not _has_new_candidate_delta(prev[0], unit[0]):
            collapsed_notes.append(
                "<<<collapsed: same Action params without candidate delta>>>"
            )
            continue
        out.append(unit)
    if collapsed_notes and out:
        draft, action, obs, step = out[-1]
        # 旁注挂在当前链末保留步，供调试；生成 Thought 前会消毒
        note = collapsed_notes[-1]
        if note not in draft:
            out[-1] = (f"{draft}\n{note}", action, obs, step)
    return out


def _unit_fact_cluster_key(
    draft: str,
) -> str:
    """语义簇指纹：video_fact_ids + subject_scope + update_kind。"""
    intent = parse_evidence_intent(draft)
    if intent is None:
        return strip_evidence_intent(draft)[:80]
    kind = (
        intent.step_kind.value
        if intent.step_kind is not None
        else CoarseStepKind.OBSERVE.value
    )
    return fact_update_fingerprint(
        list(intent.video_fact_ids),
        intent.subject_scope,
        kind,
    )


def _collapse_semantic_fact_clusters(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]]:
    """相同事实簇且无新候选状态增量时合并（即使 bbox/措辞不同）。"""
    if not units:
        return units
    out: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]] = []
    seen_keys: set[str] = set()
    for unit in units:
        draft, action, obs, step = unit
        key = _unit_fact_cluster_key(draft)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(unit)
    return out if out else units[:1]


def _is_fullframe_zoom(action: Action) -> bool:
    """zoom_inspect 且 bbox 为近似全图框。"""
    if action.tool != "zoom_inspect":
        return False
    bbox = action.params.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        vals = [float(x) for x in bbox]
    except (TypeError, ValueError):
        return False
    # [0,0,1,1] xyxy 或 [0,0,1,1] 当作 xywh 也近似全图
    if all(abs(vals[i] - [0.0, 0.0, 1.0, 1.0][i]) < 0.05 for i in range(4)):
        return True
    if (
        abs(vals[0]) < 0.05
        and abs(vals[1]) < 0.05
        and abs(vals[2] - 1.0) < 0.05
        and abs(vals[3] - 1.0) < 0.05
    ):
        return True
    return False


def _obs_brief_text(obs: ObservationExecutionResult) -> str:
    """Observation 简要文本，用于信息增益粗检。"""
    if obs.status == "skipped" or obs.observation is None:
        return ""
    return json.dumps(obs.observation, ensure_ascii=False, sort_keys=True)


def _intent_target_key(draft: str) -> str:
    """从 thought_draft 取观察目标键（用于 richness）。"""
    intent = parse_evidence_intent(draft)
    if intent is None:
        return strip_evidence_intent(draft)[:80]
    feats = ",".join(intent.target_features)
    return f"{intent.target_object}|{feats}|{intent.suggested_bbox}"


def _validate_coarse_projection_richness(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> None:
    """投影后递进可写性：Action 目标、Obs 增益与线索覆盖。

    同 bbox 不等于同一步；不同 bbox 也不自动代表新增证据。
    """
    if not units:
        raise ValueError("COARSE 投影后递进可写性不足：无步骤")

    # 连续同 Action 且无候选增量 → 应已被折叠；若仍出现则失败
    for i in range(1, len(units)):
        same_a = _action_fingerprint(units[i][1]) == _action_fingerprint(
            units[i - 1][1]
        )
        same_o = obs_fingerprint(
            None if units[i][2].status == "skipped" else units[i][2].observation
        ) == obs_fingerprint(
            None
            if units[i - 1][2].status == "skipped"
            else units[i - 1][2].observation
        )
        no_delta = not _has_new_candidate_delta(units[i - 1][0], units[i][0])
        if same_a and (same_o or no_delta):
            raise ValueError(
                f"COARSE 投影后递进可写性不足：Step {i} 与 Step {i + 1} "
                "重复相同 tool+params 且无候选增量"
            )

    fps = [_action_fingerprint(a) for _d, a, _o, _s in units]
    obs_fps = [
        obs_fingerprint(None if o.status == "skipped" else o.observation)
        for _d, _a, o, _s in units
    ]
    if len(fps) >= 2 and len(set(fps)) == 1 and len(set(obs_fps)) == 1:
        raise ValueError(
            "COARSE 投影后递进可写性不足：全部步骤为相同 tool+params 且 Obs 无差异"
        )

    # 目标键 + Obs 信息增益：至少应有两类观察目标或两类 Obs
    targets = {_intent_target_key(d) for d, _a, _o, _s in units}
    meaningful_obs = [fp for fp in obs_fps if fp]
    if len(units) >= 3 and len(targets) == 1 and len(set(meaningful_obs)) <= 1:
        raise ValueError(
            "COARSE 投影后递进可写性不足：多步共享同一观察目标且 Observation 无信息增益"
        )

    full_zooms = [
        (i, units[i])
        for i in range(len(units))
        if _is_fullframe_zoom(units[i][1])
    ]
    if len(full_zooms) >= 2:
        for j in range(1, len(full_zooms)):
            i_prev, u_prev = full_zooms[j - 1]
            i_cur, u_cur = full_zooms[j]
            if i_cur != i_prev + 1:
                continue
            prev_txt = _obs_brief_text(u_prev[2])
            cur_txt = _obs_brief_text(u_cur[2])
            if not prev_txt or not cur_txt or prev_txt == cur_txt:
                raise ValueError(
                    "COARSE 投影后递进可写性不足：连续全图 zoom_inspect "
                    "且 Observation 无新增证据"
                )


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
    draft_clean = strip_evidence_intent(thought_draft)
    intent = parse_evidence_intent(thought_draft)
    intent_line = ""
    kind = "observe"
    if intent is not None:
        kind = intent.step_kind.value
        intent_line = (
            f"evidence_intent: step_kind={kind} "
            f"target={intent.target_object!r} "
            f"features={intent.target_features} "
            f"video_fact_ids={intent.video_fact_ids} "
            f"source_claims={intent.source_claims} "
            f"relation={intent.expected_spatial_relation!r}\n"
        )
    return (
        f"### Step {index} [{kind}]\n"
        f"{intent_line}"
        f"thought_draft: {draft_clean}\n"
        f"action: tool={action.tool} params={action.params!r}\n"
        f"observation_status: {obs.status}\n"
        f"observation: {obs_repr!r}\n"
    )


def _video_context_from_units(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> Optional[VideoChainContext]:
    """从首个含嵌入标记的 ThoughtDraft 解析 VideoChainContext。"""
    for draft, _a, _o, _s in units:
        ctx = parse_video_context(draft)
        if ctx is not None:
            return ctx
    return None


def _build_coarse_evidence_ledger(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    *,
    given_clues: list[str],
    candidate_hypotheses: list[str],
) -> CoarseEvidenceLedger:
    """在生成 Thought 前构建内部证据账本。"""
    from pipeline.evidence_routing import (
        SpatialRelationEntry,
        VisualFactEntry,
        source_concepts_from_facts,
    )

    facts: list[VisualFactEntry] = []
    relations: list[SpatialRelationEntry] = []
    ui_steps: list[int] = []
    source_concepts: list[str] = []
    video_fact_claims: dict[str, str] = {}
    ctx = _video_context_from_units(units)
    raw_clues = list(given_clues)
    working_scope: Optional[str] = None
    video_facts: list[Any] = []
    if ctx is not None:
        raw_clues = [c.text for c in ctx.raw_given_clues]
        given_clues = list(raw_clues)
        if ctx.working_scope is not None:
            working_scope = ctx.working_scope.region
        source_concepts = source_concepts_from_facts(
            ctx.video_facts,
            working_scope=ctx.working_scope,
            raw_clues=ctx.raw_given_clues,
        )
        video_fact_claims = {
            fact.fact_id: fact.quote
            for fact in ctx.video_facts
            if fact.kind != "stall"
        }
        candidate_hypotheses = list(ctx.candidate_hypotheses)
        video_facts = list(ctx.video_facts)

    step_map: dict[str, list[int]] = {}
    annotation_updates: list[CandidateUpdateEntry] = []
    for i, (draft, action, obs, _step) in enumerate(units, start=1):
        if "<<<UI_REMOVED_STEPS:" in draft:
            m = re.search(r"<<<UI_REMOVED_STEPS:(\[[^\]]*\])>>>", draft)
            if m:
                try:
                    ui_steps = [int(x) for x in json.loads(m.group(1))]
                except (TypeError, ValueError, json.JSONDecodeError):
                    ui_steps = []
        intent = parse_evidence_intent(draft)
        if intent is not None:
            for tok in intent.source_concepts:
                if tok and tok not in source_concepts:
                    source_concepts.append(tok)
            for fid in intent.video_fact_ids:
                step_map.setdefault(fid, []).append(i)
            if intent.step_kind is CoarseStepKind.UPDATE:
                annotation_updates.append(
                    CandidateUpdateEntry(
                        kind=RangeUpdateKind.EXCLUDE,
                        old_candidates=[],
                        new_candidates=[],
                        excluded=[],
                        evidence_steps=[i],
                        video_fact_ids=list(intent.video_fact_ids),
                        exclusion_reason="",
                        subject_scope=intent.subject_scope,
                        spatial_anchor=intent.spatial_anchor,
                    )
                )
        brief = _obs_brief_text(obs)
        if not brief or obs.status in ("empty", "error", "skipped"):
            continue
        summary = brief[:240]
        vf_ids: list[str] = []
        if intent is not None and intent.target_features:
            summary = (
                f"{intent.target_object}: {','.join(intent.target_features)}; "
                f"obs={brief[:160]}"
            )
            vf_ids = list(intent.video_fact_ids)
        facts.append(
            VisualFactEntry(
                step_index=i,
                summary=summary,
                source_tool=action.tool,
                video_fact_ids=vf_ids,
            )
        )
        if intent is not None and intent.expected_spatial_relation:
            relations.append(
                SpatialRelationEntry(
                    description=intent.expected_spatial_relation,
                    supporting_fact_steps=[i],
                    subject_scope=intent.subject_scope,
                    spatial_anchor=intent.spatial_anchor,
                )
            )

    # Obs 全 empty 时仍可用逐视频来源声明播种 visual_facts，避免无图确认时直接断链
    if not facts and video_fact_claims:
        for i, (draft, action, _obs, _step) in enumerate(units, start=1):
            intent = parse_evidence_intent(draft)
            if intent is None or not intent.source_claims:
                continue
            facts.append(
                VisualFactEntry(
                    step_index=i,
                    summary="; ".join(intent.source_claims)[:240],
                    source_tool=action.tool,
                    video_fact_ids=list(intent.video_fact_ids),
                )
            )
            if intent.expected_spatial_relation:
                relations.append(
                    SpatialRelationEntry(
                        description=intent.expected_spatial_relation,
                        supporting_fact_steps=[i],
                        subject_scope=intent.subject_scope,
                        spatial_anchor=intent.spatial_anchor,
                    )
                )

    candidate_updates = build_candidate_updates_from_facts(
        video_facts,
        evidence_steps=step_map,
    )
    if not candidate_updates and annotation_updates:
        candidate_updates = annotation_updates

    return CoarseEvidenceLedger(
        raw_given_clues=raw_clues,
        working_scope=working_scope,
        given_clues=list(given_clues) if given_clues else raw_clues,
        candidate_hypotheses=list(candidate_hypotheses),
        visual_facts=facts,
        spatial_relations=relations,
        candidate_updates=candidate_updates,
        collapsed_evidence=[],
        unusable_ui_steps=ui_steps,
        source_concepts=source_concepts,
        video_fact_claims=video_fact_claims,
    )


# 内部脚手架标记与事实 ID：不得出现在生成 prompt 的参考段与任何产出文本中
_INTERNAL_ID_RE = re.compile(r"\bvf\d+_\d+_[a-z_]+\b")
_INTERNAL_MARKER_RE = re.compile(r"<<<.*?>>>", re.DOTALL)


def _sanitize_intent_reference(draft: str) -> str:
    """thought_draft → 该步意图参考：剥离内部标记、事实 ID 与多余空白。

    参考只说明「本步要验证什么」，供逐步生成使用；禁止照抄进 Thought。
    """
    text = strip_evidence_intent(draft or "")
    text = _INTERNAL_MARKER_RE.sub("", text)
    text = _INTERNAL_ID_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


def _align_intent_reference_to_action(reference: str, tool_name: str) -> str:
    """按本步 Action.tool 改写意图参考：去掉异工具话术，标明本步工具。"""
    text = reference or ""
    text = re.sub(
        r"web_search|以图搜图|打开网页|网页搜索|搜索引擎|google\s*maps",
        "观察",
        text,
        flags=re.I,
    )
    if tool_name in COARSE_CORE_TOOLS:
        text = re.sub(
            r"(?:调用|调取)\s*(?:卫星|遥感)(?:检索|接口|API|服务)?",
            "查看画面",
            text,
            flags=re.I,
        )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return f"本步工具为 {tool_name}；说明为何调用它。"
    return f"本步工具为 {tool_name}；观察目标：{text}"[:300]


def _thought_mentions_tool(thought: str, tool_name: str) -> bool:
    """Thought 是否体现本步 tool（snake_case 或常见中文动机词）。"""
    if re.search(
        re.escape(tool_name).replace("_", r"[_ ]?"), thought, re.I
    ):
        return True
    aliases: dict[str, re.Pattern[str]] = {
        "zoom_inspect": re.compile(r"放大|缩放|局部观察|细看|inspect", re.I),
        "ocr": re.compile(r"OCR|读字|识别文字|路牌|店招", re.I),
        "sun_position_calc": re.compile(r"太阳|阴影|日照|方位角", re.I),
        "compare_images_for_geolocation": re.compile(r"比对|对比.*图", re.I),
        "lookup_historical_satellite_map": re.compile(
            r"历史卫星|遥感|卫星地图|历史影像", re.I
        ),
        "lookup_historical_map_layout": re.compile(r"历史地图|地图布局", re.I),
        "annotate_geographic_environment_on_image": re.compile(
            r"标注|地理环境", re.I
        ),
        "detect_terrain_features": re.compile(r"地形|地貌特征", re.I),
        "analyze_terrain_ambiguity": re.compile(r"地形|视觉误差|误判", re.I),
        "analyze_terrain_visual_illusion": re.compile(
            r"视觉错觉|视觉误差|误认", re.I
        ),
        "find_specific_features_in_satellite_map": re.compile(
            r"卫星.*特征|地物匹配", re.I
        ),
    }
    pat = aliases.get(tool_name)
    return bool(pat and pat.search(thought))


def _thought_action_mismatch_issues(thought: str, tool_name: str) -> list[str]:
    """检测 Thought 是否提及非本步 Action 的工具话术，或未引出本步 tool。"""
    issues: list[str] = []
    for foreign, pat in _FOREIGN_TOOL_RHETORIC:
        if foreign != tool_name and pat.search(thought):
            issues.append(f"mentions_foreign_tool:{foreign}")
    if tool_name in COARSE_CORE_TOOLS and _CORE_TOOL_SATELLITE_MISMATCH_RE.search(
        thought
    ):
        issues.append("core_tool_with_satellite_api_rhetoric")
    if not _thought_mentions_tool(thought, tool_name):
        issues.append(f"missing_tool_motivation:{tool_name}")
    return issues


def _context_header_lines(
    agent_role: AgentRole,
    answer_timestamp: float,
    *,
    user_query: str,
    coarse_handoff: Optional[LocationHypothesis],
    fine_handoff: Optional[SubmitAnswerResult],
    revision_context: Optional[RevisionContext],
) -> list[str]:
    """逐步生成与角色输出共用的上下文头（不含 groundtruth）。"""
    lines = [
        f"agent_role: {agent_role.value}",
        f"user_query（任务与已知线索）: {user_query}",
        f"answer_timestamp: {answer_timestamp}",
    ]
    if agent_role in (AgentRole.COARSE, AgentRole.FINE):
        lines.append("时间规则：只使用 answer_timestamp 之前的证据。")
    else:
        lines.append(
            "时间规则：VERIFIER 可使用答案宣布后的验证片段，"
            "但直接宣布答案的语句不能作为验证证据。"
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
            if agent_role == AgentRole.COARSE:
                sanitized = sanitize_verification_for_coarse_prompt(
                    revision_context.verification_result
                )
                lines.append(
                    "previous_verification(abstract codes only): "
                    f"{json.dumps(sanitized, ensure_ascii=False)}"
                )
            else:
                lines.append(
                    "previous_verification: "
                    f"{revision_context.verification_result.model_dump_json()}"
                )
        if revision_context.video_segment is not None:
            lines.append(f"video_segment: {revision_context.video_segment!r}")
    return lines


def _obs_repr_for_prompt(
    action: Action,
    obs: ObservationExecutionResult,
) -> Optional[dict[str, Any]]:
    """步骤 Observation 在 prompt 中的表示；terminal 步为 None。"""
    if obs.status == "skipped" or action.tool == "submit_answer":
        return None
    return obs.observation


def _prior_steps_block(
    thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> str:
    """已完成的前序步骤 → prompt 文本（完整 T/A/O）。"""
    if not thoughts:
        return "（无，本步是第一步）"
    lines: list[str] = []
    for i, (thought, (_d, action, obs, _s)) in enumerate(
        zip(thoughts, units, strict=False), start=1
    ):
        lines.append(f"### 已完成 Step {i}")
        lines.append(f"Thought: {thought}")
        lines.append(f"Action: tool={action.tool} params={action.params!r}")
        lines.append(f"Observation: {_obs_repr_for_prompt(action, obs)!r}")
    return "\n".join(lines)


def _build_step_prompt(
    agent_role: AgentRole,
    header_lines: list[str],
    prior_thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    step_index: int,
) -> str:
    """第 step_index 步（0-based）的逐步生成 prompt。

    上下文只含前序完整 T/A/O 与本步 Action + 消毒意图参考；
    不含本步 Observation 与任何后续步信息。
    """
    draft, action, _obs, _step = units[step_index]
    total = len(units)
    reference = _align_intent_reference_to_action(
        _sanitize_intent_reference(draft), action.tool
    )
    lines = [
        f"你正在为一个图片地理定位 Agent 撰写第 {step_index + 1}/{total} 步的 Thought。",
        *header_lines,
        "\n## 已完成的前序步骤",
        _prior_steps_block(prior_thoughts, units),
        "\n## 本步将执行的 Action（已确定，不可更改）",
        f"tool={action.tool} params={action.params!r}",
        "\n## 本步意图参考（仅说明观察目标；禁止照抄，禁止当作已知结论）\n"
        f"{reference}",
    ]
    lines.extend(
        [
            "\n## 要求",
            "1. Thought 只能基于 user_query、图像与前序 Observation 已出现的信息；",
            f"2. 必须说明为何调用本步工具 `{action.tool}`（1~3 句）；"
            "动机必须与该 tool 一致；",
            "3. 禁止提及未出现在本步 Action 的工具名"
            "（尤其 web_search / map_query / reverse_image_search）；"
            f"当本步为 `{action.tool}` 时，勿写成正在调用其它检索/卫星 API；",
            "4. 禁止提前写出本步工具将返回的内容；",
            "5. 禁止旁白叙事（博主/网友/视频等字眼）与来源话术；",
            "6. 禁止出现内部编号或标记。",
            "只输出本步 Thought。",
        ]
    )
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


def _generate_thoughts_stepwise(
    agent_role: AgentRole,
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    header_lines: list[str],
    image_path: str,
) -> list[str]:
    """逐步因果生成：第 t 步只见前 t-1 步完整 T/A/O 与本步 Action。"""
    thoughts: list[str] = []
    for idx in range(len(units)):
        tool_name = units[idx][1].tool
        prompt = _build_step_prompt(agent_role, header_lines, thoughts, units, idx)
        result = call_structured(prompt, _StepThought, images=[image_path])
        thought = result.thought.strip()
        mismatch = _thought_action_mismatch_issues(thought, tool_name)
        if mismatch:
            retry_prompt = (
                prompt
                + "\n\n## 上次 Thought 被拒原因\n"
                + f"与本步 Action tool=`{tool_name}` 不对齐：{mismatch}。\n"
                + f"请重写：只解释为何调用 `{tool_name}`，不要提其它工具；"
                "禁止复述任何工具返回结果或 Observation 事实"
                "（本提示不提供本步 Observation）。\n"
            )
            result = call_structured(
                retry_prompt, _StepThought, images=[image_path]
            )
            thought = result.thought.strip()
        thoughts.append(thought)
    return thoughts


def _trajectory_text_block(
    thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> str:
    """完整轨迹 → prompt 文本（polish / 角色输出 / judge 共用）。"""
    lines: list[str] = []
    for i, (thought, (_d, action, obs, _s)) in enumerate(
        zip(thoughts, units, strict=True), start=1
    ):
        lines.append(f"### Step {i}")
        lines.append(f"Thought: {thought}")
        lines.append(f"Action: tool={action.tool} params={action.params!r}")
        lines.append(f"Observation: {_obs_repr_for_prompt(action, obs)!r}")
    return "\n".join(lines)


def _polish_thoughts(
    agent_role: AgentRole,
    thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
) -> list[str]:
    """整链润色：只改措辞与衔接；忠实性对比失败的步回退为润色前文本。

    polish / 忠实性调用失败或条数不符时整体回退，不阻断流程。
    """
    n = len(units)
    polish_prompt = (
        "请对下列图片地理定位推理链的 Thought 逐步润色：改善措辞、步间衔接与"
        "流畅度，使其读起来是自然的第一人称地理推理。\n"
        "严禁：改动事实、结论、候选地点、工具意图；增删步骤；调换顺序；"
        "引入新信息或内部编号。\n"
        f"输出 thoughts 必须恰好 {n} 条，与步骤一一对应。\n\n"
        "## 风格示例\n"
        f"{fewshot_block_for_role(agent_role)}\n\n"
        "## 待润色轨迹\n"
        f"{_trajectory_text_block(thoughts, units)}"
    )
    try:
        polished = call_structured(polish_prompt, _PolishedThoughts)
    except Exception:  # noqa: BLE001
        return list(thoughts)
    cleaned = [(t or "").strip() for t in polished.thoughts]
    if len(cleaned) != n or any(not t for t in cleaned):
        return list(thoughts)

    pairs = "\n".join(
        f"### Step {i}\n润色前: {orig}\n润色后: {new}"
        for i, (orig, new) in enumerate(zip(thoughts, cleaned, strict=True), start=1)
    )
    faith_prompt = (
        "对比每一步润色前后的 Thought，找出「润色后」新增、删除或改变了"
        "事实/结论/候选地点/工具意图的步骤（仅措辞与衔接变化不算）。\n"
        "在 unfaithful_steps 中列出这些步骤的序号（1-based）；全部忠实则为空列表。\n\n"
        f"{pairs}"
    )
    try:
        check = call_structured(faith_prompt, _FaithfulnessCheck)
    except Exception:  # noqa: BLE001
        return list(thoughts)
    out = list(cleaned)
    for step_no in check.unfaithful_steps:
        if 1 <= step_no <= n:
            out[step_no - 1] = thoughts[step_no - 1]
    return out


def _hard_check_issues(
    thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    *,
    obs_overlap_threshold: float = 0.6,
    narration_overlap_threshold: float = 0.55,
) -> list[str]:
    """轻量程序化硬校验：内部 ID 泄漏 / 本步 Observation 复述 / 旁白复述。"""
    issues: list[str] = []
    for i, (thought, (_d, _action, obs, step)) in enumerate(
        zip(thoughts, units, strict=True), start=1
    ):
        if _INTERNAL_ID_RE.search(thought) or _INTERNAL_MARKER_RE.search(thought):
            issues.append(f"internal_id_leak: Step {i}")
        obs_text = _obs_brief_text(obs)
        if (
            obs_text
            and _narration_overlap_ratio(thought, obs_text)
            >= obs_overlap_threshold
        ):
            issues.append(f"thought_observation_redundancy: Step {i}")
        narr = (step.move.narration or "").strip()
        if (
            narr
            and _narration_overlap_ratio(thought, narr)
            >= narration_overlap_threshold
        ):
            issues.append(f"narration_copy: Step {i}")
    return issues


_JUDGE_RUBRIC = (
    "按以下固定 rubric 对轨迹整体打分（score ∈ [0,1]）；"
    "Agent1/COARSE 优先「可用可学」入库，勿因轻微瑕疵整段判废：\n"
    "A. 递进性：每一步应推进定位（建立可用证据或排除/收窄候选）。"
    "**严重**：无增益地重复同 tool+params，或连续空转 zoom（Observation 无新证据）；"
    "投影已折叠后的短链不算空转；\n"
    "B. 因果与对齐：Thought 只基于此前已出现的信息，动机与本步 Action 一致，"
    "无预知本步结果、无跳步；Action 与 Observation 模态应一致"
    "（如 zoom_inspect 不应表现为独立卫星 API 调用）；"
    "短语已在视频来源事实中出现、但前序 Obs 尚未建立时，判「链内过早/预知」，"
    "勿写成「无视频来源」；\n"
    "C. 来源接地：事实性描述可由工作范围/前序 Observation/视频来源事实支撑；"
    "若短语可被上方「视频来源事实」清单蕴含，**不得**判「无视频来源/凭空发明」；"
    "无来源仅适用于清单与前序 Obs 均无法支撑的属性；"
    "Obs 多写了画面未见、但旁白或来源事实已有的地名，属**轻微瑕疵**，不单独判严重；\n"
    "D. 结论闭包：最终输出从推理链已建立的候选中得出，无首次跳入的新地点；\n"
    "E. 语言：自然流畅的第一人称地理推理，无旁白叙事体、内部编号与模板腔。"
    "ASR/字幕个别错字不单独判严重。\n"
    "评分基准：五项全部良好 ≥0.8；一项存在严重问题 ≤0.4；"
    "两项以上严重问题 ≤0.2；"
    "**仅因**轻微 Obs 过写或 ASR/字幕错字 → **不应** ≤0.4。"
    "issues 逐条写明扣分原因。"
)


def _judge_evidence_block(ledger: CoarseEvidenceLedger) -> str:
    """judge 用视频来源事实清单（核对来源接地；不进入生成 prompt）。"""
    lines = [
        "## 视频来源事实（仅供核对来源接地；轨迹文本中不应出现这些编号）",
        f"working_scope: {ledger.working_scope or '（无）'}",
        f"given_clues: {ledger.given_clues}",
        f"candidate_hypotheses(待证，非证据): {ledger.candidate_hypotheses}",
    ]
    for fid, quote in ledger.video_fact_claims.items():
        lines.append(f"- {fid}: {quote}")
    return "\n".join(lines)


def _judge_trajectory(
    agent_role: AgentRole,
    thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    *,
    user_query: str,
    coarse_output: Optional[LocationHypothesis] = None,
    fine_output: Optional[SubmitAnswerResult] = None,
    verifier_output: Optional[VerificationResult] = None,
    evidence_ledger: Optional[CoarseEvidenceLedger] = None,
) -> _TrajectoryJudgement:
    """固定 rubric 的轨迹质量评分（无 groundtruth）。"""
    lines = [
        "你是训练数据质量裁判。评估下列图片地理定位 Agent 轨迹。",
        _JUDGE_RUBRIC,
        f"agent_role: {agent_role.value}",
        f"user_query: {user_query}",
    ]
    if evidence_ledger is not None:
        lines.append(_judge_evidence_block(evidence_ledger))
    lines.append("\n## 轨迹")
    lines.append(_trajectory_text_block(thoughts, units))
    if coarse_output is not None:
        lines.append(f"\nfinal coarse_output: {coarse_output.model_dump_json()}")
    if fine_output is not None:
        lines.append(f"\nfinal fine_output: {fine_output.model_dump_json()}")
    if verifier_output is not None:
        lines.append(
            f"\nfinal verifier_output: {verifier_output.model_dump_json()}"
        )
    return call_structured("\n".join(lines), _TrajectoryJudgement)


def _generate_role_output(
    agent_role: AgentRole,
    thoughts: list[str],
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    header_lines: list[str],
    image_path: str,
) -> tuple[Optional[LocationHypothesis], Optional[VerificationResult]]:
    """由完整 TAO 生成角色结构化输出（FINE 从末步 submit_answer 程序化抽取）。"""
    if agent_role == AgentRole.FINE:
        return None, None
    context = "\n".join(
        [
            *header_lines,
            "\n## 完整推理链",
            _trajectory_text_block(thoughts, units),
        ]
    )
    if agent_role == AgentRole.COARSE:
        prompt = (
            "根据下列完整推理链输出 LocationHypothesis。\n"
            "要求：possible_regions 仅填同层级规范行政区；"
            "命名自然区域写入 reasoning_summary / key_clues_remaining；"
            "reasoning_summary 概括「工作范围 → 关键证据 → 排除/收窄 → 候选」；"
            "不得给出精确 POI 或坐标；不得引入推理链之外的新地点。\n\n"
            f"{context}"
        )
        hyp = call_structured(prompt, LocationHypothesis, images=[image_path])
        return hyp, None
    prompt = (
        "根据下列完整验证推理链输出 VerificationResult"
        "（把 fine_handoff 当作待验证候选，不得当作已知正确答案）。\n"
        "verdict 必须与推理链中的核对结果一致；"
        "fail 时写明 failed_checks 与 suggested_recheck。\n\n"
        f"{context}"
    )
    ver = call_structured(prompt, VerificationResult, images=[image_path])
    return None, ver


def _to_trajectory_steps(
    units: list[tuple[str, Action, ObservationExecutionResult, NormalizedStep]],
    thoughts: list[str],
) -> list[TrajectoryStep]:
    """脚手架 + 生成 Thought → TrajectoryStep；terminal 步 observation 均为 None。"""
    out: list[TrajectoryStep] = []
    for thought, (_draft, action, obs, _step) in zip(thoughts, units, strict=True):
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


def _extract_clue_buckets(
    narrations: list[str],
    agent_role: AgentRole,
) -> tuple[list[str], list[str]]:
    """抽取 given_clues 与 candidate_hypotheses；失败则空。不含 groundtruth。"""
    if not narrations:
        return [], []
    if agent_role == AgentRole.VERIFIER:
        return [], []
    joined = "\n".join(f"- {n}" for n in narrations[:40])
    prompt = (
        "从下列视频旁白片段中，分类抽取地名线索（不含真值、不编造）。\n"
        "A. given_clues：问题设置阶段由网友/评论/求助者**直接给出**的软先验地名"
        "（尚未进入博主地貌纠错/候选排除推理）。\n"
        "B. candidate_hypotheses：推理过程中**首次由博主演绎提出**的待证候选"
        "（如纠正后提出的自然区域/城市），不得当作给定线索。\n"
        "规则：\n"
        "1. 只保留地名短语本身，不要保留「网友说/评论说」等来源话术；\n"
        "2. 排除博主宣布最终答案、揭晓坐标的句子；\n"
        "3. 排除纯视觉描述（无明确地名）；\n"
        "4. 两类互斥；若不确定归入 candidate_hypotheses；\n"
        "5. 可只填 hints（兼容）：将视为 given_clues。\n"
        f"agent_role: {agent_role.value}\n"
        f"narrations:\n{joined}\n"
    )
    try:
        result = call_structured(prompt, _ExternalHints)
    except Exception:  # noqa: BLE001
        return [], []

    def _clean(items: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for h in items:
            text = h.strip()
            if not text or text in seen:
                continue
            text = re.sub(
                r"^(?:网友|粉丝|评论|弹幕|有人)(?:说|提到|给出|告诉)[：:，,\s]*",
                "",
                text,
            ).strip()
            if text and text not in seen:
                seen.add(text)
                cleaned.append(text)
        return cleaned[:8]

    given = _clean(list(result.given_clues) or list(result.hints))
    cands = _clean(list(result.candidate_hypotheses))
    # 互斥：候选不得重复进入 given
    given_set = set(given)
    cands = [c for c in cands if c not in given_set]
    return given, cands


def _extract_external_hints(
    narrations: list[str],
    agent_role: AgentRole,
) -> list[str]:
    """兼容旧接口：仅返回 given_clues。"""
    given, _cands = _extract_clue_buckets(narrations, agent_role)
    return given


def _user_query_for_role(
    agent_role: AgentRole,
    *,
    coarse_handoff: Optional[LocationHypothesis],
    fine_handoff: Optional[SubmitAnswerResult],
    external_hints: Optional[list[str]] = None,
    working_scope_region: Optional[str] = None,
) -> str:
    """构造 user_query（不含 groundtruth；COARSE 仅注入有效拍摄地 working_scope）。"""
    if agent_role == AgentRole.COARSE:
        if working_scope_region:
            from pipeline.evidence_routing import WorkingScope

            return format_working_scope_user_query(
                WorkingScope(region=working_scope_region)
            )
        # 无有效拍摄地工作范围：不注入人物属性等 external_hints
        return "请根据图像进行粗定位，缩小到可能的国家/地区。"
    hint_suffix = ""
    if external_hints:
        hint_suffix = "\n已知线索：" + "；".join(external_hints)
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
    """逐步因果生成 → polish → 轻量硬校验 → judge best-of-k 拒绝采样。

    第 t 步 Thought 只见前 t-1 步完整 T/A/O 与本步 Action（不含本步 Observation）。
    全部候选低于 STAGE5_JUDGE_THRESHOLD → raise TrajectoryQualityRejected。
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

    # 外部线索：COARSE 仅用 stage3 VideoChainContext（投影前解析，避免首步被剔除丢失）
    narrations = _collect_pre_answer_narrations(units, answer_timestamp)
    video_ctx: Optional[VideoChainContext] = None
    if agent_role == AgentRole.COARSE:
        for draft, _a, _o, _s in units:
            video_ctx = parse_video_context(draft)
            if video_ctx is not None:
                break
        if video_ctx is None:
            for step in steps:
                video_ctx = parse_video_context(step.thought_draft)
                if video_ctx is not None:
                    break
        units = _project_coarse_units(units)
        # 投影后若首步无嵌入，则回写 VideoChainContext，保证 user_query/账本可用
        if video_ctx is not None and units:
            d0, a0, o0, s0 = units[0]
            if parse_video_context(d0) is None:
                units[0] = (embed_video_context(d0, video_ctx), a0, o0, s0)

    given_clues: list[str] = []
    candidate_hypotheses: list[str] = []
    working_scope_region: Optional[str] = None
    if agent_role == AgentRole.COARSE and video_ctx is not None:
        given_clues = [c.text for c in video_ctx.raw_given_clues]
        candidate_hypotheses = list(video_ctx.candidate_hypotheses)
        if video_ctx.working_scope is not None:
            working_scope_region = video_ctx.working_scope.region
    elif agent_role != AgentRole.COARSE:
        given_clues, candidate_hypotheses = _extract_clue_buckets(
            narrations, agent_role
        )
    # COARSE 无 video_ctx 时：不再从全部旁白自由重抽地名
    external_hints = list(given_clues)
    user_query = _user_query_for_role(
        agent_role,
        coarse_handoff=coarse_handoff,
        fine_handoff=fine_handoff,
        external_hints=external_hints if agent_role != AgentRole.COARSE else None,
        working_scope_region=working_scope_region,
    )
    evidence_ledger: Optional[CoarseEvidenceLedger] = None
    if agent_role == AgentRole.COARSE:
        evidence_ledger = _build_coarse_evidence_ledger(
            units,
            given_clues=given_clues,
            candidate_hypotheses=candidate_hypotheses,
        )
        if (
            not evidence_ledger.visual_facts
            and not evidence_ledger.video_fact_claims
            and len(units) > 0
            and all(
                u[2].status in ("empty", "error")
                or _is_ui_overlay_observation(u[2])
                for u in units
            )
        ):
            raise ValueError(
                "COARSE 证据不足：无可用 visual_facts 且无 video_fact_claims"
                "（UI/empty Observation）。"
                "通常由 stage3 来源抽取失败后的坏链或 stage4 Obs 全部 empty 引起；"
                "请检查上游日志后 --force-from stage3 重跑。"
            )
    header_lines = _context_header_lines(
        agent_role,
        answer_timestamp,
        user_query=user_query,
        coarse_handoff=coarse_handoff,
        fine_handoff=fine_handoff,
        revision_context=revision_context,
    )

    settings = get_settings()
    k = max(1, settings.STAGE5_BEST_OF_K)
    threshold = settings.STAGE5_JUDGE_THRESHOLD

    # FINE 的最终答案由末步 submit_answer 固定给出，仅供 judge 参考
    fine_preview: Optional[SubmitAnswerResult] = None
    if agent_role == AgentRole.FINE:
        try:
            fine_preview = SubmitAnswerResult.model_validate(units[-1][1].params)
        except ValidationError:
            fine_preview = None

    best_score = -1.0
    best_thoughts: Optional[list[str]] = None
    best_coarse: Optional[LocationHypothesis] = None
    best_verifier: Optional[VerificationResult] = None
    attempts_log: list[str] = []

    for attempt in range(1, k + 1):
        thoughts = _generate_thoughts_stepwise(
            agent_role, units, header_lines, image_path
        )
        thoughts = _polish_thoughts(agent_role, thoughts, units)
        hard_issues = _hard_check_issues(thoughts, units)
        if hard_issues:
            attempts_log.append(
                f"候选{attempt} 硬校验不通过: " + "；".join(hard_issues[:5])
            )
            continue
        cand_coarse, cand_verifier = _generate_role_output(
            agent_role, thoughts, units, header_lines, image_path
        )
        judgement = _judge_trajectory(
            agent_role,
            thoughts,
            units,
            user_query=user_query,
            coarse_output=cand_coarse,
            fine_output=fine_preview,
            verifier_output=cand_verifier,
            evidence_ledger=evidence_ledger,
        )
        attempts_log.append(
            f"候选{attempt} score={judgement.score:.2f}"
            + ("：" + "；".join(judgement.issues[:3]) if judgement.issues else "")
        )
        if judgement.score > best_score:
            best_score = judgement.score
            best_thoughts = thoughts
            best_coarse = cand_coarse
            best_verifier = cand_verifier

    if best_thoughts is None or best_score < threshold:
        raise TrajectoryQualityRejected(
            f"{agent_role.value} best-of-{k} 全部候选低于阈值 {threshold}："
            + " | ".join(attempts_log)
        )

    thoughts = best_thoughts
    coarse_output: Optional[LocationHypothesis] = best_coarse
    fine_output: Optional[SubmitAnswerResult] = None
    verifier_output: Optional[VerificationResult] = best_verifier

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
            if agent_role == AgentRole.COARSE:
                revision_input = sanitize_revision_input_for_coarse_shard(
                    revision_input
                )

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
        stage5_judge_score=best_score,
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

    COARSE 拒绝采样失败仍抛出；FINE/VERIFIER 失败则跳过该角色（不阻断 COARSE 入库）。
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

    result: dict[AgentRole, Trajectory] = {AgentRole.COARSE: coarse}

    try:
        fine = reconstruct_single_trajectory(
            all_steps[AgentRole.FINE],
            all_observations[AgentRole.FINE],
            AgentRole.FINE,
            answer_timestamp,
            image_path,
            coarse_handoff=coarse.coarse_output,
        )
    except TrajectoryQualityRejected as exc:
        logger.warning("FINE 轨迹废弃（不阻断 COARSE）: %s", exc)
        return result

    if fine.fine_output is None:
        logger.warning("Agent2 未产出 fine_output；跳过 FINE/VERIFIER")
        return result

    result[AgentRole.FINE] = fine

    try:
        verifier = reconstruct_single_trajectory(
            all_steps[AgentRole.VERIFIER],
            all_observations[AgentRole.VERIFIER],
            AgentRole.VERIFIER,
            answer_timestamp,
            image_path,
            coarse_handoff=coarse.coarse_output,
            fine_handoff=fine.fine_output,
        )
    except TrajectoryQualityRejected as exc:
        logger.warning("VERIFIER 轨迹废弃（不阻断 COARSE/FINE）: %s", exc)
        return result

    if verifier.verifier_output is None:
        logger.warning("Agent3 未产出 verifier_output；跳过 VERIFIER")
        return result

    result[AgentRole.VERIFIER] = verifier
    return result


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
