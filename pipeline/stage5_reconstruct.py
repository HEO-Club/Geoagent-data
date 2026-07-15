"""stage5：三 Agent 主轨迹与 revision 轨迹重构。

本阶段禁止访问 groundtruth；函数签名不得包含 groundtruth。
LLM 仅改写前向推理 Thought，并产出角色结构化输出。
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from pipeline.llm import call_structured
from pipeline.schemas import (
    Action,
    AgentRole,
    LocationHypothesis,
    NormalizedStep,
    ObservationExecutionResult,
    RevisionContext,
    RevisionSource,
    SubmitAnswerResult,
    Trajectory,
    TrajectoryStep,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# LLM 结构化输出（仅本模块内部使用）
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 角色提示（不含任何真值坐标 / 地名）
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS: dict[AgentRole, str] = {
    AgentRole.COARSE: (
        "你是粗定位 Agent（COARSE）。根据图像宏观特征演绎缩小到国家/地区级别。"
        "禁止给出最终城市、精确地点或坐标。推理必须前向，不得使用后见之明。"
    ),
    AgentRole.FINE: (
        "你是精定位 Agent（FINE）。在粗定位假设基础上验证并锁定坐标。"
        "只有在 Observation 支持后才能使用具体地点；最后一步必须 submit_answer。"
        "推理必须前向，不得使用后见之明。"
    ),
    AgentRole.VERIFIER: (
        "你是验证 Agent（VERIFIER）。把候选 SubmitAnswerResult 与图像特征交叉验证。"
        "你看不到真实答案；只能判断候选是否与图像自洽。推理必须前向。"
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
) -> str:
    """构造轨迹改写 prompt；明确禁止后见之明与真值泄漏。"""
    lines = [
        "请将下列脚手架改写为前向 ReAct 推理链。",
        "要求：",
        "1. thoughts 列表长度必须与 Step 数量完全一致；",
        "2. Thought 只能使用该步及之前 Observation 中已出现的信息；",
        "3. 禁止后见之明；禁止编造未在 Observation 中出现的精确答案；",
        "4. 不得在 Thought 中写入真实定位答案或由真值反推的地址。",
        f"agent_role: {agent_role.value}",
        f"answer_timestamp: {answer_timestamp}",
    ]
    if agent_role in (AgentRole.COARSE, AgentRole.FINE):
        lines.append(
            "时间规则：COARSE/FINE 默认只使用 answer_timestamp 之前的证据。"
        )
    else:
        lines.append(
            "时间规则：VERIFIER 可使用答案宣布后的验证片段，"
            "但博主直接宣布答案的语句不能作为验证证据。"
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


def _user_query_for_role(
    agent_role: AgentRole,
    *,
    coarse_handoff: Optional[LocationHypothesis],
    fine_handoff: Optional[SubmitAnswerResult],
) -> str:
    """构造 user_query（不含 groundtruth）。"""
    if agent_role == AgentRole.COARSE:
        return "请根据图像进行粗定位，缩小到可能的国家/地区。"
    if agent_role == AgentRole.FINE:
        hyp = coarse_handoff.model_dump_json() if coarse_handoff else "{}"
        return f"请在粗定位假设基础上精确定位并 submit_answer。假设：{hyp}"
    cand = fine_handoff.model_dump_json() if fine_handoff else "{}"
    return f"请交叉验证以下候选定位结果是否与图像自洽：{cand}"


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

    units = _expand_action_units(steps, observations)
    prompt = _build_scaffold_prompt(
        agent_role,
        units,
        answer_timestamp,
        coarse_handoff=coarse_handoff,
        fine_handoff=fine_handoff,
        revision_context=revision_context,
    )

    # 角色结构化输出
    coarse_output: Optional[LocationHypothesis] = None
    fine_output: Optional[SubmitAnswerResult] = None
    verifier_output: Optional[VerificationResult] = None

    if agent_role == AgentRole.COARSE:
        bundle = call_structured(
            prompt
            + "\n\n请输出改写后的 thoughts，以及最终 LocationHypothesis（coarse_output）。",
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
    else:
        bundle_v = call_structured(
            prompt
            + "\n\n请输出改写后的 thoughts，以及 VerificationResult（把 fine_handoff 当候选）。",
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
        user_query=_user_query_for_role(
            agent_role,
            coarse_handoff=coarse_handoff,
            fine_handoff=fine_handoff,
        ),
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
            for target in (AgentRole.FINE, AgentRole.COARSE):
                subset = _steps_overlapping_segment(all_steps[target], segment)
                actionable = [s for s in subset if s.actions]
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
                break  # 每个 segment 只生成一条优先命中的返工轨迹

    return results
