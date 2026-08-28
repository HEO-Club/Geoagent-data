"""标准 Trajectory ↔ DatasetEntry JSONL。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pipeline.config import get_settings
from pipeline.schemas.clues import WorkingScope
from pipeline.schemas.dataset import ChatMessage, DatasetEntry
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.tools import ToolForest, ToolParameterAudit
from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from pipeline.stage3_normalize_format.compile_params import (
    ParamCompilerFn,
    apply_compile_and_revalidate,
    build_compile_request,
    compile_params_batch,
)
from pipeline.stage3_normalize_format.map_tools import ensure_tool_trees
from pipeline.stage3_normalize_format.params import (
    initial_parameter_context,
    normalize_and_validate_tool_inputs,
    update_parameter_context,
)
from pipeline.stage3_normalize_format.trees import (
    find_tree_for_name,
    resolve_canonical_name,
    resolve_operation,
)
from pipeline.tool_catalog_v2 import render_tool_contract_guidance

DEFAULT_SYSTEM_PROMPT = (
    "You are a geolocation agent. Use internal Thought events for reasoning and call "
    "tools only when an external executor is needed. Tool calls follow "
    "Thought/Action/Observation; the final answer uses final_answer. "
    "Every non-terminal Action uses params.operation, params.purpose and params.inputs."
)
DEFAULT_USER_QUERY = "Locate the place shown in the image."


def build_user_query(working_scope: WorkingScope | None = None) -> str:
    """由 working_scope 生成训练用 user_query（仅展示短语，无来源话术）。"""
    if working_scope is None or not working_scope.region.strip():
        return DEFAULT_USER_QUERY
    return f"{DEFAULT_USER_QUERY}\nWorking scope: {working_scope.region.strip()}"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _normalize_observation(
    observation: dict[str, Any] | None,
    *,
    is_terminal: bool,
) -> dict[str, Any] | None:
    if is_terminal:
        return None
    if observation is None:
        return {"result": None}
    return observation


def _resolve_image_paths(
    *,
    image_paths: list[str] | None,
    image_path: str | None,
) -> list[str]:
    """解析任务图路径；无图返回空列表，不伪造占位路径。"""
    if image_paths:
        cleaned = [p.strip() for p in image_paths if str(p).strip()]
        if cleaned:
            return cleaned
    if image_path and str(image_path).strip():
        return [str(image_path).strip()]
    return []


def _freeform_call_parts(step: Any) -> tuple[str | None, str, dict[str, Any]]:
    """Accept both legacy flat params and the v2 Stage 2 call contract."""

    params = dict(step.params or {})
    nested_inputs = params.get("inputs")
    if isinstance(nested_inputs, dict):
        operation = str(params.get("operation") or "").strip() or None
        purpose = str(params.get("purpose") or step.thought).strip() or step.thought
        return operation, purpose, dict(nested_inputs)
    operation = str(params.pop("operation", "") or "").strip() or None
    purpose = str(params.pop("purpose", "") or step.thought).strip() or step.thought
    return operation, purpose, params


def remap_trajectory(
    freeform: FreeFormTrajectory,
    forest: ToolForest,
    *,
    system_prompt: str,
    user_query: str,
    image_paths: list[str] | None = None,
    image_path: str | None = None,
    trajectory_id: str | None = None,
    parameter_audits: list[ToolParameterAudit] | None = None,
    param_compiler: ParamCompilerFn | None = None,
    compile_params: bool | None = None,
) -> Trajectory:
    """自由链 → 单 Agent 标准 Trajectory（canonical tools）。

    规则别名作种子后，每个有 schema 的 tool_call 用 LLM 对照 Thought 编译 params。
    """
    settings = get_settings()
    do_compile = (
        settings.STAGE3_COMPILE_PARAMS if compile_params is None else compile_params
    )

    resolved_image_paths = _resolve_image_paths(
        image_paths=image_paths, image_path=image_path
    )
    available_context = initial_parameter_context(resolved_image_paths)

    # 第一轮：规则别名 + 上下文补全（种子）；每个有 schema 的 tool_call 再编译
    drafts: list[dict[str, Any]] = []
    compile_requests = []
    for step_index, step in enumerate(freeform.steps, start=1):
        if step.event_type == "reasoning":
            drafts.append(
                {
                    "kind": "reasoning",
                    "step": step,
                    "step_index": step_index,
                }
            )
            continue

        assert step.tool is not None
        canonical = resolve_canonical_name(forest, step.tool) or step.tool
        tree = find_tree_for_name(forest, canonical)
        # ``final_answer`` 是阶段2的保留终端契约；即使旧 tool 树曾将其
        # 错误映射到非终端 canonical，也不得在答案后补伪造的 result=null。
        is_terminal = step.tool == "final_answer" or (
            bool(tree.canonical.is_terminal) if tree else False
        )
        if is_terminal:
            drafts.append(
                {
                    "kind": "final",
                    "step": step,
                    "step_index": step_index,
                    "canonical": canonical,
                    "normalized_params": dict(step.params or {}),
                }
            )
            continue

        requested_operation, purpose, raw_inputs = _freeform_call_parts(step)
        operation = requested_operation or resolve_operation(forest, step.tool)
        if not operation and tree and tree.canonical.operations:
            operation = tree.canonical.operations[0].name
        context_snapshot = dict(available_context)
        parameter_audit = normalize_and_validate_tool_inputs(
            forest,
            tool=canonical,
            operation=operation or "execute",
            inputs=raw_inputs,
            step_index=step_index,
            available_context=available_context,
        )
        req = None
        if do_compile:
            req = build_compile_request(
                forest=forest,
                audit=parameter_audit,
                thought=step.thought,
                available_context=context_snapshot,
            )
            if req is not None:
                compile_requests.append(req)
        drafts.append(
            {
                "kind": "tool_call",
                "step": step,
                "step_index": step_index,
                "canonical": canonical,
                "audit": parameter_audit,
                "compile_request": req,
                "purpose": purpose,
            }
        )
        update_parameter_context(available_context, parameter_audit)

    fills: dict[int, Any] = {}
    if compile_requests and do_compile:
        # 生产：默认 LLM；测试：须注入 param_compiler（ALLOW_REAL_API=false 时不自动打网）
        if param_compiler is not None:
            fills = compile_params_batch(
                compile_requests, compiler=param_compiler
            )
        elif settings.ALLOW_REAL_API:
            fills = compile_params_batch(compile_requests, compiler=None)
        # else: 测试环境未注入编译器 → 保留规则审计（失败开放）

    # 第二轮：对有编译结果的步骤合并并重校验，组装 Trajectory
    steps: list[TrajectoryStep] = []
    for draft in drafts:
        kind = draft["kind"]
        step = draft["step"]
        if kind == "reasoning":
            steps.append(
                TrajectoryStep(
                    event_type="reasoning",
                    thought=step.thought,
                    action=None,
                    observation=None,
                )
            )
            continue
        if kind == "final":
            steps.append(
                TrajectoryStep(
                    event_type="final",
                    thought=step.thought,
                    action=Action(
                        tool=draft["canonical"],
                        params=draft["normalized_params"],
                    ),
                    observation=_normalize_observation(
                        step.observation, is_terminal=True
                    ),
                )
            )
            continue

        audit: ToolParameterAudit = draft["audit"]
        req = draft.get("compile_request")
        if req is not None and draft["step_index"] in fills:
            audit = apply_compile_and_revalidate(
                forest, req, fills[draft["step_index"]]
            )
        if parameter_audits is not None:
            parameter_audits.append(audit)
        normalized_params = {
            "operation": audit.operation,
            "purpose": draft["purpose"],
            "inputs": audit.normalized_inputs,
        }
        steps.append(
            TrajectoryStep(
                event_type="tool_call",
                thought=step.thought,
                action=Action(tool=draft["canonical"], params=normalized_params),
                observation=_normalize_observation(
                    step.observation, is_terminal=False
                ),
            )
        )

    return Trajectory(
        id=trajectory_id or str(uuid.uuid4()),
        system_prompt=system_prompt,
        user_query=user_query,
        image_paths=resolved_image_paths,
        steps=steps,
    )


def _format_assistant_content(step: TrajectoryStep) -> str:
    if step.event_type == "reasoning":
        return f"Thought: {step.thought}"
    assert step.action is not None
    action_payload = {"tool": step.action.tool, "params": step.action.params}
    return f"Thought: {step.thought}\nAction: {_json_dumps(action_payload)}"


def trajectory_to_messages(traj: Trajectory) -> list[ChatMessage]:
    """将 Trajectory 展开为 chat messages（loss：仅 assistant）。"""
    user_content = traj.user_query
    if traj.image_paths:
        images_block = "\n".join(f"[Image: {p}]" for p in traj.image_paths)
        user_content = f"{traj.user_query}\n{images_block}"
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=traj.system_prompt),
        ChatMessage(role="user", content=user_content),
    ]
    for step in traj.steps:
        messages.append(
            ChatMessage(role="assistant", content=_format_assistant_content(step))
        )
        if step.observation is not None:
            messages.append(
                ChatMessage(role="tool", content=_json_dumps(step.observation))
            )
    return messages


def format_dataset_entry(traj: Trajectory, *, source_video: str) -> DatasetEntry:
    """Trajectory → 单条 JSONL DatasetEntry。quality_score 由阶段4回写。"""
    return DatasetEntry(
        id=traj.id,
        source_video=source_video,
        messages=trajectory_to_messages(traj),
        quality_score=None,
    )


def _tool_mapping_audit(
    raw_events: list[dict[str, Any]],
    freeform: FreeFormTrajectory,
    forest: ToolForest,
) -> dict[str, Any]:
    mappings: list[dict[str, Any]] = []
    raw_tools: set[str] = set()
    canonicals: set[str] = set()
    demoted = 0
    tool_calls_before = 0
    tool_calls_after = 0
    for index, (raw, step) in enumerate(zip(raw_events, freeform.steps), start=1):
        raw_type = str(raw["event_type"])
        raw_tool = raw.get("tool")
        if raw_type == "tool_call" and raw_tool:
            tool_calls_before += 1
            raw_tools.add(str(raw_tool))
        if raw_type == "tool_call" and step.event_type == "reasoning":
            demoted += 1

        canonical = None
        operation = None
        if step.event_type == "tool_call" and step.tool:
            tool_calls_after += 1
            canonical = resolve_canonical_name(forest, step.tool) or step.tool
            requested_operation, _, _ = _freeform_call_parts(step)
            operation = (
                requested_operation
                or resolve_operation(forest, step.tool)
                or "execute"
            )
            canonicals.add(canonical)
        mappings.append(
            {
                "index": index,
                "raw_event_type": raw_type,
                "raw_tool": raw_tool,
                "normalized_event_type": step.event_type,
                "canonical_tool": canonical,
                "operation": operation,
                "reclassified": bool(
                    raw_tool
                    and step.tool
                    and str(raw_tool).strip().lower() != str(step.tool).strip().lower()
                ),
            }
        )

    reclassified_count = sum(1 for item in mappings if item.get("reclassified"))
    return {
        "total_events": len(freeform.steps),
        "reasoning_events": sum(
            step.event_type == "reasoning" for step in freeform.steps
        ),
        "tool_calls_before_stage3": tool_calls_before,
        "tool_calls_after_stage3": tool_calls_after,
        "pseudo_tools_demoted": demoted,
        "tools_reclassified": reclassified_count,
        "unique_raw_tools": len(raw_tools),
        "unique_canonical_tools": len(canonicals),
        "canonical_compression_ratio": (
            round(len(canonicals) / len(raw_tools), 4) if raw_tools else 0.0
        ),
        "mappings": mappings,
    }


def run_stage3(
    freeform: FreeFormTrajectory,
    *,
    trees_path: str | Path | None = None,
    out_trajectory_path: str | None = None,
    out_jsonl_path: str | None = None,
    image_path: str = "",
    image_paths: list[str] | None = None,
    shard_id: str | None = None,
    system_prompt: str | None = None,
    user_query: str | None = None,
    matcher=None,
    param_compiler: ParamCompilerFn | None = None,
    compile_params: bool | None = None,
) -> DatasetEntry:
    """阶段3 一站式：树维护 → 参数归一/审计 → DatasetEntry。"""
    settings = get_settings()
    forest_path = Path(trees_path) if trees_path else Path(settings.TOOL_TREES_PATH)
    raw_events = [
        {
            "event_type": step.event_type,
            "tool": step.tool,
            "params": dict(step.params or {}),
        }
        for step in freeform.steps
    ]
    forest = ensure_tool_trees(freeform, forest_path, matcher=matcher)
    resolved_system_prompt = system_prompt or (
        DEFAULT_SYSTEM_PROMPT + "\n\n" + render_tool_contract_guidance(forest)
    )
    resolved_query = (
        user_query
        if user_query is not None
        else build_user_query(freeform.working_scope)
    )
    parameter_audits: list[ToolParameterAudit] = []
    traj = remap_trajectory(
        freeform,
        forest,
        system_prompt=resolved_system_prompt,
        user_query=resolved_query,
        image_paths=image_paths,
        image_path=image_path or None,
        trajectory_id=shard_id,
        parameter_audits=parameter_audits,
        param_compiler=param_compiler,
        compile_params=compile_params,
    )
    entry = format_dataset_entry(traj, source_video=freeform.source_video)

    video_id = freeform.source_video
    traj_path = (
        Path(out_trajectory_path)
        if out_trajectory_path
        else (Path(settings.INTERMEDIATE_DIR) / video_id / "stage3_trajectory.json")
    )
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    audit_path = traj_path.with_name("stage3_tool_mapping.json")
    audit_path.write_text(
        json.dumps(
            _tool_mapping_audit(raw_events, freeform, forest),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    parameter_audit_path = traj_path.with_name("stage3_parameter_audit.json")
    parameter_audit_path.write_text(
        json.dumps(
            {
                "schema_version": "canonical_inputs_v2",
                "calls": [item.model_dump() for item in parameter_audits],
                "valid_calls": sum(item.valid for item in parameter_audits),
                "total_calls": len(parameter_audits),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if out_jsonl_path:
        shard = Path(out_jsonl_path)
    else:
        name = (shard_id or video_id).strip() or video_id
        shard = Path(settings.OUTPUT_DIR) / "shards" / f"{name}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(entry.model_dump_json() + "\n", encoding="utf-8")

    return entry
