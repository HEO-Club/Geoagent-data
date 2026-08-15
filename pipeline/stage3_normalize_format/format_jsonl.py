"""标准 Trajectory ↔ DatasetEntry JSONL。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

from pipeline.config import get_settings
from pipeline.schemas.clues import WorkingScope
from pipeline.schemas.dataset import ChatMessage, DatasetEntry
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.tools import ToolForest
from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from pipeline.stage3_normalize_format.map_tools import ensure_tool_trees
from pipeline.stage3_normalize_format.trees import (
    find_tree_for_name,
    resolve_canonical_name,
    resolve_operation,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a geolocation agent. Use internal Thought events for reasoning and call "
    "tools only when an external executor is needed. Tool calls follow "
    "Thought/Action/Observation; the final answer uses final_answer."
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
    observation: Optional[dict[str, Any]],
    *,
    is_terminal: bool,
) -> Optional[dict[str, Any]]:
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
    if image_paths:
        cleaned = [p.strip() for p in image_paths if str(p).strip()]
        if cleaned:
            return cleaned
    if image_path and str(image_path).strip():
        return [str(image_path).strip()]
    return ["unknown.jpg"]


def remap_trajectory(
    freeform: FreeFormTrajectory,
    forest: ToolForest,
    *,
    system_prompt: str,
    user_query: str,
    image_paths: list[str] | None = None,
    image_path: str | None = None,
    trajectory_id: str | None = None,
) -> Trajectory:
    """自由链 → 单 Agent 标准 Trajectory（canonical tools）。"""
    steps: list[TrajectoryStep] = []
    for step in freeform.steps:
        if step.event_type == "reasoning":
            steps.append(
                TrajectoryStep(
                    event_type="reasoning",
                    thought=step.thought,
                    action=None,
                    observation=None,
                )
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
            normalized_params = dict(step.params or {})
            event_type = "final"
        else:
            operation = resolve_operation(forest, step.tool)
            if not operation and tree and tree.canonical.operations:
                operation = tree.canonical.operations[0].name
            normalized_params = {
                "operation": operation or "execute",
                "purpose": step.thought,
                "inputs": dict(step.params or {}),
            }
            event_type = "tool_call"
        steps.append(
            TrajectoryStep(
                event_type=event_type,
                thought=step.thought,
                action=Action(tool=canonical, params=normalized_params),
                observation=_normalize_observation(
                    step.observation, is_terminal=is_terminal
                ),
            )
        )
    return Trajectory(
        id=trajectory_id or str(uuid.uuid4()),
        system_prompt=system_prompt,
        user_query=user_query,
        image_paths=_resolve_image_paths(
            image_paths=image_paths, image_path=image_path
        ),
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
    images_block = "\n".join(f"[Image: {p}]" for p in traj.image_paths)
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=traj.system_prompt),
        ChatMessage(
            role="user",
            content=f"{traj.user_query}\n{images_block}",
        ),
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
    """Trajectory → 单条 JSONL DatasetEntry。"""
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
            operation = resolve_operation(forest, step.tool) or "execute"
            canonicals.add(canonical)
        mappings.append(
            {
                "index": index,
                "raw_event_type": raw_type,
                "raw_tool": raw_tool,
                "normalized_event_type": step.event_type,
                "canonical_tool": canonical,
                "operation": operation,
            }
        )

    return {
        "total_events": len(freeform.steps),
        "reasoning_events": sum(
            step.event_type == "reasoning" for step in freeform.steps
        ),
        "tool_calls_before_stage3": tool_calls_before,
        "tool_calls_after_stage3": tool_calls_after,
        "pseudo_tools_demoted": demoted,
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
) -> DatasetEntry:
    """阶段3 一站式：树维护 → 重写 → DatasetEntry。"""
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
    resolved_query = (
        user_query
        if user_query is not None
        else build_user_query(freeform.working_scope)
    )
    traj = remap_trajectory(
        freeform,
        forest,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        user_query=resolved_query,
        image_paths=image_paths,
        image_path=image_path or None,
        trajectory_id=shard_id,
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

    if out_jsonl_path:
        shard = Path(out_jsonl_path)
    else:
        name = (shard_id or video_id).strip() or video_id
        shard = Path(settings.OUTPUT_DIR) / "shards" / f"{name}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(entry.model_dump_json() + "\n", encoding="utf-8")

    return entry
