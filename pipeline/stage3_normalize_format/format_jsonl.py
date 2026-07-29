"""标准 Trajectory ↔ DatasetEntry JSONL。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

from pipeline.config import get_settings
from pipeline.schemas.dataset import ChatMessage, DatasetEntry
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.tools import ToolForest
from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from pipeline.stage3_normalize_format.map_tools import ensure_tool_trees
from pipeline.stage3_normalize_format.trees import resolve_canonical_name

DEFAULT_SYSTEM_PROMPT = (
    "You are a geolocation agent. Reason step by step with Thought/Action/Observation."
)
DEFAULT_USER_QUERY = "Locate the place shown in the image."


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


def remap_trajectory(
    freeform: FreeFormTrajectory,
    forest: ToolForest,
    *,
    system_prompt: str,
    user_query: str,
    image_path: str,
    trajectory_id: str | None = None,
) -> Trajectory:
    """自由链 → 单 Agent 标准 Trajectory（canonical tools）。"""
    steps: list[TrajectoryStep] = []
    for step in freeform.steps:
        canonical = resolve_canonical_name(forest, step.tool) or step.tool
        tree = None
        for t in forest.trees:
            if t.canonical.name == canonical:
                tree = t
                break
        is_terminal = bool(tree.canonical.is_terminal) if tree else False
        steps.append(
            TrajectoryStep(
                thought=step.thought,
                action=Action(tool=canonical, params=dict(step.params or {})),
                observation=_normalize_observation(
                    step.observation, is_terminal=is_terminal
                ),
            )
        )
    return Trajectory(
        id=trajectory_id or str(uuid.uuid4()),
        system_prompt=system_prompt,
        user_query=user_query,
        image_path=image_path,
        steps=steps,
    )


def _format_assistant_content(step: TrajectoryStep) -> str:
    action_payload = {"tool": step.action.tool, "params": step.action.params}
    return f"Thought: {step.thought}\nAction: {_json_dumps(action_payload)}"


def trajectory_to_messages(traj: Trajectory) -> list[ChatMessage]:
    """将 Trajectory 展开为 chat messages（loss：仅 assistant）。"""
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=traj.system_prompt),
        ChatMessage(
            role="user",
            content=f"{traj.user_query}\n[Image: {traj.image_path}]",
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


def run_stage3(
    freeform: FreeFormTrajectory,
    *,
    trees_path: str | Path | None = None,
    out_trajectory_path: str | None = None,
    out_jsonl_path: str | None = None,
    image_path: str = "",
    system_prompt: str | None = None,
    user_query: str | None = None,
    matcher=None,
) -> DatasetEntry:
    """阶段3 一站式：树维护 → 重写 → DatasetEntry。"""
    settings = get_settings()
    forest_path = Path(trees_path) if trees_path else Path(settings.TOOL_TREES_PATH)
    forest = ensure_tool_trees(freeform, forest_path, matcher=matcher)
    traj = remap_trajectory(
        freeform,
        forest,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        user_query=user_query or DEFAULT_USER_QUERY,
        image_path=image_path,
    )
    entry = format_dataset_entry(traj, source_video=freeform.source_video)

    video_id = freeform.source_video
    traj_path = Path(out_trajectory_path) if out_trajectory_path else (
        Path(settings.INTERMEDIATE_DIR) / video_id / "stage3_trajectory.json"
    )
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")

    if out_jsonl_path:
        shard = Path(out_jsonl_path)
    else:
        shard = Path(settings.OUTPUT_DIR) / "shards" / f"{video_id}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(entry.model_dump_json() + "\n", encoding="utf-8")

    return entry
