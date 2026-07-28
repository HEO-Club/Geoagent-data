"""stage7：Trajectory → DatasetEntry，按 Agent 分片写 JSONL，单 writer 合并。

宁缺毋滥：verified=False 的样本不写入训练分片 / 最终 JSONL。
禁止多个协程直接追加同一个最终 JSONL。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pipeline.evidence_routing import sanitize_revision_input_for_coarse_shard
from pipeline.schemas import (
    AgentRole,
    ChatMessage,
    DatasetEntry,
    Trajectory,
    TrajectoryStep,
)

# agent_role → 分片后缀 / 最终文件名
_ROLE_SHARD_SUFFIX: dict[AgentRole, str] = {
    AgentRole.COARSE: "agent1",
    AgentRole.FINE: "agent2",
    AgentRole.VERIFIER: "agent3",
}

_ROLE_FINAL_NAME: dict[AgentRole, str] = {
    AgentRole.COARSE: "agent1_coarse.jsonl",
    AgentRole.FINE: "agent2_fine.jsonl",
    AgentRole.VERIFIER: "agent3_verifier.jsonl",
}


def _json_dumps(obj: Any) -> str:
    """稳定 JSON 序列化（UTF-8，不转义非 ASCII）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _format_assistant_content(step: TrajectoryStep) -> str:
    """assistant 内容 = Thought + Action（计入 loss）。"""
    action_payload = {"tool": step.action.tool, "params": step.action.params}
    return f"Thought: {step.thought}\nAction: {_json_dumps(action_payload)}"


def _format_tool_content(observation: Optional[dict[str, Any]]) -> str:
    """tool 内容 = Observation JSON（全部 mask）。"""
    if observation is None:
        return ""
    return _json_dumps(observation)


def _final_output_message(traj: Trajectory) -> Optional[ChatMessage]:
    """将角色结构化输出追加为最后一条 assistant（若尚未体现在 terminal Action 中）。"""
    if traj.agent_role == AgentRole.COARSE and traj.coarse_output is not None:
        return ChatMessage(
            role="assistant",
            content=(
                "Final output (LocationHypothesis):\n"
                + traj.coarse_output.model_dump_json()
            ),
        )
    if traj.agent_role == AgentRole.VERIFIER and traj.verifier_output is not None:
        return ChatMessage(
            role="assistant",
            content=(
                "Final output (VerificationResult):\n"
                + traj.verifier_output.model_dump_json()
            ),
        )
    # FINE：答案已在 submit_answer Action 中，不再重复
    return None


def trajectory_to_messages(traj: Trajectory) -> list[ChatMessage]:
    """将 Trajectory 展开为 chat messages（loss mask：仅 assistant 计 loss）。"""
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
        # terminal 步 observation 为 None → 不写 tool message
        if step.observation is not None:
            messages.append(
                ChatMessage(
                    role="tool",
                    content=_format_tool_content(step.observation),
                )
            )
    final_msg = _final_output_message(traj)
    if final_msg is not None:
        messages.append(final_msg)
    return messages


def _resolve_traj_meta(traj: Trajectory, meta: dict[str, Any]) -> dict[str, Any]:
    """合并视频级 meta 与按 trajectory id 的报告字段。"""
    reports = meta.get("reports") or {}
    per = reports.get(traj.id) or {}
    merged = {**meta, **per}
    # 去掉嵌套结构，避免误写入 DatasetEntry
    merged.pop("reports", None)
    return merged


def to_dataset_entry(traj: Trajectory, meta: dict[str, Any]) -> DatasetEntry:
    """将单条 Trajectory 转为 DatasetEntry。

    meta 必填字段（视频级或 reports[traj.id] 覆盖）：
    - source_video: str
    - groundtruth: tuple[float, float]
    - quality_score: float
    - verified: bool
    可选：distance_error_km
    """
    resolved = _resolve_traj_meta(traj, meta)
    required = ("source_video", "groundtruth", "quality_score", "verified")
    missing = [k for k in required if k not in resolved]
    if missing:
        raise KeyError(f"to_dataset_entry meta 缺少字段: {missing}")

    gt = resolved["groundtruth"]
    if isinstance(gt, list):
        gt = (float(gt[0]), float(gt[1]))
    elif isinstance(gt, tuple):
        gt = (float(gt[0]), float(gt[1]))
    else:
        raise TypeError(f"groundtruth 类型非法: {type(gt)}")

    revision_input = traj.revision_input
    if traj.agent_role == AgentRole.COARSE:
        revision_input = sanitize_revision_input_for_coarse_shard(revision_input)

    return DatasetEntry(
        id=traj.id,
        source_video=str(resolved["source_video"]),
        agent_role=traj.agent_role,
        groundtruth=gt,
        messages=trajectory_to_messages(traj),
        coarse_handoff=traj.coarse_handoff,
        fine_handoff=traj.fine_handoff,
        is_revision=traj.is_revision,
        parent_trajectory_id=traj.parent_trajectory_id,
        revision_round=traj.revision_round,
        revision_source=traj.revision_source,
        revision_input=revision_input,
        quality_score=float(resolved["quality_score"]),
        verified=bool(resolved["verified"]),
        distance_error_km=(
            None
            if resolved.get("distance_error_km") is None
            else float(resolved["distance_error_km"])
        ),
    )


def _shard_path(output_dir: Path, video_id: str, role: AgentRole) -> Path:
    suffix = _ROLE_SHARD_SUFFIX[role]
    return output_dir / "shards" / f"{video_id}_{suffix}.jsonl"


def _write_jsonl(path: Path, entries: list[DatasetEntry]) -> None:
    """覆写式写入 JSONL（每视频分片独立，避免并发交错）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [e.model_dump_json() for e in entries]
    text = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(text, encoding="utf-8")


def format_all_and_save(
    trajectories: list[Trajectory],
    meta: dict[str, Any],
    output_dir: str,
    video_id: str,
) -> list[DatasetEntry]:
    """按 agent_role 写入 ``data/output/shards/{video_id}_agent{1|2|3}.jsonl``。

    仅 verified=True 的条目写入分片（宁缺毋滥）。
    返回全部转换后的 DatasetEntry（含未通过验证的，便于 intermediate 审计）。
    最终 JSONL 由 :func:`merge_jsonl_shards` 单 writer 合并；禁止多协程直写最终文件。
    """
    out = Path(output_dir)
    all_entries: list[DatasetEntry] = []
    by_role: dict[AgentRole, list[DatasetEntry]] = {
        AgentRole.COARSE: [],
        AgentRole.FINE: [],
        AgentRole.VERIFIER: [],
    }

    for traj in trajectories:
        entry = to_dataset_entry(traj, meta)
        all_entries.append(entry)
        if entry.verified:
            by_role[entry.agent_role].append(entry)

    for role, entries in by_role.items():
        _write_jsonl(_shard_path(out, video_id, role), entries)

    return all_entries


def merge_jsonl_shards(output_dir: str | Path) -> dict[str, int]:
    """单 writer：合并 shards/ 下全部分片到三个最终 JSONL。

    返回各最终文件写入的行数。调用方应在批处理全部结束后、无并发写分片时调用。
    """
    out = Path(output_dir)
    shards_dir = out / "shards"
    counts: dict[str, int] = {}

    for role, final_name in _ROLE_FINAL_NAME.items():
        suffix = _ROLE_SHARD_SUFFIX[role]
        pattern = f"*_{suffix}.jsonl"
        lines: list[str] = []
        if shards_dir.is_dir():
            for shard in sorted(shards_dir.glob(pattern)):
                content = shard.read_text(encoding="utf-8")
                for line in content.splitlines():
                    if line.strip():
                        lines.append(line)
        final_path = out / final_name
        final_path.parent.mkdir(parents=True, exist_ok=True)
        text = ("\n".join(lines) + "\n") if lines else ""
        final_path.write_text(text, encoding="utf-8")
        counts[final_name] = len(lines)

    return counts
