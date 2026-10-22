"""从阶段3/4 产物目录收集可蒸馏轨迹。不读取 groundtruth。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.schemas.trajectory import Trajectory
from pipeline.skill_distill.schemas import SkipRecord

_TRAJECTORY_NAME = "stage3_trajectory.json"
_CONFIDENCE_NAME = "stage4_confidence.json"


@dataclass(frozen=True)
class AdmittedTrajectory:
    """通过质量门的一条轨迹。"""

    trajectory_id: str
    path: Path
    trajectory: Trajectory
    quality_score: float | None


def discover_trajectories(input_dir: Path) -> list[Path]:
    """递归查找阶段3轨迹。只认固定文件名，避免扫到其他 JSON。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"蒸馏输入目录不存在：{input_dir}")
    return sorted(input_dir.rglob(_TRAJECTORY_NAME))


def admit_trajectories(
    input_dir: Path,
    *,
    min_quality: float = 0.85,
    allow_unscored: bool = False,
) -> tuple[list[AdmittedTrajectory], list[SkipRecord]]:
    """按阶段4分数与 decision 过滤轨迹。

    有阶段4时，``reject`` 或分数低于 ``min_quality`` 的轨迹不进入蒸馏。
    没有阶段4时默认跳过；批量结果尚未评分时可打开 ``allow_unscored``。
    """
    admitted: list[AdmittedTrajectory] = []
    skipped: list[SkipRecord] = []
    for path in discover_trajectories(input_dir):
        trajectory_id = path.parent.name
        try:
            trajectory = Trajectory.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            skipped.append(
                SkipRecord(
                    trajectory_id=trajectory_id,
                    path=str(path),
                    reason=f"轨迹无法解析：{exc}",
                )
            )
            continue
        score, decision, score_reason = _read_confidence(path.parent / _CONFIDENCE_NAME)
        if score_reason == "missing":
            if not allow_unscored:
                skipped.append(
                    SkipRecord(
                        trajectory_id=trajectory.id or trajectory_id,
                        path=str(path),
                        reason="缺少 stage4_confidence.json",
                    )
                )
                continue
        elif score_reason is not None:
            skipped.append(
                SkipRecord(
                    trajectory_id=trajectory.id or trajectory_id,
                    path=str(path),
                    reason=score_reason,
                )
            )
            continue
        if decision == "reject":
            skipped.append(
                SkipRecord(
                    trajectory_id=trajectory.id or trajectory_id,
                    path=str(path),
                    reason="stage4 decision=reject",
                )
            )
            continue
        if score is not None and score < min_quality:
            skipped.append(
                SkipRecord(
                    trajectory_id=trajectory.id or trajectory_id,
                    path=str(path),
                    reason=f"quality_score={score:.3f} 低于 {min_quality:.2f}",
                )
            )
            continue
        admitted.append(
            AdmittedTrajectory(
                trajectory_id=trajectory.id or trajectory_id,
                path=path,
                trajectory=trajectory,
                quality_score=score,
            )
        )
    return admitted, skipped


def render_trajectory(trajectory: Trajectory, *, observation_chars: int = 400) -> str:
    """把轨迹收成拆句用的短文本。不附带图片，也不附带目录外材料。"""
    lines = [
        f"轨迹: {trajectory.id}",
        f"问题: {trajectory.user_query}",
    ]
    for index, step in enumerate(trajectory.steps, start=1):
        lines.append(f"步骤{index} [{step.event_type}] {step.thought}")
        if step.action is not None:
            params = json.dumps(step.action.params, ensure_ascii=False)
            lines.append(f"  调用: {step.action.tool} {params}")
        if step.observation is not None:
            observation = json.dumps(step.observation, ensure_ascii=False)
            if len(observation) > observation_chars:
                observation = observation[:observation_chars] + "…"
            lines.append(f"  回执: {observation}")
    return "\n".join(lines)


def _read_confidence(path: Path) -> tuple[float | None, str | None, str | None]:
    """返回 (分数, decision, 错误原因)。文件不存在时错误原因为 missing。"""
    if not path.is_file():
        return None, None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, None, f"阶段4文件无法解析：{exc}"
    if not isinstance(payload, dict):
        return None, None, "阶段4文件不是对象"
    decision = payload.get("decision")
    decision_text = decision.strip() if isinstance(decision, str) else None
    raw_score = payload.get("quality_score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return None, decision_text, "阶段4缺少 quality_score"
    return float(raw_score), decision_text, None
