"""证据校准的轨迹质量置信度。

分数不是模型的主观自报概率。每个维度由可追溯检查组成；没有审核证据时
使用中性分并降低 audit_coverage，避免把“尚未发现错误”误写成“已经证实正确”。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from pipeline.schemas.audit import AnswerStatus, GeoTaskSpec, TaskStatus
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.quality import (
    QualityCheck,
    QualityDimensionName,
    QualityDimensionScore,
    QualityIssue,
    SemanticQualityReview,
    TrajectoryQualityReport,
)
from pipeline.schemas.tools import ToolForest
from pipeline.schemas.trajectory import Trajectory

DIMENSION_WEIGHTS: dict[QualityDimensionName, float] = {
    "evidence_grounding": 0.30,
    "final_answer_support": 0.20,
    "tool_parameter_validity": 0.20,
    "reasoning_consistency": 0.15,
    "input_alignment": 0.10,
    "sft_format": 0.05,
}

SOFT_QUALITY_CAPS = {
    "trajectory_uses_selected_images": 0.75,
    "task_gate": 0.80,
    "canonical_tools_known": 0.70,
    "operation_known": 0.70,
    "operation_inputs_validated": 0.70,
}

ACCEPT_SCORE = 0.85
ACCEPT_COVERAGE = 0.70
REVIEW_SCORE = 0.65
REJECT_MIN_COVERAGE = 0.70
PROVISIONAL_SCORE = 0.78
UNKNOWN_SCORE = 0.50


def _find_tree_for_name(forest: ToolForest, name: str):
    key = name.strip().lower()
    for tree in forest.trees:
        if tree.canonical.name.lower() == key:
            return tree
        if any(variant.lower() == key for variant in tree.variants):
            return tree
    return None


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[\s，。；、,.;:：()（）\[\]【】'\"“”‘’_-]+", "", text)


def _valid_location(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(isinstance(x, str) and x.strip() for x in value)
    return False


def _location_similarity(expected: Any, actual: Any) -> float:
    expected_norm = _norm_text(expected)
    actual_norm = _norm_text(actual)
    if not expected_norm or not actual_norm:
        return 0.0
    if expected_norm in actual_norm or actual_norm in expected_norm:
        return 1.0
    return SequenceMatcher(None, expected_norm, actual_norm).ratio()


def _iter_leaf_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_leaf_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_leaf_strings(child)
    elif value is not None:
        yield str(value)


class _Builder:
    def __init__(self) -> None:
        self.checks: dict[QualityDimensionName, list[QualityCheck]] = defaultdict(list)
        self.issues: list[QualityIssue] = []
        self.hard_failures: list[str] = []

    def add(
        self,
        dimension: QualityDimensionName,
        code: str,
        score: float,
        message: str,
        *,
        observed: bool = True,
        importance: float = 1.0,
        severity: str | None = None,
        step_index: int | None = None,
        evidence: list[str] | None = None,
    ) -> None:
        refs = list(evidence or [])
        self.checks[dimension].append(
            QualityCheck(
                code=code,
                score=score,
                observed=observed,
                importance=importance,
                message=message,
                step_index=step_index,
                evidence=refs,
            )
        )
        if severity:
            issue = QualityIssue(
                code=code,
                dimension=dimension,
                severity=severity,
                message=message,
                step_index=step_index,
                evidence=refs,
            )
            self.issues.append(issue)
            if severity == "hard_fail":
                self.hard_failures.append(code)

    def unknown(
        self,
        dimension: QualityDimensionName,
        code: str,
        message: str,
        *,
        importance: float = 1.0,
    ) -> None:
        self.add(
            dimension,
            code,
            UNKNOWN_SCORE,
            message,
            observed=False,
            importance=importance,
            severity="warning",
        )

    def dimensions(self) -> list[QualityDimensionScore]:
        result: list[QualityDimensionScore] = []
        for name, weight in DIMENSION_WEIGHTS.items():
            items = self.checks.get(name) or [
                QualityCheck(
                    code="dimension_not_audited",
                    score=UNKNOWN_SCORE,
                    observed=False,
                    message="该维度尚未审核",
                )
            ]
            total = sum(item.importance for item in items)
            score = sum(item.score * item.importance for item in items) / total
            coverage = (
                sum(item.importance for item in items if item.observed) / total
            )
            result.append(
                QualityDimensionScore(
                    name=name,
                    weight=weight,
                    score=round(score, 4),
                    audit_coverage=round(coverage, 4),
                    checks=items,
                )
            )
        return result


def _check_format(builder: _Builder, trajectory: Trajectory) -> tuple[list[int], Any]:
    final_indices = [
        index
        for index, step in enumerate(trajectory.steps, start=1)
        if step.event_type == "final"
    ]
    if len(final_indices) != 1:
        builder.add(
            "sft_format",
            "final_event_count",
            0.0,
            f"final 事件数量必须为 1，实际为 {len(final_indices)}",
            severity="hard_fail",
            importance=2.0,
        )
        final_step = None
    else:
        builder.add(
            "sft_format",
            "final_event_count",
            1.0,
            "final 事件数量正确",
            importance=2.0,
        )
        final_step = trajectory.steps[final_indices[0] - 1]

    if final_indices and final_indices[-1] == len(trajectory.steps):
        builder.add("sft_format", "final_is_last", 1.0, "final 位于轨迹末步")
    else:
        builder.add(
            "sft_format",
            "final_is_last",
            0.0,
            "final 缺失或后面仍有事件",
            severity="hard_fail",
        )

    bad_shapes = 0
    for index, step in enumerate(trajectory.steps, start=1):
        if step.event_type == "reasoning" and (
            step.action is not None or step.observation is not None
        ) or step.event_type == "tool_call" and step.action is None or step.event_type == "final" and (
            step.action is None
            or step.action.tool != "final_answer"
            or step.observation is not None
        ):
            bad_shapes += 1
    builder.add(
        "sft_format",
        "event_shapes",
        1.0 if bad_shapes == 0 else 0.0,
        "事件结构符合 reasoning/tool_call/final 契约"
        if bad_shapes == 0
        else f"发现 {bad_shapes} 个事件结构错误",
        severity="hard_fail" if bad_shapes else None,
        importance=2.0,
    )
    return final_indices, final_step


def _check_final(
    builder: _Builder,
    final_step: Any,
    task: GeoTaskSpec | None,
    semantic_review: SemanticQualityReview | None,
) -> None:
    location = None
    if final_step is not None and final_step.action is not None:
        location = final_step.action.params.get("location")
        only_location = set(final_step.action.params) == {"location"}
        builder.add(
            "final_answer_support",
            "final_contract",
            1.0 if only_location else 0.3,
            "最终参数严格为 location"
            if only_location
            else "最终参数含 location 之外字段",
            severity="error" if not only_location else None,
        )

    if _valid_location(location):
        builder.add(
            "final_answer_support",
            "location_present",
            1.0,
            "最终 location 非空且类型正确",
            importance=2.0,
        )
    else:
        builder.add(
            "final_answer_support",
            "location_present",
            0.0,
            "最终 location 缺失、为空或类型错误",
            severity="hard_fail",
            importance=2.0,
        )

    if task is None or not str(task.final_location_text or "").strip():
        builder.unknown(
            "final_answer_support",
            "answer_reference_unavailable",
            "没有 task 级 resolved 答案，无法核验最终地点完整性",
            importance=2.0,
        )
    else:
        expected = task.final_location_text
        actual = location[0] if isinstance(location, list) and len(location) == 1 else location
        similarity = _location_similarity(expected, actual)
        matched = similarity >= 0.55
        strong_mismatch = similarity < 0.35
        builder.add(
            "final_answer_support",
            "answer_matches_task",
            1.0 if matched else (0.4 if not strong_mismatch else 0.0),
            "最终地点与 Stage 1.5 resolved 答案一致"
            if matched
            else (
                f"最终地点与 task 答案相似度不足（{similarity:.2f}）："
                f"expected={expected!r}, actual={actual!r}"
            ),
            severity=(
                "hard_fail" if strong_mismatch else ("warning" if not matched else None)
            ),
            importance=2.0,
        )

    if semantic_review is None:
        builder.unknown(
            "final_answer_support",
            "final_semantic_support_not_reviewed",
            "尚未由独立审核 Agent 检查最终答案是否由证据链推出",
            importance=2.0,
        )
    else:
        builder.add(
            "final_answer_support",
            "final_semantic_support",
            semantic_review.final_answer_support,
            "独立审核 Agent 对最终答案支撑度的评分",
            importance=2.0,
        )


def _final_observation_items(audit: dict[str, Any]) -> list[dict[str, Any]]:
    passes = audit.get("passes")
    if not isinstance(passes, list) or not passes:
        return []
    last = passes[-1]
    if not isinstance(last, dict):
        return []
    items = last.get("items")
    return [item for item in items or [] if isinstance(item, dict)]


def _check_evidence(
    builder: _Builder,
    trajectory: Trajectory,
    observation_audit: dict[str, Any] | None,
    semantic_review: SemanticQualityReview | None,
) -> None:
    tool_steps = [
        (index, step)
        for index, step in enumerate(trajectory.steps, start=1)
        if step.event_type == "tool_call"
    ]
    missing = [index for index, step in tool_steps if step.observation is None]
    if missing:
        builder.add(
            "evidence_grounding",
            "tool_observation_missing",
            0.0,
            f"{len(missing)} 个真实 Tool 调用缺少 Observation",
            severity="hard_fail",
            importance=2.0,
            evidence=[f"step:{i}" for i in missing],
        )
    else:
        builder.add(
            "evidence_grounding",
            "tool_observation_present",
            1.0,
            "所有真实 Tool 调用均有 Observation",
            importance=1.0,
        )

    if observation_audit is None:
        builder.unknown(
            "evidence_grounding",
            "observation_direct_evidence_not_audited",
            "尚无严格 Observation 直接证据审计，不能仅因字段非空判为真实",
            importance=4.0,
        )
    else:
        items = _final_observation_items(observation_audit)
        verdicts = [str(item.get("verdict") or "").lower() for item in items]
        supported = sum(value == "supported" for value in verdicts)
        complete = bool(items) and supported == len(items)
        audit_accepted = observation_audit.get("accepted") is not False
        if complete and audit_accepted:
            builder.add(
                "evidence_grounding",
                "observation_direct_evidence",
                1.0,
                f"严格审计末轮 {supported}/{len(items)} 个 Observation 均有直接证据",
                importance=4.0,
            )
        else:
            bad = [value for value in verdicts if value != "supported"]
            builder.add(
                "evidence_grounding",
                "observation_direct_evidence",
                0.0,
                f"严格审计未通过：supported={supported}/{len(items)}，其他={bad}",
                severity="hard_fail",
                importance=4.0,
            )

    numeric_claims = 0
    for _, step in tool_steps:
        for leaf in _iter_leaf_strings(step.observation):
            if re.search(r"\d+(?:\.\d+)?\s*(?:米|公里|度|年|%|％|秒|分钟)", leaf):
                numeric_claims += 1
    if numeric_claims and observation_audit is None:
        builder.add(
            "evidence_grounding",
            "precise_claims_need_evidence",
            0.4,
            f"发现 {numeric_claims} 条精确数值回执，但没有直接证据审计",
            severity="warning",
            importance=1.5,
        )

    if semantic_review is not None:
        builder.add(
            "evidence_grounding",
            "semantic_grounding_review",
            semantic_review.evidence_grounding,
            "独立审核 Agent 对 Thought/Observation 证据落地度的评分",
            importance=2.0,
        )


def _check_tools(
    builder: _Builder,
    trajectory: Trajectory,
    forest: ToolForest,
    parameter_audits: list[dict[str, Any]] | None,
    semantic_review: SemanticQualityReview | None,
) -> None:
    calls = [
        (index, step)
        for index, step in enumerate(trajectory.steps, start=1)
        if step.event_type == "tool_call"
    ]
    unknown_tools: list[str] = []
    bad_operations: list[str] = []
    bad_envelopes: list[int] = []
    for index, step in calls:
        assert step.action is not None
        tree = _find_tree_for_name(forest, step.action.tool)
        if tree is None:
            unknown_tools.append(step.action.tool)
            continue
        params = step.action.params
        if set(params) != {"operation", "purpose", "inputs"}:
            bad_envelopes.append(index)
        raw_operation = str(params.get("operation") or "").strip().lower()
        operation = raw_operation
        for candidate in tree.canonical.operations:
            aliases = {str(alias).strip().lower() for alias in candidate.aliases}
            if raw_operation == candidate.name or raw_operation in aliases:
                operation = candidate.name
                break
        known_ops = {item.name for item in tree.canonical.operations}
        if operation not in known_ops:
            bad_operations.append(f"step:{index}:{step.action.tool}.{operation}")

    builder.add(
        "tool_parameter_validity",
        "canonical_tools_known",
        1.0 if not unknown_tools else 0.2,
        "全部调用均命中 Canonical Tool"
        if not unknown_tools
        else f"存在未登记 Tool：{sorted(set(unknown_tools))}",
        severity="error" if unknown_tools else None,
        importance=2.0,
    )
    builder.add(
        "tool_parameter_validity",
        "operation_known",
        1.0 if not bad_operations else 0.0,
        "全部 operation 均属于对应执行器"
        if not bad_operations
        else f"存在未知 operation：{bad_operations}",
        severity="error" if bad_operations else None,
        importance=2.0,
    )
    builder.add(
        "tool_parameter_validity",
        "canonical_envelope",
        1.0 if not bad_envelopes else 0.2,
        "全部调用采用 operation + purpose + inputs"
        if not bad_envelopes
        else f"以下步骤未采用规范外层参数：{bad_envelopes}",
        severity="error" if bad_envelopes else None,
    )

    if parameter_audits is None:
        builder.unknown(
            "tool_parameter_validity",
            "operation_inputs_not_validated",
            "尚未执行 operation 级 input_schema 校验",
            importance=3.0,
        )
    else:
        readiness_values = {
            "ready": 1.0,
            "context_resolvable": 0.85,
            "repairable": 0.55,
            "invalid": 0.0,
        }
        readiness = [
            str(
                audit.get("readiness")
                or ("ready" if audit.get("valid") else "repairable")
            )
            for audit in parameter_audits
        ]
        readiness_score = (
            sum(readiness_values.get(value, 0.5) for value in readiness)
            / len(readiness)
            if readiness
            else 1.0
        )
        counts = {
            value: sum(item == value for item in readiness)
            for value in readiness_values
        }
        invalid = counts["invalid"]
        repairable = counts["repairable"]
        builder.add(
            "tool_parameter_validity",
            "operation_inputs_validated",
            readiness_score,
            "参数调用状态："
            + ", ".join(f"{name}={count}" for name, count in counts.items()),
            severity=(
                "error"
                if invalid
                else ("warning" if repairable else None)
            ),
            importance=3.0,
        )

    if semantic_review is not None:
        builder.add(
            "tool_parameter_validity",
            "tool_semantic_review",
            semantic_review.tool_semantics,
            "独立审核 Agent 对 Tool/思考边界和调用语义的评分",
            importance=2.0,
        )


def _check_reasoning(
    builder: _Builder,
    trajectory: Trajectory,
    semantic_review: SemanticQualityReview | None,
) -> None:
    substantive = [step.thought.strip() for step in trajectory.steps if step.thought.strip()]
    unique_ratio = len(set(substantive)) / max(1, len(substantive))
    builder.add(
        "reasoning_consistency",
        "nonempty_nonduplicate_thoughts",
        unique_ratio,
        f"Thought 非空且去重比例为 {unique_ratio:.2f}",
        observed=True,
        importance=1.0,
        severity="warning" if unique_ratio < 0.8 else None,
    )
    if semantic_review is None:
        builder.unknown(
            "reasoning_consistency",
            "reasoning_semantics_not_reviewed",
            "尚未由独立审核 Agent 检查候选产生、排除和收敛是否连贯",
            importance=4.0,
        )
    else:
        builder.add(
            "reasoning_consistency",
            "reasoning_semantic_review",
            semantic_review.reasoning_consistency,
            "独立审核 Agent 对逻辑一致性的评分",
            importance=4.0,
        )


def _check_input(
    builder: _Builder,
    trajectory: Trajectory,
    task: GeoTaskSpec | None,
    trajectory_consistency: dict[str, Any] | None,
    semantic_review: SemanticQualityReview | None,
) -> None:
    if task is None:
        builder.unknown(
            "input_alignment",
            "stage15_context_unavailable",
            "缺少 Stage 1.5 task/选图审计，无法确认输入图与题目对齐",
            importance=4.0,
        )
    else:
        resolved = task.answer_status == AnswerStatus.resolved
        if not resolved or task.status == TaskStatus.rejected:
            builder.add(
                "input_alignment",
                "task_gate",
                0.0,
                f"task 无法用于下游：status={task.status}, answer={task.answer_status}",
                severity="hard_fail",
                importance=2.0,
            )
        elif task.status == TaskStatus.needs_review:
            builder.add(
                "input_alignment",
                "task_gate",
                0.5,
                f"Stage 1.5 task 仍需复核：{task.status_reason or '未给出原因'}",
                severity="warning",
                importance=2.0,
            )
        else:
            builder.add(
                "input_alignment",
                "task_gate",
                1.0,
                "Stage 1.5 task 已 accepted 且答案 resolved",
                importance=2.0,
            )
        selected = [item for item in task.frame_assessments if item.selected]
        if not selected:
            if task.image_paths:
                builder.unknown(
                    "input_alignment",
                    "selected_frame_audit_unavailable",
                    "存在选中图片，但缺少逐帧审计记录",
                    importance=3.0,
                )
            else:
                builder.add(
                    "input_alignment",
                    "selected_frame_audit",
                    0.0,
                    "Stage 1.5 没有选中图片或已选帧审计记录",
                    severity="hard_fail",
                    importance=3.0,
                )
        else:
            leaked = [item for item in selected if item.answer_leakage]
            aligned_paths = {_norm_text(path) for path in trajectory.image_paths}
            task_paths = {_norm_text(path) for path in task.image_paths}
            path_match = bool(aligned_paths and task_paths and aligned_paths == task_paths)
            quality = sum(item.quality_score for item in selected) / len(selected)
            support_values = [item.chain_support_score for item in selected]
            support = sum(support_values) / len(support_values)
            clean_ratio = sum(item.clean_source for item in selected) / len(selected)
            score = 0.4 * quality + 0.35 * support + 0.25 * clean_ratio
            if leaked:
                score = 0.0
            builder.add(
                "input_alignment",
                "selected_frame_quality",
                score,
                f"选图质量={quality:.2f}，定位链支撑={support:.2f}，干净图比例={clean_ratio:.2f}",
                severity="hard_fail" if leaked else ("warning" if score < 0.65 else None),
                importance=3.0,
            )
            builder.add(
                "input_alignment",
                "trajectory_uses_selected_images",
                1.0 if path_match else 0.0,
                "Stage 3 图片与 Stage 1.5 选图完全一致"
                if path_match
                else "Stage 3 image_paths 与 Stage 1.5 选图不一致",
                severity="error" if not path_match else None,
                importance=2.0,
            )

    if trajectory_consistency is None:
        builder.unknown(
            "input_alignment",
            "trajectory_image_consistency_unavailable",
            "没有轨迹—图片一致性检查结果",
            importance=1.0,
        )
    else:
        conflict = bool(trajectory_consistency.get("conflict"))
        confidence = float(trajectory_consistency.get("confidence") or 0.0)
        builder.add(
            "input_alignment",
            "trajectory_image_consistency",
            0.0 if conflict else 1.0,
            str(trajectory_consistency.get("reason") or "轨迹—图片无高置信冲突"),
            observed=confidence > 0.0,
            severity="hard_fail" if conflict else None,
            importance=1.0,
        )

    if semantic_review is not None:
        builder.add(
            "input_alignment",
            "input_semantic_review",
            semantic_review.input_alignment,
            "独立审核 Agent 对图片、字幕与任务对齐的评分",
            importance=2.0,
        )


def score_trajectory_quality(
    freeform: FreeFormTrajectory,
    trajectory: Trajectory,
    forest: ToolForest,
    *,
    task: GeoTaskSpec | None = None,
    observation_audit: dict[str, Any] | None = None,
    trajectory_consistency: dict[str, Any] | None = None,
    parameter_audits: list[dict[str, Any]] | None = None,
    semantic_review: SemanticQualityReview | None = None,
) -> TrajectoryQualityReport:
    """计算一条轨迹的分维度质量置信度与审核覆盖率。"""

    builder = _Builder()
    _, final_step = _check_format(builder, trajectory)
    _check_final(builder, final_step, task, semantic_review)
    _check_evidence(builder, trajectory, observation_audit, semantic_review)
    _check_tools(builder, trajectory, forest, parameter_audits, semantic_review)
    _check_reasoning(builder, trajectory, semantic_review)
    _check_input(builder, trajectory, task, trajectory_consistency, semantic_review)

    if semantic_review is not None:
        for issue in semantic_review.issues:
            builder.issues.append(issue)
            if issue.severity == "hard_fail":
                builder.hard_failures.append(issue.code)
        builder.hard_failures.extend(semantic_review.hard_failures)

    parameter_readiness = [
        str(
            audit.get("readiness")
            or ("ready" if audit.get("valid") else "repairable")
        )
        for audit in parameter_audits or []
    ]
    has_invalid_parameters = "invalid" in parameter_readiness
    has_repairable_parameters = "repairable" in parameter_readiness
    dimensions = builder.dimensions()
    raw_quality_score = sum(item.weight * item.score for item in dimensions)
    applied_soft_caps: dict[str, float] = {}
    for issue in builder.issues:
        cap = SOFT_QUALITY_CAPS.get(issue.code)
        if cap is None:
            continue
        if issue.code == "task_gate" and issue.severity != "warning":
            continue
        if issue.code == "operation_inputs_validated" and not has_invalid_parameters:
            continue
        applied_soft_caps[issue.code] = cap
    quality_score = min([raw_quality_score, *applied_soft_caps.values()])
    audit_coverage = sum(item.weight * item.audit_coverage for item in dimensions)
    hard_failures = sorted(set(builder.hard_failures))
    has_nonparameter_error = any(
        issue.severity == "error" and issue.dimension != "tool_parameter_validity"
        for issue in builder.issues
    )
    dimension_by_name = {item.name: item for item in dimensions}

    if hard_failures:
        decision = "reject"
    elif has_nonparameter_error or has_invalid_parameters or audit_coverage < ACCEPT_COVERAGE:
        decision = "needs_review"
    elif has_repairable_parameters:
        decision = "parameter_repair"
    elif quality_score >= ACCEPT_SCORE and audit_coverage >= ACCEPT_COVERAGE:
        decision = "accept"
    elif (
        quality_score >= PROVISIONAL_SCORE
        and dimension_by_name["evidence_grounding"].score >= 0.9
        and dimension_by_name["sft_format"].score >= 0.9
    ):
        decision = "provisional_pass"
    elif audit_coverage < REJECT_MIN_COVERAGE or quality_score >= REVIEW_SCORE:
        decision = "needs_review"
    else:
        decision = "reject"

    return TrajectoryQualityReport(
        source_video=freeform.source_video,
        trajectory_id=trajectory.id,
        quality_score=quality_score,
        audit_coverage=audit_coverage,
        decision=decision,
        dimensions=dimensions,
        hard_failures=hard_failures,
        issues=builder.issues,
        metadata={
            "weights": DIMENSION_WEIGHTS,
            "raw_quality_score": round(raw_quality_score, 4),
            "applied_soft_caps": applied_soft_caps,
            "thresholds": {
                "accept_score": ACCEPT_SCORE,
                "accept_coverage": ACCEPT_COVERAGE,
                "review_score": REVIEW_SCORE,
                "reject_min_coverage": REJECT_MIN_COVERAGE,
                "provisional_score": PROVISIONAL_SCORE,
            },
            "parameter_readiness_counts": {
                value: parameter_readiness.count(value)
                for value in ("ready", "context_resolvable", "repairable", "invalid")
            },
            "freeform_event_count": len(freeform.steps),
            "trajectory_event_count": len(trajectory.steps),
        },
    )


def load_optional_json(value: str | None) -> dict[str, Any] | None:
    """脚本共用：读取可选 JSON 文本。"""

    if not value:
        return None
    data = json.loads(value)
    return data if isinstance(data, dict) else None
