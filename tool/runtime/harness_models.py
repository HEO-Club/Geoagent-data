"""Tool harness 的可持久化执行报告模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from pipeline.schemas.tools import ToolParameterAudit


class ReferenceBinding(BaseModel):
    """一次精确运行时引用的解析记录。"""

    input_path: str
    reference: str
    source: str
    dependency_step: int | None = None
    value_type: str
    value_preview: str


class HarnessObservation(BaseModel):
    """可 JSON 序列化的执行器回执。"""

    ok: bool
    result: dict[str, Any] | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    session: str | None = None
    error: str | None = None
    error_code: str | None = None


class ToolExecutionRecord(BaseModel):
    """轨迹中一个事件的执行状态。"""

    step_index: int = Field(ge=1)
    event_type: Literal["reasoning", "tool_call", "final"]
    status: Literal[
        "running",
        "skipped_reasoning",
        "succeeded",
        "failed",
        "blocked",
        "invalid",
        "terminal",
    ]
    thought: str
    call_id: str
    tool: str | None = None
    operation: str | None = None
    purpose: str | None = None
    raw_inputs: dict[str, Any] = Field(default_factory=dict)
    resolved_inputs: dict[str, Any] = Field(default_factory=dict)
    references: list[ReferenceBinding] = Field(default_factory=list)
    dependency_steps: list[int] = Field(default_factory=list)
    parameter_audit: ToolParameterAudit | None = None
    observation: HarnessObservation | None = None
    observation_fingerprint: str | None = None
    artifact_fingerprints: dict[str, dict[str, Any]] = Field(default_factory=dict)
    final_location: str | list[str] | None = None
    started_at: str = ""
    finished_at: str = ""
    duration_ms: float | None = None
    reused_from_checkpoint: bool = False
    error_code: str | None = None
    error: str | None = None
    note: str = ""


class HarnessReport(BaseModel):
    """一条 Stage 3 Trajectory 的真实 Tool replay 报告。"""

    schema_version: Literal["tool_harness_v1"] = "tool_harness_v1"
    run_id: str
    trajectory_id: str
    trajectory_fingerprint: str
    catalog_path: str
    catalog_fingerprint: str
    executor_fingerprint: str
    runtime_fingerprint: str
    execution_policy: dict[str, Any] = Field(default_factory=dict)
    status: Literal["running", "completed", "completed_with_errors"] = "running"
    started_at: str
    finished_at: str = ""
    resumed: bool = False
    source_observations_ignored: bool = True
    records: list[ToolExecutionRecord] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    final_location: str | list[str] | None = None
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "HarnessObservation",
    "HarnessReport",
    "ReferenceBinding",
    "ToolExecutionRecord",
]
