"""顺序执行 Stage 3 Trajectory 中真实 Tool 调用的可恢复 harness。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from pipeline.schemas.tools import ToolForest, ToolParameterAudit
from pipeline.schemas.trajectory import Trajectory, TrajectoryStep
from pipeline.stage3_normalize_format.params import (
    normalize_and_validate_tool_inputs,
)
from tool.contract import Observation, RuntimeContext
from tool.runtime.harness_models import (
    HarnessObservation,
    HarnessReport,
    ToolExecutionRecord,
)
from tool.runtime.image_store import FilesystemImageStore
from tool.runtime.references import (
    ReferenceResolutionError,
    contains_credential_fields,
    redact_credentials,
    resolve_inputs,
)
from tool.runtime.result_store import FilesystemResultStore

Dispatcher = Callable[..., Observation]
_OUTER_KEYS = {"operation", "purpose", "inputs"}
_REUSABLE_STATUSES = {"skipped_reasoning", "succeeded", "terminal"}
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class HarnessError(Exception):
    """运行配置、轨迹或检查点不一致。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class ToolHarnessConfig:
    """harness 行为配置；真实 API 开关仍只从环境变量读取。"""

    workspace_dir: Path = Path("data/tool_runs")
    catalog_path: Path = Path("canonical_tool_catalog_v2.json")
    resume: bool = True
    retry_failed: bool = True
    stop_on_error: bool = False
    allowed_file_roots: list[Path] = field(default_factory=list)
    provider_revision: str = ""


class ToolHarness:
    """把规范 Trajectory 安全执行成独立 HarnessReport。"""

    def __init__(
        self,
        config: ToolHarnessConfig | None = None,
        *,
        context: RuntimeContext | None = None,
        dispatcher: Dispatcher | None = None,
    ) -> None:
        self.config = config or ToolHarnessConfig()
        self.base_context = context or RuntimeContext()
        self.dispatcher = dispatcher
        self._last_run_dir: Path | None = None

    @property
    def last_run_dir(self) -> Path | None:
        return self._last_run_dir

    @property
    def last_report_path(self) -> Path | None:
        return self._last_run_dir / "harness_report.json" if self._last_run_dir else None

    def run(self, trajectory: Trajectory) -> HarnessReport:
        _validate_trajectory_contract(trajectory)
        run_dir = self._select_run_dir(trajectory.id)
        run_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(run_dir / ".harness.lock"))
        try:
            with lock.acquire(timeout=0):
                return self._run_locked(trajectory, run_dir)
        except Timeout as exc:
            raise HarnessError(
                f"同一轨迹已有 harness 正在运行: {trajectory.id}",
                "run_already_locked",
            ) from exc

    def _run_locked(self, trajectory: Trajectory, run_dir: Path) -> HarnessReport:
        catalog_path = self.config.catalog_path.resolve()
        if not catalog_path.is_file():
            raise HarnessError(f"找不到 Tool 目录: {catalog_path}", "catalog_not_found")
        try:
            forest = ToolForest.model_validate_json(
                catalog_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise HarnessError(
                f"Canonical Tool 目录无法解析: {catalog_path}",
                "catalog_invalid",
            ) from exc
        trajectory_fingerprint = _fingerprint(trajectory.model_dump(mode="json"))
        catalog_fingerprint = _file_fingerprint(catalog_path)
        executor_fingerprint = _executor_fingerprint(self.dispatcher)
        execution_policy = {"stop_on_error": self.config.stop_on_error}
        runtime_fingerprint = _fingerprint(
            _runtime_identity(
                self.base_context,
                self.config,
                trajectory=trajectory,
                dispatcher=self.dispatcher,
            )
        )
        self._last_run_dir = run_dir
        report_path = run_dir / "harness_report.json"
        events_path = run_dir / "events.jsonl"
        report = self._load_or_create_report(
            trajectory=trajectory,
            trajectory_fingerprint=trajectory_fingerprint,
            catalog_path=catalog_path,
            catalog_fingerprint=catalog_fingerprint,
            executor_fingerprint=executor_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            execution_policy=execution_policy,
            report_path=report_path,
        )
        ctx = self._runtime_context(trajectory, run_dir)
        _save_report(report_path, report)
        _append_event(events_path, "run_started", None, report=report)
        records = {record.step_index: record for record in report.records}
        step_results: dict[int, dict[str, Any]] = {}
        rerun_steps: set[int] = set()
        halted = False

        for step_index, step in enumerate(trajectory.steps, start=1):
            existing = records.get(step_index)
            dirty_dependency = bool(
                existing is not None
                and set(existing.dependency_steps).intersection(rerun_steps)
            )
            if (
                existing is not None
                and not dirty_dependency
                and self._can_reuse(existing, ctx)
            ):
                reused = existing.model_copy(update={"reused_from_checkpoint": True})
                records[step_index] = reused
                self._restore_context(reused, ctx, step_results)
                if reused.status in {"failed", "blocked", "invalid"}:
                    halted = halted or self.config.stop_on_error
                continue

            if existing is not None:
                rerun_steps.add(step_index)

            if existing is not None and existing.status == "running":
                _append_warning_once(
                    report,
                    f"步骤 {step_index} 上次中断于 running，将安全重跑；远程只读请求可能产生重复计费。",
                )

            if step.event_type == "reasoning":
                record = self._reasoning_record(step_index, step, trajectory_fingerprint)
            elif step.event_type == "final":
                record = self._final_record(step_index, step, trajectory_fingerprint)
            elif halted:
                record = self._blocked_after_failure(
                    step_index,
                    step,
                    trajectory_fingerprint,
                )
            else:
                record = self._execute_tool_step(
                    step_index=step_index,
                    step=step,
                    trajectory_fingerprint=trajectory_fingerprint,
                    forest=forest,
                    ctx=ctx,
                    step_results=step_results,
                    report=report,
                    records=records,
                    report_path=report_path,
                    events_path=events_path,
                )
                if record.status in {"failed", "blocked", "invalid"}:
                    halted = self.config.stop_on_error

            records[step_index] = record
            self._restore_context(record, ctx, step_results)
            report.records = [records[index] for index in sorted(records)]
            _append_event(events_path, "step_finished", record)
            _save_report(report_path, report)

        report.records = [records[index] for index in sorted(records)]
        report.final_location = next(
            (
                record.final_location
                for record in reversed(report.records)
                if record.status == "terminal"
            ),
            None,
        )
        report.counts = _counts(report.records)
        has_errors = any(
            record.status in {"failed", "blocked", "invalid", "running"}
            for record in report.records
        )
        report.status = "completed_with_errors" if has_errors else "completed"
        report.finished_at = _utcnow()
        _save_report(report_path, report)
        _append_event(events_path, "run_finished", None, report=report)
        return report

    def _select_run_dir(self, trajectory_id: str) -> Path:
        safe_id = _safe_id(trajectory_id)
        base = self.config.workspace_dir.resolve() / safe_id
        if self.config.resume or not (base / "harness_report.json").is_file():
            return base
        return self.config.workspace_dir.resolve() / f"{safe_id}_{uuid.uuid4().hex[:8]}"

    def _load_or_create_report(
        self,
        *,
        trajectory: Trajectory,
        trajectory_fingerprint: str,
        catalog_path: Path,
        catalog_fingerprint: str,
        executor_fingerprint: str,
        runtime_fingerprint: str,
        execution_policy: dict[str, Any],
        report_path: Path,
    ) -> HarnessReport:
        if self.config.resume and report_path.is_file():
            try:
                report = HarnessReport.model_validate_json(
                    report_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise HarnessError("harness 检查点损坏，拒绝猜测恢复", "checkpoint_corrupt") from exc
            if report.trajectory_fingerprint != trajectory_fingerprint:
                raise HarnessError(
                    "轨迹内容已变化，拒绝复用旧检查点",
                    "trajectory_fingerprint_mismatch",
                )
            if report.catalog_fingerprint != catalog_fingerprint:
                raise HarnessError(
                    "Canonical Tool 目录已变化，拒绝复用旧检查点",
                    "catalog_fingerprint_mismatch",
                )
            if report.executor_fingerprint != executor_fingerprint:
                raise HarnessError(
                    "Tool 执行器代码已变化，拒绝复用旧检查点",
                    "executor_fingerprint_mismatch",
                )
            if report.runtime_fingerprint != runtime_fingerprint:
                raise HarnessError(
                    "初始运行时上下文或 Provider 配置已变化，拒绝复用旧检查点",
                    "runtime_fingerprint_mismatch",
                )
            if report.execution_policy != execution_policy:
                raise HarnessError(
                    "stop_on_error 等执行策略已变化，拒绝复用旧检查点",
                    "execution_policy_mismatch",
                )
            _validate_checkpoint_records(report, trajectory, trajectory_fingerprint)
            report.resumed = True
            report.status = "running"
            report.finished_at = ""
            return report
        return HarnessReport(
            run_id=uuid.uuid4().hex,
            trajectory_id=trajectory.id,
            trajectory_fingerprint=trajectory_fingerprint,
            catalog_path=str(catalog_path),
            catalog_fingerprint=catalog_fingerprint,
            executor_fingerprint=executor_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            execution_policy=execution_policy,
            started_at=_utcnow(),
        )

    def _runtime_context(self, trajectory: Trajectory, run_dir: Path) -> RuntimeContext:
        ctx = replace(
            self.base_context,
            current_images=list(self.base_context.current_images),
            extras=dict(self.base_context.extras),
            allowed_file_roots=(
                list(self.base_context.allowed_file_roots)
                if self.base_context.allowed_file_roots is not None
                else None
            ),
        )
        if ctx.result_store is None:
            ctx.result_store = FilesystemResultStore(run_dir / "results")
        if ctx.image_store is None:
            ctx.image_store = FilesystemImageStore(run_dir / "images")
        raw_images = list(trajectory.image_paths) or list(ctx.current_images)
        if not raw_images and ctx.current_image:
            raw_images = [ctx.current_image]
        registered_images: list[str] = []
        for index, raw_path in enumerate(raw_images, start=1):
            path = Path(raw_path)
            if path.is_file():
                registered_images.append(
                    ctx.image_store.register(path, image_id=f"input_{index:04d}")
                )
            else:
                registered_images.append(raw_path)
        if registered_images:
            ctx.current_images = registered_images
            ctx.current_image = registered_images[0]
        roots = [run_dir, *self.config.allowed_file_roots]
        roots.extend(ctx.allowed_file_roots or [])
        roots.extend(Path(path).resolve() for path in trajectory.image_paths)
        ctx.allowed_file_roots = _unique_paths(roots)
        ctx.extras.setdefault("artifact_dir", str(run_dir / "artifacts"))
        return ctx

    def _can_reuse(self, record: ToolExecutionRecord, ctx: RuntimeContext) -> bool:
        if record.observation is not None:
            current_hash = _fingerprint(record.observation.model_dump(mode="json"))
            if record.observation_fingerprint != current_hash:
                return False
        if record.status in {"failed", "blocked", "invalid", "running"}:
            return not self.config.retry_failed and record.status != "running"
        if record.status not in _REUSABLE_STATUSES:
            return False
        observation = record.observation
        if observation is None:
            return True
        for key, value in observation.artifacts.items():
            if key.endswith("_path") and isinstance(value, str):
                expected = record.artifact_fingerprints.get(key)
                if expected is None or _file_identity(Path(value)) != expected:
                    return False
        result = observation.result or {}
        result_id = result.get("result_id")
        if isinstance(result_id, str) and ctx.result_store is not None:
            try:
                stored = ctx.result_store.get(result_id)
            except (KeyError, TypeError, ValueError):
                return False
            report_payload = dict(result)
            report_payload.pop("result_id", None)
            if _fingerprint(stored) != _fingerprint(report_payload):
                return False
        return True

    def _execute_tool_step(
        self,
        *,
        step_index: int,
        step: TrajectoryStep,
        trajectory_fingerprint: str,
        forest: ToolForest,
        ctx: RuntimeContext,
        step_results: dict[int, dict[str, Any]],
        report: HarnessReport,
        records: dict[int, ToolExecutionRecord],
        report_path: Path,
        events_path: Path,
    ) -> ToolExecutionRecord:
        assert step.action is not None
        call_id = _call_id(trajectory_fingerprint, step_index, step)
        params = dict(step.action.params)
        raw_inputs = params.get("inputs")
        operation = params.get("operation")
        purpose = params.get("purpose")
        safe_thought = _safe_text(step.thought)
        safe_purpose = _safe_text(str(purpose or ""))
        started_at = _utcnow()
        started_clock = time.perf_counter()

        contract_error = _outer_contract_error(params, operation, purpose, raw_inputs)
        if contract_error is not None:
            return ToolExecutionRecord(
                step_index=step_index,
                event_type="tool_call",
                status="invalid",
                thought=safe_thought,
                call_id=call_id,
                tool=step.action.tool,
                operation=str(operation or ""),
                purpose=safe_purpose,
                raw_inputs=redact_credentials(raw_inputs if isinstance(raw_inputs, dict) else {}),
                started_at=started_at,
                finished_at=_utcnow(),
                duration_ms=_elapsed_ms(started_clock),
                error_code="invalid_outer_contract",
                error=contract_error,
                note=contract_error,
            )
        assert isinstance(raw_inputs, dict)
        assert isinstance(operation, str)
        assert isinstance(purpose, str)
        if safe_thought != step.thought or safe_purpose != purpose:
            return ToolExecutionRecord(
                step_index=step_index,
                event_type="tool_call",
                status="invalid",
                thought=safe_thought,
                call_id=call_id,
                tool=step.action.tool,
                operation=operation,
                purpose=safe_purpose,
                raw_inputs=redact_credentials(raw_inputs),
                started_at=started_at,
                finished_at=_utcnow(),
                duration_ms=_elapsed_ms(started_clock),
                error_code="credential_in_action_text",
                error="Thought 或 purpose 含疑似凭证，已脱敏并阻止执行",
                note="credential_in_action_text",
            )
        if contains_credential_fields(raw_inputs):
            return ToolExecutionRecord(
                step_index=step_index,
                event_type="tool_call",
                status="invalid",
                thought=safe_thought,
                call_id=call_id,
                tool=step.action.tool,
                operation=operation,
                purpose=safe_purpose,
                raw_inputs=redact_credentials(raw_inputs),
                started_at=started_at,
                finished_at=_utcnow(),
                duration_ms=_elapsed_ms(started_clock),
                error_code="credential_in_inputs",
                error="密钥必须由运行时环境注入，禁止进入 Tool inputs",
                note="credential_in_inputs：密钥必须由运行时环境注入，禁止进入 Tool inputs",
            )

        try:
            resolved = resolve_inputs(
                raw_inputs,
                ctx=ctx,
                step_results=step_results,
                current_step=step_index,
            )
        except ReferenceResolutionError as exc:
            return ToolExecutionRecord(
                step_index=step_index,
                event_type="tool_call",
                status="blocked",
                thought=safe_thought,
                call_id=call_id,
                tool=step.action.tool,
                operation=operation,
                purpose=safe_purpose,
                raw_inputs=redact_credentials(raw_inputs),
                dependency_steps=(
                    [exc.dependency_step] if exc.dependency_step is not None else []
                ),
                started_at=started_at,
                finished_at=_utcnow(),
                duration_ms=_elapsed_ms(started_clock),
                error_code=exc.error_code,
                error=str(exc),
                note=f"{exc.error_code}: {exc}",
            )

        available_context = _available_context(ctx)
        try:
            audit = normalize_and_validate_tool_inputs(
                forest,
                tool=step.action.tool,
                operation=operation,
                inputs=resolved.value,
                step_index=step_index,
                available_context=available_context,
            )
        except Exception as exc:  # noqa: BLE001 - 单步目录/归一化错误不得击穿批次
            safe_error = f"{type(exc).__name__}: {_safe_text(str(exc))}"
            return ToolExecutionRecord(
                step_index=step_index,
                event_type="tool_call",
                status="invalid",
                thought=safe_thought,
                call_id=call_id,
                tool=step.action.tool,
                operation=operation,
                purpose=safe_purpose,
                raw_inputs=redact_credentials(raw_inputs),
                resolved_inputs=redact_credentials(resolved.value),
                references=resolved.bindings,
                dependency_steps=resolved.dependency_steps,
                started_at=started_at,
                finished_at=_utcnow(),
                duration_ms=_elapsed_ms(started_clock),
                error_code="parameter_validation_exception",
                error=safe_error,
                note="Canonical 参数归一化器异常；该步骤已隔离",
            )
        safe_audit = ToolParameterAudit.model_validate(
            redact_credentials(audit.model_dump(mode="json"))
        )
        dependency_steps = sorted(
            set(resolved.dependency_steps).union(_audit_context_dependencies(audit, ctx))
        )
        if contains_credential_fields(audit.normalized_inputs):
            return ToolExecutionRecord(
                step_index=step_index,
                event_type="tool_call",
                status="invalid",
                thought=safe_thought,
                call_id=call_id,
                tool=audit.tool,
                operation=audit.operation,
                purpose=safe_purpose,
                raw_inputs=redact_credentials(raw_inputs),
                resolved_inputs=redact_credentials(audit.normalized_inputs),
                references=resolved.bindings,
                dependency_steps=dependency_steps,
                parameter_audit=safe_audit,
                started_at=started_at,
                finished_at=_utcnow(),
                duration_ms=_elapsed_ms(started_clock),
                error_code="credential_in_resolved_inputs",
                error="运行时上下文注入了疑似凭证，已阻止执行",
                note="credential_in_resolved_inputs",
            )
        if audit.readiness not in {"ready", "context_resolvable"}:
            status = "invalid" if audit.readiness == "invalid" else "blocked"
            return ToolExecutionRecord(
                step_index=step_index,
                event_type="tool_call",
                status=status,
                thought=safe_thought,
                call_id=call_id,
                tool=audit.tool,
                operation=audit.operation,
                purpose=safe_purpose,
                raw_inputs=redact_credentials(raw_inputs),
                resolved_inputs=redact_credentials(resolved.value),
                references=resolved.bindings,
                dependency_steps=dependency_steps,
                parameter_audit=safe_audit,
                started_at=started_at,
                finished_at=_utcnow(),
                duration_ms=_elapsed_ms(started_clock),
                error_code=f"parameter_{audit.readiness}",
                error=f"parameter_readiness={audit.readiness}",
                note=f"parameter_readiness={audit.readiness}",
            )

        running = ToolExecutionRecord(
            step_index=step_index,
            event_type="tool_call",
            status="running",
            thought=safe_thought,
            call_id=call_id,
            tool=audit.tool,
            operation=audit.operation,
            purpose=safe_purpose,
            raw_inputs=redact_credentials(raw_inputs),
            resolved_inputs=redact_credentials(audit.normalized_inputs),
            references=resolved.bindings,
            dependency_steps=dependency_steps,
            parameter_audit=safe_audit,
            started_at=started_at,
        )
        records[step_index] = running
        report.records = [records[index] for index in sorted(records)]
        _save_report(report_path, report)
        _append_event(events_path, "step_started", running)

        previous_call_id = ctx.extras.get("harness_call_id")
        previous_step_index = ctx.extras.get("harness_step_index")
        ctx.extras["harness_call_id"] = call_id
        ctx.extras["harness_step_index"] = step_index
        try:
            observation = self._dispatch(
                audit.tool,
                audit.operation,
                purpose=safe_purpose,
                inputs=audit.normalized_inputs,
                ctx=ctx,
            )
        except Exception as exc:  # noqa: BLE001 - harness 必须隔离任意第三方异常
            observation = Observation(
                ok=False,
                error=f"{type(exc).__name__}: {_safe_text(str(exc))}",
                error_code="executor_exception",
            )
        finally:
            _restore_extra(ctx.extras, "harness_call_id", previous_call_id)
            _restore_extra(ctx.extras, "harness_step_index", previous_step_index)
        credential_output = (
            (
                observation.result is not None
                and contains_credential_fields(observation.result)
            )
            or contains_credential_fields(observation.artifacts)
            or contains_credential_fields(observation.session)
        )
        try:
            snapshot = _observation_snapshot(observation)
            if credential_output:
                snapshot = snapshot.model_copy(
                    update={
                        "ok": False,
                        "result": redact_credentials(snapshot.result),
                        "error": "执行器回执包含凭证字段，已阻止进入 harness 报告",
                        "error_code": "credential_in_observation",
                    }
                )
            snapshot = _canonicalize_observation(snapshot)
        except Exception as exc:  # noqa: BLE001 - Provider 回执边界必须兜住任意对象
            snapshot = HarnessObservation(
                ok=False,
                error=(
                    "执行器回执不是有限、可 JSON 序列化的 Observation: "
                    f"{type(exc).__name__}: {_safe_text(str(exc))}"
                ),
                error_code="invalid_observation",
            )
        status = "succeeded" if snapshot.ok else "failed"
        return running.model_copy(
            update={
                "status": status,
                "observation": snapshot,
                "observation_fingerprint": _fingerprint(
                    snapshot.model_dump(mode="json")
                ),
                "artifact_fingerprints": _artifact_file_identities(
                    snapshot.artifacts
                ),
                "finished_at": _utcnow(),
                "duration_ms": _elapsed_ms(started_clock),
                "error_code": snapshot.error_code if not snapshot.ok else None,
                "error": snapshot.error if not snapshot.ok else None,
            }
        )

    def _dispatch(
        self,
        tool_name: str,
        operation: str,
        *,
        purpose: str,
        inputs: dict[str, Any],
        ctx: RuntimeContext,
    ) -> Observation:
        if self.dispatcher is None:
            from tool import execute

            dispatcher = execute
        else:
            dispatcher = self.dispatcher
        observation = dispatcher(
            tool_name,
            operation,
            purpose=purpose,
            inputs=inputs,
            ctx=ctx,
        )
        if not isinstance(observation, Observation):
            raise TypeError("Tool dispatcher 必须返回 Observation")
        return observation

    @staticmethod
    def _reasoning_record(
        step_index: int,
        step: TrajectoryStep,
        fingerprint: str,
    ) -> ToolExecutionRecord:
        now = _utcnow()
        return ToolExecutionRecord(
            step_index=step_index,
            event_type="reasoning",
            status="skipped_reasoning",
            thought=_safe_text(step.thought),
            call_id=_call_id(fingerprint, step_index, step),
            started_at=now,
            finished_at=now,
            duration_ms=0.0,
            note="reasoning 只记录，不调用外部执行器",
        )

    @staticmethod
    def _final_record(
        step_index: int,
        step: TrajectoryStep,
        fingerprint: str,
    ) -> ToolExecutionRecord:
        assert step.action is not None
        location = step.action.params.get("location")
        now = _utcnow()
        return ToolExecutionRecord(
            step_index=step_index,
            event_type="final",
            status="terminal",
            thought=_safe_text(step.thought),
            call_id=_call_id(fingerprint, step_index, step),
            tool="final_answer",
            operation="submit",
            purpose="提交最终地点",
            raw_inputs={"location": location},
            resolved_inputs={"location": location},
            final_location=location,
            started_at=now,
            finished_at=now,
            duration_ms=0.0,
            note="终端合同校验通过；不调用外部服务",
        )

    @staticmethod
    def _blocked_after_failure(
        step_index: int,
        step: TrajectoryStep,
        fingerprint: str,
    ) -> ToolExecutionRecord:
        now = _utcnow()
        tool_name = step.action.tool if step.action else None
        return ToolExecutionRecord(
            step_index=step_index,
            event_type=step.event_type,
            status="blocked",
            thought=_safe_text(step.thought),
            call_id=_call_id(fingerprint, step_index, step),
            tool=tool_name,
            started_at=now,
            finished_at=now,
            duration_ms=0.0,
            error_code="prior_step_failed",
            error="stop_on_error 已阻止后续 Tool",
            note="prior_step_failed：stop_on_error 已阻止后续 Tool",
        )

    @staticmethod
    def _restore_context(
        record: ToolExecutionRecord,
        ctx: RuntimeContext,
        step_results: dict[int, dict[str, Any]],
    ) -> None:
        if record.status != "succeeded" or record.observation is None:
            return
        result = record.observation.result
        if not isinstance(result, dict):
            return
        step_results[record.step_index] = result
        ctx.previous_tool_result = result
        ctx.extras["_previous_tool_step"] = record.step_index
        area = record.resolved_inputs.get("area")
        if isinstance(area, str) and area.strip():
            ctx.active_area = area.strip()
            ctx.extras["_active_area_step"] = record.step_index
        session = record.observation.session
        if session is None:
            candidate = result.get("session") or result.get("session_id")
            session = candidate if isinstance(candidate, str) else None
        if session:
            ctx.active_session = session
            ctx.extras["_active_session_step"] = record.step_index


def _validate_trajectory_contract(trajectory: Trajectory) -> None:
    finals = [index for index, step in enumerate(trajectory.steps, start=1) if step.event_type == "final"]
    if len(finals) != 1:
        raise HarnessError(
            f"轨迹必须且只能包含一个 final，当前为 {len(finals)}",
            "invalid_final_count",
        )
    if finals[0] != len(trajectory.steps):
        raise HarnessError("final 必须是轨迹最后一步", "final_not_last")
    final = trajectory.steps[-1]
    assert final.action is not None
    if set(final.action.params) != {"location"}:
        raise HarnessError(
            "final_answer params 必须且只能包含 location",
            "invalid_final_contract",
        )
    location = final.action.params.get("location")
    valid_string = isinstance(location, str) and bool(location.strip())
    valid_list = (
        isinstance(location, list)
        and bool(location)
        and all(isinstance(item, str) and item.strip() for item in location)
    )
    if not (valid_string or valid_list):
        raise HarnessError("location 必须是非空字符串或字符串数组", "invalid_final_location")
    if contains_credential_fields(location):
        raise HarnessError("location 含疑似凭证，拒绝写入报告", "credential_in_final")


def _validate_checkpoint_records(
    report: HarnessReport,
    trajectory: Trajectory,
    trajectory_fingerprint: str,
) -> None:
    seen: set[int] = set()
    for record in report.records:
        if record.step_index in seen or not 1 <= record.step_index <= len(trajectory.steps):
            raise HarnessError("检查点步骤编号重复或越界", "checkpoint_record_invalid")
        seen.add(record.step_index)
        step = trajectory.steps[record.step_index - 1]
        if record.event_type != step.event_type:
            raise HarnessError("检查点事件类型与轨迹不一致", "checkpoint_record_invalid")
        if record.call_id != _call_id(trajectory_fingerprint, record.step_index, step):
            raise HarnessError("检查点 call_id 与轨迹不一致", "checkpoint_record_invalid")
        if record.status == "succeeded" and (
            record.observation is None or not record.observation.ok
        ):
            raise HarnessError("成功记录缺少成功 Observation", "checkpoint_record_invalid")
        if record.status == "failed" and (
            record.observation is None or record.observation.ok
        ):
            raise HarnessError("失败记录缺少失败 Observation", "checkpoint_record_invalid")
        if record.status == "terminal":
            assert step.action is not None
            if record.final_location != step.action.params.get("location"):
                raise HarnessError("终端记录与原始 location 不一致", "checkpoint_record_invalid")


def _outer_contract_error(
    params: dict[str, Any],
    operation: Any,
    purpose: Any,
    inputs: Any,
) -> str | None:
    extras = sorted(set(params) - _OUTER_KEYS)
    if extras:
        return f"outer_contract_extra_fields：{extras}"
    if not isinstance(operation, str) or not operation.strip():
        return "outer_contract_missing_operation"
    if not isinstance(purpose, str) or not purpose.strip():
        return "outer_contract_missing_purpose"
    if not isinstance(inputs, dict):
        return "outer_contract_inputs_not_object"
    return None


def _available_context(ctx: RuntimeContext) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for name in (
        "current_image",
        "current_images",
        "previous_tool_result",
        "active_area",
        "active_session",
    ):
        value = getattr(ctx, name)
        if value not in (None, [], {}):
            context[name] = value
    return context


def _audit_context_dependencies(
    audit: ToolParameterAudit,
    ctx: RuntimeContext,
) -> set[int]:
    source_steps = {
        "source_result": ctx.extras.get("_previous_tool_step"),
        "session": ctx.extras.get("_active_session_step"),
        "area": ctx.extras.get("_active_area_step"),
    }
    dependencies: set[int] = set()
    for issue in audit.issues:
        if issue.code != "required_input_from_context" or issue.field is None:
            continue
        step = source_steps.get(issue.field)
        if isinstance(step, int) and step >= 1:
            dependencies.add(step)
    return dependencies


def _observation_snapshot(observation: Observation) -> HarnessObservation:
    return HarnessObservation(
        ok=observation.ok,
        result=redact_credentials(observation.result),
        artifacts=redact_credentials(observation.artifacts),
        session=(
            redact_credentials(observation.session)
            if isinstance(observation.session, str)
            else observation.session
        ),
        error=_safe_text(observation.error) if observation.error else None,
        error_code=observation.error_code,
    )


def _canonicalize_observation(observation: HarnessObservation) -> HarnessObservation:
    """把回执收敛到稳定 JSON 值；拒绝对象实例、NaN 和 Infinity。"""

    encoded = json.dumps(
        observation.model_dump(mode="python"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return HarnessObservation.model_validate_json(encoded)


def _save_report(path: Path, report: HarnessReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(report.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _append_event(
    path: Path,
    event: str,
    record: ToolExecutionRecord | None,
    *,
    report: HarnessReport | None = None,
) -> None:
    payload: dict[str, Any] = {"timestamp": _utcnow(), "event": event}
    if record is not None:
        payload.update(
            {
                "step_index": record.step_index,
                "call_id": record.call_id,
                "status": record.status,
                "tool": record.tool,
                "operation": record.operation,
                "error_code": (
                    record.error_code
                    or (record.observation.error_code if record.observation else None)
                ),
                "error": record.error,
            }
        )
    if report is not None:
        payload.update({"run_id": report.run_id, "status": report.status, "counts": report.counts})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _counts(records: list[ToolExecutionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {"total": len(records)}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executor_fingerprint(dispatcher: Dispatcher | None) -> str:
    tool_root = Path(__file__).resolve().parents[1]
    repository_root = tool_root.parent
    runtime_contract_files = [
        repository_root / "pipeline" / "schemas" / "tools.py",
        repository_root / "pipeline" / "schemas" / "trajectory.py",
        repository_root / "pipeline" / "stage3_normalize_format" / "params.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(tool_root.rglob("*.py")):
        _update_source_digest(digest, path, repository_root)
    for path in runtime_contract_files:
        _update_source_digest(digest, path, repository_root)
    digest.update(_callable_identity(dispatcher).encode("utf-8"))
    return digest.hexdigest()


def _runtime_identity(
    context: RuntimeContext,
    config: ToolHarnessConfig,
    *,
    trajectory: Trajectory,
    dispatcher: Dispatcher | None,
) -> dict[str, Any]:
    providers = {
        key: f"{type(value).__module__}.{type(value).__qualname__}"
        for key, value in sorted(context.extras.items())
        if key.endswith(("_client", "_engine", "_probe"))
    }
    return {
        "initial_context": redact_credentials(
            {
                "current_image": context.current_image,
                "current_images": context.current_images,
                "previous_tool_result": context.previous_tool_result,
                "active_area": context.active_area,
                "active_session": context.active_session,
            }
        ),
        "allowed_file_roots": sorted(
            {
                os.path.normcase(str(Path(path).resolve()))
                for path in [
                    *config.allowed_file_roots,
                    *(context.allowed_file_roots or []),
                ]
            }
        ),
        "providers": providers,
        "provider_revision": (
            config.provider_revision
            or str(context.extras.get("harness_provider_revision") or "")
        ),
        "input_files": _input_file_identities(trajectory, context),
        "dispatcher": _callable_identity(dispatcher),
    }


def _callable_identity(dispatcher: Dispatcher | None) -> str:
    if dispatcher is None:
        return "tool.execute"
    module = getattr(dispatcher, "__module__", type(dispatcher).__module__)
    name = getattr(dispatcher, "__qualname__", type(dispatcher).__qualname__)
    return f"{module}.{name}"


def _update_source_digest(
    digest: Any,
    path: Path,
    repository_root: Path,
) -> None:
    digest.update(path.relative_to(repository_root).as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")


def _input_file_identities(
    trajectory: Trajectory,
    context: RuntimeContext,
) -> list[dict[str, Any]]:
    candidates: list[str] = [
        *trajectory.image_paths,
        *context.current_images,
    ]
    if context.current_image:
        candidates.append(context.current_image)
    for step in trajectory.steps:
        if step.action is not None:
            candidates.extend(_string_leaves(step.action.params.get("inputs", {})))

    identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate.startswith("$"):
            continue
        try:
            path = Path(candidate).resolve()
            if not path.is_file():
                continue
            key = os.path.normcase(str(path))
            if key in seen:
                continue
            seen.add(key)
            stat = path.stat()
            identities.append(
                {
                    "path": key,
                    "size_bytes": stat.st_size,
                    "sha256": _file_fingerprint(path),
                }
            )
        except OSError:
            continue
    return sorted(identities, key=lambda item: item["path"])


def _artifact_file_identities(
    artifacts: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for key, value in artifacts.items():
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        identity = _file_identity(Path(value))
        if identity is not None:
            identities[key] = identity
    return identities


def _file_identity(path: Path) -> dict[str, Any] | None:
    try:
        resolved = path.resolve()
        if not resolved.is_file():
            return None
        stat = resolved.stat()
        return {
            "path": os.path.normcase(str(resolved)),
            "size_bytes": stat.st_size,
            "sha256": _file_fingerprint(resolved),
        }
    except OSError:
        return None


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(_string_leaves(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_string_leaves(item))
        return result
    return []


def _call_id(fingerprint: str, step_index: int, step: TrajectoryStep) -> str:
    payload = {
        "trajectory": fingerprint,
        "step_index": step_index,
        "step": step.model_dump(mode="json"),
    }
    return "call_" + _fingerprint(payload)[:20]


def _safe_id(value: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", value.strip()).strip("._")
    stem = cleaned[:80] or "trajectory"
    return f"{stem}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def _unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(text: str) -> str:
    safe = re.sub(
        r"(?:sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._-]{16,})",
        "***REDACTED***",
        text,
        flags=re.IGNORECASE,
    )
    safe = re.sub(
        r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[=:]\s*[^\s&,;]+",
        r"\1=***REDACTED***",
        safe,
    )
    return safe[:1000]


def _append_warning_once(report: HarnessReport, warning: str) -> None:
    if warning not in report.warnings:
        report.warnings.append(warning)


def _restore_extra(extras: dict[str, Any], key: str, previous: Any) -> None:
    if previous is None:
        extras.pop(key, None)
    else:
        extras[key] = previous


__all__ = ["HarnessError", "ToolHarness", "ToolHarnessConfig"]
