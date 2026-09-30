"""本机 MCP 会话：一份 RuntimeContext，调用落到 tool.execute。"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.schemas.tools import ToolForest
from pipeline.stage3_normalize_format.trees import load_forest
from tool.contract import Observation, RuntimeContext
from tool.mcp.catalog import (
    DEFAULT_CATALOG_PATH,
    PURPOSE_FIELD,
    McpToolSpec,
    specs_from_forest,
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


class McpSessionError(Exception):
    """会话配置无效，例如初始图片不在允许目录内。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass
class McpSession:
    """一个 stdio 进程对应一次任务；进程内复用 RuntimeContext。"""

    workspace_dir: Path
    catalog_path: Path = DEFAULT_CATALOG_PATH
    dispatcher: Dispatcher | None = None
    allowed_file_roots: list[Path] = field(default_factory=list)
    initial_images: list[Path] = field(default_factory=list)
    forest: ToolForest | None = None
    ctx: RuntimeContext = field(init=False)
    specs: dict[str, McpToolSpec] = field(init=False)
    step_results: dict[int, dict[str, Any]] = field(init=False)
    step_index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = Path(self.catalog_path)
        self.forest = self.forest or load_forest(self.catalog_path)
        self.specs = {spec.name: spec for spec in specs_from_forest(self.forest)}
        self.step_results = {}
        self.step_index = 0
        self.ctx = self._build_context()

    @classmethod
    def from_env(cls, *, dispatcher: Dispatcher | None = None) -> McpSession:
        """按环境变量创建会话；未设置允许目录时禁止任意本机路径。"""

        workspace = Path(
            os.environ.get("GEOAGENT_MCP_WORKSPACE", "data/mcp_sessions")
        ) / str(os.getpid())
        catalog = Path(os.environ.get("TOOL_CATALOG_PATH", str(DEFAULT_CATALOG_PATH)))
        return cls(
            workspace_dir=workspace,
            catalog_path=catalog,
            dispatcher=dispatcher,
            allowed_file_roots=_split_paths(os.environ.get("GEOAGENT_MCP_FILE_ROOTS")),
            initial_images=_split_paths(os.environ.get("GEOAGENT_MCP_IMAGES")),
        )

    def list_specs(self) -> list[McpToolSpec]:
        """当前会话对外暴露的 MCP 工具表。"""

        return list(self.specs.values())

    def call(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """执行一次 MCP 工具调用，始终返回结构化 JSON 对象。"""

        args = dict(arguments or {})
        spec = self.specs.get(name)
        if spec is None:
            return _failure(
                f"未知 MCP 工具: {name}",
                "unknown_mcp_tool",
            )
        tool = spec.tool
        operation = spec.operation

        purpose = args.get(PURPOSE_FIELD)
        if not isinstance(purpose, str) or not purpose.strip():
            return _failure("purpose 必须是非空字符串", "missing_purpose")
        inputs = {key: value for key, value in args.items() if key != PURPOSE_FIELD}
        if contains_credential_fields(purpose) or contains_credential_fields(
            {"purpose": purpose}
        ):
            return _failure(
                "密钥必须由运行时环境注入，禁止进入 purpose",
                "credential_in_purpose",
            )
        if contains_credential_fields(inputs):
            return _failure(
                "密钥必须由运行时环境注入，禁止进入 Tool inputs",
                "credential_in_inputs",
            )
        path_error = _disallowed_path_error(inputs, self.ctx)
        if path_error is not None:
            return _failure(path_error, "path_not_allowed")

        next_step = self.step_index + 1
        try:
            resolved = resolve_inputs(
                inputs,
                ctx=self.ctx,
                step_results=self.step_results,
                current_step=next_step,
            )
        except ReferenceResolutionError as exc:
            return _failure(str(exc), exc.error_code)
        if contains_credential_fields(resolved.value):
            return _failure(
                "运行时上下文注入了疑似凭证，已阻止执行",
                "credential_in_resolved_inputs",
            )

        self.step_index = next_step
        dispatcher = self.dispatcher
        if dispatcher is None:
            from tool import execute as dispatcher
        try:
            observation = dispatcher(
                tool,
                operation,
                purpose=purpose.strip(),
                inputs=resolved.value,
                ctx=self.ctx,
            )
        except Exception as exc:  # noqa: BLE001 - MCP 不得因执行器异常断开连接
            return _failure(
                f"{type(exc).__name__}: {exc}",
                "executor_exception",
            )

        payload = observation_payload(observation)
        if payload["ok"] and isinstance(observation.result, dict):
            _update_context_after_success(
                ctx=self.ctx,
                step_index=self.step_index,
                result=observation.result,
                resolved_inputs=resolved.value,
                observation=observation,
                step_results=self.step_results,
            )
        return payload

    def _build_context(self) -> RuntimeContext:
        image_store = FilesystemImageStore(self.workspace_dir / "images")
        result_store = FilesystemResultStore(self.workspace_dir / "results")
        roots = _unique_paths([self.workspace_dir, *self.allowed_file_roots])
        registered: list[str] = []
        for index, raw in enumerate(self.initial_images, start=1):
            path = Path(raw)
            if not path.is_file():
                raise McpSessionError(
                    f"初始图片不存在: {path}",
                    "image_not_found",
                )
            resolved = path.resolve()
            if not any(_is_within(resolved, root) for root in roots):
                raise McpSessionError(
                    f"初始图片不在运行时允许目录内: {resolved}",
                    "path_not_allowed",
                )
            registered.append(
                image_store.register(resolved, image_id=f"input_{index:04d}")
            )
        return RuntimeContext(
            current_image=registered[0] if registered else None,
            current_images=list(registered),
            extras={"artifact_dir": str(self.workspace_dir / "artifacts")},
            image_store=image_store,
            result_store=result_store,
            allowed_file_roots=roots,
        )


def observation_payload(observation: Observation) -> dict[str, Any]:
    """把 Observation 收成可 JSON 序列化的 MCP 回执。"""

    artifacts = observation.artifacts if isinstance(observation.artifacts, dict) else {}
    return _jsonable(
        {
            "ok": bool(observation.ok),
            "result": redact_credentials(observation.result),
            "artifacts": _strip_artifact_paths(redact_credentials(artifacts)),
            "session": (
                redact_credentials(observation.session)
                if isinstance(observation.session, str)
                else observation.session
            ),
            "error": observation.error,
            "error_code": observation.error_code,
        }
    )


def _update_context_after_success(
    *,
    ctx: RuntimeContext,
    step_index: int,
    result: dict[str, Any],
    resolved_inputs: dict[str, Any],
    observation: Observation,
    step_results: dict[int, dict[str, Any]],
) -> None:
    step_results[step_index] = result
    ctx.previous_tool_result = result
    ctx.extras["_previous_tool_step"] = step_index
    area = resolved_inputs.get("area")
    if isinstance(area, str) and area.strip():
        ctx.active_area = area.strip()
        ctx.extras["_active_area_step"] = step_index
    session = observation.session
    if session is None:
        candidate = result.get("session") or result.get("session_id")
        session = candidate if isinstance(candidate, str) else None
    if isinstance(session, str) and session.strip():
        ctx.active_session = session.strip()
        ctx.extras["_active_session_step"] = step_index
    image_id = result.get("image_id")
    if isinstance(image_id, str) and image_id.strip() and _image_registered(
        ctx, image_id.strip()
    ):
        token = image_id.strip()
        ctx.current_image = token
        if token not in ctx.current_images:
            ctx.current_images.append(token)


def _image_registered(ctx: RuntimeContext, image_id: str) -> bool:
    store = ctx.image_store
    if store is None:
        return False
    try:
        store.resolve(image_id)
    except FileNotFoundError:
        return False
    return True


def _disallowed_path_error(value: Any, ctx: RuntimeContext) -> str | None:
    if isinstance(value, dict):
        for item in value.values():
            error = _disallowed_path_error(item, ctx)
            if error is not None:
                return error
        return None
    if isinstance(value, list):
        for item in value:
            error = _disallowed_path_error(item, ctx)
            if error is not None:
                return error
        return None
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or token.startswith("$"):
        return None
    if ctx.image_store is not None:
        try:
            ctx.image_store.resolve(token)
            return None
        except FileNotFoundError:
            pass
    path = Path(token)
    if not path.is_file():
        return None
    resolved = path.resolve()
    roots = [Path(root).resolve() for root in (ctx.allowed_file_roots or [])]
    if not roots or not any(_is_within(resolved, root) for root in roots):
        return f"文件不在运行时允许目录内: {resolved}"
    return None


def _strip_artifact_paths(artifacts: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in artifacts.items():
        if str(key).endswith("_path"):
            continue
        if isinstance(value, str) and Path(value).is_absolute():
            continue
        cleaned[key] = value
    return cleaned


def _split_paths(raw: str | None) -> list[Path]:
    if raw is None or not raw.strip():
        return []
    return [Path(part.strip()) for part in raw.split(";") if part.strip()]


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = Path(path).resolve()
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _failure(message: str, error_code: str) -> dict[str, Any]:
    return {
        "ok": False,
        "result": None,
        "artifacts": {},
        "session": None,
        "error": message,
        "error_code": error_code,
    }


def _jsonable(value: Any) -> Any:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    )
    return json.loads(encoded)
