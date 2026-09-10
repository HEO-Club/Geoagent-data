"""受限、可审计的 Tool input 引用解析；不使用 eval 或通用 JSONPath。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from tool.contract import RuntimeContext
from tool.runtime.harness_models import ReferenceBinding

_BASE_RE = re.compile(
    r"^\$(?P<base>current_image|current_images|previous_tool_result|active_area|"
    r"active_session|step_(?P<step>\d+)_tool_result)(?P<suffix>.*)$"
)
_DOT_KEY_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_:-]*)")
_INDEX_RE = re.compile(r"\[(\d+)\]")
_REFERENCE_LIKE_RE = re.compile(r"^\$[A-Za-z_]")
_CREDENTIAL_KEY_RE = re.compile(
    r"(^|_)(api_?key|access_?token|client_?secret|password|authorization)($|_)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._-]{16,})",
    re.IGNORECASE,
)


class ReferenceResolutionError(Exception):
    """引用未知、越界或依赖未成功。"""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        reference: str,
        input_path: str,
        dependency_step: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.reference = reference
        self.input_path = input_path
        self.dependency_step = dependency_step


@dataclass(frozen=True)
class ResolvedInputs:
    value: dict[str, Any]
    bindings: list[ReferenceBinding]
    dependency_steps: list[int]


def resolve_inputs(
    inputs: dict[str, Any],
    *,
    ctx: RuntimeContext,
    step_results: dict[int, dict[str, Any]],
    current_step: int,
) -> ResolvedInputs:
    bindings: list[ReferenceBinding] = []
    dependencies: set[int] = set()
    resolved = _resolve_value(
        inputs,
        input_path="$",
        ctx=ctx,
        step_results=step_results,
        current_step=current_step,
        bindings=bindings,
        dependencies=dependencies,
    )
    assert isinstance(resolved, dict)
    return ResolvedInputs(
        value=resolved,
        bindings=bindings,
        dependency_steps=sorted(dependencies),
    )


def contains_credential_fields(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _is_credential_key(str(key)) or contains_credential_fields(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_credential_fields(item) for item in value)
    if isinstance(value, str):
        return bool(_SECRET_VALUE_RE.search(value))
    return False


def redact_credentials(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "***REDACTED***"
                if _is_credential_key(str(key))
                else redact_credentials(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub("***REDACTED***", value)
    return value


def _resolve_value(
    value: Any,
    *,
    input_path: str,
    ctx: RuntimeContext,
    step_results: dict[int, dict[str, Any]],
    current_step: int,
    bindings: list[ReferenceBinding],
    dependencies: set[int],
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _resolve_value(
                item,
                input_path=f"{input_path}.{key}",
                ctx=ctx,
                step_results=step_results,
                current_step=current_step,
                bindings=bindings,
                dependencies=dependencies,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_value(
                item,
                input_path=f"{input_path}[{index}]",
                ctx=ctx,
                step_results=step_results,
                current_step=current_step,
                bindings=bindings,
                dependencies=dependencies,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    match = _BASE_RE.fullmatch(value.strip())
    if match is None:
        if _REFERENCE_LIKE_RE.match(value.strip()):
            raise ReferenceResolutionError(
                f"不支持或格式错误的运行时引用: {value}",
                error_code="unknown_reference",
                reference=value,
                input_path=input_path,
            )
        return value

    base = match.group("base")
    dependency_step: int | None = None
    if base == "current_image":
        source = "runtime.current_image"
        resolved = ctx.current_image
    elif base == "current_images":
        source = "runtime.current_images"
        resolved = list(ctx.current_images)
    elif base == "previous_tool_result":
        source = "runtime.previous_tool_result"
        resolved = ctx.previous_tool_result
        dependency_step = _context_step(ctx, "_previous_tool_step")
    elif base == "active_area":
        source = "runtime.active_area"
        resolved = ctx.active_area
        dependency_step = _context_step(ctx, "_active_area_step")
    elif base == "active_session":
        source = "runtime.active_session"
        resolved = ctx.active_session
        dependency_step = _context_step(ctx, "_active_session_step")
    else:
        dependency_step = int(match.group("step"))
        if dependency_step >= current_step:
            raise ReferenceResolutionError(
                f"步骤 {current_step} 不得引用当前或未来步骤 {dependency_step}",
                error_code="forward_reference",
                reference=value,
                input_path=input_path,
                dependency_step=dependency_step,
            )
        source = f"step_{dependency_step}_tool_result"
        resolved = step_results.get(dependency_step)
    if dependency_step is not None:
        dependencies.add(dependency_step)

    if resolved is None:
        raise ReferenceResolutionError(
            f"运行时引用尚无可用值: {value}",
            error_code="unresolved_reference",
            reference=value,
            input_path=input_path,
            dependency_step=dependency_step,
        )
    resolved = _walk_suffix(
        resolved,
        match.group("suffix"),
        reference=value,
        input_path=input_path,
        dependency_step=dependency_step,
    )
    if contains_credential_fields(resolved):
        raise ReferenceResolutionError(
            "禁止通过运行时引用读取凭证字段",
            error_code="credential_reference_forbidden",
            reference=value,
            input_path=input_path,
            dependency_step=dependency_step,
        )
    bindings.append(
        ReferenceBinding(
            input_path=input_path,
            reference=value,
            source=source,
            dependency_step=dependency_step,
            value_type=type(resolved).__name__,
            value_preview=_preview(resolved),
        )
    )
    return resolved


def _walk_suffix(
    value: Any,
    suffix: str,
    *,
    reference: str,
    input_path: str,
    dependency_step: int | None,
) -> Any:
    current = value
    cursor = 0
    while cursor < len(suffix):
        dot = _DOT_KEY_RE.match(suffix, cursor)
        if dot is not None:
            key = dot.group(1)
            if _is_credential_key(key):
                raise ReferenceResolutionError(
                    "禁止引用凭证字段",
                    error_code="credential_reference_forbidden",
                    reference=reference,
                    input_path=input_path,
                    dependency_step=dependency_step,
                )
            if not isinstance(current, dict) or key not in current:
                raise ReferenceResolutionError(
                    f"引用字段不存在: {key}",
                    error_code="reference_path_missing",
                    reference=reference,
                    input_path=input_path,
                    dependency_step=dependency_step,
                )
            current = current[key]
            cursor = dot.end()
            continue
        index_match = _INDEX_RE.match(suffix, cursor)
        if index_match is not None:
            index = int(index_match.group(1))
            if not isinstance(current, list) or index >= len(current):
                raise ReferenceResolutionError(
                    f"引用数组下标越界: {index}",
                    error_code="reference_index_out_of_range",
                    reference=reference,
                    input_path=input_path,
                    dependency_step=dependency_step,
                )
            current = current[index]
            cursor = index_match.end()
            continue
        raise ReferenceResolutionError(
            f"不支持的引用路径语法: {suffix[cursor:]}",
            error_code="invalid_reference_path",
            reference=reference,
            input_path=input_path,
            dependency_step=dependency_step,
        )
    return current


def _is_credential_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return bool(_CREDENTIAL_KEY_RE.search(normalized))


def _context_step(ctx: RuntimeContext, name: str) -> int | None:
    value = ctx.extras.get(name)
    return value if isinstance(value, int) and value >= 1 else None


def _preview(value: Any) -> str:
    safe = redact_credentials(value)
    try:
        text = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        text = repr(safe)
    return text if len(text) <= 160 else text[:157] + "..."


__all__ = [
    "ReferenceResolutionError",
    "ResolvedInputs",
    "contains_credential_fields",
    "redact_credentials",
    "resolve_inputs",
]
