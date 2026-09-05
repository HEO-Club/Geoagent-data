"""真实 Tool 执行器的公共合同。

独立于 `pipeline/`：蒸馏阶段禁止调用真实 Tool API；本包只给日后运行时 / MCP 使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tool.runtime.image_store import ImageStore


@dataclass
class RuntimeContext:
    """一次定位会话内跨 tool 共享的引用，由运行时注入。"""

    current_image: str | None = None
    current_images: list[str] = field(default_factory=list)
    previous_tool_result: dict[str, Any] | None = None
    active_area: str | None = None
    active_session: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    image_store: ImageStore | None = None


@dataclass
class Observation:
    """执行器回执信封；`result` 对应目录中的 observation_fields.result。"""

    ok: bool
    result: dict[str, Any] | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    session: str | None = None
    error: str | None = None
    error_code: str | None = None


def not_implemented(
    tool: str,
    operation: str,
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """尚未实现的 operation 的统一占位回执。"""

    del purpose, inputs, ctx
    return Observation(
        ok=False,
        result=None,
        error=f"{tool}.{operation} 尚未实现真实执行器",
        error_code="not_implemented",
    )
