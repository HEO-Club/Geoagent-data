"""会话内真实 Tool 结果仓库，供后续调用通过稳定 result_id 引用。"""

from __future__ import annotations

import copy
import threading
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ResultStore(Protocol):
    """保存和读取结构化 Tool 结果。"""

    def put(self, namespace: str, payload: dict[str, Any]) -> str:
        """保存结果并返回会话内唯一 ID。"""

    def get(self, result_id: str) -> dict[str, Any]:
        """读取结果；未知 ID 时抛出 KeyError。"""


class InMemoryResultStore:
    """线程安全的会话内结果仓库；不跨进程持久化。"""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._counter = 0
        self._lock = threading.RLock()

    def put(self, namespace: str, payload: dict[str, Any]) -> str:
        prefix = "".join(ch for ch in namespace.lower() if ch.isalnum() or ch == "_")
        prefix = prefix.strip("_") or "result"
        with self._lock:
            self._counter += 1
            result_id = f"{prefix}_{self._counter:04d}"
            self._items[result_id] = copy.deepcopy(payload)
            return result_id

    def get(self, result_id: str) -> dict[str, Any]:
        with self._lock:
            if result_id not in self._items:
                raise KeyError(result_id)
            return copy.deepcopy(self._items[result_id])


def store_result(
    payload: dict[str, Any],
    *,
    namespace: str,
    ctx: Any,
) -> str | None:
    store = ctx.result_store if ctx is not None else None
    if store is None:
        return None
    return store.put(namespace, payload)


def resolve_result(raw: Any, ctx: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    if raw == "$previous_tool_result" and ctx is not None:
        previous = ctx.previous_tool_result
        return copy.deepcopy(previous) if isinstance(previous, dict) else None
    if not isinstance(raw, str) or not raw.strip() or ctx is None:
        return None
    store = ctx.result_store
    if store is None:
        return None
    try:
        return store.get(raw.strip())
    except KeyError:
        return None


__all__ = ["InMemoryResultStore", "ResultStore", "resolve_result", "store_result"]
