"""会话内真实 Tool 结果仓库，供后续调用通过稳定 result_id 引用。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_RESULT_ID_RE = re.compile(r"^[a-z0-9_]+_[a-f0-9]{16}$")
_CREDENTIAL_KEY_RE = re.compile(
    r"(^|_)(api_?key|access_?token|client_?secret|password|authorization)($|_)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._-]{16,})",
    re.IGNORECASE,
)


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
        _reject_credentials(payload)
        prefix = re.sub(r"[^a-z0-9_]+", "", namespace.lower())
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


class FilesystemResultStore:
    """原子写入的持久化结果仓库；每个结果一个 JSON，不维护易损全局索引。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def put(self, namespace: str, payload: dict[str, Any]) -> str:
        _reject_credentials(payload)
        prefix = re.sub(r"[^a-z0-9_]+", "", namespace.lower())
        prefix = prefix.strip("_") or "result"
        result_id = f"{prefix}_{uuid.uuid4().hex[:16]}"
        envelope = {
            "schema_version": "tool_result_v1",
            "result_id": result_id,
            "namespace": prefix,
            "created_at": _utcnow(),
            "payload": copy.deepcopy(payload),
            "payload_sha256": _payload_fingerprint(payload),
        }
        with self._lock:
            _atomic_json(self.root / f"{result_id}.json", envelope)
        return result_id

    def get(self, result_id: str) -> dict[str, Any]:
        envelope = self.get_record(result_id)
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise TypeError(f"结果文件 payload 不是对象: {result_id}")
        return copy.deepcopy(payload)

    def get_record(self, result_id: str) -> dict[str, Any]:
        token = result_id.strip()
        if not _RESULT_ID_RE.fullmatch(token):
            raise KeyError(result_id)
        path = self.root / f"{token}.json"
        if not path.is_file():
            raise KeyError(result_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"结果文件损坏: {result_id}") from exc
        if not isinstance(raw, dict) or raw.get("result_id") != token:
            raise ValueError(f"结果文件 ID 不匹配: {result_id}")
        payload = raw.get("payload")
        expected_hash = raw.get("payload_sha256")
        if not isinstance(payload, dict) or expected_hash != _payload_fingerprint(payload):
            raise ValueError(f"结果文件 payload 哈希不匹配: {result_id}")
        return raw

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(
                path.stem
                for path in self.root.glob("*.json")
                if _RESULT_ID_RE.fullmatch(path.stem)
            )


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
        result_id = raw.get("result_id")
        if (
            isinstance(result_id, str)
            and ctx is not None
            and ctx.result_store is not None
            and len(raw) == 1
        ):
            try:
                return ctx.result_store.get(result_id)
            except (KeyError, TypeError, ValueError):
                return None
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
    except (KeyError, TypeError, ValueError):
        return None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_credentials(value: Any) -> None:
    if _contains_credentials(value):
        raise ValueError("ResultStore 拒绝持久化凭证字段或疑似密钥值")


def _contains_credentials(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if _CREDENTIAL_KEY_RE.search(normalized) or _contains_credentials(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_credentials(item) for item in value)
    return isinstance(value, str) and bool(_SECRET_VALUE_RE.search(value))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


__all__ = [
    "FilesystemResultStore",
    "InMemoryResultStore",
    "ResultStore",
    "resolve_result",
    "store_result",
]
