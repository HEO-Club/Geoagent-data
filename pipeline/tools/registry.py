"""tool_registry.json 读写：跨进程文件锁、原子写与注册。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import editdistance
from filelock import FileLock

from pipeline.config import get_settings
from pipeline.schemas import AgentRole, ToolDefinition

SEMANTIC_SIMILARITY_THRESHOLD = 0.85
NAME_EDIT_DISTANCE_REJECT = 3


def _registry_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(get_settings().TOOL_REGISTRY_PATH)


def _lock_path(registry_path: Path) -> Path:
    return registry_path.with_suffix(registry_path.suffix + ".lock")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_raw(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.is_file():
        return []
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tool_registry.json 根节点必须为数组")
    return data


def _atomic_write(registry_path: Path, tools: list[ToolDefinition]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [t.model_dump(mode="json") for t in tools]
    tmp = registry_path.with_suffix(registry_path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        fh.fileno() and __import__("os").fsync(fh.fileno())
    tmp.replace(registry_path)


def load_registry(path: Optional[str | Path] = None) -> dict[str, ToolDefinition]:
    """从 tool_registry.json 读取全部 tool 定义。"""
    registry_path = _registry_path(path)
    items = _read_raw(registry_path)
    result: dict[str, ToolDefinition] = {}
    for item in items:
        tool = ToolDefinition.model_validate(item)
        if tool.name in result:
            raise ValueError(f"registry 中存在重复 tool 名: {tool.name}")
        result[tool.name] = tool
    return result


def _token_jaccard(a: str, b: str) -> float:
    ta = set(a.lower().replace("_", " ").split())
    tb = set(b.lower().replace("_", " ").split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _semantic_similarity(a: ToolDefinition, b: ToolDefinition) -> float:
    """名称 token + 描述 token 的简单相似度（无外部 embedding）。"""
    name_sim = _token_jaccard(a.name.replace("_", " "), b.name.replace("_", " "))
    desc_sim = _token_jaccard(a.description, b.description)
    return max(name_sim, 0.6 * name_sim + 0.4 * desc_sim)


def _assert_no_duplicates(new_tool: ToolDefinition, existing: dict[str, ToolDefinition]) -> None:
    if new_tool.name in existing:
        raise ValueError(f"Tool 名称已存在: {new_tool.name}")
    for other in existing.values():
        dist = editdistance.eval(new_tool.name, other.name)
        if dist <= NAME_EDIT_DISTANCE_REJECT:
            raise ValueError(
                f"Tool 名称与现有 {other.name!r} 编辑距离={dist}≤{NAME_EDIT_DISTANCE_REJECT}，拒绝注册"
            )
        sim = _semantic_similarity(new_tool, other)
        if sim >= SEMANTIC_SIMILARITY_THRESHOLD:
            raise ValueError(
                f"Tool 与现有 {other.name!r} 语义相似度={sim:.2f}≥"
                f"{SEMANTIC_SIMILARITY_THRESHOLD}，拒绝注册"
            )


def _assert_derived_from(tool: ToolDefinition, existing: dict[str, ToolDefinition]) -> None:
    missing = [n for n in tool.derived_from_existing_tools if n not in existing]
    if missing:
        raise ValueError(f"derived_from_existing_tools 不存在于 Registry: {missing}")


def register_tool(tool: ToolDefinition, path: Optional[str | Path] = None) -> None:
    """写入新 tool（文件锁 + 重读 + 去重 + schema 校验 + 原子替换）。自动注入 created_at。"""
    registry_path = _registry_path(path)
    lock = FileLock(str(_lock_path(registry_path)))
    with lock:
        existing = load_registry(registry_path)
        _assert_no_duplicates(tool, existing)
        _assert_derived_from(tool, existing)
        payload = tool.model_dump(mode="json")
        payload["created_at"] = _utcnow_iso()
        # ToolDefinition.model_validate 强制完整 schema（含 F 规则）
        final = ToolDefinition.model_validate(payload)
        existing[final.name] = final
        _atomic_write(registry_path, list(existing.values()))


def get_tools_for_agent(
    role: AgentRole,
    path: Optional[str | Path] = None,
) -> list[ToolDefinition]:
    """按 allowed_agents 过滤。"""
    registry = load_registry(path)
    return [t for t in registry.values() if role in t.allowed_agents]
