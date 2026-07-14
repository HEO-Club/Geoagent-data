"""tool_registry.json 读写：跨进程文件锁、原子写、注册与升档。"""

from __future__ import annotations

import importlib
import json
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import editdistance
from filelock import FileLock

from pipeline.config import get_settings
from pipeline.schemas import AgentRole, ToolDefinition, ToolTier
from pipeline.tools.validation import validate_action_params, validate_observation

SEMANTIC_SIMILARITY_THRESHOLD = 0.85
NAME_EDIT_DISTANCE_REJECT = 3

StageRerunner = Callable[[str, list[str]], dict[str, Any]]


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
    """写入新 tool（文件锁 + 重读 + 去重 + 原子替换）。自动注入 created_at。"""
    registry_path = _registry_path(path)
    lock = FileLock(str(_lock_path(registry_path)))
    with lock:
        existing = load_registry(registry_path)
        _assert_no_duplicates(tool, existing)
        _assert_derived_from(tool, existing)
        payload = tool.model_dump(mode="json")
        if not payload.get("created_at"):
            payload["created_at"] = _utcnow_iso()
        else:
            payload["created_at"] = _utcnow_iso()
        # 新建 Draft 不得为 production
        if payload.get("tier") == ToolTier.PRODUCTION.value:
            raise ValueError("新建 Tool 的初始 tier 不得为 production")
        payload["tier"] = ToolTier.DRAFT.value
        payload["executor_ref"] = None
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


def _import_executor(executor_ref: str) -> Callable[..., Any]:
    if "." not in executor_ref:
        raise ValueError(f"executor_ref 必须为可导入路径: {executor_ref!r}")
    module_path, _, attr = executor_ref.rpartition(".")
    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr, None)
    if fn is None or not callable(fn):
        raise ValueError(f"executor_ref 不可调用: {executor_ref!r}")
    return fn


def _example_params(tool: ToolDefinition) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for field in tool.params:
        if field.required or field.default is None:
            params[field.name] = field.example
        # optional with default: leave unset so defaults apply; still include for map_query
    if tool.name == "map_query":
        # 保证交叉约束至少有一个
        if "query" not in params and "latlng" not in params:
            params["query"] = next(p.example for p in tool.params if p.name == "query")
    return params


def _smoke_test_executor(tool: ToolDefinition, executor_ref: str, image_path: str) -> dict[str, Any]:
    """import + 样例 params 校验 + 真实执行 + observation schema。"""
    execute_fn = _import_executor(executor_ref)
    params = validate_action_params(tool, _example_params(tool))
    observation = execute_fn(params, image_path)
    validated = validate_observation(tool, observation)
    if validated is None:
        raise ValueError("smoke test: 非 terminal Tool 不应返回 None observation")
    return validated


def _default_find_affected_video_ids(tool_name: str, output_dir: Path) -> list[str]:
    """扫描 output JSONL 中 draft_tool_names 含该 tool 的样本，返回 video_id 列表。"""
    if not output_dir.is_dir():
        return []
    affected: set[str] = set()
    for jsonl in output_dir.rglob("*.jsonl"):
        try:
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                names = row.get("draft_tool_names") or []
                if tool_name in names:
                    vid = row.get("source_video") or row.get("id") or jsonl.stem
                    affected.add(str(vid))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(affected)


def _default_stage_rerunner(tool_name: str, video_ids: list[str]) -> dict[str, Any]:
    """阶段尚未实现时的默认重跑钩子：无样本则空成功；有样本则明确失败。"""
    if not video_ids:
        return {"rerun": [], "note": "no affected samples"}
    raise NotImplementedError(
        f"promote_tool 需重跑 stage4→stage7，但阶段模块尚未实现；"
        f"受影响样本: {video_ids}（tool={tool_name}）。"
        f"请注入 stage_rerunner 回调，或待 stage 落地后再升档历史数据。"
    )


def promote_tool(
    tool_name: str,
    executor_ref: str,
    *,
    path: Optional[str | Path] = None,
    image_path: str = "",
    stage_rerunner: Optional[StageRerunner] = None,
    find_affected: Optional[Callable[[str], list[str]]] = None,
) -> dict[str, Any]:
    """升档流程：备份 → import/smoke/schema → 原子更新 → 重跑受影响样本 → 失败回滚。"""
    settings = get_settings()
    registry_path = _registry_path(path)
    intermediate_dir = Path(settings.INTERMEDIATE_DIR)
    output_dir = Path(settings.OUTPUT_DIR)
    backup_root = Path(settings.CACHE_DIR) / "promote_backups" / f"{tool_name}_{_utcnow_iso().replace(':', '')}"
    backup_root.mkdir(parents=True, exist_ok=True)

    lock = FileLock(str(_lock_path(registry_path)))
    with lock:
        registry = load_registry(registry_path)
        if tool_name not in registry:
            raise KeyError(f"Tool 不存在: {tool_name}")
        tool = registry[tool_name]
        if tool.is_terminal:
            raise ValueError("terminal Tool 无需升档 executor")

        # 1. 备份 registry
        shutil.copy2(registry_path, backup_root / "tool_registry.json")
        if intermediate_dir.is_dir():
            shutil.copytree(intermediate_dir, backup_root / "intermediate", dirs_exist_ok=True)
        if output_dir.is_dir():
            shutil.copytree(output_dir, backup_root / "output", dirs_exist_ok=True)

        report: dict[str, Any] = {
            "tool_name": tool_name,
            "executor_ref": executor_ref,
            "backup_dir": str(backup_root),
            "status": "pending",
        }

        try:
            # 2–4. import + smoke + schema
            smoke_image = image_path or str(backup_root / "smoke_placeholder.txt")
            if not Path(smoke_image).exists():
                Path(smoke_image).write_text("smoke", encoding="utf-8")
            smoke_obs = _smoke_test_executor(tool, executor_ref, smoke_image)
            report["smoke_observation"] = smoke_obs

            # 5. 原子更新 registry
            updated = tool.model_copy(
                update={
                    "tier": ToolTier.PRODUCTION,
                    "executor_ref": executor_ref,
                }
            )
            # model_validate 会检查 executor 可导入
            updated = ToolDefinition.model_validate(updated.model_dump(mode="json"))
            registry[tool_name] = updated
            _atomic_write(registry_path, list(registry.values()))

            # 6–7. 查找并重跑
            finder = find_affected or (
                lambda name: _default_find_affected_video_ids(name, output_dir)
            )
            affected = finder(tool_name)
            report["affected_videos"] = affected
            rerunner = stage_rerunner or _default_stage_rerunner
            rerun_result = rerunner(tool_name, affected)
            report["rerun"] = rerun_result
            report["status"] = "success"
            report["tier"] = ToolTier.PRODUCTION.value
            return report
        except Exception as exc:
            # 8. 回滚
            bak_registry = backup_root / "tool_registry.json"
            if bak_registry.is_file():
                shutil.copy2(bak_registry, registry_path)
            bak_inter = backup_root / "intermediate"
            if bak_inter.is_dir():
                if intermediate_dir.exists():
                    shutil.rmtree(intermediate_dir)
                shutil.copytree(bak_inter, intermediate_dir)
            bak_out = backup_root / "output"
            if bak_out.is_dir():
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                shutil.copytree(bak_out, output_dir)
            report["status"] = "rolled_back"
            report["error"] = str(exc)
            raise
