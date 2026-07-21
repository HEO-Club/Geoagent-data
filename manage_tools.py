"""Tool 库 CLI：list / stats / register。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.schemas import ToolDefinition
from pipeline.tools.registry import load_registry, register_tool


def cmd_list() -> int:
    registry = load_registry()
    tools = list(registry.values())
    for t in tools:
        print(
            f"{t.name}\tterminal={t.is_terminal}\t"
            f"params={len(t.params)}\tobs_fields={len(t.observation_fields)}\t"
            f"agents={[a.value for a in t.allowed_agents]}"
        )
    print(f"total: {len(tools)}")
    return 0


def cmd_stats() -> int:
    registry = load_registry()
    agents: Counter[str] = Counter()
    for t in registry.values():
        for a in t.allowed_agents:
            agents[a.value] += 1
    report: dict[str, Any] = {
        "total": len(registry),
        "terminal": [n for n, t in registry.items() if t.is_terminal],
        "non_terminal": [n for n, t in registry.items() if not t.is_terminal],
        "by_agent_allow": dict(agents),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_register(from_json: str) -> int:
    path = Path(from_json)
    if not path.is_file():
        print(f"文件不存在: {path}", file=sys.stderr)
        return 1
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        tools = [ToolDefinition.model_validate(item) for item in raw]
    else:
        tools = [ToolDefinition.model_validate(raw)]
    try:
        for tool in tools:
            register_tool(tool)
            print(f"registered: {tool.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"register failed: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage tool_registry.json")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出 tools")
    sub.add_parser("stats", help="统计 registry")

    p_reg = sub.add_parser("register", help="从 JSON 注册 tool（schema 校验后写入）")
    p_reg.add_argument(
        "--from-json",
        required=True,
        help="单个 ToolDefinition JSON 或 ToolDefinition 数组",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return cmd_list()
    if args.command == "stats":
        return cmd_stats()
    if args.command == "register":
        return cmd_register(args.from_json)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
