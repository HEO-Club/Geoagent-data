"""Tool 生命周期 CLI：list / promote / stats。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any

from pipeline.tools.registry import load_registry, promote_tool


def cmd_list(tier: str | None) -> int:
    registry = load_registry()
    tools = list(registry.values())
    if tier:
        tools = [t for t in tools if t.tier.value == tier]
    for t in tools:
        print(
            f"{t.name}\ttier={t.tier.value}\tterminal={t.is_terminal}\t"
            f"executor_ref={t.executor_ref}\tagents={[a.value for a in t.allowed_agents]}"
        )
    print(f"total: {len(tools)}")
    return 0


def cmd_stats() -> int:
    registry = load_registry()
    tiers = Counter(t.tier.value for t in registry.values())
    agents: Counter[str] = Counter()
    for t in registry.values():
        for a in t.allowed_agents:
            agents[a.value] += 1
    report: dict[str, Any] = {
        "total": len(registry),
        "by_tier": dict(tiers),
        "by_agent_allow": dict(agents),
        "production": [n for n, t in registry.items() if t.tier.value == "production"],
        "draft": [n for n, t in registry.items() if t.tier.value == "draft"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_promote(tool_name: str, executor_ref: str) -> int:
    try:
        report = promote_tool(tool_name, executor_ref)
    except Exception as exc:  # noqa: BLE001
        print(f"promote failed / rolled back: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage tool_registry.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出 tools")
    p_list.add_argument("--tier", choices=["draft", "production"], default=None)

    sub.add_parser("stats", help="统计 draft/production")

    p_promote = sub.add_parser("promote", help="显式升档 tool")
    p_promote.add_argument("tool_name")
    p_promote.add_argument(
        "--executor-ref",
        required=True,
        help="可导入路径，如 pipeline.tools.sun_position.execute",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return cmd_list(args.tier)
    if args.command == "stats":
        return cmd_stats()
    if args.command == "promote":
        return cmd_promote(args.tool_name, args.executor_ref)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
