"""执行 Stage 3 trajectory 中的真实 Tool 调用，输出独立可恢复报告。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pipeline.schemas.trajectory import Trajectory
from tool.runtime import HarnessError, ToolHarness, ToolHarnessConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay canonical Tool calls without overwriting the source trajectory.",
    )
    parser.add_argument("--trajectory", required=True, help="stage3_trajectory.json 路径")
    parser.add_argument(
        "--workspace",
        default="data/tool_runs",
        help="harness 检查点、结果和事件日志目录",
    )
    parser.add_argument(
        "--catalog",
        default="canonical_tool_catalog_v2.json",
        help="Canonical Tool 目录路径",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不复用同 ID 检查点，自动创建新的 run 目录",
    )
    parser.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="恢复时保留旧失败记录，不重新执行失败步骤",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="某次 Tool 失败后阻止后续 Tool；默认继续独立步骤并记录错误",
    )
    parser.add_argument(
        "--allow-root",
        action="append",
        default=[],
        help="允许 Tool 读取的额外本地目录；可重复指定",
    )
    parser.add_argument(
        "--provider-revision",
        default="",
        help="非密钥 Provider/数据配置版本；变化时使旧检查点失效",
    )
    return parser.parse_args()


def main() -> int:
    _configure_windows_stdout()
    args = _parse_args()
    try:
        trajectory_path = Path(args.trajectory).resolve()
        trajectory = Trajectory.model_validate_json(
            trajectory_path.read_text(encoding="utf-8")
        )
        harness = ToolHarness(
            ToolHarnessConfig(
                workspace_dir=Path(args.workspace),
                catalog_path=Path(args.catalog),
                resume=not args.no_resume,
                retry_failed=not args.no_retry_failed,
                stop_on_error=args.stop_on_error,
                allowed_file_roots=[Path(item) for item in args.allow_root],
                provider_revision=args.provider_revision,
            )
        )
        report = harness.run(trajectory)
    except HarnessError as exc:
        print(
            json.dumps(
                {"ok": False, "error_code": exc.error_code, "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_code": "invalid_input_or_io",
                    "error": str(exc)[:1000],
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "ok": report.status == "completed",
                "status": report.status,
                "trajectory_id": report.trajectory_id,
                "counts": report.counts,
                "final_location": report.final_location,
                "report": str(harness.last_report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.status == "completed" else 1


def _configure_windows_stdout() -> None:
    """确保 PowerShell 中中文 JSON 摘要不因系统代码页变成乱码。"""

    if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
