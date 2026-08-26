"""导出 Canonical Tool v2 说明文件与生产 ToolForest。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.tool_catalog_v2 import build_tool_forest_v2
from pipeline.tool_catalog_v2_proposal import build_tool_catalog_v2


def main() -> None:
    proposal = build_tool_catalog_v2()
    path = Path("docs/tool_catalog_v2_proposed.json")
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime_path = Path("canonical_tool_catalog_v2.json")
    runtime_path.write_text(
        build_tool_forest_v2().model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(proposal["stats"], ensure_ascii=False, indent=2))
    print(f"report={path}")
    print(f"runtime={runtime_path}")


if __name__ == "__main__":
    main()
