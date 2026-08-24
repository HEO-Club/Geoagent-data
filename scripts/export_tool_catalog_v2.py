"""导出更细 Canonical Tool v2 提案，不改生产配置。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.tool_catalog_v2_proposal import build_tool_catalog_v2


def main() -> None:
    proposal = build_tool_catalog_v2()
    path = Path("docs/tool_catalog_v2_proposed.json")
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(proposal["stats"], ensure_ascii=False, indent=2))
    print(f"report={path}")


if __name__ == "__main__":
    main()
