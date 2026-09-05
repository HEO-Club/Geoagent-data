"""真实 Tool 包目录必须与 canonical_tool_catalog_v2.json 对齐。"""

from __future__ import annotations

import json
from pathlib import Path

from tool import TOOLS, execute
from tool.contract import Observation

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "canonical_tool_catalog_v2.json"
TOOL_ROOT = REPO_ROOT / "tool"


def _catalog_tools() -> list[tuple[str, list[str]]]:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    items: list[tuple[str, list[str]]] = []
    for tree in payload["trees"]:
        canonical = tree["canonical"]
        name = str(canonical["name"])
        operations = [str(op["name"]) for op in canonical.get("operations") or []]
        items.append((name, operations))
    return items


def test_tool_package_matches_catalog_layout() -> None:
    catalog = _catalog_tools()
    assert len(catalog) == 31
    assert sum(len(operations) for _, operations in catalog) == 57
    assert set(TOOLS) == {name for name, _ in catalog}
    for tool_name, operations in catalog:
        package_dir = TOOL_ROOT / tool_name
        assert package_dir.is_dir(), tool_name
        assert (package_dir / "__init__.py").is_file(), tool_name
        assert set(TOOLS[tool_name].OPERATIONS) == set(operations)
        for operation in operations:
            assert (package_dir / f"{operation}.py").is_file(), (tool_name, operation)


IMPLEMENTED_OPERATIONS = {
    ("image_edit", "crop"),
    ("image_edit", "zoom"),
    ("image_edit", "enhance"),
    ("image_measure", "measure"),
    ("image_compare", "compare"),
    ("ocr_read", "recognize"),
    ("ocr_read", "decode"),
    ("reverse_image_search", "search"),
    ("reverse_image_search", "search_crop"),
    ("media_metadata_read", "exif"),
    ("media_metadata_read", "file"),
}


def test_unimplemented_catalog_operations_return_placeholder() -> None:
    for tool_name, operations in _catalog_tools():
        for operation in operations:
            observation = execute(
                tool_name,
                operation,
                purpose="scaffold",
                inputs={},
            )
            assert isinstance(observation, Observation)
            assert observation.ok is False
            if (tool_name, operation) in IMPLEMENTED_OPERATIONS:
                assert observation.error_code != "not_implemented"
                assert observation.error_code == "missing_input"
            else:
                assert observation.error_code == "not_implemented"


def test_unknown_tool_and_operation_are_structured_errors() -> None:
    unknown_tool = execute("not_a_tool", "query", purpose="x", inputs={})
    assert unknown_tool.error_code == "unknown_tool"
    unknown_op = execute("osm_query", "not_an_op", purpose="x", inputs={})
    assert unknown_op.error_code == "unknown_operation"
