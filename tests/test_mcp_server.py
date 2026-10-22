"""本机 stdio MCP：目录工具表、会话调用与拒绝路径。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mcp.types import Tool
from PIL import Image

from tool.contract import Observation, RuntimeContext
from tool.mcp.catalog import DEFAULT_CATALOG_PATH, load_mcp_tool_specs
from tool.mcp.session import McpSession, McpSessionError
from tool.mcp.server import SERVER_NAME, create_server

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "canonical_tool_catalog_v2.json"


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(path)
    return path


def _session(
    tmp_path: Path,
    *,
    dispatcher: Any,
    allowed_file_roots: list[Path] | None = None,
    initial_images: list[Path] | None = None,
) -> McpSession:
    return McpSession(
        workspace_dir=tmp_path / "session",
        catalog_path=CATALOG,
        dispatcher=dispatcher,
        allowed_file_roots=list(allowed_file_roots or []),
        initial_images=list(initial_images or []),
    )


def test_mcp_tool_table_matches_non_terminal_catalog() -> None:
    specs = load_mcp_tool_specs(CATALOG)
    names = [spec.name for spec in specs]
    assert CATALOG == DEFAULT_CATALOG_PATH
    assert len(specs) == 53
    assert "final_answer__submit" not in names
    assert "satellite_imagery_query__oblique_view" not in names
    assert "route_query__route" not in names
    assert "infrastructure_registry__construction" not in names
    assert "infrastructure_registry__permit" not in names
    assert "administrative_registry__administrative" not in names
    assert "administrative_registry__directory" not in names
    assert "web_search__keyword_search" in names
    assert "osm_query__near" in names
    keyword = next(spec for spec in specs if spec.name == "web_search__keyword_search")
    properties = keyword.input_schema["properties"]
    assert "purpose" in properties
    assert "query" in properties
    assert properties["purpose"]["type"] == "string"
    assert "type" not in properties["query"]
    assert keyword.input_schema["required"][:2] == ["purpose", "query"]
    for spec in specs:
        Tool(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
        )


def test_call_forwards_purpose_and_inputs_to_dispatcher(tmp_path: Path) -> None:
    calls: list[tuple[str, str, str, dict[str, Any]]] = []

    def fake_execute(
        tool: str,
        operation: str,
        *,
        purpose: str,
        inputs: dict[str, Any],
        ctx: RuntimeContext | None = None,
    ) -> Observation:
        del ctx
        calls.append((tool, operation, purpose, dict(inputs)))
        return Observation(ok=True, result={"hits": 2, "query": inputs.get("query")})

    session = _session(tmp_path, dispatcher=fake_execute)
    payload = session.call(
        "web_search__keyword_search",
        {
            "purpose": "检索候选地名",
            "query": "郑州 铁路桥",
            "top_k": 5,
        },
    )
    assert payload["ok"] is True
    assert payload["result"] == {"hits": 2, "query": "郑州 铁路桥"}
    assert payload["error"] is None
    assert calls == [
        ("web_search", "keyword_search", "检索候选地名", {"query": "郑州 铁路桥", "top_k": 5})
    ]


def test_previous_tool_result_and_current_image_survive_across_calls(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, Any]] = []
    allowed = tmp_path / "allowed"
    source = _png(allowed / "source.png")
    derived = _png(allowed / "derived.png")

    def fake_execute(
        tool: str,
        operation: str,
        *,
        purpose: str,
        inputs: dict[str, Any],
        ctx: RuntimeContext | None = None,
    ) -> Observation:
        del tool, operation, purpose
        assert ctx is not None
        seen.append(
            {
                "inputs": dict(inputs),
                "current_image": ctx.current_image,
            }
        )
        if len(seen) == 1:
            image_id = ctx.image_store.register(derived, image_id="img_derived")
            return Observation(ok=True, result={"image_id": image_id, "label": "crop"})
        return Observation(
            ok=True,
            result={
                "image": inputs.get("image"),
                "source": inputs.get("source"),
            },
        )

    session = _session(
        tmp_path,
        dispatcher=fake_execute,
        allowed_file_roots=[allowed],
        initial_images=[source],
    )
    first = session.call(
        "image_edit__crop",
        {"purpose": "裁出桥塔", "image": "$current_image"},
    )
    assert first["ok"] is True
    assert first["result"]["image_id"] == "img_derived"
    assert session.ctx.current_image == "img_derived"

    second = session.call(
        "ocr_read__recognize",
        {
            "purpose": "读裁剪图文字",
            "image": "$current_image",
            "source": "$previous_tool_result",
        },
    )
    assert second["ok"] is True
    assert second["result"]["image"] == "img_derived"
    assert second["result"]["source"]["image_id"] == "img_derived"
    assert seen[0]["current_image"] == "input_0001"
    assert seen[1]["current_image"] == "img_derived"


def test_credentials_and_paths_outside_roots_are_rejected(tmp_path: Path) -> None:
    def boom(*args: Any, **kwargs: Any) -> Observation:
        raise AssertionError((args, kwargs))

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = _png(tmp_path / "secret" / "photo.png")
    session = _session(
        tmp_path,
        dispatcher=boom,
        allowed_file_roots=[allowed],
    )

    cred = session.call(
        "web_search__keyword_search",
        {
            "purpose": "检索",
            "query": "郑州",
            "api_key": "sk-not-a-real-key-value",
        },
    )
    assert cred["ok"] is False
    assert cred["error_code"] == "credential_in_inputs"

    blocked = session.call(
        "image_edit__crop",
        {"purpose": "打开外部文件", "image": str(secret)},
    )
    assert blocked["ok"] is False
    assert blocked["error_code"] == "path_not_allowed"


def test_initial_image_outside_roots_raises(tmp_path: Path) -> None:
    outside = _png(tmp_path / "outside" / "a.png")
    with pytest.raises(McpSessionError) as caught:
        _session(
            tmp_path,
            dispatcher=lambda *a, **k: Observation(ok=True, result={}),
            allowed_file_roots=[tmp_path / "allowed"],
            initial_images=[outside],
        )
    assert caught.value.error_code == "path_not_allowed"


def test_create_server_uses_session_tool_table(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        dispatcher=lambda *a, **k: Observation(ok=True, result={}),
    )
    server = create_server(session)
    assert server.name == SERVER_NAME


def test_artifacts_drop_local_paths(tmp_path: Path) -> None:
    def fake_execute(
        tool: str,
        operation: str,
        *,
        purpose: str,
        inputs: dict[str, Any],
        ctx: RuntimeContext | None = None,
    ) -> Observation:
        del tool, operation, purpose, inputs, ctx
        return Observation(
            ok=True,
            result={"image_id": "img_keep"},
            artifacts={
                "image_id": "img_keep",
                "image_path": str(tmp_path / "session" / "images" / "img_keep.png"),
            },
        )

    session = _session(tmp_path, dispatcher=fake_execute)
    payload = session.call(
        "image_edit__enhance",
        {"purpose": "增强", "image": "img_keep"},
    )
    assert payload["ok"] is True
    assert payload["artifacts"] == {"image_id": "img_keep"}
    assert "image_path" not in payload["artifacts"]
