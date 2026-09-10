"""持久化 ResultStore 与 ImageStore 的恢复测试。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from tool.runtime import FilesystemImageStore, FilesystemResultStore


def test_filesystem_result_store_persists_and_reopens(tmp_path: Path) -> None:
    root = tmp_path / "results"
    first = FilesystemResultStore(root)
    result_id = first.put("osm", {"elements": [{"id": 1}]})
    assert result_id.startswith("osm_")
    assert first.get(result_id) == {"elements": [{"id": 1}]}

    reopened = FilesystemResultStore(root)
    assert reopened.get(result_id) == {"elements": [{"id": 1}]}
    assert reopened.list_ids() == [result_id]
    record = reopened.get_record(result_id)
    assert record["schema_version"] == "tool_result_v1"
    assert record["namespace"] == "osm"
    assert len(record["payload_sha256"]) == 64

    fallback_id = reopened.put("中文 namespace", {"ok": True})
    assert fallback_id.startswith("namespace_")
    assert reopened.get(fallback_id) == {"ok": True}


def test_filesystem_result_store_rejects_traversal_and_corruption(tmp_path: Path) -> None:
    store = FilesystemResultStore(tmp_path / "results")
    with pytest.raises(KeyError):
        store.get("../outside")

    corrupt_id = "osm_0123456789abcdef"
    (store.root / f"{corrupt_id}.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="损坏"):
        store.get(corrupt_id)

    with pytest.raises(ValueError, match="凭证"):
        store.put("unsafe", {"api_key": "secret-value"})
    assert not any(path.name.startswith("unsafe_") for path in store.root.glob("*.json"))


def test_filesystem_result_store_detects_payload_tampering(tmp_path: Path) -> None:
    store = FilesystemResultStore(tmp_path / "results")
    result_id = store.put("osm", {"count": 1})
    path = store.root / f"{result_id}.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["count"] = 999
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="哈希不匹配"):
        store.get(result_id)


def test_filesystem_result_store_concurrent_ids_are_unique(tmp_path: Path) -> None:
    store = FilesystemResultStore(tmp_path / "results")
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda index: store.put("geo", {"index": index}), range(32)))
    assert len(ids) == len(set(ids)) == 32
    assert len(store.list_ids()) == 32


def test_filesystem_image_store_restores_derived_ids(tmp_path: Path) -> None:
    root = tmp_path / "images"
    image = Image.new("RGB", (4, 4), (1, 2, 3))
    first = FilesystemImageStore(root)
    first_id = first.put(image, source_id="source", suffix="png")
    assert first_id == "img_0001"

    reopened = FilesystemImageStore(root)
    assert reopened.resolve(first_id).is_file()
    second_id = reopened.put(image, source_id=first_id, suffix="png")
    assert second_id == "img_0002"
