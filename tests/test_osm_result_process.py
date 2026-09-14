"""OSM 本地结果筛选与导出测试；禁止真实网络。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from tool import execute
from tool.contract import RuntimeContext
from tool.runtime import InMemoryResultStore


def _source() -> dict:
    return {
        "crs": "EPSG:4326",
        "attribution": "© OpenStreetMap contributors",
        "elements": [
            {
                "osm_type": "node",
                "osm_id": 1,
                "name": "黄河桥 A",
                "latitude": 34.9500,
                "longitude": 113.5000,
                "tags": {"bridge": "yes", "material": "stone"},
            },
            {
                "osm_type": "node",
                "osm_id": 2,
                "name": "黄河桥 B",
                "latitude": 34.9600,
                "longitude": 113.5100,
                "tags": {"bridge": "yes", "material": "steel"},
            },
            {
                "osm_type": "way",
                "osm_id": 3,
                "name": "普通道路",
                "geometry": [[113.49, 34.94], [113.53, 34.98]],
                "tags": {"highway": "primary"},
            },
            {
                "osm_type": "relation",
                "osm_id": 4,
                "name": "无几何对象",
                "tags": {"type": "site"},
            },
        ],
    }


def _ctx(tmp_path: Path) -> tuple[RuntimeContext, str]:
    store = InMemoryResultStore()
    result_id = store.put("osm", _source())
    return (
        RuntimeContext(
            result_store=store,
            extras={"artifact_dir": str(tmp_path / "artifacts")},
        ),
        result_id,
    )


def test_filter_by_tags_name_bbox_and_persists_result(tmp_path: Path) -> None:
    ctx, result_id = _ctx(tmp_path)
    observation = execute(
        "osm_result_process",
        "filter",
        purpose="筛选石桥",
        inputs={
            "source_result": result_id,
            "filters": {
                "tags": {"bridge": "*", "material": "stone"},
                "name_contains": "黄河",
                "bbox": [113.49, 34.94, 113.505, 34.955],
            },
        },
        ctx=ctx,
    )
    assert observation.ok is True
    result = observation.result
    assert result["input_count"] == 4
    assert result["output_count"] == 1
    assert result["elements"][0]["osm_id"] == 1
    assert result["result_id"].startswith("osm_filtered_")
    stored = ctx.result_store.get(result["result_id"])
    assert stored["elements"][0]["name"] == "黄河桥 A"


def test_filter_spatial_and_missing_geometry_accounting(tmp_path: Path) -> None:
    ctx, result_id = _ctx(tmp_path)
    observation = execute(
        "osm_result_process",
        "filter",
        purpose="筛选范围内对象",
        inputs={
            "source_result": result_id,
            "filters": {},
            "spatial_filter": {
                "relation": "within",
                "geometry": {"bbox": [113.49, 34.94, 113.52, 34.97]},
            },
        },
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result["output_count"] == 2
    assert observation.result["spatial_filter"]["missing_geometry_count"] == 1
    assert {item["osm_id"] for item in observation.result["elements"]} == {1, 2}


def test_limit_is_applied_after_spatial_filter(tmp_path: Path) -> None:
    ctx, result_id = _ctx(tmp_path)
    observation = execute(
        "osm_result_process",
        "filter",
        purpose="先空间筛选再限量",
        inputs={
            "source_result": result_id,
            "filters": {"limit": 1},
            "spatial_filter": {
                "relation": "within",
                "geometry": {"bbox": [113.505, 34.955, 113.515, 34.965]},
            },
        },
        ctx=ctx,
    )
    assert observation.ok is True
    assert [item["osm_id"] for item in observation.result["elements"]] == [2]


@pytest.mark.parametrize("export_format", ["json", "geojson", "csv", "kml"])
def test_export_supported_formats(tmp_path: Path, export_format: str) -> None:
    ctx, result_id = _ctx(tmp_path)
    observation = execute(
        "osm_result_process",
        "export",
        purpose="导出已有结果",
        inputs={"source_result": result_id, "format": export_format},
        ctx=ctx,
    )
    assert observation.ok is True
    path = Path(observation.artifacts["export_path"])
    assert path.is_file()
    assert observation.result["exported_count"] == 4
    assert observation.result["sha256"]
    if export_format == "json":
        assert len(json.loads(path.read_text(encoding="utf-8"))["elements"]) == 4
    elif export_format == "geojson":
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) == 4
    elif export_format == "csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 4
    else:
        root = ElementTree.parse(path).getroot()
        assert root.tag.endswith("kml")


def test_export_without_geometry_and_refuses_lossy_osm_xml(tmp_path: Path) -> None:
    ctx, result_id = _ctx(tmp_path)
    plain = execute(
        "osm_result_process",
        "export",
        purpose="仅导出属性",
        inputs={"source_result": result_id, "format": "geojson", "include_geometry": False},
        ctx=ctx,
    )
    payload = json.loads(Path(plain.artifacts["export_path"]).read_text(encoding="utf-8"))
    assert all(feature["geometry"] is None for feature in payload["features"])

    lossy = execute(
        "osm_result_process",
        "export",
        purpose="拒绝伪造 OSM XML",
        inputs={"source_result": result_id, "format": "osm_xml"},
        ctx=ctx,
    )
    assert lossy.ok is False
    assert lossy.error_code == "lossy_format_unsupported"


def test_filter_rejects_unknown_condition_and_missing_source(tmp_path: Path) -> None:
    ctx, result_id = _ctx(tmp_path)
    unknown = execute(
        "osm_result_process",
        "filter",
        purpose="未知条件",
        inputs={"source_result": result_id, "filters": {"guess": "yes"}},
        ctx=ctx,
    )
    assert unknown.error_code == "unsupported_filter"

    missing = execute(
        "osm_result_process",
        "filter",
        purpose="缺结果",
        inputs={"source_result": "osm_missing", "filters": {}},
        ctx=ctx,
    )
    assert missing.error_code == "needs_acquisition"

    mercator_id = ctx.result_store.put(
        "mercator",
        {"crs": "EPSG:3857", "elements": [{"latitude": 1, "longitude": 1}]},
    )
    wrong_crs = execute(
        "osm_result_process",
        "filter",
        purpose="拒绝静默坐标系误读",
        inputs={"source_result": mercator_id, "filters": {}},
        ctx=ctx,
    )
    assert wrong_crs.error_code == "unsupported_crs"

    unsafe_regex = execute(
        "osm_result_process",
        "filter",
        purpose="拒绝高风险正则",
        inputs={"source_result": result_id, "filters": {"name_regex": "(a+)+$"}},
        ctx=ctx,
    )
    assert unsafe_regex.error_code == "unsafe_regex"


def test_csv_escapes_formula_like_names(tmp_path: Path) -> None:
    store = InMemoryResultStore()
    result_id = store.put(
        "osm",
        {
            "elements": [
                {
                    "osm_type": "node",
                    "osm_id": 1,
                    "name": "=HYPERLINK(\"https://example.invalid\")",
                    "latitude": 0,
                    "longitude": 0,
                }
            ]
        },
    )
    ctx = RuntimeContext(
        result_store=store,
        extras={"artifact_dir": str(tmp_path / "artifacts")},
    )
    observation = execute(
        "osm_result_process",
        "export",
        purpose="安全 CSV",
        inputs={"source_result": result_id, "format": "csv"},
        ctx=ctx,
    )
    with Path(observation.artifacts["export_path"]).open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["name"].startswith("'=")
