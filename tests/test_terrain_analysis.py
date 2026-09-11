"""terrain_analysis.terrain 本地 Horn DEM 测试；禁止真实付费 API 与网络。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.terrain_analysis._terrain import BBox, DemGrid

_WEST = 0.0
_SOUTH = 0.0
_EAST = 0.001
_NORTH = 0.0005
_ROWS = 11
_COLS = 21
_FLAT_M = 100.0
_METERS_PER_DEG = 111_320.0
_WIDTH_M = (_EAST - _WEST) * _METERS_PER_DEG


def _nested_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            found.update(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_nested_keys(item))
    return found


def _area() -> dict[str, float]:
    return {"west": _WEST, "south": _SOUTH, "east": _EAST, "north": _NORTH}


def _grid(
    values: list[list[float]] | np.ndarray,
    *,
    kind: str = "dem",
    west: float = _WEST,
    south: float = _SOUTH,
    east: float = _EAST,
    north: float = _NORTH,
    dataset: str = "local_grid",
) -> dict[str, Any]:
    return {
        "values": values,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "kind": kind,
        "crs": "wgs84",
        "dataset": dataset,
        "confirmed_location": "MUST NOT LEAK",
    }


def _flat_values(*, elev: float = _FLAT_M, rows: int = _ROWS, cols: int = _COLS) -> list[list[float]]:
    return [[elev] * cols for _ in range(rows)]


def _east_ramp_values() -> list[list[float]]:
    dx = (_EAST - _WEST) / _COLS
    values: list[list[float]] = []
    for _row in range(_ROWS):
        row: list[float] = []
        for col in range(_COLS):
            lon = _WEST + (col + 0.5) * dx
            row.append((lon - _WEST) * _METERS_PER_DEG)
        values.append(row)
    return values


def _terrain(
    *,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    return execute(
        "terrain_analysis",
        "terrain",
        purpose="分析地形",
        inputs=inputs,
        ctx=ctx,
    )


def test_empty_inputs_are_missing_input() -> None:
    observation = _terrain(inputs={})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_missing_metrics_is_missing_input() -> None:
    observation = _terrain(inputs={"area": _area()})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "metrics" in observation.error


def test_missing_area_and_path_is_missing_input() -> None:
    observation = _terrain(inputs={"metrics": ["elevation"]})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_place_name_area_is_missing_input() -> None:
    observation = _terrain(inputs={"area": "郑州附近黄河沿线", "metrics": ["elevation"]})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "geocode" in observation.error


def test_unknown_metric_is_invalid_input() -> None:
    observation = _terrain(
        inputs={
            "area": _area(),
            "metrics": ["viewshed"],
            "terrain": _grid(_flat_values()),
        }
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"
    assert observation.error is not None
    assert "metrics" in observation.error


def test_profile_without_path_is_missing_input() -> None:
    observation = _terrain(
        inputs={
            "area": _area(),
            "metrics": ["profile"],
            "terrain": _grid(_flat_values()),
        }
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "path" in observation.error


def test_flat_dem_elevation_statistics() -> None:
    observation = _terrain(
        inputs={
            "area": _area(),
            "metrics": ["elevation", "relief"],
            "terrain": _grid(_flat_values()),
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["operation"] == "terrain"
    assert observation.result["method"] == "horn_dem"
    stats = observation.result["statistics"]["elevation"]
    assert stats["min_m"] == pytest.approx(_FLAT_M)
    assert stats["max_m"] == pytest.approx(_FLAT_M)
    assert stats["mean_m"] == pytest.approx(_FLAT_M)
    assert observation.result["statistics"]["relief"]["relief_m"] == pytest.approx(0.0)
    assert observation.result["statistics"]["relief"]["relief_class"] == "plain"
    assert observation.result["applied"]["dem_source"] == "grid"
    assert "confirmed_location" not in _nested_keys(observation.result)
    joined = "".join(observation.result["assumptions"])
    assert "Horn" in joined
    assert "栅格" in joined


def test_east_ramp_slope_and_west_facing_aspect() -> None:
    observation = _terrain(
        inputs={
            "area": _area(),
            "metrics": ["slope", "aspect"],
            "terrain": _grid(_east_ramp_values()),
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    slope = observation.result["statistics"]["slope"]
    assert slope["mean_deg"] == pytest.approx(45.0, abs=2.0)
    aspect_mean = observation.result["statistics"]["aspect"]["mean_deg"]
    assert aspect_mean is not None
    assert aspect_mean == pytest.approx(270.0, abs=15.0)


def test_path_profile_monotonic_distance_and_end_elevation() -> None:
    observation = _terrain(
        inputs={
            "path": [[_WEST + 0.0001, 0.00025], [_EAST - 0.0001, 0.00025]],
            "metrics": ["profile", "elevation"],
            "terrain": _grid(_east_ramp_values()),
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    profile = observation.result["profile"]
    assert len(profile) >= 2
    distances = [item["distance_m"] for item in profile]
    assert distances == sorted(distances)
    assert distances[0] == pytest.approx(0.0, abs=0.1)
    assert profile[0]["elevation_m"] < profile[-1]["elevation_m"]
    start_expected = 0.0001 * _METERS_PER_DEG
    end_expected = (_EAST - _WEST - 0.0001) * _METERS_PER_DEG
    assert profile[0]["elevation_m"] == pytest.approx(start_expected, rel=0.15)
    assert profile[-1]["elevation_m"] == pytest.approx(end_expected, rel=0.15)


def test_previous_tool_result_terrain_grid() -> None:
    terrain = _grid(_flat_values())
    ctx = RuntimeContext(previous_tool_result={"ok": True, "result": terrain, "error_code": None})
    observation = _terrain(
        inputs={
            "area": _area(),
            "metrics": ["elevation"],
            "terrain": "$previous_tool_result",
        },
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["dem_source"] == "grid"
    assert observation.result["statistics"]["elevation"]["mean_m"] == pytest.approx(_FLAT_M)


def test_injected_grid_without_terrain_field() -> None:
    ctx = RuntimeContext(extras={"terrain_analysis_dem": _grid(_flat_values())})
    observation = _terrain(inputs={"area": _area(), "metrics": ["elevation"]}, ctx=ctx)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["dem_source"] == "injected"


def test_geotiff_path_roundtrip(tmp_path: Any) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    path = tmp_path / "flat.tif"
    transform = from_bounds(_WEST, _SOUTH, _EAST, _NORTH, _COLS, _ROWS)
    array = np.full((_ROWS, _COLS), _FLAT_M, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_ROWS,
        width=_COLS,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(array, 1)
    observation = _terrain(
        inputs={
            "area": _area(),
            "metrics": ["elevation"],
            "terrain": str(path),
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["dem_source"] == "geotiff"
    assert observation.result["statistics"]["elevation"]["mean_m"] == pytest.approx(_FLAT_M)


def test_fake_opentopography_provider_receives_bbox() -> None:
    class FakeDemProvider:
        name = "fake_ot"

        def __init__(self) -> None:
            self.requests: list[BBox] = []

        def fetch(self, bbox: BBox, *, demtype: str) -> DemGrid:
            self.requests.append(bbox)
            assert demtype
            values = np.asarray(_flat_values(), dtype=float)
            dx = (_EAST - _WEST) / _COLS
            dy = (_SOUTH - _NORTH) / _ROWS
            return DemGrid(
                values=values,
                transform=(dx, 0.0, _WEST, 0.0, dy, _NORTH),
                kind="dem",
                crs="wgs84",
                dataset=demtype,
            )

    fake = FakeDemProvider()
    ctx = RuntimeContext(extras={"terrain_analysis_provider": fake})
    observation = _terrain(inputs={"area": _area(), "metrics": ["elevation"]}, ctx=ctx)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["dem_source"] == "fake_ot"
    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request.west == pytest.approx(_WEST)
    assert request.south == pytest.approx(_SOUTH)
    assert request.east == pytest.approx(_EAST)
    assert request.north == pytest.approx(_NORTH)


def test_allow_real_api_false_without_local_dem_is_engine_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    observation = _terrain(inputs={"area": _area(), "metrics": ["elevation"]})
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_remote_dem_url_is_rejected() -> None:
    observation = _terrain(
        inputs={
            "area": _area(),
            "metrics": ["elevation"],
            "terrain": "https://example.com/dem.tif",
        }
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"
    assert observation.error is not None
    assert "远程" in observation.error


def test_coarse_resolution_assumption_and_integer_relief() -> None:
    west, south, east, north = 0.0, 0.0, 0.03, 0.03
    rows, cols = 5, 5
    values = [[10.4 + col * 12.6 for col in range(cols)] for _ in range(rows)]
    observation = _terrain(
        inputs={
            "area": {"west": west, "south": south, "east": east, "north": north},
            "metrics": ["elevation", "relief"],
            "terrain": _grid(
                values,
                west=west,
                south=south,
                east=east,
                north=north,
                dataset="COP30",
            ),
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["resolution_m"] >= 30
    relief = observation.result["statistics"]["relief"]["relief_m"]
    assert relief == float(round(relief))
    mean_m = observation.result["statistics"]["elevation"]["mean_m"]
    assert mean_m == float(round(mean_m))
    joined = "".join(observation.result["assumptions"])
    assert "局部高差" in joined
    assert observation.result["statistics"]["relief"]["relief_class"] in {"plain", "hill", "mountain"}
