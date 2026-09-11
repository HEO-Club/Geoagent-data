"""visibility_analysis.sightline 点到点视线测试；禁止真实付费 API 与网络。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tool import execute
from tool.contract import Observation, RuntimeContext

_WEST = 0.0
_SOUTH = 0.0
_EAST = 0.02
_NORTH = 0.01
_ROWS = 21
_COLS = 41
_FLAT_M = 100.0
_RIDGE_M = 500.0
_OBS = {"lon": 0.002, "lat": 0.005}
_TGT = {"lon": 0.018, "lat": 0.005}
_RIDGE_LON = 0.01


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


def _grid(
    values: list[list[float]] | np.ndarray,
    *,
    kind: str = "dem",
) -> dict[str, Any]:
    return {
        "values": values,
        "west": _WEST,
        "south": _SOUTH,
        "east": _EAST,
        "north": _NORTH,
        "kind": kind,
        "crs": "wgs84",
        "confirmed_location": "MUST NOT LEAK",
    }


def _flat_values(*, elev: float = _FLAT_M) -> list[list[float]]:
    return [[elev] * _COLS for _ in range(_ROWS)]


def _ridge_values() -> list[list[float]]:
    values = _flat_values()
    dx = (_EAST - _WEST) / _COLS
    dy = (_SOUTH - _NORTH) / _ROWS
    ridge_col = int((_RIDGE_LON - _WEST) / dx)
    lat_center = 0.005
    for row in range(_ROWS):
        lat = _NORTH + dy * (row + 0.5)
        if abs(lat - lat_center) > 0.0018:
            continue
        for col in range(ridge_col - 1, ridge_col + 2):
            if 0 <= col < _COLS:
                values[row][col] = _RIDGE_M
    return values


def _sightline(*, inputs: dict[str, Any], ctx: RuntimeContext | None = None) -> Observation:
    return execute(
        "visibility_analysis",
        "sightline",
        purpose="核验视线",
        inputs=inputs,
        ctx=ctx,
    )


@pytest.fixture(autouse=True)
def _disable_curvature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISIBILITY_CURVATURE", "false")


def test_empty_inputs_are_missing_input() -> None:
    observation = _sightline(inputs={})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_missing_terrain_is_missing_input() -> None:
    observation = _sightline(inputs={"observer": _OBS, "target": _TGT})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "terrain" in observation.error


def test_place_name_is_rejected() -> None:
    observation = _sightline(
        inputs={
            "observer": "郑州黄河铁路桥",
            "target": _TGT,
            "terrain": _grid(_flat_values()),
        }
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"
    assert observation.error is not None
    assert "地名" in observation.error


def test_terrain_place_name_is_rejected() -> None:
    observation = _sightline(
        inputs={"observer": _OBS, "target": _TGT, "terrain": "嵩山DEM"}
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"


def test_remote_dem_url_is_rejected() -> None:
    observation = _sightline(
        inputs={
            "observer": _OBS,
            "target": _TGT,
            "terrain": "https://example.com/dem.tif",
        }
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"
    assert observation.error is not None
    assert "下载" in observation.error


def test_flat_dem_is_visible() -> None:
    observation = _sightline(
        inputs={"observer": _OBS, "target": _TGT, "terrain": _grid(_flat_values())}
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["visible"] is True
    assert observation.result["method"] == "geodesic_dem_sample"
    assert observation.result["first_obstruction"] is None
    assert observation.result["terrain_kind"] == "dem"
    assert observation.result["distance_m"] > 1000.0
    assert observation.result["profile"]
    assert "applied" in observation.result
    assert observation.result["coverage"]["accounts_for_buildings"] is False
    assert observation.result["coverage"]["accounts_for_trees"] is False
    joined = "".join(observation.result["assumptions"])
    assert "裸地 DEM" in joined
    assert "viewshed" in joined
    assert "confirmed_location" not in _nested_keys(observation.result)


def test_ridge_blocks_sightline() -> None:
    observation = _sightline(
        inputs={"observer": _OBS, "target": _TGT, "terrain": _grid(_ridge_values())}
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["visible"] is False
    obstruction = observation.result["first_obstruction"]
    assert obstruction is not None
    assert obstruction["lon"] == pytest.approx(_RIDGE_LON, abs=0.002)
    assert obstruction["terrain_m"] > obstruction["los_m"]
    assert obstruction["terrain_m"] > _FLAT_M


def test_raised_observer_clears_ridge() -> None:
    observer = {**_OBS, "height_m": 1000.0}
    observation = _sightline(
        inputs={"observer": observer, "target": _TGT, "terrain": _grid(_ridge_values())}
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["visible"] is True
    assert observation.result["first_obstruction"] is None
    assert observation.result["applied"]["observer"]["height_m"] == 1000.0


def test_through_points_detour_is_visible() -> None:
    blocked = _sightline(
        inputs={"observer": _OBS, "target": _TGT, "terrain": _grid(_ridge_values())}
    )
    detour = _sightline(
        inputs={
            "observer": _OBS,
            "target": _TGT,
            "through_points": [{"lon": 0.01, "lat": 0.0085}],
            "terrain": _grid(_ridge_values()),
        }
    )
    assert blocked.ok is True and blocked.result is not None
    assert blocked.result["visible"] is False
    assert detour.ok is True and detour.result is not None
    assert detour.result["visible"] is True
    assert len(detour.result["applied"]["through_points"]) == 1


def test_polygon_target_is_unsupported_viewshed() -> None:
    observation = _sightline(
        inputs={
            "observer": _OBS,
            "target": {
                "type": "Polygon",
                "coordinates": [[
                    [0.0, 0.0],
                    [0.02, 0.0],
                    [0.02, 0.01],
                    [0.0, 0.01],
                    [0.0, 0.0],
                ]],
            },
            "terrain": _grid(_flat_values()),
        }
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_mode"
    assert observation.error is not None
    assert "可视域" in observation.error


def test_bbox_target_is_unsupported_viewshed() -> None:
    observation = _sightline(
        inputs={
            "observer": _OBS,
            "target": [0.0, 0.0, 0.02, 0.01],
            "terrain": _grid(_flat_values()),
        }
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_mode"


def test_mode_viewshed_is_ignored_extra_field() -> None:
    observation = _sightline(
        inputs={
            "observer": _OBS,
            "target": _TGT,
            "terrain": _grid(_flat_values()),
            "mode": "viewshed",
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["visible"] is True
    assert observation.extensions == {"mode": "viewshed"}
    assert "mode" not in observation.result


def test_previous_tool_result_terrain_grid() -> None:
    terrain = _grid(_flat_values())
    ctx = RuntimeContext(
        previous_tool_result={"ok": True, "result": terrain, "error_code": None}
    )
    observation = _sightline(
        inputs={"observer": _OBS, "target": _TGT, "terrain": "$previous_tool_result"},
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["visible"] is True
    assert observation.result["applied"]["terrain_source"] == "grid"


def test_injected_sampler_without_terrain_field() -> None:
    class FakeSampler:
        kind = "dem"
        crs = "wgs84"

        def sample(self, lon: float, lat: float) -> float:
            del lon, lat
            return _FLAT_M

        def resolution_m(self) -> float:
            return 50.0

        def bbox(self) -> tuple[float, float, float, float]:
            return (_WEST, _SOUTH, _EAST, _NORTH)

    ctx = RuntimeContext(extras={"visibility_terrain": FakeSampler()})
    observation = _sightline(inputs={"observer": _OBS, "target": _TGT}, ctx=ctx)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["visible"] is True
    assert observation.result["applied"]["terrain_source"] == "injected"


def test_dsm_assumption_does_not_claim_full_cover() -> None:
    observation = _sightline(
        inputs={
            "observer": _OBS,
            "target": _TGT,
            "terrain": _grid(_flat_values(), kind="dsm"),
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["terrain_kind"] == "dsm"
    joined = "".join(observation.result["assumptions"])
    assert "不能声称" in joined
    assert observation.result["coverage"]["accounts_for_buildings"] is False


def test_out_of_coverage_is_engine_unavailable() -> None:
    observation = _sightline(
        inputs={
            "observer": {"lon": 1.0, "lat": 1.0},
            "target": {"lon": 1.1, "lat": 1.0},
            "terrain": _grid(_flat_values()),
        }
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"


def test_curvature_blocks_long_flat_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISIBILITY_CURVATURE", "true")
    rows, cols = 11, 21
    values = [[_FLAT_M] * cols for _ in range(rows)]
    terrain = {
        "values": values,
        "west": 0.0,
        "south": -0.05,
        "east": 1.0,
        "north": 0.05,
        "kind": "dem",
        "crs": "wgs84",
    }
    observation = _sightline(
        inputs={
            "observer": {"lon": 0.05, "lat": 0.0},
            "target": {"lon": 0.95, "lat": 0.0},
            "terrain": terrain,
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["visible"] is False
    assert observation.result["applied"]["curvature"] is True
    joined = "".join(observation.result["assumptions"])
    assert "曲率" in joined


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
    observation = _sightline(
        inputs={
            "observer": _OBS,
            "target": _TGT,
            "terrain": str(path),
        }
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["visible"] is True
    assert observation.result["applied"]["terrain_source"] == "geotiff"
