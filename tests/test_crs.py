"""WGS84 / GCJ-02 换算；禁止真实付费 API。"""

from __future__ import annotations

import math

from tool._crs import (
    CRS_GCJ02,
    CRS_WGS84,
    format_gcj02_lonlat,
    gcj02_to_wgs84,
    in_china,
    is_gcj02,
    is_wgs84,
    location_to_wgs84,
    shift_amap_coord_string,
    to_gcj02,
    to_wgs84,
    transform_bbox,
    transform_coordinates,
    wgs84_to_gcj02,
)

_ZHENGZHOU = (113.67, 34.89)
_BEIJING = (116.397, 39.908)
_LONDON = (-0.1276, 51.5074)


def _approx_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """粗测两点球面距离（米）。"""

    lon1, lat1 = a
    lon2, lat2 = b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    hav = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * 6371000.0 * math.asin(min(1.0, math.sqrt(hav)))


def test_china_points_are_offset_tens_of_meters() -> None:
    for lon, lat in (_ZHENGZHOU, _BEIJING):
        gcj = wgs84_to_gcj02(lon, lat)
        assert gcj != (lon, lat)
        offset_m = _approx_m((lon, lat), gcj)
        assert 50.0 < offset_m < 700.0


def test_round_trip_error_is_sub_meter() -> None:
    for lon, lat in (_ZHENGZHOU, _BEIJING):
        gcj = wgs84_to_gcj02(lon, lat)
        back = gcj02_to_wgs84(*gcj)
        assert _approx_m((lon, lat), back) < 1.0


def test_outside_china_is_identity() -> None:
    assert wgs84_to_gcj02(*_LONDON) == _LONDON
    assert gcj02_to_wgs84(*_LONDON) == _LONDON
    assert in_china(*_LONDON) is False
    assert in_china(*_ZHENGZHOU) is True


def test_to_wgs84_does_not_double_shift() -> None:
    lon, lat = _ZHENGZHOU
    assert to_wgs84(lon, lat, CRS_WGS84) == (lon, lat)
    assert to_wgs84(lon, lat, None) == (lon, lat)
    shifted = to_wgs84(lon, lat, CRS_GCJ02)
    assert shifted != (lon, lat)


def test_to_gcj02_skips_already_gcj() -> None:
    lon, lat = _ZHENGZHOU
    assert to_gcj02(lon, lat, CRS_GCJ02) == (lon, lat)
    assert to_gcj02(lon, lat, CRS_WGS84) != (lon, lat)


def test_aliases() -> None:
    assert is_wgs84("EPSG:4326")
    assert is_wgs84("crs84")
    assert is_gcj02("GCJ-02")
    assert is_gcj02("amap")
    assert is_gcj02("gaode")


def test_transform_bbox_and_coordinates() -> None:
    west, south, east, north = transform_bbox(
        113.60,
        34.70,
        113.80,
        34.90,
        from_crs=CRS_WGS84,
        to_crs=CRS_GCJ02,
    )
    assert west != 113.60 or south != 34.70
    coords = transform_coordinates(
        [[113.67, 34.89], [113.80, 34.75]],
        from_crs=CRS_GCJ02,
        to_crs=CRS_WGS84,
    )
    assert coords[0] != [113.67, 34.89]
    assert coords[0][0] < 113.67


def test_location_and_amap_string_helpers() -> None:
    payload = location_to_wgs84(113.67, 34.89, CRS_GCJ02)
    assert payload["crs"] == CRS_WGS84
    assert payload["lon"] != 113.67
    wire = format_gcj02_lonlat(113.65, 34.75)
    assert wire != "113.650000,34.750000"
    shifted = shift_amap_coord_string("113.650000,34.750000")
    assert shifted == wire
    pair = shift_amap_coord_string("113.600000,34.900000|113.800000,34.700000")
    assert "|" in pair
    assert pair != "113.600000,34.900000|113.800000,34.700000"
