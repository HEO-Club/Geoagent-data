"""WGS84 / GCJ-02 近似换算；Agent 可见坐标统一为 WGS84。

高德国内接口使用 GCJ-02。本模块只做公开偏移公式的米级近似，
不引入第三方依赖。中国境外坐标原样返回。
"""

from __future__ import annotations

import math
from typing import Any

CRS_WGS84 = "wgs84"
CRS_GCJ02 = "gcj02"

_WGS84_ALIASES = frozenset(
    {
        "wgs84",
        "wgs 84",
        "wgs1984",
        "epsg:4326",
        "epsg4326",
        "4326",
        "crs84",
        "ogc:crs84",
        "urn:ogc:def:crs:ogc:1.3:crs84",
        "urn:ogc:def:crs:epsg::4326",
        "urn:ogc:def:crs:epsg:4326",
    }
)
_GCJ02_ALIASES = frozenset(
    {
        "gcj02",
        "gcj-02",
        "gcj_02",
        "gcj 02",
        "mars",
        "mars84",
        "amap",
        "gaode",
    }
)

_PI = math.pi
_A = 6378245.0
_EE = 0.00669342162296594323
_REVERSE_ITERS = 12
_REVERSE_EPS = 1e-12


def normalize_crs_tag(tag: str | None) -> str:
    """把常见别名收成 wgs84 / gcj02；其它标签原样小写返回。"""

    if tag is None or not str(tag).strip():
        return CRS_WGS84
    text = str(tag).strip().lower()
    compact = text.replace(" ", "")
    if text in _WGS84_ALIASES or compact in _WGS84_ALIASES:
        return CRS_WGS84
    if compact.startswith("urn:ogc:def:crs:epsg::") and compact.endswith("4326"):
        return CRS_WGS84
    if text in _GCJ02_ALIASES or compact in _GCJ02_ALIASES:
        return CRS_GCJ02
    return text


def is_wgs84(tag: str | None) -> bool:
    """是否按 WGS84 解释。"""

    return normalize_crs_tag(tag) == CRS_WGS84


def is_gcj02(tag: str | None) -> bool:
    """是否按 GCJ-02 解释。"""

    return normalize_crs_tag(tag) == CRS_GCJ02


def in_china(lon: float, lat: float) -> bool:
    """公开算法使用的中国范围；范围外不做火星偏移。"""

    return 72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271


def wgs84_to_gcj02(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 → GCJ-02；境外原样返回。"""

    if not in_china(lon, lat):
        return lon, lat
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    rad_lat = lat / 180.0 * _PI
    magic = math.sin(rad_lat)
    magic = 1 - _EE * magic * magic
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrt_magic) * _PI)
    dlon = (dlon * 180.0) / (_A / sqrt_magic * math.cos(rad_lat) * _PI)
    return lon + dlon, lat + dlat


def gcj02_to_wgs84(lon: float, lat: float) -> tuple[float, float]:
    """GCJ-02 → WGS84；迭代求逆，境外原样返回。"""

    if not in_china(lon, lat):
        return lon, lat
    wgs_lon, wgs_lat = lon, lat
    for _ in range(_REVERSE_ITERS):
        tmp_lon, tmp_lat = wgs84_to_gcj02(wgs_lon, wgs_lat)
        dlon = tmp_lon - lon
        dlat = tmp_lat - lat
        wgs_lon -= dlon
        wgs_lat -= dlat
        if abs(dlon) < _REVERSE_EPS and abs(dlat) < _REVERSE_EPS:
            break
    return wgs_lon, wgs_lat


def to_wgs84(lon: float, lat: float, crs: str | None = None) -> tuple[float, float]:
    """任意标签点转到 WGS84；已是 WGS84 则不二次偏移。"""

    if is_gcj02(crs):
        return gcj02_to_wgs84(lon, lat)
    return lon, lat


def to_gcj02(lon: float, lat: float, crs: str | None = None) -> tuple[float, float]:
    """任意标签点转到 GCJ-02；已是 GCJ-02 则不二次偏移。"""

    if is_gcj02(crs):
        return lon, lat
    return wgs84_to_gcj02(lon, lat)


def transform_point(
    lon: float,
    lat: float,
    *,
    from_crs: str | None,
    to_crs: str | None,
) -> tuple[float, float]:
    """两点之间按标签换算；相同或未知标签原样返回。"""

    src = normalize_crs_tag(from_crs)
    dst = normalize_crs_tag(to_crs)
    if src == dst:
        return lon, lat
    if src == CRS_WGS84 and dst == CRS_GCJ02:
        return wgs84_to_gcj02(lon, lat)
    if src == CRS_GCJ02 and dst == CRS_WGS84:
        return gcj02_to_wgs84(lon, lat)
    return lon, lat


def transform_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    from_crs: str | None,
    to_crs: str | None,
) -> tuple[float, float, float, float]:
    """转换四角再取外包，避免偏移场导致边框歪斜。"""

    corners = (
        transform_point(west, south, from_crs=from_crs, to_crs=to_crs),
        transform_point(east, south, from_crs=from_crs, to_crs=to_crs),
        transform_point(west, north, from_crs=from_crs, to_crs=to_crs),
        transform_point(east, north, from_crs=from_crs, to_crs=to_crs),
    )
    lons = [item[0] for item in corners]
    lats = [item[1] for item in corners]
    return min(lons), min(lats), max(lons), max(lats)


def transform_coordinates(
    coords: Any,
    *,
    from_crs: str | None,
    to_crs: str | None,
) -> Any:
    """递归转换 GeoJSON 风格坐标（叶子为 [lon, lat, ...]）。"""

    if not isinstance(coords, (list, tuple)):
        return coords
    if not coords:
        return [] if isinstance(coords, list) else ()
    first = coords[0]
    if isinstance(first, (int, float)):
        if len(coords) < 2:
            return list(coords)
        lon = float(coords[0])
        lat = float(coords[1])
        out_lon, out_lat = transform_point(lon, lat, from_crs=from_crs, to_crs=to_crs)
        rest = [float(item) if isinstance(item, (int, float)) else item for item in coords[2:]]
        return [out_lon, out_lat, *rest]
    converted = [
        transform_coordinates(item, from_crs=from_crs, to_crs=to_crs) for item in coords
    ]
    return converted


def wgs84_location(lon: float, lat: float) -> dict[str, float | str]:
    """Agent 可见点对象。"""

    return {"lon": lon, "lat": lat, "crs": CRS_WGS84}


def location_to_wgs84(
    lon: float,
    lat: float,
    crs: str | None = None,
) -> dict[str, float | str]:
    """把带标签的点译成 Agent 可见 WGS84。"""

    out_lon, out_lat = to_wgs84(lon, lat, crs)
    return wgs84_location(out_lon, out_lat)


def format_lonlat(lon: float, lat: float, *, decimals: int = 6) -> str:
    """高德 / OSRM 风格 `lon,lat` 字符串。"""

    return f"{lon:.{decimals}f},{lat:.{decimals}f}"


def format_gcj02_lonlat(lon: float, lat: float, *, crs: str | None = None, decimals: int = 6) -> str:
    """把内部 WGS84（或已是 GCJ-02）点格式化成高德请求坐标。"""

    out_lon, out_lat = to_gcj02(lon, lat, crs)
    return format_lonlat(out_lon, out_lat, decimals=decimals)


def shift_amap_coord_string(raw: str, *, from_crs: str | None = CRS_WGS84) -> str:
    """转换 `lon,lat` 或 `lon,lat|lon,lat` 高德坐标串。"""

    chunks: list[str] = []
    for part in raw.split("|"):
        chunk = part.strip()
        if not chunk:
            continue
        bits = chunk.split(",")
        if len(bits) != 2:
            chunks.append(chunk)
            continue
        try:
            lon = float(bits[0].strip())
            lat = float(bits[1].strip())
        except ValueError:
            chunks.append(chunk)
            continue
        chunks.append(format_gcj02_lonlat(lon, lat, crs=from_crs))
    return "|".join(chunks)


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * _PI) + 40.0 * math.sin(y / 3.0 * _PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * _PI) + 320.0 * math.sin(y * _PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * _PI) + 40.0 * math.sin(x / 3.0 * _PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * _PI) + 300.0 * math.sin(x / 30.0 * _PI)) * 2.0 / 3.0
    return ret
