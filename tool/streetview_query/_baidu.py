"""百度全景网页内部接口适配器；坐标在边界转为百度墨卡托。"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlencode

from tool._crs import gcj02_to_wgs84, wgs84_to_gcj02
from tool.streetview_query._streetview import (
    EngineUnavailableError,
    ImageSearchRequest,
    _http_bytes,
    _http_json,
)

_PROVIDER_BAIDU = "baidu"
_DEFAULT_ENDPOINT = "https://mapsv0.bdimg.com/"
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_XPI = math.pi * 3000.0 / 180.0
_MCBAND = [12890594.86, 8362377.87, 5591021, 3481989.83, 1678043.12, 0]
_MC2LL = [
    [1.410526172116255e-8, 0.00000898305509648872, -1.9939833816331, 200.9824383106796, -187.2403703815547, 91.6087516669843, -23.38765649603339, 2.57121317296198, -0.03801003308653, 17337981.2],
    [-7.435856389565537e-9, 0.000008983055097726239, -0.78625201886289, 96.32687599759846, -1.85204757529826, -59.36935905485877, 47.40033549296737, -16.50741931063887, 2.28786674699375, 10260144.86],
    [-3.030883460898826e-8, 0.00000898305509983578, 0.30071316287616, 59.74293618442277, 7.357984074871, -25.38371002664745, 13.45380521110908, -3.29883767235584, 0.32710905363475, 6856817.37],
    [-1.981981304930552e-8, 0.000008983055099779535, 0.03278182852591, 40.31678527705744, 0.65659298677277, -4.44255534477492, 0.85341911805263, 0.12923347998204, -0.04625736007561, 4482777.06],
    [3.09191371068437e-9, 0.000008983055096812155, 0.00006995724062, 23.10934304144901, -0.00023663490511, -0.6321817810242, -0.00663494467273, 0.03430082397953, -0.00466043876332, 2555164.4],
    [2.890871144776878e-9, 0.000008983055095805407, -3.068298e-8, 7.47137025468032, -0.00000353937994, -0.02145144861037, -0.00001234426596, 0.00010322952773, -0.00000323890364, 826088.5],
]
_LLBAND = [75, 60, 45, 30, 15, 0]
_LL2MC = [
    [-0.0015702102444, 111320.7020616939, 1704480524535203, -10338987376042340, 26112667856603880, -35149669176653700, 26595700718403920, -10725012454188240, 1800819912950474, 82.5],
    [0.0008277824516172526, 111320.7020463578, 647795574.6671607, -4082003173.641316, 10774905663.51142, -15171875531.51559, 12053065338.62167, -5124939663.577472, 913311935.9512032, 67.5],
    [0.00337398766765, 111320.7020202162, 4481351.045890365, -23393751.19931662, 79682215.47186455, -115964993.2797253, 97236711.15602145, -43661946.33752821, 8477230.501135234, 52.5],
    [0.00220636496208, 111320.7020209128, 51751.86112841131, 3796837.749470245, 992013.7397791013, -1221952.21711287, 1340652.697009075, -620943.6990984312, 144416.9293806241, 37.5],
    [-0.0003441963504368392, 111320.7020576856, 278.2353980772752, 2485758.690035394, 6070.750963243378, 54821.18345352118, 9540.606633304236, -2710.55326746645, 1405.483844121726, 22.5],
    [-0.0003218135878613132, 111320.7020701615, 0.00369383431289, 823725.6402795718, 0.46104986909093, 2351.343141331292, 1.58060784298199, 8.77738589078284, 0.37238884252424, 7.45],
]


class BaiduPanoramaProvider:
    """百度全景 qt=qsdata / sdata / pr3d；不扫区域网格。"""

    name = _PROVIDER_BAIDU
    crs = "wgs84"
    renders_perspective = True

    def __init__(
        self,
        *,
        endpoint: str = _DEFAULT_ENDPOINT,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        user_agent: str = _DEFAULT_UA,
    ) -> None:
        self._endpoint = endpoint if endpoint.endswith("/") else f"{endpoint}/"
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent
        self._cache: dict[str, dict[str, Any]] = {}

    def search_images(self, request: ImageSearchRequest) -> list[dict[str, Any]]:
        """只查 bbox 中心的最近点，不按网格扫一片。"""

        lat = (request.bbox.south + request.bbox.north) / 2.0
        lon = (request.bbox.west + request.bbox.east) / 2.0
        x, y = wgs84_to_bdmc(lon, lat)
        payload = self._get_json(
            {"qt": "qsdata", "x": f"{x:.0f}", "y": f"{y:.0f}"},
            error_prefix="百度全景最近点",
        )
        content = payload.get("content") if isinstance(payload, dict) else None
        if not content:
            return []
        if not isinstance(content, dict):
            raise EngineUnavailableError("百度全景最近点回执无法解析")
        panoid = str(content.get("id") or "").strip()
        if not panoid:
            return []
        record = self.get_image(panoid)
        return [record] if record.get("id") else []

    def get_image(self, image_id: str) -> dict[str, Any]:
        token = image_id.strip()
        if not token:
            raise EngineUnavailableError("百度全景 ID 为空")
        cached = self._cache.get(token)
        if cached is not None:
            return dict(cached)
        payload = self._get_json({"qt": "sdata", "sid": token}, error_prefix="百度全景详情")
        rows = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise EngineUnavailableError("百度全景详情回执无法解析")
        record = _record_from_sdata(rows[0], panoid=token)
        self._cache[token] = record
        for neighbor in record.get("neighbors") or []:
            if isinstance(neighbor, dict) and neighbor.get("id"):
                nid = str(neighbor["id"])
                self._cache.setdefault(nid, dict(neighbor))
        return dict(record)

    def list_sequence(self, sequence_id: str) -> list[str]:
        record = self._cache.get(sequence_id.strip())
        if record is None:
            return []
        ids = [str(record.get("id") or sequence_id)]
        for item in record.get("neighbors") or []:
            if isinstance(item, dict):
                token = str(item.get("id") or "").strip()
                if token and token not in ids:
                    ids.append(token)
        return ids

    def fetch_image_bytes(self, url: str) -> bytes:
        return _http_bytes(
            url,
            headers={"User-Agent": self._user_agent},
            timeout_sec=self._timeout_sec,
            error_prefix="百度全景截图",
        )

    def render_view(
        self,
        image_id: str,
        *,
        heading: float,
        pitch: float,
        fov: float,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
    ) -> bytes:
        """按朝向请求透视截图，不再本地投影整幅全景。"""

        params = {
            "qt": "pr3d",
            "fovy": f"{max(1.0, min(179.0, fov)):.0f}",
            "quality": "80",
            "panoid": image_id.strip(),
            "heading": f"{heading:.1f}",
            "pitch": f"{pitch:.1f}",
            "width": str(min(width, 1024)),
            "height": str(height),
        }
        url = f"{self._endpoint}?{urlencode(params)}"
        data = self.fetch_image_bytes(url)
        if not data:
            raise EngineUnavailableError("百度全景截图为空")
        return data

    def _get_json(self, params: dict[str, str], *, error_prefix: str) -> dict[str, Any]:
        url = f"{self._endpoint}?{urlencode(params)}"
        payload = _http_json(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self._user_agent,
            },
            timeout_sec=self._timeout_sec,
            error_prefix=error_prefix,
        )
        if not isinstance(payload, dict):
            raise EngineUnavailableError(f"{error_prefix} 回执不是 JSON 对象")
        return payload


def _record_from_sdata(raw: dict[str, Any], *, panoid: str) -> dict[str, Any]:
    x = _scaled_meters(raw.get("X", raw.get("x")))
    y = _scaled_meters(raw.get("Y", raw.get("y")))
    lon, lat = bdmc_to_wgs84(x, y) if x is not None and y is not None else (None, None)
    neighbors: list[dict[str, Any]] = []
    seen: set[str] = {panoid}
    for road in raw.get("Roads") or []:
        if not isinstance(road, dict):
            continue
        for pano in road.get("Panos") or []:
            if not isinstance(pano, dict):
                continue
            pid = str(pano.get("PID") or pano.get("id") or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            px = _scaled_meters(pano.get("X", pano.get("x")))
            py = _scaled_meters(pano.get("Y", pano.get("y")))
            nlon, nlat = bdmc_to_wgs84(px, py) if px is not None and py is not None else (None, None)
            neighbors.append(
                {
                    "id": pid,
                    "lat": nlat,
                    "lon": nlon,
                    "is_pano": True,
                    "sequence": panoid,
                }
            )
    history: list[dict[str, Any]] = []
    for item in raw.get("TimeLine") or []:
        if not isinstance(item, dict):
            continue
        hid = str(item.get("ID") or item.get("id") or "").strip()
        if not hid or hid == panoid:
            continue
        history.append(
            {
                "id": hid,
                "lat": lat,
                "lon": lon,
                "captured_at": item.get("Date") or item.get("date"),
                "is_pano": True,
            }
        )
    return {
        "id": panoid,
        "lat": lat,
        "lon": lon,
        "captured_at": raw.get("Date") or raw.get("date"),
        "compass_angle": raw.get("MoveDir"),
        "sequence": panoid,
        "is_pano": True,
        "neighbors": neighbors,
        "history": history,
    }


def _scaled_meters(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) / 100.0
    if isinstance(raw, str) and raw.strip():
        try:
            return float(raw.strip()) / 100.0
        except ValueError:
            return None
    return None


def _poly(x: float, y: float, coeff: list[float]) -> tuple[float, float]:
    fx = coeff[0] + coeff[1] * abs(x)
    t = abs(y) / coeff[9]
    fy = (
        coeff[2]
        + coeff[3] * t
        + coeff[4] * t**2
        + coeff[5] * t**3
        + coeff[6] * t**4
        + coeff[7] * t**5
        + coeff[8] * t**6
    )
    return (-fx if x < 0 else fx), (-fy if y < 0 else fy)


def gcj02_to_bd09(lat: float, lon: float) -> tuple[float, float]:
    """GCJ-02 → BD-09，返回 lat, lon。"""

    z = math.hypot(lon, lat) + 0.00002 * math.sin(lat * _XPI)
    theta = math.atan2(lat, lon) + 0.000003 * math.cos(lon * _XPI)
    return z * math.sin(theta) + 0.006, z * math.cos(theta) + 0.0065


def bd09_to_gcj02(lat: float, lon: float) -> tuple[float, float]:
    """BD-09 → GCJ-02，返回 lat, lon。"""

    x, y = lon - 0.0065, lat - 0.006
    z = math.hypot(x, y) - 0.00002 * math.sin(y * _XPI)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * _XPI)
    return z * math.sin(theta), z * math.cos(theta)


def bd09_to_bdmc(lat: float, lon: float) -> tuple[float, float]:
    """BD-09 → 百度墨卡托平面坐标。"""

    clamped = max(min(lat, 74.0), -74.0)
    coeff = next(_LL2MC[index] for index, band in enumerate(_LLBAND) if abs(clamped) >= band)
    return _poly(lon, clamped, coeff)


def bdmc_to_bd09(x: float, y: float) -> tuple[float, float]:
    """百度墨卡托 → BD-09，返回 lat, lon。"""

    coeff = next(_MC2LL[index] for index, band in enumerate(_MCBAND) if abs(y) >= band)
    lon, lat = _poly(x, y, coeff)
    return lat, lon


def wgs84_to_bdmc(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 → 百度墨卡托。"""

    gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
    bd_lat, bd_lon = gcj02_to_bd09(gcj_lat, gcj_lon)
    return bd09_to_bdmc(bd_lat, bd_lon)


def bdmc_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """百度墨卡托 → WGS84 lon, lat。"""

    bd_lat, bd_lon = bdmc_to_bd09(x, y)
    gcj_lat, gcj_lon = bd09_to_gcj02(bd_lat, bd_lon)
    return gcj02_to_wgs84(gcj_lon, gcj_lat)
