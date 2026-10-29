"""location_verify 答案验证执行器测试；禁止真实付费 API。

通过注入假街景后端、假 OCR 引擎与假地理编码后端，覆盖 matched / inconclusive /
错误路径。image_compare 为本地算法，直接跑真实图片。
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.runtime import FilesystemImageStore


class FakeGeocodeProvider:
    """假地理编码后端：固定返回涛源镇坐标。"""

    name = "fake"
    crs = "wgs84"

    def lookup(self, request: Any) -> dict[str, Any]:
        del request
        return {
            "results": [
                {
                    "formatted_address": "云南省丽江市永胜县涛源镇",
                    "name": "涛源镇",
                    "addresstype": "town",
                    "location": {"lat": "26.40", "lon": "100.66"},
                }
            ]
        }


class FakeStreetviewProvider:
    """假街景后端：按 bbox 返回固定影像，缩略图返回给定字节。"""

    name = "fake"
    crs = "wgs84"

    def __init__(self, images: list[dict[str, Any]], png: bytes) -> None:
        self.images = {str(item["id"]): item for item in images}
        self.png = png

    def search_images(self, request: Any) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for item in self.images.values():
            lon, lat = _image_lonlat(item)
            if lon is None or lat is None:
                continue
            if (
                request.bbox.west <= lon <= request.bbox.east
                and request.bbox.south <= lat <= request.bbox.north
            ):
                hits.append(dict(item))
        return hits

    def get_image(self, image_id: str) -> dict[str, Any]:
        return dict(self.images.get(image_id, {"id": image_id}))

    def list_sequence(self, sequence_id: str) -> list[str]:
        del sequence_id
        return list(self.images.keys())

    def fetch_image_bytes(self, url: str) -> bytes:
        del url
        return self.png


class FakeOcrEngine:
    """假 OCR 引擎：固定返回给定文本。"""

    name = "fake"

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def detect(self, image_rgb: Image.Image) -> list[dict[str, Any]]:
        del image_rgb
        return list(self.items)


def _image_lonlat(item: dict[str, Any]) -> tuple[float | None, float | None]:
    geom = item.get("geometry") or item.get("computed_geometry")
    if isinstance(geom, dict):
        coords = geom.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return float(coords[0]), float(coords[1])
    return None, None


def _textured_bytes(width: int = 200, height: int = 160) -> bytes:
    image = Image.new("RGB", (width, height), (20, 40, 60))
    draw = ImageDraw.Draw(image)
    for index in range(16):
        x = 8 + index * 11
        y = 8 + (index % 6) * 24
        draw.rectangle([x, y, x + 22, y + 20], fill=(index * 15, 200 - index * 10, 80 + index * 8))
        draw.ellipse([x + 44, y, x + 64, y + 18], fill=(255 - index * 15, index * 16, 120))
    draw.line([(0, 0), (width, height)], fill=(255, 255, 0), width=2)
    draw.line([(0, height), (width, 0)], fill=(0, 255, 255), width=2)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _query_image(tmp_path: Path, png: bytes) -> Path:
    path = tmp_path / "query.png"
    Image.open(BytesIO(png)).save(path)
    return path


def _verify(
    tmp_path: Path,
    *,
    query_path: Path,
    streetview: FakeStreetviewProvider,
    ocr: FakeOcrEngine,
    geocode: FakeGeocodeProvider,
    inputs: dict[str, Any],
) -> Observation:
    store = FilesystemImageStore(tmp_path / "images")
    query_id = store.register(query_path, image_id="input_0001")
    ctx = RuntimeContext(
        current_image=query_id,
        current_images=[query_id],
        image_store=store,
        extras={
            "artifact_dir": str(tmp_path / "artifacts"),
            "streetview_query_provider": streetview,
            "ocr_engine": ocr,
            "geocode_provider": geocode,
        },
    )
    return execute(
        "location_verify",
        "verify",
        purpose="验证最终定位答案",
        inputs=inputs,
        ctx=ctx,
    )


def test_missing_location_is_structured_error() -> None:
    observation = execute("location_verify", "verify", purpose="验证", inputs={}, ctx=None)
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_verify_matched_when_image_and_ocr_match(tmp_path: Path) -> None:
    png = _textured_bytes()
    query_path = _query_image(tmp_path, png)

    streetview = FakeStreetviewProvider(
        images=[
            {
                "id": "img_a",
                "captured_at": 1_557_619_200_000,
                "compass_angle": 90.0,
                "geometry": {"type": "Point", "coordinates": [113.67, 34.89]},
                "sequence": "seq_east",
                "is_pano": False,
                "thumb_1024_url": "https://example.test/a.jpg",
            }
        ],
        png=png,
    )
    ocr = FakeOcrEngine([{"text": "江湾大酒店", "confidence": 0.9, "bbox": [0, 0, 120, 24]}])
    geocode = FakeGeocodeProvider()

    observation = _verify(
        tmp_path,
        query_path=query_path,
        streetview=streetview,
        ocr=ocr,
        geocode=geocode,
        inputs={"location": [113.67, 34.89], "compare_method": "feature"},
    )

    assert observation.ok is True
    result = observation.result
    assert result is not None
    assert result["coverage"] is True
    assert result["verdict"] == "matched"
    assert result["signals"]["image"]["conclusion"] == "match"
    assert result["signals"]["ocr"]["conclusion"] == "match"
    assert result["signals"]["ocr"]["common"] == ["江湾大酒店"]
    assert result["confidence"] == 0.85


def test_verify_inconclusive_when_no_coverage(tmp_path: Path) -> None:
    png = _textured_bytes()
    query_path = _query_image(tmp_path, png)

    streetview = FakeStreetviewProvider(images=[], png=png)
    ocr = FakeOcrEngine([])
    geocode = FakeGeocodeProvider()

    observation = _verify(
        tmp_path,
        query_path=query_path,
        streetview=streetview,
        ocr=ocr,
        geocode=geocode,
        inputs={"location": "云南省丽江市永胜县涛源镇"},
    )

    assert observation.ok is True
    result = observation.result
    assert result is not None
    assert result["coverage"] is False
    assert result["verdict"] == "inconclusive"
    assert result["location"]["match_level"] == "town"
    assert result["location"]["resolved"] == {"lon": 100.66, "lat": 26.40, "crs": "wgs84"}
