"""satellite_imagery_compare 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.satellite_imagery_query._query import PreviewRequest, SceneSearchRequest


class FakeSatelliteProvider:
    """测试替身：按空间/时间返回目录命中与 PNG，不访问网络。"""

    name = "fake"
    crs = "wgs84"

    def __init__(
        self,
        scenes: list[dict[str, Any]] | None = None,
        *,
        png_by_scene: dict[str, bytes] | None = None,
        png: bytes | None = None,
    ) -> None:
        self.scenes = list(scenes if scenes is not None else _sample_scenes())
        self.png_by_scene = dict(png_by_scene or {})
        self.png = png if png is not None else _png_bytes()
        self.search_calls: list[SceneSearchRequest] = []
        self.preview_calls: list[PreviewRequest] = []

    def search_scenes(self, request: SceneSearchRequest) -> list[dict[str, Any]]:
        self.search_calls.append(request)
        hits: list[dict[str, Any]] = []
        for item in self.scenes:
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            lon = (float(bbox[0]) + float(bbox[2])) / 2.0
            lat = (float(bbox[1]) + float(bbox[3])) / 2.0
            if not (
                request.bbox.west <= lon <= request.bbox.east
                and request.bbox.south <= lat <= request.bbox.north
            ):
                continue
            collection = str(item.get("collection") or "")
            if request.collections and collection not in request.collections:
                continue
            captured = _scene_date(item)
            if request.datetime and captured is not None and not _in_datetime(request.datetime, captured):
                continue
            hits.append(dict(item))
        return hits

    def fetch_preview(self, request: PreviewRequest) -> bytes:
        self.preview_calls.append(request)
        scene_id = request.scene.scene_id
        if scene_id in self.png_by_scene:
            return self.png_by_scene[scene_id]
        return self.png


def _png_bytes(color: tuple[int, int, int] = (30, 90, 40), size: tuple[int, int] = (8, 8)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _textured_png(size: tuple[int, int] = (32, 32)) -> bytes:
    image = Image.new("RGB", size, (20, 40, 60))
    draw = ImageDraw.Draw(image)
    for index in range(8):
        x = 2 + index * 4
        y = 2 + (index % 4) * 7
        draw.rectangle([x, y, x + 6, y + 5], fill=(index * 30, 200 - index * 10, 80))
        draw.ellipse([x + 10, y, x + 16, y + 6], fill=(255 - index * 20, index * 25, 140))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _sample_scenes() -> list[dict[str, Any]]:
    return [
        _scene("S2_2024", "2024-06-12T03:11:00Z", 3.2, 10.0),
        _scene("S2_2018", "2018-06-08T02:40:00Z", 4.0, 10.0),
        _scene(
            "S2_ELSEWHERE",
            "2024-06-12T03:11:00Z",
            1.0,
            10.0,
            bbox=[10.0, 50.0, 10.2, 50.2],
        ),
    ]


def _scene(
    scene_id: str,
    datetime_text: str,
    cloud_cover: float,
    gsd: float,
    *,
    bbox: list[float] | None = None,
    collection: str = "sentinel-2-l2a",
) -> dict[str, Any]:
    return {
        "id": scene_id,
        "collection": collection,
        "bbox": bbox or [113.5, 34.7, 113.8, 34.9],
        "properties": {
            "datetime": datetime_text,
            "eo:cloud_cover": cloud_cover,
            "gsd": gsd,
            "confirmed_location": "MUST NOT LEAK",
            "raw_content": "FULL STAC JSON MUST NOT LEAK",
        },
    }


def _scene_date(item: dict[str, Any]) -> date | None:
    props = item.get("properties")
    if not isinstance(props, dict):
        return None
    raw = str(props.get("datetime") or "")
    if len(raw) < 10:
        return None
    return date.fromisoformat(raw[:10])


def _in_datetime(spec: str, captured: date) -> bool:
    if "/" not in spec:
        return True
    start_raw, end_raw = spec.split("/", 1)
    start = date.fromisoformat(start_raw[:10])
    end = date.fromisoformat(end_raw[:10])
    return start <= captured <= end


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


def _call(
    operation: str,
    *,
    inputs: dict[str, Any] | None = None,
    provider: FakeSatelliteProvider | None = None,
    ctx: RuntimeContext | None = None,
    tmp_path: Path | None = None,
) -> Observation:
    payload: dict[str, Any] = dict(inputs or {})
    engine = provider if provider is not None else FakeSatelliteProvider()
    runtime = ctx
    if runtime is None:
        extras: dict[str, Any] = {"satellite_imagery_query_provider": engine}
        if tmp_path is not None:
            extras["artifact_dir"] = str(tmp_path / "out")
        runtime = RuntimeContext(extras=extras)
    else:
        if provider is not None or "satellite_imagery_query_provider" not in runtime.extras:
            runtime.extras["satellite_imagery_query_provider"] = engine
        if tmp_path is not None and "artifact_dir" not in runtime.extras:
            runtime.extras["artifact_dir"] = str(tmp_path / "out")
    return execute(
        "satellite_imagery_compare",
        operation,
        purpose="测试",
        inputs=payload,
        ctx=runtime,
    )


def test_compare_time_missing_times_is_missing_input() -> None:
    observation = execute(
        "satellite_imagery_compare",
        "compare_time",
        purpose="缺输入",
        inputs={},
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_compare_time_missing_area_is_missing_input() -> None:
    observation = _call("compare_time", inputs={"times": ["2018", "2024"]})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_compare_candidates_missing_candidates_or_template_is_missing_input() -> None:
    observation = execute(
        "satellite_imagery_compare",
        "compare_candidates",
        purpose="缺输入",
        inputs={},
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    missing_template = _call(
        "compare_candidates",
        inputs={
            "candidates": [
                {"bbox": [113.5, 34.7, 113.8, 34.9]},
                {"bbox": [10.0, 50.0, 10.2, 50.2]},
            ]
        },
    )
    assert missing_template.ok is False
    assert missing_template.error_code == "missing_input"


def test_candidates_count_only_is_missing_input() -> None:
    observation = _call(
        "compare_candidates",
        inputs={"candidates": 3, "template": "T字路口旁有河"},
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "数量" in observation.error


def test_place_name_without_geometry_is_missing_input() -> None:
    observation = _call(
        "compare_time",
        inputs={"area": "郑州市", "times": ["2018", "2024"]},
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "geocode" in observation.error


def test_compare_time_two_years_returns_overlay_and_dates(tmp_path: Path) -> None:
    provider = FakeSatelliteProvider(
        png_by_scene={
            "S2_2018": _png_bytes((180, 40, 40)),
            "S2_2024": _png_bytes((30, 90, 40)),
        }
    )
    observation = _call(
        "compare_time",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "times": ["2018", "2024"],
            "comparison": "建设变化",
        },
        provider=provider,
        tmp_path=tmp_path,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["operation"] == "compare_time"
    captured = {item["captured_at"] for item in observation.result["sources"]}
    assert captured == {"2018-06-08", "2024-06-12"}
    pair = observation.result["pairs"][0]
    assert pair["mae"] > 0
    assert pair["overlay_image_id"]
    assert observation.artifacts.get("overlay_image_id")
    assert Path(str(observation.artifacts["image_path"])).is_file()
    assert observation.result["applied"]["comparison"] == "建设变化"
    assert "demolished" not in _nested_keys(observation.result)
    assert "confirmed_location" not in _nested_keys(observation.result)
    assert any("伪变化" in item or "SCL" in item for item in observation.result["assumptions"])


def test_compare_time_one_year_missing_is_insufficient_imagery() -> None:
    observation = _call(
        "compare_time",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "times": ["2010", "2024"],
        },
    )
    assert observation.ok is False
    assert observation.error_code == "insufficient_imagery"


def test_resolution_mismatch_is_not_comparable() -> None:
    provider = FakeSatelliteProvider(
        scenes=[
            _scene("S2_2024", "2024-06-12T03:11:00Z", 3.2, 10.0),
            _scene(
                "LS_2018",
                "2018-06-08T02:40:00Z",
                5.0,
                30.0,
                collection="landsat-ot-l2",
            ),
        ]
    )
    observation = _call(
        "compare_time",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "times": ["2018", "2024"],
        },
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["comparable"] is False
    assert observation.result["conditions"]["resolution_consistent"] is False
    notes = " ".join(observation.result["conditions"]["notes"])
    assert "分辨率" in notes
    assert "比较条件不一致" in notes


def test_compare_candidates_text_template_has_no_location_conclusion(tmp_path: Path) -> None:
    observation = _call(
        "compare_candidates",
        inputs={
            "candidates": [
                {"bbox": [113.5, 34.7, 113.8, 34.9]},
                {"bbox": [10.0, 50.0, 10.2, 50.2]},
            ],
            "template": "T字路口旁有河",
        },
        tmp_path=tmp_path,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["template_kind"] == "text"
    assert len(observation.result["sources"]) == 2
    assert "confirmed_location" not in _nested_keys(observation.result)
    assert "best_location" not in _nested_keys(observation.result)
    assert any("语义匹配" in item for item in observation.result["assumptions"])
    ranked_ids = {item["image_id"] for item in observation.result["ranked"]}
    source_ids = {item["image_id"] for item in observation.result["sources"]}
    assert ranked_ids == source_ids


def test_image_template_prefers_matching_texture(tmp_path: Path) -> None:
    textured = _textured_png()
    template_path = tmp_path / "template.png"
    template_path.write_bytes(textured)
    provider = FakeSatelliteProvider(
        png_by_scene={
            "S2_2024": textured,
            "S2_ELSEWHERE": _png_bytes((200, 10, 10), size=(32, 32)),
        }
    )
    observation = _call(
        "compare_candidates",
        inputs={
            "candidates": [
                {"bbox": [113.5, 34.7, 113.8, 34.9]},
                {"bbox": [10.0, 50.0, 10.2, 50.2]},
            ],
            "template": str(template_path),
        },
        provider=provider,
        tmp_path=tmp_path,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["template_kind"] == "image"
    ranked = observation.result["ranked"]
    assert ranked[0]["score"] > ranked[1]["score"]
    assert ranked[0]["scene_id"] == "S2_2024"


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "satellite_imagery_compare",
        "compare_time",
        purpose="闸门",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "times": ["2018", "2024"],
        },
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_previous_candidates_from_context(tmp_path: Path) -> None:
    ctx = RuntimeContext(
        extras={
            "previous_candidates": [
                {"bbox": [113.5, 34.7, 113.8, 34.9]},
                {"bbox": [10.0, 50.0, 10.2, 50.2]},
            ],
            "artifact_dir": str(tmp_path / "out"),
        }
    )
    observation = _call(
        "compare_candidates",
        inputs={"candidates": "$previous_candidates", "template": "河汊"},
        ctx=ctx,
        tmp_path=tmp_path,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert len(observation.result["sources"]) == 2
