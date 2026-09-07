"""image_measure 本地量测执行器测试；禁止真实付费 API。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tool import execute
from tool.contract import Observation, RuntimeContext


def _solid(path: Path, color: tuple[int, int, int], width: int = 40, height: int = 24) -> Path:
    Image.new("RGB", (width, height), color).save(path)
    return path


def _gradient(path: Path, width: int = 40, height: int = 24) -> Path:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    assert pixels is not None
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (x * 6, y * 10, 80)
    image.save(path)
    return path


def _measure(
    tmp_path: Path,
    *,
    source: Path | None = None,
    inputs: dict[str, object],
    ctx: RuntimeContext | None = None,
) -> Observation:
    image = source if source is not None else _solid(tmp_path / "src.png", (10, 20, 30))
    payload = {"image": str(image), **inputs}
    return execute(
        "image_measure",
        "measure",
        purpose="量测",
        inputs=payload,
        ctx=ctx,
    )


def test_bbox_distance_uses_longer_side(tmp_path: Path) -> None:
    observation = _measure(tmp_path, inputs={"measurement": "distance", "region": [10, 4, 20, 20]})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["value"] == 16.0
    assert observation.result["unit"] == "px"
    assert observation.result["method"] == "bbox_longer_side"
    assert observation.result["measurement"] == "distance"
    assert observation.result["pixel_value"] == 16.0
    assert observation.result["components"]["width_px"] == 10.0
    assert observation.result["components"]["height_px"] == 16.0
    assert "无尺度参照" in "".join(observation.result["assumptions"])
    assert "米" not in str(observation.result["value"])
    assert observation.result["unit"] != "m"


def test_bbox_distance_axis_override(tmp_path: Path) -> None:
    observation = _measure(
        tmp_path,
        inputs={"measurement": "distance", "region": [10, 4, 20, 20], "axis": "horizontal"},
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["value"] == 10.0
    assert observation.result["method"] == "bbox_horizontal"
    assert observation.result["applied"]["axis"] == "horizontal"


def test_reference_scale_converts_distance(tmp_path: Path) -> None:
    source = _solid(tmp_path / "wide.png", (10, 20, 30), width=100, height=24)
    observation = _measure(
        tmp_path,
        source=source,
        inputs={
            "measurement": "distance",
            "region": [0, 0, 80, 10],
            "axis": "horizontal",
            "reference": {"region": [0, 0, 40, 8], "known_length": 10, "unit": "m", "axis": "horizontal"},
        },
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["unit"] == "m"
    assert observation.result["method"] == "reference_scale"
    assert observation.result["pixel_value"] == 80.0
    assert observation.result["value"] == pytest.approx(20.0)
    assert observation.result["scale"] == pytest.approx(0.25)
    assert observation.result["reference_applied"]["known_length"] == 10.0
    assert any("均匀" in item for item in observation.result["assumptions"])
    assert any("未做透视" in item for item in observation.result["assumptions"])


def test_vague_reference_is_rejected(tmp_path: Path) -> None:
    observation = _measure(
        tmp_path,
        inputs={"measurement": "distance", "region": [0, 0, 20, 8], "reference": "参照桥"},
    )
    assert observation.ok is False
    assert observation.error_code == "unresolved_reference"
    assert observation.result is None


def test_distance_without_reference_stays_pixels(tmp_path: Path) -> None:
    observation = _measure(tmp_path, inputs={"measurement": "distance", "region": [2, 2, 22, 10]})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["unit"] == "px"
    assert observation.result["method"] == "bbox_longer_side"
    assert "scale" not in observation.result
    assert observation.result["value"] == 20.0


def test_single_box_ratio_is_aspect(tmp_path: Path) -> None:
    observation = _measure(tmp_path, inputs={"measurement": "ratio", "region": [0, 0, 20, 10]})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["value"] == pytest.approx(2.0)
    assert observation.result["unit"] == "ratio"
    assert observation.result["method"] == "aspect_ratio"


def test_two_region_ratio_is_pixel_proportion(tmp_path: Path) -> None:
    source = _solid(tmp_path / "tall.png", (10, 20, 30), width=40, height=48)
    observation = _measure(
        tmp_path,
        source=source,
        inputs={
            "measurement": "ratio",
            "region": [0, 0, 10, 40],
            "reference": {"region": [0, 0, 20, 8]},
        },
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["method"] == "relative_proportion"
    assert observation.result["unit"] == "ratio"
    assert observation.result["value"] == pytest.approx(2.0)
    assert "scale" not in observation.result


def test_two_boxes_in_region_ratio(tmp_path: Path) -> None:
    observation = _measure(
        tmp_path,
        inputs={"measurement": "ratio", "region": [[0, 0, 8, 20], [0, 0, 10, 5]]},
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["value"] == pytest.approx(2.0)
    assert observation.result["method"] == "relative_proportion"


def test_angle_between_two_lines(tmp_path: Path) -> None:
    observation = _measure(
        tmp_path,
        inputs={
            "measurement": "angle",
            "region": [[[0, 0], [10, 0]], [[0, 0], [0, 10]]],
        },
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["unit"] == "deg"
    assert observation.result["method"] == "line_angle"
    assert observation.result["value"] == pytest.approx(90.0)


def test_angle_forty_five_degrees(tmp_path: Path) -> None:
    observation = _measure(
        tmp_path,
        inputs={
            "measurement": "angle",
            "region": [[[0, 0], [10, 0]], [[0, 0], [10, 10]]],
        },
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["value"] == pytest.approx(45.0)


def test_angle_with_single_box_fails(tmp_path: Path) -> None:
    observation = _measure(tmp_path, inputs={"measurement": "angle", "region": [4, 4, 16, 12]})
    assert observation.ok is False
    assert observation.error_code == "invalid_region"


def test_area_pixels_and_scaled_area(tmp_path: Path) -> None:
    pixels = _measure(tmp_path, inputs={"measurement": "area", "region": [0, 0, 40, 20]})
    assert pixels.ok is True
    assert pixels.result is not None
    assert pixels.result["value"] == 800.0
    assert pixels.result["unit"] == "px"
    assert pixels.result["method"] == "bbox_area"

    scaled = _measure(
        tmp_path,
        inputs={
            "measurement": "area",
            "region": [0, 0, 40, 20],
            "reference": {"region": [0, 0, 20, 4], "known_length": 10, "unit": "m", "axis": "horizontal"},
        },
    )
    assert scaled.ok is True
    assert scaled.result is not None
    assert scaled.result["method"] == "reference_scale"
    assert scaled.result["scale"] == pytest.approx(0.5)
    assert scaled.result["value"] == pytest.approx(200.0)
    assert scaled.result["unit"] == "m"
    assert any("未做透视" in item for item in scaled.result["assumptions"])


def test_color_region_mean_and_full_image(tmp_path: Path) -> None:
    source = _solid(tmp_path / "red.png", (12, 34, 56), width=16, height=12)
    region = _measure(
        tmp_path,
        source=source,
        inputs={"measurement": "color", "region": [0, 0, 8, 6]},
    )
    assert region.ok is True
    assert region.result is not None
    assert region.result["unit"] == "rgb"
    assert region.result["method"] == "mean_rgb"
    assert region.result["value"] == {"r": 12, "g": 34, "b": 56, "hex": "#0c2238"}

    full = _measure(tmp_path, source=source, inputs={"measurement": "color"})
    assert full.ok is True
    assert full.result is not None
    assert full.result["value"]["hex"] == "#0c2238"
    assert full.result["applied"]["full_image"] is True


def test_color_gradient_region(tmp_path: Path) -> None:
    source = _gradient(tmp_path / "grad.png", width=10, height=4)
    observation = _measure(
        tmp_path,
        source=source,
        inputs={"measurement": "color", "region": [0, 0, 2, 1]},
    )
    assert observation.ok is True
    assert observation.result is not None
    # 像素 (0,0)=(0,0,80) 与 (1,0)=(6,0,80)
    assert observation.result["value"]["r"] == 3
    assert observation.result["value"]["g"] == 0
    assert observation.result["value"]["b"] == 80


def test_current_image_and_named_region(tmp_path: Path) -> None:
    source = _solid(tmp_path / "src.png", (1, 2, 3))
    ctx = RuntimeContext(
        current_image=str(source),
        extras={"named_regions": {"bridge_tower_top": [2, 2, 12, 8]}},
    )
    observation = execute(
        "image_measure",
        "measure",
        purpose="当前图",
        inputs={"image": "$current_image", "measurement": "distance", "region": "bridge_tower_top"},
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["region"] == [2, 2, 12, 8]
    assert observation.result["value"] == 10.0


def test_missing_and_invalid_inputs(tmp_path: Path) -> None:
    missing_image = execute(
        "image_measure",
        "measure",
        purpose="缺图",
        inputs={"measurement": "distance", "region": [0, 0, 4, 4]},
    )
    assert missing_image.ok is False
    assert missing_image.error_code == "missing_input"

    source = _solid(tmp_path / "src.png", (0, 0, 0))
    missing_measurement = execute(
        "image_measure",
        "measure",
        purpose="缺量",
        inputs={"image": str(source), "region": [0, 0, 4, 4]},
    )
    assert missing_measurement.ok is False
    assert missing_measurement.error_code == "missing_input"

    invalid = execute(
        "image_measure",
        "measure",
        purpose="非法量",
        inputs={"image": str(source), "measurement": "volume", "region": [0, 0, 4, 4]},
    )
    assert invalid.ok is False
    assert invalid.error_code == "invalid_measurement"

    missing_file = execute(
        "image_measure",
        "measure",
        purpose="不存在",
        inputs={"image": str(tmp_path / "nope.png"), "measurement": "color"},
    )
    assert missing_file.ok is False
    assert missing_file.error_code == "image_not_found"

    missing_region = execute(
        "image_measure",
        "measure",
        purpose="缺区域",
        inputs={"image": str(source), "measurement": "distance"},
    )
    assert missing_region.ok is False
    assert missing_region.error_code == "missing_input"


def test_incomplete_scale_reference_is_rejected(tmp_path: Path) -> None:
    observation = _measure(
        tmp_path,
        inputs={
            "measurement": "distance",
            "region": [0, 0, 20, 8],
            "reference": {"region": [0, 0, 10, 4], "known_length": 5},
        },
    )
    assert observation.ok is False
    assert observation.error_code == "unresolved_reference"


def test_segment_distance(tmp_path: Path) -> None:
    observation = _measure(
        tmp_path,
        inputs={"measurement": "distance", "region": [[0, 0], [30, 0]]},
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["method"] == "segment_length"
    assert observation.result["value"] == 30.0
    assert observation.result["unit"] == "px"


def test_json_string_reference(tmp_path: Path) -> None:
    observation = _measure(
        tmp_path,
        inputs={
            "measurement": "distance",
            "region": [0, 0, 40, 6],
            "axis": "horizontal",
            "reference": '{"region":[0,0,20,4],"known_length":8,"unit":"m","axis":"horizontal"}',
        },
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["value"] == pytest.approx(16.0)
    assert observation.result["unit"] == "m"
