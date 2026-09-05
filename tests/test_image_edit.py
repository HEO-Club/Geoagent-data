"""image_edit 本地 Pillow 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from tool import execute
from tool.contract import RuntimeContext
from tool.runtime import FilesystemImageStore


def _checkerboard(path: Path, width: int = 40, height: int = 24) -> Path:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    assert pixels is not None
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (255, 255, 255) if (x + y) % 2 == 0 else (0, 0, 0)
    image.save(path)
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


def test_crop_pixel_box_and_padding(tmp_path: Path) -> None:
    source = _gradient(tmp_path / "src.png")
    observation = execute(
        "image_edit",
        "crop",
        purpose="裁出标志",
        inputs={"image": str(source), "region": [10, 4, 20, 14], "padding": 2},
        ctx=RuntimeContext(extras={"artifact_dir": str(tmp_path / "out")}),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["width"] == 14
    assert observation.result["height"] == 14
    assert observation.result["applied"]["region"] == [8, 2, 22, 16]
    assert observation.result["applied"]["padding"] == 2
    assert observation.result["detail_invented"] is False
    cropped = Image.open(observation.artifacts["image_path"])
    assert cropped.size == (14, 14)


def test_crop_normalized_box(tmp_path: Path) -> None:
    source = _gradient(tmp_path / "src.png", width=100, height=80)
    observation = execute(
        "image_edit",
        "crop",
        purpose="归一化框",
        inputs={"image": str(source), "region": [0.1, 0.25, 0.5, 0.75]},
        ctx=RuntimeContext(extras={"artifact_dir": str(tmp_path / "out")}),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["region"] == [10, 20, 50, 60]
    assert observation.result["width"] == 40
    assert observation.result["height"] == 40


def test_zoom_uses_lanczos_not_nearest(tmp_path: Path) -> None:
    source = _checkerboard(tmp_path / "src.png")
    observation = execute(
        "image_edit",
        "zoom",
        purpose="放大棋盘",
        inputs={"image": str(source), "region": [0, 0, 10, 10], "scale": 3},
        ctx=RuntimeContext(extras={"artifact_dir": str(tmp_path / "out")}),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["width"] == 30
    assert observation.result["height"] == 30
    assert observation.result["applied"]["scale"] == 3.0
    assert observation.result["detail_invented"] is False

    cropped = Image.open(source).crop((0, 0, 10, 10))
    nearest = cropped.resize((30, 30), Image.Resampling.NEAREST)
    actual = Image.open(observation.artifacts["image_path"]).convert("RGB")
    assert actual.size == (30, 30)
    assert actual.tobytes() != nearest.tobytes()


def test_zoom_default_scale_is_two(tmp_path: Path) -> None:
    source = _gradient(tmp_path / "src.png")
    observation = execute(
        "image_edit",
        "zoom",
        purpose="默认倍数",
        inputs={"image": str(source), "region": [0, 0, 8, 6]},
        ctx=RuntimeContext(extras={"artifact_dir": str(tmp_path / "out")}),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["scale"] == 2.0
    assert observation.result["width"] == 16
    assert observation.result["height"] == 12


def test_enhance_brightness_is_reproducible(tmp_path: Path) -> None:
    source = _gradient(tmp_path / "src.png")
    ctx = RuntimeContext(extras={"artifact_dir": str(tmp_path / "out")})
    inputs = {
        "image": str(source),
        "adjustments": {"brightness": 0.2, "unknown_key": 1},
    }
    first = execute("image_edit", "enhance", purpose="提亮", inputs=inputs, ctx=ctx)
    second = execute("image_edit", "enhance", purpose="提亮", inputs=inputs, ctx=ctx)
    assert first.ok is True and second.ok is True
    assert first.result is not None
    assert first.result["applied"]["adjustments"]["brightness"] == 0.2
    assert first.result["applied"]["ignored"]["unknown_key"] == 1
    pixels_a = Image.open(first.artifacts["image_path"]).convert("RGB").tobytes()
    pixels_b = Image.open(second.artifacts["image_path"]).convert("RGB").tobytes()
    original = Image.open(source).convert("RGB").tobytes()
    assert pixels_a == pixels_b
    assert pixels_a != original


def test_enhance_region_returns_that_crop(tmp_path: Path) -> None:
    source = _gradient(tmp_path / "src.png")
    observation = execute(
        "image_edit",
        "enhance",
        purpose="局部提亮",
        inputs={
            "image": str(source),
            "region": {"x1": 4, "y1": 4, "x2": 12, "y2": 10},
            "adjustments": {"contrast": 0.3, "shadows": 0.2},
        },
        ctx=RuntimeContext(extras={"artifact_dir": str(tmp_path / "out")}),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["width"] == 8
    assert observation.result["height"] == 6
    assert observation.result["applied"]["region"] == [4, 4, 12, 10]


def test_current_image_and_store_chain(tmp_path: Path) -> None:
    source = _checkerboard(tmp_path / "src.png")
    store = FilesystemImageStore(tmp_path / "store")
    ctx = RuntimeContext(current_image=str(source), image_store=store)
    cropped = execute(
        "image_edit",
        "crop",
        purpose="先裁",
        inputs={"image": "$current_image", "region": [0, 0, 10, 10]},
        ctx=ctx,
    )
    assert cropped.ok is True
    assert cropped.result is not None
    image_id = cropped.result["image_id"]
    assert image_id == "img_0001"
    zoomed = execute(
        "image_edit",
        "zoom",
        purpose="再放大",
        inputs={"image": image_id, "region": [0, 0, 10, 10], "scale": 2},
        ctx=ctx,
    )
    assert zoomed.ok is True
    assert zoomed.result is not None
    assert zoomed.result["source_image_id"] == image_id
    assert zoomed.result["image_id"] == "img_0002"
    assert zoomed.result["width"] == 20
    assert zoomed.result["height"] == 20
    assert Path(zoomed.artifacts["image_path"]).is_file()


def test_named_region_without_mapping_is_unresolved(tmp_path: Path) -> None:
    source = _gradient(tmp_path / "src.png")
    observation = execute(
        "image_edit",
        "crop",
        purpose="命名区域",
        inputs={"image": str(source), "region": "bridge_tower_top"},
        ctx=RuntimeContext(extras={"artifact_dir": str(tmp_path / "out")}),
    )
    assert observation.ok is False
    assert observation.error_code == "unresolved_region"


def test_named_region_with_mapping(tmp_path: Path) -> None:
    source = _gradient(tmp_path / "src.png")
    observation = execute(
        "image_edit",
        "crop",
        purpose="映射区域",
        inputs={"image": str(source), "region": "bridge_tower_top"},
        ctx=RuntimeContext(
            extras={
                "artifact_dir": str(tmp_path / "out"),
                "named_regions": {"bridge_tower_top": [2, 2, 8, 6]},
            }
        ),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["region"] == [2, 2, 8, 6]
    assert observation.result["width"] == 6
    assert observation.result["height"] == 4


def test_missing_image_and_region_are_structured_errors(tmp_path: Path) -> None:
    missing_image = execute(
        "image_edit",
        "crop",
        purpose="缺图",
        inputs={"region": [0, 0, 1, 1]},
    )
    assert missing_image.ok is False
    assert missing_image.error_code == "missing_input"

    source = _gradient(tmp_path / "src.png")
    missing_region = execute(
        "image_edit",
        "zoom",
        purpose="缺区域",
        inputs={"image": str(source)},
    )
    assert missing_region.ok is False
    assert missing_region.error_code == "missing_input"

    missing_file = execute(
        "image_edit",
        "enhance",
        purpose="文件不存在",
        inputs={"image": str(tmp_path / "nope.png")},
    )
    assert missing_file.ok is False
    assert missing_file.error_code == "image_not_found"


def test_invalid_scale_and_adjustments(tmp_path: Path) -> None:
    source = _gradient(tmp_path / "src.png")
    bad_scale = execute(
        "image_edit",
        "zoom",
        purpose="倍数非法",
        inputs={"image": str(source), "region": [0, 0, 4, 4], "scale": 0.5},
    )
    assert bad_scale.ok is False
    assert bad_scale.error_code == "invalid_scale"

    bad_adjustments = execute(
        "image_edit",
        "enhance",
        purpose="参数非法",
        inputs={"image": str(source), "adjustments": "bright"},
    )
    assert bad_adjustments.ok is False
    assert bad_adjustments.error_code == "invalid_adjustments"
