"""image_compare 本地比较执行器测试；禁止真实付费 API。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tool import execute
from tool.contract import Observation, RuntimeContext


def _textured(path: Path, width: int = 200, height: int = 160) -> Path:
    image = Image.new("RGB", (width, height), (20, 40, 60))
    draw = ImageDraw.Draw(image)
    for index in range(12):
        x = 8 + index * 15
        y = 8 + (index % 5) * 28
        draw.rectangle(
            [x, y, x + 20, y + 18],
            fill=(index * 20, 200 - index * 10, 80 + index * 8),
        )
        draw.ellipse(
            [x + 40, y, x + 58, y + 16],
            fill=(255 - index * 15, index * 18, 120),
        )
    draw.line([(0, 0), (width, height)], fill=(255, 255, 0), width=2)
    draw.line([(0, height), (width, 0)], fill=(0, 255, 255), width=2)
    image.save(path)
    return path


def _solid(path: Path, color: tuple[int, int, int], width: int = 40, height: int = 24) -> Path:
    Image.new("RGB", (width, height), color).save(path)
    return path


def _compare(
    tmp_path: Path,
    *,
    images: list[Path],
    inputs: dict[str, object] | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {"images": [str(path) for path in images]}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    if runtime is None:
        runtime = RuntimeContext(extras={"artifact_dir": str(tmp_path / "out")})
    return execute(
        "image_compare",
        "compare",
        purpose="比较",
        inputs=payload,
        ctx=runtime,
    )


def test_identical_copies_have_high_inlier_ratio_and_registration(tmp_path: Path) -> None:
    source = _textured(tmp_path / "a.png")
    copy = tmp_path / "b.png"
    Image.open(source).save(copy)
    observation = _compare(tmp_path, images=[source, copy], inputs={"method": "feature"})
    assert observation.ok is True
    assert observation.result is not None
    pair = observation.result["pairs"][0]
    assert pair["match_count"] > 0
    assert pair["inlier_ratio"] >= 0.5
    assert pair["homography"] is not None
    assert observation.artifacts.get("registration_image_id")
    assert Path(observation.artifacts["image_path"]).is_file()
    matrix = pair["homography"]
    assert matrix[0][0] == pytest.approx(1.0, abs=0.15)
    assert matrix[1][1] == pytest.approx(1.0, abs=0.15)
    assert matrix[0][2] == pytest.approx(0.0, abs=8.0)
    assert matrix[1][2] == pytest.approx(0.0, abs=8.0)


def test_translated_crop_homography_is_approximately_translation(tmp_path: Path) -> None:
    source = _textured(tmp_path / "src.png")
    cropped = Image.open(source).crop((24, 16, 160, 130))
    crop_path = tmp_path / "crop.png"
    cropped.save(crop_path)
    observation = _compare(tmp_path, images=[source, crop_path], inputs={"method": "feature"})
    assert observation.ok is True
    assert observation.result is not None
    pair = observation.result["pairs"][0]
    assert pair["match_count"] > 0
    assert pair["homography"] is not None
    assert pair["homography"][0][2] == pytest.approx(24.0, abs=8.0)
    assert pair["homography"][1][2] == pytest.approx(16.0, abs=8.0)


def test_different_solids_have_near_zero_matches_without_identity_claim(tmp_path: Path) -> None:
    red = _solid(tmp_path / "red.png", (220, 10, 10))
    blue = _solid(tmp_path / "blue.png", (10, 20, 220))
    observation = _compare(tmp_path, images=[red, blue], inputs={"method": "feature"})
    assert observation.ok is True
    assert observation.result is not None
    pair = observation.result["pairs"][0]
    assert pair["match_count"] == 0
    assert pair["inlier_count"] == 0
    assert pair["homography"] is None
    dumped = str(observation.result) + str(observation.artifacts)
    assert "高度一致" not in dumped
    assert any("证据" in item and "地点" in item for item in observation.result["assumptions"])


def test_pixel_mae_on_same_size_and_rejects_unaligned(tmp_path: Path) -> None:
    left = _solid(tmp_path / "left.png", (10, 20, 30), width=40, height=24)
    right_img = Image.new("RGB", (40, 24), (10, 20, 30))
    right_img.putpixel((2, 2), (200, 200, 200))
    right = tmp_path / "right.png"
    right_img.save(right)
    close = _compare(tmp_path, images=[left, right], inputs={"method": "pixel"})
    assert close.ok is True
    assert close.result is not None
    pair = close.result["pairs"][0]
    assert pair["mae"] > 0
    assert pair["changed_ratio"] > 0
    assert pair["method_used"] == "pixel"
    assert close.artifacts.get("diff_image_id")

    wide = _solid(tmp_path / "wide.png", (10, 20, 30), width=80, height=24)
    unaligned = _compare(tmp_path, images=[left, wide], inputs={"method": "pixel"})
    assert unaligned.ok is False
    assert unaligned.error_code == "images_not_aligned"
    assert unaligned.result is None


def test_histogram_similar_beats_different(tmp_path: Path) -> None:
    red_a = _solid(tmp_path / "red_a.png", (220, 12, 12))
    red_b = _solid(tmp_path / "red_b.png", (210, 18, 16))
    blue = _solid(tmp_path / "blue.png", (12, 18, 220))
    similar = _compare(tmp_path, images=[red_a, red_b], inputs={"method": "histogram"})
    different = _compare(tmp_path, images=[red_a, blue], inputs={"method": "histogram"})
    assert similar.ok is True and different.ok is True
    assert similar.result is not None and different.result is not None
    similar_corr = similar.result["pairs"][0]["histogram"]["correlation"]
    different_corr = different.result["pairs"][0]["histogram"]["correlation"]
    assert similar_corr > different_corr


def test_geometry_distinguishes_aspect_ratios(tmp_path: Path) -> None:
    wide = _solid(tmp_path / "wide.png", (10, 20, 30), width=80, height=20)
    tall = _solid(tmp_path / "tall.png", (10, 20, 30), width=20, height=80)
    observation = _compare(tmp_path, images=[wide, tall], inputs={"method": "geometry"})
    assert observation.ok is True
    assert observation.result is not None
    pair = observation.result["pairs"][0]
    assert pair["aspect_ratio"] == pytest.approx(16.0)
    assert pair["geometry"]["aspect_a"] == pytest.approx(4.0)
    assert pair["geometry"]["aspect_b"] == pytest.approx(0.25)
    assert "match_count" not in pair


def test_auto_selects_feature_then_histogram(tmp_path: Path) -> None:
    source = _textured(tmp_path / "tex.png")
    copy = tmp_path / "tex_copy.png"
    Image.open(source).save(copy)
    matched = _compare(tmp_path, images=[source, copy], inputs={"method": "auto"})
    assert matched.ok is True
    assert matched.result is not None
    assert matched.result["applied"]["auto_selected"] == "feature"
    assert matched.result["pairs"][0]["method_used"] == "feature"
    assert "inlier_ratio" in matched.result["pairs"][0]

    red = _solid(tmp_path / "auto_red.png", (220, 10, 10))
    blue = _solid(tmp_path / "auto_blue.png", (10, 20, 220))
    fallback = _compare(tmp_path, images=[red, blue], inputs={"method": "auto"})
    assert fallback.ok is True
    assert fallback.result is not None
    assert fallback.result["applied"]["auto_selected"] == "histogram"
    assert fallback.result["pairs"][0]["method_used"] == "histogram"
    assert "match_count" not in fallback.result["pairs"][0]


def test_region_named_region_and_current_images(tmp_path: Path) -> None:
    source = _textured(tmp_path / "named.png")
    copy = tmp_path / "named_copy.png"
    Image.open(source).save(copy)
    boxed = _compare(
        tmp_path,
        images=[source, copy],
        inputs={"method": "feature", "region": [10, 8, 150, 120]},
    )
    assert boxed.ok is True
    assert boxed.result is not None
    assert boxed.result["applied"]["regions"][0] == [10, 8, 150, 120]

    named = execute(
        "image_compare",
        "compare",
        purpose="命名区域",
        inputs={
            "images": [str(source), str(copy)],
            "method": "geometry",
            "region": "bridge_tower_top",
        },
        ctx=RuntimeContext(
            extras={
                "artifact_dir": str(tmp_path / "named_out"),
                "named_regions": {"bridge_tower_top": [2, 2, 40, 20]},
            },
        ),
    )
    assert named.ok is True
    assert named.result is not None
    assert named.result["applied"]["regions"][0] == [2, 2, 40, 20]

    current = execute(
        "image_compare",
        "compare",
        purpose="当前图列表",
        inputs={"method": "histogram"},
        ctx=RuntimeContext(
            current_images=[str(source), str(copy)],
            extras={"artifact_dir": str(tmp_path / "cur_out")},
        ),
    )
    assert current.ok is True
    assert current.result is not None
    assert current.result["summary"]["pair_count"] == 1


def test_missing_too_few_and_invalid_method(tmp_path: Path) -> None:
    missing = execute("image_compare", "compare", purpose="缺图", inputs={})
    assert missing.ok is False
    assert missing.error_code == "missing_input"

    source = _textured(tmp_path / "one.png")
    too_few = _compare(tmp_path, images=[source], inputs={"method": "feature"})
    assert too_few.ok is False
    assert too_few.error_code == "too_few_images"

    missing_file = _compare(
        tmp_path,
        images=[source, tmp_path / "nope.png"],
        inputs={"method": "feature"},
    )
    assert missing_file.ok is False
    assert missing_file.error_code == "image_not_found"

    invalid = _compare(
        tmp_path,
        images=[source, source],
        inputs={"method": "semantic"},
    )
    assert invalid.ok is False
    assert invalid.error_code == "invalid_method"

    unresolved = _compare(
        tmp_path,
        images=[source, source],
        inputs={"method": "geometry", "region": "not_a_region"},
    )
    assert unresolved.ok is False
    assert unresolved.error_code == "unresolved_region"
