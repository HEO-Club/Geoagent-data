"""ocr_read 本地识别/解码执行器测试；禁止真实付费 API。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import pytest
from PIL import Image

from tool import execute
from tool.contract import Observation, RuntimeContext


class FakeOcrEngine:
    """测试替身：返回预设的 ROI 坐标原始识别结果。"""

    name = "fake"

    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self.items = items if items is not None else []

    def detect(self, image_rgb: Image.Image) -> list[dict[str, Any]]:
        del image_rgb
        return list(self.items)


def _solid(path: Path, color: tuple[int, int, int] = (20, 30, 40), width: int = 80, height: int = 60) -> Path:
    Image.new("RGB", (width, height), color).save(path)
    return path


def _recognize(
    tmp_path: Path,
    *,
    source: Path | None = None,
    inputs: dict[str, object] | None = None,
    engine: FakeOcrEngine | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    image = source if source is not None else _solid(tmp_path / "src.png")
    payload: dict[str, object] = {"image": str(image)}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    if runtime is None:
        runtime = RuntimeContext(
            extras={
                "artifact_dir": str(tmp_path / "out"),
                "ocr_engine": engine if engine is not None else FakeOcrEngine(),
            }
        )
    elif engine is not None:
        runtime.extras.setdefault("artifact_dir", str(tmp_path / "out"))
        runtime.extras["ocr_engine"] = engine
    return execute("ocr_read", "recognize", purpose="读字", inputs=payload, ctx=runtime)


def _decode(
    tmp_path: Path,
    *,
    source: Path,
    inputs: dict[str, object] | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {"image": str(source)}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    if runtime is None:
        runtime = RuntimeContext(extras={"artifact_dir": str(tmp_path / "out")})
    return execute("ocr_read", "decode", purpose="解码", inputs=payload, ctx=runtime)


def _make_qr(path: Path, payload: str, size: int = 240) -> Path:
    encoder = cv2.QRCodeEncoder.create()
    qr = encoder.encode(payload)
    bordered = cv2.copyMakeBorder(qr, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    scaled = cv2.resize(bordered, (size, size), interpolation=cv2.INTER_NEAREST)
    if scaled.ndim == 2:
        rgb = cv2.cvtColor(scaled, cv2.COLOR_GRAY2RGB)
    else:
        rgb = scaled
    Image.fromarray(rgb).save(path)
    return path


def test_missing_image_is_structured_error() -> None:
    missing = execute("ocr_read", "recognize", purpose="缺图", inputs={})
    assert missing.ok is False
    assert missing.error_code == "missing_input"
    missing_decode = execute("ocr_read", "decode", purpose="缺图", inputs={})
    assert missing_decode.ok is False
    assert missing_decode.error_code == "missing_input"


def test_current_image_and_file_path(tmp_path: Path) -> None:
    source = _solid(tmp_path / "sign.png")
    engine = FakeOcrEngine(
        [{"text": "江湖大桥", "confidence": 0.97, "bbox": [2, 4, 30, 18]}],
    )
    by_path = _recognize(tmp_path, source=source, engine=engine)
    assert by_path.ok is True
    assert by_path.result is not None
    assert by_path.result["full_text"] == "江湖大桥"

    current = execute(
        "ocr_read",
        "recognize",
        purpose="当前图",
        inputs={"image": "$current_image"},
        ctx=RuntimeContext(
            current_image=str(source),
            extras={"artifact_dir": str(tmp_path / "cur"), "ocr_engine": engine},
        ),
    )
    assert current.ok is True
    assert current.result is not None
    assert current.result["full_text"] == "江湖大桥"


def test_region_offsets_bbox_to_original(tmp_path: Path) -> None:
    source = _solid(tmp_path / "wide.png", width=100, height=80)
    engine = FakeOcrEngine(
        [{"text": "A12", "confidence": 0.9, "quad": [[0, 0], [10, 0], [10, 8], [0, 8]]}],
    )
    observation = _recognize(
        tmp_path,
        source=source,
        inputs={"region": [10, 20, 50, 60]},
        engine=engine,
    )
    assert observation.ok is True
    assert observation.result is not None
    item = observation.result["items"][0]
    assert item["bbox"] == [10, 20, 20, 28]
    assert item["quad"] == [[10, 20], [20, 20], [20, 28], [10, 28]]
    assert observation.result["applied"]["region"] == [10, 20, 50, 60]


def test_raw_text_is_not_corrected(tmp_path: Path) -> None:
    engine = FakeOcrEngine(
        [{"text": "江湖大桥", "confidence": 0.88, "bbox": [1, 1, 40, 16]}],
    )
    observation = _recognize(tmp_path, engine=engine)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["full_text"] == "江湖大桥"
    assert observation.result["items"][0]["text"] == "江湖大桥"
    assert "corrected_text" not in observation.result
    assert "corrected_text" not in observation.result["items"][0]
    assert observation.artifacts.get("annotated_image_id")
    assert Path(observation.artifacts["image_path"]).is_file()


def test_number_kind_keeps_digits_drops_natural_text(tmp_path: Path) -> None:
    engine = FakeOcrEngine(
        [
            {"text": "江湖大桥", "confidence": 0.9, "bbox": [0, 0, 20, 10]},
            {"text": "128号", "confidence": 0.91, "bbox": [21, 0, 40, 10]},
            {"text": "42", "confidence": 0.95, "bbox": [41, 0, 50, 10]},
        ]
    )
    observation = _recognize(
        tmp_path,
        inputs={"text_kind": "number"},
        engine=engine,
    )
    assert observation.ok is True
    assert observation.result is not None
    texts = [item["text"] for item in observation.result["items"]]
    assert texts == ["128号", "42"]
    assert observation.result["applied"]["text_kind"] == "number"


def test_invalid_text_kind_and_empty_items(tmp_path: Path) -> None:
    source = _solid(tmp_path / "blank.png")
    invalid = _recognize(tmp_path, source=source, inputs={"text_kind": "logo"})
    assert invalid.ok is False
    assert invalid.error_code == "invalid_text_kind"

    empty = _recognize(tmp_path, source=source, engine=FakeOcrEngine([]))
    assert empty.ok is True
    assert empty.result is not None
    assert empty.result["items"] == []
    assert empty.result["full_text"] == ""
    assert empty.artifacts == {}

    missing_file = _recognize(tmp_path, source=tmp_path / "nope.png", engine=FakeOcrEngine())
    assert missing_file.ok is False
    assert missing_file.error_code == "image_not_found"


def test_unknown_language_is_ignored(tmp_path: Path) -> None:
    observation = _recognize(
        tmp_path,
        inputs={"languages": ["zh", "tlh"]},
        engine=FakeOcrEngine(
            [{"text": "店", "confidence": 0.8, "bbox": [0, 0, 8, 8]}],
        ),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["languages"] == ["zh", "tlh"]
    assert observation.result["applied"]["ignored"]["languages"] == ["tlh"]


def test_named_region(tmp_path: Path) -> None:
    source = _solid(tmp_path / "named.png")
    observation = execute(
        "ocr_read",
        "recognize",
        purpose="命名区域",
        inputs={"image": str(source), "region": "bridge_tower_top"},
        ctx=RuntimeContext(
            extras={
                "artifact_dir": str(tmp_path / "named"),
                "named_regions": {"bridge_tower_top": [2, 2, 40, 20]},
                "ocr_engine": FakeOcrEngine(
                    [{"text": "塔", "confidence": 0.7, "bbox": [0, 0, 6, 6]}],
                ),
            }
        ),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["region"] == [2, 2, 40, 20]
    assert observation.result["items"][0]["bbox"] == [2, 2, 8, 8]


def test_decode_qr_payload(tmp_path: Path) -> None:
    qr = _make_qr(tmp_path / "qr.png", "geoagent-test")
    observation = _decode(tmp_path, source=qr)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["items"]
    assert observation.result["items"][0]["payload"] == "geoagent-test"
    assert observation.result["items"][0]["text"] == "geoagent-test"
    assert observation.result["items"][0]["code_type"] == "qr"
    assert observation.result["full_text"] == "geoagent-test"
    assert observation.result["applied"]["engine"] == "opencv_qr"


def test_decode_qr_in_region(tmp_path: Path) -> None:
    qr = _make_qr(tmp_path / "qr_small.png", "roi-qr", size=120)
    canvas = Image.new("RGB", (300, 220), (30, 30, 30))
    qr_im = Image.open(qr)
    canvas.paste(qr_im, (40, 30))
    composed = tmp_path / "composed.png"
    canvas.save(composed)
    observation = _decode(
        tmp_path,
        source=composed,
        inputs={"region": [40, 30, 160, 150], "code_types": ["qr"]},
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["full_text"] == "roi-qr"
    bbox = observation.result["items"][0]["bbox"]
    assert bbox[0] >= 40
    assert bbox[1] >= 30


def test_decode_barcode_unsupported_and_empty(tmp_path: Path) -> None:
    blank = _solid(tmp_path / "nocode.png")
    barcode_only = _decode(tmp_path, source=blank, inputs={"code_types": ["barcode"]})
    assert barcode_only.ok is False
    assert barcode_only.error_code == "unsupported_code_type"

    empty = _decode(tmp_path, source=blank)
    assert empty.ok is True
    assert empty.result is not None
    assert empty.result["items"] == []
    assert empty.result["full_text"] == ""

    qr = _make_qr(tmp_path / "mixed.png", "keep-qr")
    mixed = _decode(tmp_path, source=qr, inputs={"code_types": ["qr", "barcode"]})
    assert mixed.ok is True
    assert mixed.result is not None
    assert mixed.result["full_text"] == "keep-qr"
    assert mixed.result["applied"]["unsupported_code_types"] == ["barcode"]


def test_real_rapidocr_optional(tmp_path: Path) -> None:
    pytest.importorskip("rapidocr")
    image = Image.new("RGB", (240, 80), (255, 255, 255))
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((12, 24), "TEST 42", fill=(0, 0, 0), font=font)
    path = tmp_path / "hello.png"
    image.save(path)
    observation = execute(
        "ocr_read",
        "recognize",
        purpose="真实引擎",
        inputs={"image": str(path), "languages": ["en"]},
        ctx=RuntimeContext(extras={"artifact_dir": str(tmp_path / "real")}),
    )
    if not observation.ok and observation.error_code == "engine_unavailable":
        pytest.skip(observation.error or "RapidOCR 模型不可用")
    assert observation.ok is True
    assert observation.result is not None
    assert "corrected_text" not in observation.result
    joined = observation.result["full_text"].replace(" ", "").upper()
    if observation.result["items"]:
        assert any(ch.isalnum() for ch in joined)
