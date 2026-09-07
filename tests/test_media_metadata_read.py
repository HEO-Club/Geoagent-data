"""media_metadata_read 本地元数据读取测试；禁止真实付费 API。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image
from PIL.ExifTags import IFD, Base, GPS
from PIL.TiffImagePlugin import IFDRational

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.runtime import FilesystemImageStore


class FakeMediaProbe:
    """测试替身：返回预设容器字段。"""

    name = "fake"

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = dict(payload or {})

    def probe(self, path: Path) -> dict[str, Any]:
        del path
        return dict(self.payload)


def _jpeg_with_exif(
    path: Path,
    *,
    datetime_original: str = "2020:01:15 08:30:00",
    datetime_digitized: str = "2020:01:15 08:31:00",
    make: str = "TestMake",
    model: str = "TestModel",
    gps: bool = True,
) -> Path:
    image = Image.new("RGB", (32, 24), (12, 34, 56))
    exif = Image.Exif()
    exif[Base.Make] = make
    exif[Base.Model] = model
    exif[Base.Orientation] = 1
    exif_ifd = exif.get_ifd(IFD.Exif)
    exif_ifd[Base.DateTimeOriginal] = datetime_original
    exif_ifd[Base.DateTimeDigitized] = datetime_digitized
    if gps:
        gps_ifd = exif.get_ifd(IFD.GPSInfo)
        gps_ifd[GPS.GPSLatitudeRef] = "N"
        gps_ifd[GPS.GPSLatitude] = (
            IFDRational(39, 1),
            IFDRational(54, 1),
            IFDRational(0, 1),
        )
        gps_ifd[GPS.GPSLongitudeRef] = "E"
        gps_ifd[GPS.GPSLongitude] = (
            IFDRational(116, 1),
            IFDRational(24, 1),
            IFDRational(0, 1),
        )
        gps_ifd[GPS.GPSAltitudeRef] = 0
        gps_ifd[GPS.GPSAltitude] = IFDRational(50, 1)
    image.save(path, format="JPEG", exif=exif)
    return path


def _png(path: Path, width: int = 20, height: int = 10) -> Path:
    Image.new("RGB", (width, height), (1, 2, 3)).save(path)
    return path


def _video(path: Path, frames: int = 10, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :] = (index * 12, 40, 80)
        writer.write(frame)
    writer.release()
    return path


def _exif(
    tmp_path: Path,
    *,
    source: Path | None = None,
    inputs: dict[str, object] | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    path = source if source is not None else _jpeg_with_exif(tmp_path / "src.jpg")
    payload: dict[str, object] = {"file": str(path)}
    if inputs:
        payload.update(inputs)
    return execute(
        "media_metadata_read",
        "exif",
        purpose="读 EXIF",
        inputs=payload,
        ctx=ctx,
    )


def _file(
    tmp_path: Path,
    *,
    source: Path,
    inputs: dict[str, object] | None = None,
    probe: FakeMediaProbe | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {"file": str(source)}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    if runtime is None:
        extras: dict[str, Any] = {}
        if probe is not None:
            extras["media_probe"] = probe
        runtime = RuntimeContext(extras=extras)
    elif probe is not None:
        runtime.extras["media_probe"] = probe
    return execute(
        "media_metadata_read",
        "file",
        purpose="读文件元数据",
        inputs=payload,
        ctx=runtime,
    )


def _by_name(fields: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matched = [item for item in fields if item["name"] == name]
    assert matched, name
    return matched[0]


def test_exif_datetime_and_gps(tmp_path: Path) -> None:
    source = _jpeg_with_exif(tmp_path / "geo.jpg")
    observation = _exif(tmp_path, source=source)
    assert observation.ok is True
    assert observation.result is not None
    result = observation.result
    original = result["times"]["datetime_original"]
    assert original["present"] is True
    assert original["value"] == "2020:01:15 08:30:00"
    assert original["source"] == "exif"
    assert result["times"]["file_mtime"]["source"] == "filesystem"
    assert result["times"]["file_mtime"]["value"] != original["value"]
    assert result["gps"]["present"] is True
    assert result["gps"]["latitude"] == pytest.approx(39.9)
    assert result["gps"]["longitude"] == pytest.approx(116.4)
    assert result["gps"]["altitude"] == pytest.approx(50.0)
    assert _by_name(result["fields"], "make")["value"] == "TestMake"
    assert _by_name(result["fields"], "model")["value"] == "TestModel"
    joined = "".join(result["assumptions"])
    assert "可被编辑" in joined
    assert "不会用文件创建或修改时间替代" in joined


def test_png_without_exif_does_not_fallback_to_mtime(tmp_path: Path) -> None:
    source = _png(tmp_path / "plain.png")
    observation = _exif(tmp_path, source=source)
    assert observation.ok is True
    assert observation.result is not None
    times = observation.result["times"]
    assert times["datetime_original"]["present"] is False
    assert times["datetime_original"]["value"] is None
    assert times["datetime_original"]["source"] == "exif"
    assert times["file_mtime"]["present"] is True
    assert times["file_mtime"]["value"] is not None
    assert times["file_mtime"]["value"] != times["datetime_original"]["value"]
    assert "datetime_original" in observation.result["missing"]
    assert observation.result["gps"]["present"] is False


def test_fields_filter_and_unknown_ignored(tmp_path: Path) -> None:
    source = _jpeg_with_exif(tmp_path / "src.jpg")
    observation = _exif(
        tmp_path,
        source=source,
        inputs={"fields": ["拍摄时间", "make", "not_a_real_field"]},
    )
    assert observation.ok is True
    assert observation.result is not None
    names = [item["name"] for item in observation.result["fields"]]
    assert names == ["datetime_original", "make"]
    assert observation.result["applied"]["ignored"]["fields"] == ["not_a_real_field"]
    gps_only = _exif(tmp_path, source=source, inputs={"fields": "gps"})
    assert gps_only.ok is True
    assert gps_only.result is not None
    gps_names = [item["name"] for item in gps_only.result["fields"]]
    assert gps_names == ["gps_latitude", "gps_longitude", "gps_altitude"]


def test_current_image_and_missing_inputs(tmp_path: Path) -> None:
    source = _jpeg_with_exif(tmp_path / "src.jpg")
    store = FilesystemImageStore(tmp_path / "store")
    image_id = store.register(source)
    current = execute(
        "media_metadata_read",
        "exif",
        purpose="当前图",
        inputs={"file": "$current_image"},
        ctx=RuntimeContext(current_image=image_id, image_store=store),
    )
    assert current.ok is True
    assert current.result is not None
    assert current.result["times"]["datetime_original"]["value"] == "2020:01:15 08:30:00"

    missing = execute("media_metadata_read", "exif", purpose="缺文件", inputs={})
    assert missing.ok is False
    assert missing.error_code == "missing_input"

    missing_file = _exif(tmp_path, source=tmp_path / "nope.jpg")
    assert missing_file.ok is False
    assert missing_file.error_code == "image_not_found"


def test_file_video_duration_and_missing_encoded_date(tmp_path: Path) -> None:
    source = _video(tmp_path / "clip.mp4")
    observation = _file(tmp_path, source=source, probe=FakeMediaProbe())
    assert observation.ok is True
    assert observation.result is not None
    result = observation.result
    assert result["times"]["encoded_date"]["present"] is False
    assert result["times"]["encoded_date"]["source"] == "container"
    assert result["times"]["file_mtime"]["present"] is True
    assert _by_name(result["fields"], "width")["value"] == 32
    assert _by_name(result["fields"], "height")["value"] == 24
    duration = _by_name(result["fields"], "duration_sec")["value"]
    assert duration == pytest.approx(1.0)
    joined = "".join(result["assumptions"])
    assert "视频编码" in joined


def test_injected_probe_encoded_date_not_used_as_shooting_time(tmp_path: Path) -> None:
    source = _video(tmp_path / "clip.mp4")
    observation = _file(
        tmp_path,
        source=source,
        probe=FakeMediaProbe({"encoded_date": "2018-06-01T00:00:00Z", "video_codec": "h264"}),
    )
    assert observation.ok is True
    assert observation.result is not None
    times = observation.result["times"]
    assert times["encoded_date"]["present"] is True
    assert times["encoded_date"]["value"] == "2018-06-01T00:00:00Z"
    assert times["encoded_date"]["source"] == "container"
    assert times["datetime_original"]["present"] is False
    assert times["datetime_original"]["value"] is None
    assert times["file_mtime"]["value"] != times["encoded_date"]["value"]
    assert _by_name(observation.result["fields"], "video_codec")["value"] == "h264"


def test_exif_rejects_video(tmp_path: Path) -> None:
    source = _video(tmp_path / "clip.mp4")
    observation = _exif(tmp_path, source=source)
    assert observation.ok is False
    assert observation.error_code == "unsupported_format"


def test_file_ctime_does_not_fill_datetime_original(tmp_path: Path) -> None:
    source = _png(tmp_path / "plain.png")
    observation = _file(
        tmp_path,
        source=source,
        inputs={"fields": ["datetime_original", "file_ctime", "file_mtime"]},
    )
    assert observation.ok is True
    assert observation.result is not None
    original = _by_name(observation.result["fields"], "datetime_original")
    ctime = _by_name(observation.result["fields"], "file_ctime")
    mtime = _by_name(observation.result["fields"], "file_mtime")
    assert original["present"] is False
    assert original["source"] == "exif"
    assert ctime["present"] is True
    assert ctime["source"] == "filesystem"
    assert mtime["source"] == "filesystem"
    assert original["value"] not in {ctime["value"], mtime["value"]}
