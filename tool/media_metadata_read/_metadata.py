"""media_metadata_read 的本地读取：EXIF/GPS、文件系统时间与容器探测。

拍摄时间、文件修改时间、视频编码时间分栏返回；缺字段如实标 missing，
不得用文件创建/修改时间或容器时间填拍摄时间。
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import IFD, Base, GPS

from tool.contract import Observation, RuntimeContext
from tool.runtime.image_store import ImageResolveError, resolve_image_ref

_EXIF_DEFAULT_FIELDS = (
    "datetime_original",
    "datetime_digitized",
    "make",
    "model",
    "gps_latitude",
    "gps_longitude",
    "gps_altitude",
    "width",
    "height",
    "orientation",
)
_FILE_DEFAULT_FIELDS = (
    "format",
    "size_bytes",
    "width",
    "height",
    "duration_sec",
    "video_codec",
    "file_mtime",
    "encoded_date",
)
_FIELD_ALIASES = {
    "datetime_original": "datetime_original",
    "datetimeoriginal": "datetime_original",
    "date_time_original": "datetime_original",
    "shooting_time": "datetime_original",
    "taken_at": "datetime_original",
    "capture_time": "datetime_original",
    "拍摄时间": "datetime_original",
    "datetime_digitized": "datetime_digitized",
    "datetimedigitized": "datetime_digitized",
    "date_time_digitized": "datetime_digitized",
    "数字化时间": "datetime_digitized",
    "make": "make",
    "camera_make": "make",
    "厂商": "make",
    "model": "model",
    "camera": "model",
    "camera_model": "model",
    "型号": "model",
    "gps": "gps",
    "gps_latitude": "gps_latitude",
    "gpslatitude": "gps_latitude",
    "latitude": "gps_latitude",
    "纬度": "gps_latitude",
    "gps_longitude": "gps_longitude",
    "gpslongitude": "gps_longitude",
    "longitude": "gps_longitude",
    "经度": "gps_longitude",
    "gps_altitude": "gps_altitude",
    "gpsaltitude": "gps_altitude",
    "altitude": "gps_altitude",
    "海拔": "gps_altitude",
    "width": "width",
    "imagewidth": "width",
    "image_width": "width",
    "宽度": "width",
    "height": "height",
    "imageheight": "height",
    "image_height": "height",
    "高度": "height",
    "orientation": "orientation",
    "方向": "orientation",
    "format": "format",
    "container": "format",
    "mime": "format",
    "格式": "format",
    "size_bytes": "size_bytes",
    "size": "size_bytes",
    "filesize": "size_bytes",
    "大小": "size_bytes",
    "duration_sec": "duration_sec",
    "duration": "duration_sec",
    "时长": "duration_sec",
    "video_codec": "video_codec",
    "codec": "video_codec",
    "encoding": "video_codec",
    "编码": "video_codec",
    "file_mtime": "file_mtime",
    "mtime": "file_mtime",
    "modified": "file_mtime",
    "filemodifydate": "file_mtime",
    "file_modify_date": "file_mtime",
    "文件修改时间": "file_mtime",
    "encoded_date": "encoded_date",
    "creation_time": "encoded_date",
    "mediacreatedate": "encoded_date",
    "encoded_time": "encoded_date",
    "视频编码时间": "encoded_date",
    "file_ctime": "file_ctime",
    "ctime": "file_ctime",
    "filecreatedate": "file_ctime",
    "file_create_date": "file_ctime",
    "文件创建时间": "file_ctime",
}
_GPS_FIELDS = ("gps_latitude", "gps_longitude", "gps_altitude")
_ASSUMPTIONS = [
    "EXIF/GPS 可被编辑，不能当作不可改变的拍摄事实",
    "拍摄时间缺失时不会用文件创建或修改时间替代",
    "视频编码/容器时间不等于拍摄时间",
]
_FFMPEG_TIMEOUT_SEC = 30.0
_DURATION_RE = re.compile(
    r"Duration:\s*(?:(\d+):(\d+):(\d+(?:\.\d+)?)|N/A)",
    re.IGNORECASE,
)
_INPUT_FORMAT_RE = re.compile(r"Input #\d+,\s*([^,]+)")
_VIDEO_STREAM_RE = re.compile(r"Video:\s*([^,\s(]+)")
_VIDEO_SIZE_RE = re.compile(r"(\d+)x(\d+)")
_CREATION_TIME_RE = re.compile(r"^\s*creation_time\s*:\s*(\S+)", re.MULTILINE)


class MetadataInputError(Exception):
    """file / fields 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@runtime_checkable
class MediaProbe(Protocol):
    """可注入的容器探测；测试用 extras['media_probe'] 替换默认 ffmpeg。"""

    def probe(self, path: Path) -> dict[str, Any]:
        """返回 format / video_codec / encoded_date / duration_sec / width / height。"""


class FfmpegMediaProbe:
    """用 imageio-ffmpeg 自带的 ffmpeg 读容器信息；不依赖系统 ffprobe。"""

    name = "ffmpeg"

    def probe(self, path: Path) -> dict[str, Any]:
        try:
            import imageio_ffmpeg  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("未安装 imageio-ffmpeg，无法探测容器元数据") from exc
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-hide_banner",
            "-i",
            str(path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_FFMPEG_TIMEOUT_SEC,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"ffmpeg 探测失败: {exc}") from exc
        return _parse_ffmpeg_stderr((completed.stderr or "") + (completed.stdout or ""))


def execute_exif(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """读取图片 EXIF/GPS；非图片返回 unsupported_format。"""

    del purpose
    return _run("exif", inputs, ctx)


def execute_file(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """读取容器、编码与文件系统元数据；拍摄时间若存在则只来自 EXIF。"""

    del purpose
    return _run("file", inputs, ctx)


def _run(
    operation: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    try:
        file_id, path = _resolve_file(inputs, ctx)
        selected, ignored = _parse_fields(inputs.get("fields"), operation)
        snapshot = _read_snapshot(path, operation, ctx)
    except ImageResolveError as exc:
        return _fail(str(exc), exc.error_code)
    except MetadataInputError as exc:
        return _fail(str(exc), exc.error_code)

    if operation == "exif" and snapshot.image is None:
        return _fail("不是可读取的图片，无法读取 EXIF", "unsupported_format")

    records = [_field_record(name, snapshot) for name in selected]
    times = {
        "datetime_original": _time_slot(
            snapshot.datetime_original, "exif"
        ),
        "file_mtime": _time_slot(snapshot.file_mtime, "filesystem"),
        "encoded_date": _time_slot(snapshot.encoded_date, "container"),
    }
    result: dict[str, Any] = {
        "operation": operation,
        "file_id": file_id,
        "fields": records,
        "times": times,
        "missing": [item["name"] for item in records if not item["present"]],
        "applied": _applied(operation, selected, ignored, snapshot),
        "assumptions": list(_ASSUMPTIONS),
    }
    if operation == "exif":
        result["gps"] = {
            "present": snapshot.gps_latitude is not None
            and snapshot.gps_longitude is not None,
            "latitude": snapshot.gps_latitude,
            "longitude": snapshot.gps_longitude,
            "altitude": snapshot.gps_altitude,
        }
    return Observation(
        ok=True,
        result=result,
        artifacts={"file_path": str(path)},
    )


class _Snapshot:
    """一次读取得到的规范字段；缺失保持 None，不做时间回退。"""

    def __init__(self) -> None:
        self.image: Image.Image | None = None
        self.engine: str = "pillow"
        self.datetime_original: str | None = None
        self.datetime_digitized: str | None = None
        self.make: str | None = None
        self.model: str | None = None
        self.gps_latitude: float | None = None
        self.gps_longitude: float | None = None
        self.gps_altitude: float | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.orientation: int | None = None
        self.format: str | None = None
        self.size_bytes: int | None = None
        self.duration_sec: float | None = None
        self.video_codec: str | None = None
        self.file_mtime: str | None = None
        self.file_ctime: str | None = None
        self.encoded_date: str | None = None


def _read_snapshot(
    path: Path,
    operation: str,
    ctx: RuntimeContext | None,
) -> _Snapshot:
    snapshot = _Snapshot()
    stat = path.stat()
    snapshot.size_bytes = int(stat.st_size)
    snapshot.file_mtime = _fs_timestamp(stat.st_mtime)
    snapshot.file_ctime = _fs_timestamp(stat.st_ctime)

    loaded = _open_still_image(path)
    if loaded is not None:
        image, exif, fmt = loaded
        snapshot.image = image
        snapshot.width, snapshot.height = image.size
        snapshot.format = fmt
        _fill_exif(snapshot, exif)
        snapshot.engine = "pillow"
        return snapshot

    if operation == "exif":
        return snapshot

    _fill_video(snapshot, path)
    probe = _safe_probe(path, ctx)
    if probe:
        snapshot.engine = f"opencv+{probe[1]}"
        _apply_container(snapshot, probe[0])
    else:
        snapshot.engine = "opencv"
        if snapshot.format is None:
            snapshot.format = path.suffix.lstrip(".").lower() or None
    return snapshot


def _fill_exif(snapshot: _Snapshot, exif: Image.Exif) -> None:
    snapshot.make = _as_text(exif.get(Base.Make))
    snapshot.model = _as_text(exif.get(Base.Model))
    orientation = exif.get(Base.Orientation)
    if isinstance(orientation, int):
        snapshot.orientation = orientation
    elif orientation is not None:
        parsed = _to_int(orientation)
        snapshot.orientation = parsed

    exif_ifd = _ifd(exif, IFD.Exif)
    snapshot.datetime_original = _as_text(exif_ifd.get(Base.DateTimeOriginal))
    if snapshot.datetime_original is None:
        snapshot.datetime_original = _as_text(exif.get(Base.DateTimeOriginal))
    snapshot.datetime_digitized = _as_text(exif_ifd.get(Base.DateTimeDigitized))
    if snapshot.datetime_digitized is None:
        snapshot.datetime_digitized = _as_text(exif.get(Base.DateTimeDigitized))

    gps_ifd = _ifd(exif, IFD.GPSInfo)
    snapshot.gps_latitude = _gps_coordinate(
        gps_ifd.get(GPS.GPSLatitude),
        gps_ifd.get(GPS.GPSLatitudeRef),
    )
    snapshot.gps_longitude = _gps_coordinate(
        gps_ifd.get(GPS.GPSLongitude),
        gps_ifd.get(GPS.GPSLongitudeRef),
    )
    snapshot.gps_altitude = _gps_altitude(
        gps_ifd.get(GPS.GPSAltitude),
        gps_ifd.get(GPS.GPSAltitudeRef),
    )


def _fill_video(snapshot: _Snapshot, path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width > 0:
            snapshot.width = width
        if height > 0:
            snapshot.height = height
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0 and frames > 0:
            snapshot.duration_sec = frames / fps
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = _fourcc_to_text(fourcc)
        if codec:
            snapshot.video_codec = codec
        snapshot.format = path.suffix.lstrip(".").lower() or snapshot.format
    finally:
        capture.release()


def _apply_container(snapshot: _Snapshot, payload: dict[str, Any]) -> None:
    fmt = _as_text(payload.get("format"))
    if fmt:
        snapshot.format = fmt
    codec = _as_text(payload.get("video_codec"))
    if codec:
        snapshot.video_codec = codec
    encoded = _as_text(payload.get("encoded_date"))
    if encoded:
        snapshot.encoded_date = encoded
    duration = _to_float(payload.get("duration_sec"))
    if duration is not None and duration >= 0:
        snapshot.duration_sec = duration
    width = _to_int(payload.get("width"))
    if width is not None and width > 0:
        snapshot.width = width
    height = _to_int(payload.get("height"))
    if height is not None and height > 0:
        snapshot.height = height


def _safe_probe(
    path: Path,
    ctx: RuntimeContext | None,
) -> tuple[dict[str, Any], str] | None:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("media_probe", _MISSING)
    if injected is False:
        return None
    if injected is not _MISSING and injected is not None:
        if not isinstance(injected, MediaProbe):
            raise MetadataInputError(
                "media_probe 必须提供 probe(path)",
                "engine_unavailable",
            )
        name = str(getattr(injected, "name", "injected"))
        try:
            payload = injected.probe(path)
        except Exception as exc:
            raise MetadataInputError(
                f"容器探测失败: {exc}",
                "engine_unavailable",
            ) from exc
        if not isinstance(payload, dict):
            raise MetadataInputError("media_probe 必须返回对象", "engine_unavailable")
        return payload, name
    try:
        payload = FfmpegMediaProbe().probe(path)
    except Exception:
        return None
    return payload, "ffmpeg"


_MISSING = object()


def _resolve_file(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> tuple[str, Path]:
    raw = inputs.get("file")
    if not isinstance(raw, str) or not raw.strip():
        raise MetadataInputError("缺少必填输入 file", "missing_input")
    return resolve_image_ref(raw, ctx)


def _parse_fields(raw: Any, operation: str) -> tuple[list[str], list[str]]:
    if raw is None:
        defaults = _EXIF_DEFAULT_FIELDS if operation == "exif" else _FILE_DEFAULT_FIELDS
        return list(defaults), []
    tokens = _parse_string_list(raw)
    selected: list[str] = []
    ignored: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        canonical = _canonical_field(token)
        if canonical is None:
            ignored.append(token)
            continue
        names = _GPS_FIELDS if canonical == "gps" else (canonical,)
        for name in names:
            if name not in seen:
                selected.append(name)
                seen.add(name)
    return selected, ignored


def _parse_string_list(raw: Any) -> list[str]:
    value: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        loaded = _try_json(stripped)
        if isinstance(loaded, list):
            value = loaded
        elif "," in stripped:
            value = [part.strip() for part in stripped.split(",") if part.strip()]
        else:
            value = [stripped] if stripped else []
    if not isinstance(value, list):
        raise MetadataInputError("fields 必须是字符串或字符串列表", "invalid_input")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise MetadataInputError("fields 必须是字符串或字符串列表", "invalid_input")
        if item.strip():
            items.append(item.strip())
    return items


def _canonical_field(token: str) -> str | None:
    stripped = token.strip()
    lowered = stripped.lower().replace("-", "_").replace(" ", "_")
    return _FIELD_ALIASES.get(stripped) or _FIELD_ALIASES.get(lowered)


def _field_record(name: str, snapshot: _Snapshot) -> dict[str, Any]:
    value = getattr(snapshot, name)
    source = {
        "datetime_original": "exif",
        "datetime_digitized": "exif",
        "make": "exif",
        "model": "exif",
        "gps_latitude": "exif",
        "gps_longitude": "exif",
        "gps_altitude": "exif",
        "orientation": "exif",
        "file_mtime": "filesystem",
        "file_ctime": "filesystem",
        "size_bytes": "filesystem",
        "encoded_date": "container",
        "duration_sec": "container",
        "video_codec": "container",
        "format": "container",
        "width": "container",
        "height": "container",
    }.get(name, "container")
    present = value is not None
    return {
        "name": name,
        "present": present,
        "value": value if present else None,
        "source": source,
    }


def _time_slot(value: str | None, source: str) -> dict[str, Any]:
    present = value is not None and value != ""
    return {
        "present": present,
        "value": value if present else None,
        "source": source,
    }


def _applied(
    operation: str,
    selected: list[str],
    ignored: list[str],
    snapshot: _Snapshot,
) -> dict[str, Any]:
    applied: dict[str, Any] = {
        "operation": operation,
        "fields": list(selected),
        "engine": snapshot.engine,
    }
    if ignored:
        applied["ignored"] = {"fields": ignored}
    return applied


def _open_still_image(path: Path) -> tuple[Image.Image, Image.Exif, str | None] | None:
    try:
        with Image.open(path) as opened:
            opened.load()
            copied = opened.copy()
            return copied, opened.getexif(), opened.format
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return None


def _ifd(exif: Image.Exif, ifd: IFD) -> dict[int, Any]:
    try:
        loaded = exif.get_ifd(ifd)
    except Exception:
        return {}
    return dict(loaded) if loaded else {}


def _gps_coordinate(values: Any, ref: Any) -> float | None:
    if isinstance(values, (int, float)):
        degrees = float(values)
    elif isinstance(values, (tuple, list)):
        parts = [_to_float(item) for item in values]
        if any(part is None for part in parts):
            return None
        if len(parts) == 3:
            degrees = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
        elif len(parts) == 1:
            degrees = parts[0]
        else:
            return None
    else:
        degrees = _to_float(values)
        if degrees is None:
            return None
    token = (_as_text(ref) or "").upper()
    if token in {"S", "W"} and degrees > 0:
        return -degrees
    if token in {"N", "E"} and degrees < 0:
        return abs(degrees)
    return degrees


def _gps_altitude(value: Any, ref: Any) -> float | None:
    altitude = _to_float(value)
    if altitude is None:
        return None
    if _altitude_below_sea(ref):
        return -abs(altitude)
    return altitude


def _altitude_below_sea(ref: Any) -> bool:
    if ref in {1, "1"}:
        return True
    if isinstance(ref, (bytes, bytearray)):
        return len(ref) > 0 and ref[0] == 1
    text = _as_text(ref)
    return text == "1"


def _fs_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _as_text(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace").rstrip("\x00").strip()
        return text or None
    if isinstance(value, (int, float)):
        return None
    text = str(value).strip()
    return text or None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        denominator = float(value.denominator)
        if denominator == 0:
            return None
        return float(value.numerator) / denominator
    if isinstance(value, (tuple, list)) and len(value) == 2:
        denominator = _to_float(value[1])
        numerator = _to_float(value[0])
        if numerator is None or denominator in (None, 0.0):
            return None
        return numerator / denominator
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(round(number))


def _fourcc_to_text(value: int) -> str | None:
    if value <= 0:
        return None
    chars = "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
    text = "".join(ch for ch in chars if ch.isprintable()).strip()
    return text or None


def _parse_ffmpeg_stderr(text: str) -> dict[str, Any]:
    header = _input_header(text)
    payload: dict[str, Any] = {}
    match_format = _INPUT_FORMAT_RE.search(header)
    if match_format:
        payload["format"] = match_format.group(1).strip()
    match_time = _CREATION_TIME_RE.search(header)
    if match_time:
        payload["encoded_date"] = match_time.group(1).strip()
    match_duration = _DURATION_RE.search(header)
    if match_duration and match_duration.group(1) is not None:
        hours = int(match_duration.group(1))
        minutes = int(match_duration.group(2))
        seconds = float(match_duration.group(3))
        payload["duration_sec"] = hours * 3600 + minutes * 60 + seconds
    match_codec = _VIDEO_STREAM_RE.search(header)
    if match_codec:
        payload["video_codec"] = match_codec.group(1).strip()
    match_size = _VIDEO_SIZE_RE.search(header)
    if match_size:
        payload["width"] = int(match_size.group(1))
        payload["height"] = int(match_size.group(2))
    return payload


def _input_header(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("Stream mapping:") or line.startswith("Output #"):
            break
        if line.startswith("At least one output"):
            break
        lines.append(line)
    return "\n".join(lines)


def _try_json(text: str) -> Any:
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _fail(error: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=error, error_code=error_code)
