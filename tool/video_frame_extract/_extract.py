"""video_frame_extract 共享执行器：按媒体时间戳抽帧并登记图片 ID。

使用 imageio-ffmpeg 自带的 ffmpeg 做 output seeking（``-ss`` 在 ``-i`` 之后），
按 PTS 定位，而不是「秒数 × 标称帧率」换算帧号。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from tool._gate import allow_real_api as _allow_real_api
from tool.contract import Observation, RuntimeContext

_DEFAULT_FETCH_TIMEOUT_SEC = 60.0
_DEFAULT_MAX_BYTES = 100 * 1024 * 1024
_DEFAULT_USER_AGENT = "geoagent-dataset/1.0 (video_frame_extract; local)"
_VIDEO_KEYS = ("video", "video_id", "result_id", "链接", "视频")
_VIEW_KEYS = ("view", "视角", "方向")
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v", ".mpeg", ".mpg"}
_ASSUMPTIONS = [
    "抽帧按媒体时间戳定位，不是秒数乘标称帧率",
    "extracted_time_sec 是实际帧 PTS，可变帧率时可能与请求时间不同",
    "只能提取已有画面，不能判断该帧是否为正确题图",
    "view 只作记录，不改变抽帧位置",
]
_DURATION_RE = re.compile(
    r"Duration:\s*(?:(\d+):(\d+):(\d+(?:\.\d+)?)|N/A)",
    re.IGNORECASE,
)
_PTS_TIME_RE = re.compile(r"pts_time:(-?\d+(?:\.\d+)?)")
_SHOWINFO_SIZE_RE = re.compile(r"\bs:(\d+)x(\d+)\b")
_TIMECODE_RE = re.compile(
    r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$",
)

class ExtractInputError(Exception):
    """video / timestamps 等输入无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class ExtractFailedError(Exception):
    """单次抽帧失败（越界、解码失败或输出为空）。"""

    def __init__(self, message: str, error_code: str = "extract_failed") -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """默认下载器被闸门拒绝，或注入对象不合合同。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code

@dataclass(frozen=True)
class ExtractedFrame:
    """一次抽帧的媒体时间与画面尺寸。"""

    requested_time_sec: float
    extracted_time_sec: float
    width: int
    height: int

@runtime_checkable
class FrameExtractor(Protocol):
    """可注入的抽帧后端；测试用 extras['frame_extractor'] 替换默认 ffmpeg。"""

    def extract(
        self,
        video: Path,
        timestamp_sec: float,
        out_path: Path,
    ) -> ExtractedFrame:
        """按媒体时间戳写出一帧并返回实际 PTS 与尺寸。"""

@runtime_checkable
class VideoFetcher(Protocol):
    """可注入的远程视频下载器；测试用 extras['video_fetcher'] 替换。"""

    def fetch(self, url: str, dest: Path) -> Path:
        """把 URL 下载到 dest（已存在且非空则可复用），返回本地路径。"""

class FfmpegFrameExtractor:
    """用 imageio-ffmpeg 自带的 ffmpeg 按 PTS 精确抽一帧。"""

    name = "ffmpeg"

    def extract(
        self,
        video: Path,
        timestamp_sec: float,
        out_path: Path,
    ) -> ExtractedFrame:
        ffmpeg = _ffmpeg_exe()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-i",
            str(video),
            "-ss",
            f"{timestamp_sec:.3f}",
            "-frames:v",
            "1",
            "-vf",
            "showinfo",
            "-q:v",
            "2",
            "-y",
            str(out_path),
        ]
        timeout = max(30.0, float(timestamp_sec) * 3.0 + 15.0)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            out_path.unlink(missing_ok=True)
            raise ExtractFailedError(f"ffmpeg 抽帧失败: {exc}") from exc
        log = (completed.stderr or "") + (completed.stdout or "")
        if completed.returncode != 0 or not out_path.is_file() or out_path.stat().st_size <= 0:
            out_path.unlink(missing_ok=True)
            detail = log.strip()[-500:]
            raise ExtractFailedError(f"ffmpeg 抽帧失败: {detail or 'empty output'}")
        pts, width, height = _parse_showinfo(log)
        if width is None or height is None:
            with Image.open(out_path) as opened:
                width, height = opened.size
        extracted = pts if pts is not None else timestamp_sec
        return ExtractedFrame(
            requested_time_sec=timestamp_sec,
            extracted_time_sec=extracted,
            width=int(width),
            height=int(height),
        )

    def probe_duration(self, video: Path) -> float | None:
        """读取容器 Duration（秒）；探测失败时返回 None。"""

        ffmpeg = _ffmpeg_exe()
        command = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-i",
            str(video),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return _parse_duration((completed.stderr or "") + (completed.stdout or ""))

class HttpVideoFetcher:
    """HTTP(S) 下载远程视频；受 ALLOW_REAL_API 闸门约束。"""

    name = "http"

    def __init__(
        self,
        *,
        timeout_sec: float = _DEFAULT_FETCH_TIMEOUT_SEC,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        user_agent: str = _DEFAULT_USER_AGENT,
    ) -> None:
        self.timeout_sec = timeout_sec
        self.max_bytes = max_bytes
        self.user_agent = user_agent

    def fetch(self, url: str, dest: Path) -> Path:
        if not _allow_real_api():
            raise EngineUnavailableError(
                "ALLOW_REAL_API=false，禁止下载远程视频",
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and dest.stat().st_size > 0:
            return dest
        request = urllib.request.Request(url, method="GET")
        request.add_header("User-Agent", self.user_agent)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = _read_limited(response, self.max_bytes)
        except urllib.error.URLError as exc:
            raise ExtractInputError(
                f"无法下载视频: {getattr(exc, 'reason', exc)}",
                "video_not_found",
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise ExtractInputError(f"无法下载视频: {exc}", "video_not_found") from exc
        if body is None:
            raise ExtractInputError("远程视频超过大小上限", "video_not_found")
        dest.write_bytes(body)
        if dest.stat().st_size <= 0:
            dest.unlink(missing_ok=True)
            raise ExtractInputError("远程视频下载为空", "video_not_found")
        return dest

def execute_frame_retrieve(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """从已找到的视频按时间戳抽帧，登记图片 ID 并记录实际 PTS。"""

    del purpose
    try:
        video_ref = _parse_video_ref(inputs)
        timestamps = _parse_timestamps(inputs.get("timestamps"))
        view = _parse_view(inputs)
        video_path = _resolve_video(video_ref, ctx)
        extractor = _resolve_extractor(ctx)
    except ExtractInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

    duration = _probe_duration(extractor, video_path)
    artifact = _artifact_dir(ctx)
    frames: list[dict[str, Any]] = []
    image_ids: list[str] = []
    image_paths: list[str] = []
    missing: list[dict[str, Any]] = []

    for index, timestamp_sec in enumerate(timestamps):
        if duration is not None and timestamp_sec > duration:
            missing.append(
                {
                    "requested_time_sec": timestamp_sec,
                    "error": "时间点超出视频时长",
                }
            )
            continue
        out_path = artifact / f"frame_{index}_{_stamp(timestamp_sec)}.jpg"
        try:
            extracted = extractor.extract(video_path, timestamp_sec, out_path)
        except (ExtractFailedError, OSError, RuntimeError) as exc:
            out_path.unlink(missing_ok=True)
            missing.append(
                {
                    "requested_time_sec": timestamp_sec,
                    "error": str(exc),
                }
            )
            continue
        if not out_path.is_file() or out_path.stat().st_size <= 0:
            out_path.unlink(missing_ok=True)
            missing.append(
                {
                    "requested_time_sec": timestamp_sec,
                    "error": "抽帧输出为空",
                }
            )
            continue
        image_id, stored = _register_frame(out_path, ctx)
        frames.append(
            {
                "image_id": image_id,
                "requested_time_sec": timestamp_sec,
                "extracted_time_sec": extracted.extracted_time_sec,
                "width": extracted.width,
                "height": extracted.height,
                "source_video": video_ref,
            }
        )
        image_ids.append(image_id)
        image_paths.append(str(stored))

    if not frames:
        detail = missing[0]["error"] if missing else "未能提取任何帧"
        return _fail(str(detail), "extract_failed")

    applied: dict[str, Any] = {
        "timestamps": timestamps,
        "seek": "ffmpeg_timestamp",
    }
    if view is not None:
        applied["view"] = view

    result: dict[str, Any] = {
        "operation": "frame_retrieve",
        "video": video_ref,
        "frames": frames,
        "applied": applied,
        "assumptions": list(_ASSUMPTIONS),
    }
    if missing:
        result["missing"] = missing
    return Observation(
        ok=True,
        result=result,
        artifacts={
            "image_ids": image_ids,
            "image_paths": image_paths,
        },
    )

def _parse_video_ref(inputs: dict[str, Any]) -> str:
    for key in _VIDEO_KEYS:
        raw = inputs.get(key)
        if raw is None or raw == "":
            continue
        if not isinstance(raw, str):
            raise ExtractInputError("video 必须是字符串", "missing_input")
        token = raw.strip()
        if token:
            return token
    raise ExtractInputError("缺少必填输入 video", "missing_input")

def _parse_timestamps(raw: Any) -> list[float]:
    if raw is None or raw == "":
        raise ExtractInputError("缺少必填输入 timestamps", "missing_input")
    value: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        loaded = _try_json(stripped)
        if loaded is not None:
            value = loaded
        elif "," in stripped:
            value = [part.strip() for part in stripped.split(",") if part.strip()]
        else:
            value = [stripped]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ExtractInputError("timestamps 必须是时间点或时间点列表", "invalid_timestamp")
    times: list[float] = []
    for item in value:
        times.append(_parse_one_timestamp(item))
    return times

def _parse_one_timestamp(raw: Any) -> float:
    if isinstance(raw, bool) or raw is None:
        raise ExtractInputError("无法解析时间点", "invalid_timestamp")
    if isinstance(raw, (int, float)):
        if float(raw) < 0:
            raise ExtractInputError("时间点不能为负数", "invalid_timestamp")
        return float(raw)
    if not isinstance(raw, str) or not raw.strip():
        raise ExtractInputError("无法解析时间点", "invalid_timestamp")
    text = raw.strip()
    match = _TIMECODE_RE.match(text)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        total = hours * 3600 + minutes * 60 + seconds
        if total < 0:
            raise ExtractInputError("时间点不能为负数", "invalid_timestamp")
        return total
    try:
        value = float(text)
    except ValueError as exc:
        raise ExtractInputError(f"无法解析时间点: {text}", "invalid_timestamp") from exc
    if value < 0:
        raise ExtractInputError("时间点不能为负数", "invalid_timestamp")
    return value

def _parse_view(inputs: dict[str, Any]) -> str | None:
    for key in _VIEW_KEYS:
        raw = inputs.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None

def _resolve_video(token: str, ctx: RuntimeContext | None) -> Path:
    path = Path(token)
    if path.is_file():
        return path.resolve()
    url = _as_http_url(token)
    if url is not None:
        return _fetch_url(url, ctx)
    looked_up = _lookup_previous_media(token, ctx)
    if looked_up is not None:
        nested_url = _as_http_url(looked_up)
        if nested_url is not None:
            return _fetch_url(nested_url, ctx)
        nested = Path(looked_up)
        if nested.is_file():
            return nested.resolve()
        raise ExtractInputError(f"找不到视频: {looked_up}", "video_not_found")
    if _looks_like_path(token):
        raise ExtractInputError(f"找不到视频: {token}", "video_not_found")
    raise ExtractInputError(f"找不到搜索结果 {token}", "unknown_result_id")

def _lookup_previous_media(result_id: str, ctx: RuntimeContext | None) -> str | None:
    previous = ctx.previous_tool_result if ctx is not None else None
    rows = previous.get("results") if isinstance(previous, dict) else None
    if not isinstance(rows, list):
        return None
    for item in rows:
        if not isinstance(item, dict):
            continue
        identities = {
            str(item.get("result_id", "")).strip(),
            str(item.get("media_id", "")).strip(),
            str(item.get("video_id", "")).strip(),
        }
        if result_id not in identities:
            continue
        for key in ("source_url", "url"):
            raw = item.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        raise ExtractInputError(f"搜索结果 {result_id} 没有有效视频地址", "video_not_found")
    return None

def _fetch_url(url: str, ctx: RuntimeContext | None) -> Path:
    fetcher = _resolve_fetcher(ctx)
    dest = _download_dest(url, ctx)
    return Path(fetcher.fetch(url, dest)).resolve()

def _download_dest(url: str, ctx: RuntimeContext | None) -> Path:
    cache = _artifact_dir(ctx) / "video_cache"
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix not in _VIDEO_SUFFIXES:
        suffix = ".mp4"
    return cache / f"{digest}{suffix}"

def _as_http_url(token: str) -> str | None:
    parsed = urllib.parse.urlparse(token)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return token
    return None

def _looks_like_path(token: str) -> bool:
    path = Path(token)
    if path.suffix.lower() in _VIDEO_SUFFIXES:
        return True
    return "/" in token or "\\" in token or path.is_absolute()

def _resolve_extractor(ctx: RuntimeContext | None) -> FrameExtractor:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("frame_extractor")
    if injected is None:
        return FfmpegFrameExtractor()
    if not isinstance(injected, FrameExtractor):
        raise EngineUnavailableError("frame_extractor 必须提供 extract(video, timestamp_sec, out_path)")
    return injected

def _resolve_fetcher(ctx: RuntimeContext | None) -> VideoFetcher:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("video_fetcher")
    if injected is None:
        return HttpVideoFetcher()
    if not isinstance(injected, VideoFetcher):
        raise EngineUnavailableError("video_fetcher 必须提供 fetch(url, dest)")
    return injected

def _probe_duration(extractor: FrameExtractor, video: Path) -> float | None:
    probe = getattr(extractor, "probe_duration", None)
    if not callable(probe):
        return None
    try:
        value = probe(video)
    except (OSError, RuntimeError, ExtractFailedError):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None

def _register_frame(path: Path, ctx: RuntimeContext | None) -> tuple[str, Path]:
    store = ctx.image_store if ctx is not None else None
    resolved = Path(path).resolve()
    if store is not None:
        image_id = store.register(resolved)
        return image_id, store.path_for(image_id)
    return resolved.stem, resolved

def _artifact_dir(ctx: RuntimeContext | None) -> Path:
    extras = ctx.extras if ctx is not None else {}
    raw = extras.get("artifact_dir")
    if raw:
        directory = Path(str(raw))
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    return Path(tempfile.mkdtemp(prefix="video_frame_extract_"))

def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ExtractFailedError("未安装 imageio-ffmpeg，无法抽帧") from exc
    return str(imageio_ffmpeg.get_ffmpeg_exe())

def _parse_showinfo(text: str) -> tuple[float | None, int | None, int | None]:
    pts: float | None = None
    width: int | None = None
    height: int | None = None
    for line in text.splitlines():
        if "pts_time:" not in line:
            continue
        match_pts = _PTS_TIME_RE.search(line)
        if match_pts:
            pts = float(match_pts.group(1))
        match_size = _SHOWINFO_SIZE_RE.search(line)
        if match_size:
            width = int(match_size.group(1))
            height = int(match_size.group(2))
    return pts, width, height

def _parse_duration(text: str) -> float | None:
    match = _DURATION_RE.search(text)
    if match is None or match.group(1) is None:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds

def _read_limited(response: Any, max_bytes: int) -> bytes | None:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)

def _stamp(timestamp_sec: float) -> str:
    return f"t{timestamp_sec:.3f}".replace(".", "p")

def _try_json(text: str) -> Any:
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def _fail(error: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=error, error_code=error_code)
