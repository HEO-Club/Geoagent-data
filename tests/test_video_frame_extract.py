"""video_frame_extract 执行器测试；禁止真实外网。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.runtime import FilesystemImageStore
from tool.video_frame_extract._extract import ExtractFailedError, ExtractedFrame


class FakeFrameExtractor:
    """测试替身：按秒级时间戳写图，可返回与请求不同的实际 PTS。"""

    name = "fake"

    def __init__(
        self,
        *,
        actual_offset: float = 0.0,
        max_time: float | None = None,
        size: tuple[int, int] = (32, 24),
        duration_sec: float | None = None,
    ) -> None:
        self.actual_offset = actual_offset
        self.max_time = max_time
        self.size = size
        self.duration_sec = duration_sec
        self.calls: list[tuple[str, float]] = []

    def extract(
        self,
        video: Path,
        timestamp_sec: float,
        out_path: Path,
    ) -> ExtractedFrame:
        self.calls.append((str(video), timestamp_sec))
        if self.max_time is not None and timestamp_sec > self.max_time:
            raise ExtractFailedError(f"时间点超出视频: {timestamp_sec}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", self.size, (10, 20, 30)).save(out_path, format="JPEG")
        return ExtractedFrame(
            requested_time_sec=timestamp_sec,
            extracted_time_sec=timestamp_sec + self.actual_offset,
            width=self.size[0],
            height=self.size[1],
        )

    def probe_duration(self, video: Path) -> float | None:
        del video
        return self.duration_sec


class FakeVideoFetcher:
    """测试替身：把预置本地视频复制到 dest，不访问网络。"""

    name = "fake"

    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls: list[tuple[str, str]] = []

    def fetch(self, url: str, dest: Path) -> Path:
        self.calls.append((url, str(dest)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.source.read_bytes())
        return dest


def _video(path: Path, frames: int = 10, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :] = (index * 12, 40, 80)
        writer.write(frame)
    writer.release()
    return path


def _retrieve(
    *,
    video: Path | None = None,
    inputs: dict[str, object] | None = None,
    extractor: FakeFrameExtractor | None = None,
    fetcher: FakeVideoFetcher | None = None,
    ctx: RuntimeContext | None = None,
    tmp_path: Path | None = None,
) -> Observation:
    payload: dict[str, object] = {}
    if video is not None:
        payload["video"] = str(video)
        payload["timestamps"] = 0.2
    if inputs:
        payload.update(inputs)
    runtime = ctx
    extras: dict[str, Any]
    if runtime is None:
        extras = {}
        if tmp_path is not None:
            extras["artifact_dir"] = str(tmp_path / "out")
        runtime = RuntimeContext(extras=extras)
    else:
        extras = runtime.extras
        if tmp_path is not None and "artifact_dir" not in extras:
            extras["artifact_dir"] = str(tmp_path / "out")
    if extractor is not None:
        extras["frame_extractor"] = extractor
    if fetcher is not None:
        extras["video_fetcher"] = fetcher
    return execute(
        "video_frame_extract",
        "frame_retrieve",
        purpose="抽取指定时刻画面",
        inputs=payload,
        ctx=runtime,
    )


def test_missing_video_and_timestamps_are_missing_input() -> None:
    empty = _retrieve(inputs={})
    assert empty.ok is False
    assert empty.error_code == "missing_input"

    no_time = _retrieve(inputs={"video": "clip.mp4"})
    assert no_time.ok is False
    assert no_time.error_code == "missing_input"


def test_ffmpeg_extracts_frame_by_seconds_and_timecode(tmp_path: Path) -> None:
    source = _video(tmp_path / "clip.mp4")
    store = FilesystemImageStore(tmp_path / "store")
    ctx = RuntimeContext(
        extras={"artifact_dir": str(tmp_path / "out")},
        image_store=store,
    )
    observation = _retrieve(
        video=source,
        inputs={"timestamps": ["00:00:00.300", 0.5]},
        ctx=ctx,
        tmp_path=tmp_path,
    )
    assert observation.ok is True
    assert observation.result is not None
    frames = observation.result["frames"]
    assert len(frames) == 2
    for item in frames:
        assert item["width"] == 32
        assert item["height"] == 24
        assert item["source_video"] == str(source)
        opened = Image.open(store.path_for(item["image_id"]))
        assert opened.size == (32, 24)
        assert abs(item["extracted_time_sec"] - item["requested_time_sec"]) < 0.2
    assert observation.result["applied"]["seek"] == "ffmpeg_timestamp"
    assert observation.artifacts["image_ids"] == [item["image_id"] for item in frames]


def test_fake_extractor_records_seconds_not_frame_index(tmp_path: Path) -> None:
    source = _video(tmp_path / "clip.mp4")
    extractor = FakeFrameExtractor(actual_offset=-0.02)
    observation = _retrieve(
        video=source,
        inputs={"timestamps": 0.5, "view": "aerial"},
        extractor=extractor,
        tmp_path=tmp_path,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert extractor.calls == [(str(source.resolve()), 0.5)]
    frame = observation.result["frames"][0]
    assert frame["requested_time_sec"] == 0.5
    assert frame["extracted_time_sec"] == 0.48
    assert observation.result["applied"]["view"] == "aerial"
    assert "view" not in frame
    assert "只能提取已有画面" in observation.result["assumptions"][2]


def test_result_id_uses_previous_source_url_and_fake_fetcher(tmp_path: Path) -> None:
    source = _video(tmp_path / "clip.mp4")
    fetcher = FakeVideoFetcher(source)
    extractor = FakeFrameExtractor()
    ctx = RuntimeContext(
        extras={"artifact_dir": str(tmp_path / "out")},
        previous_tool_result={
            "results": [
                {
                    "result_id": "ms_1",
                    "media_id": "File:Bridge_drone.webm",
                    "url": "https://commons.example/wiki/File:Bridge_drone.webm",
                    "source_url": "https://upload.example/v/v1/drone.webm",
                }
            ]
        },
    )
    observation = _retrieve(
        inputs={"video": "ms_1", "timestamps": [0.1]},
        extractor=extractor,
        fetcher=fetcher,
        ctx=ctx,
        tmp_path=tmp_path,
    )
    assert observation.ok is True
    assert fetcher.calls[0][0] == "https://upload.example/v/v1/drone.webm"
    assert extractor.calls[0][1] == 0.1
    assert observation.result is not None
    assert observation.result["video"] == "ms_1"


def test_unknown_result_id_is_rejected(tmp_path: Path) -> None:
    observation = _retrieve(
        inputs={"video": "ms_missing", "timestamps": 0.1},
        extractor=FakeFrameExtractor(),
        ctx=RuntimeContext(previous_tool_result={"results": []}),
        tmp_path=tmp_path,
    )
    assert observation.ok is False
    assert observation.error_code == "unknown_result_id"


def test_out_of_range_timestamp_goes_to_missing(tmp_path: Path) -> None:
    source = _video(tmp_path / "clip.mp4")
    extractor = FakeFrameExtractor(duration_sec=1.0, max_time=1.0)
    observation = _retrieve(
        video=source,
        inputs={"timestamps": [0.2, 9.0]},
        extractor=extractor,
        tmp_path=tmp_path,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert len(observation.result["frames"]) == 1
    assert observation.result["frames"][0]["requested_time_sec"] == 0.2
    assert observation.result["missing"][0]["requested_time_sec"] == 9.0
    assert extractor.calls == [(str(source.resolve()), 0.2)]


def test_all_timestamps_fail_is_extract_failed(tmp_path: Path) -> None:
    source = _video(tmp_path / "clip.mp4")
    extractor = FakeFrameExtractor(duration_sec=1.0)
    observation = _retrieve(
        video=source,
        inputs={"timestamps": 99},
        extractor=extractor,
        tmp_path=tmp_path,
    )
    assert observation.ok is False
    assert observation.error_code == "extract_failed"


def test_invalid_timestamp_is_rejected(tmp_path: Path) -> None:
    source = _video(tmp_path / "clip.mp4")
    observation = _retrieve(
        video=source,
        inputs={"timestamps": "not-a-time"},
        extractor=FakeFrameExtractor(),
        tmp_path=tmp_path,
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_timestamp"


def test_missing_local_video_is_video_not_found(tmp_path: Path) -> None:
    observation = _retrieve(
        inputs={"video": str(tmp_path / "absent.mp4"), "timestamps": 0.1},
        extractor=FakeFrameExtractor(),
        tmp_path=tmp_path,
    )
    assert observation.ok is False
    assert observation.error_code == "video_not_found"


def test_json_timestamp_array_and_image_store(tmp_path: Path) -> None:
    source = _video(tmp_path / "clip.mp4")
    store = FilesystemImageStore(tmp_path / "store")
    extractor = FakeFrameExtractor()
    observation = _retrieve(
        video=source,
        inputs={"timestamps": "[0.1, 0.4]"},
        extractor=extractor,
        ctx=RuntimeContext(
            extras={"artifact_dir": str(tmp_path / "out")},
            image_store=store,
        ),
        tmp_path=tmp_path,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert [item["requested_time_sec"] for item in observation.result["frames"]] == [0.1, 0.4]
    assert extractor.calls[0][1] == 0.1
    assert extractor.calls[1][1] == 0.4
    for item in observation.result["frames"]:
        assert store.path_for(item["image_id"]).is_file()
