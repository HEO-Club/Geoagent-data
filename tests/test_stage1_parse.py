"""stage1 抽帧与屏幕操作识别测试（外部依赖全部 mock）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pipeline.config import clear_settings_cache
from pipeline.stage1_parse import (
    _ScreenActionBatch,
    _ScreenActionItem,
    detect_screen_actions,
    extract_keyframes,
)


class TestExtractKeyframes:
    def test_extracts_at_requested_fps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
        clear_settings_cache()

        fake_frame = object()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 25.0  # video fps
        cap.read.return_value = (True, fake_frame)

        cv2 = MagicMock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_PROP_FPS = 5
        cv2.CAP_PROP_POS_FRAMES = 1
        cv2.imwrite.return_value = True

        monkeypatch.setitem(__import__("sys").modules, "cv2", cv2)

        paths = extract_keyframes("video.mp4", (10.0, 12.5), fps=1.0)
        # t=10.0, 11.0, 12.0 → 3 帧
        assert len(paths) == 3
        assert all(Path(p).suffix == ".jpg" for p in paths)
        assert cap.release.called
        clear_settings_cache()

    def test_empty_range_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cv2 = MagicMock()
        monkeypatch.setitem(__import__("sys").modules, "cv2", cv2)
        assert extract_keyframes("v.mp4", (5.0, 5.0), fps=1.0) == []
        cv2.VideoCapture.assert_not_called()

    def test_invalid_fps_raises(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            extract_keyframes("v.mp4", (0.0, 1.0), fps=0.0)

    def test_unreadable_video_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cap = MagicMock()
        cap.isOpened.return_value = False
        cv2 = MagicMock()
        cv2.VideoCapture.return_value = cap
        monkeypatch.setitem(__import__("sys").modules, "cv2", cv2)
        with pytest.raises(FileNotFoundError, match="无法打开"):
            extract_keyframes("missing.mp4", (0.0, 1.0))


class TestDetectScreenActions:
    def test_uses_mocked_call_structured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _ScreenActionBatch(
            actions=[
                _ScreenActionItem(
                    start_time=1.0,
                    end_time=2.0,
                    description="放大查看路牌文字",
                    visible_clues=["路牌"],
                ),
                _ScreenActionItem(
                    start_time=2.5,
                    end_time=3.5,
                    description="在搜索框输入地名",
                    visible_clues=["搜索框"],
                ),
            ]
        )

        def _fake_call(
            prompt: str,
            response_model: type[Any],
            images: list[str] | None = None,
            **_kwargs: Any,
        ) -> _ScreenActionBatch:
            assert response_model is _ScreenActionBatch
            assert images == ["f1.jpg", "f2.jpg"]
            assert "北欧" in prompt
            return batch

        monkeypatch.setattr("pipeline.stage1_parse.call_structured", _fake_call)

        actions = detect_screen_actions(
            keyframes=["f1.jpg", "f2.jpg"],
            narration_context="先看路牌，像北欧。",
            time_range=(0.0, 5.0),
        )
        assert len(actions) == 2
        assert actions[0].description == "放大查看路牌文字"
        assert actions[0].visible_clues == ["路牌"]
        assert actions[1].start_time == pytest.approx(2.5)

    def test_empty_keyframes_returns_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _should_not_run(*_a: Any, **_k: Any) -> _ScreenActionBatch:
            raise AssertionError("不应调用 LLM")

        monkeypatch.setattr("pipeline.stage1_parse.call_structured", _should_not_run)
        assert (
            detect_screen_actions(
                keyframes=[],
                narration_context="x",
                time_range=(0.0, 1.0),
            )
            == []
        )

    def test_out_of_range_clamped_via_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _ScreenActionBatch(
            actions=[
                _ScreenActionItem(
                    start_time=100.0,
                    end_time=90.0,  # 非法
                    description="缩放地图",
                    visible_clues=[],
                )
            ]
        )
        monkeypatch.setattr(
            "pipeline.stage1_parse.call_structured",
            lambda *a, **k: batch,
        )
        actions = detect_screen_actions(
            keyframes=["a.jpg"],
            narration_context="看地图",
            time_range=(10.0, 12.0),
        )
        assert len(actions) == 1
        assert 10.0 <= actions[0].start_time < 12.0
        assert actions[0].end_time <= 12.0
