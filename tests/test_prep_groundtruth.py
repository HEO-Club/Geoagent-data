"""prep_groundtruth：字幕地名抽取与 map 解析（mock map_query）。"""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.prep_groundtruth import (
    extract_place_candidates,
    lookup_groundtruth,
)
from pipeline.schemas import TranscriptSegment
from pipeline.tools import map_query


def _segs() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=0.0, end=5.0, text="看起来像一座桥旁边的公园。"),
        TranscriptSegment(
            start=200.0,
            end=210.0,
            text="由此可以基本确定,这是在郑州黄河文化公园拍的照片了。",
        ),
        TranscriptSegment(
            start=320.0,
            end=330.0,
            text="这张照片是他父亲在郑州黄河文化公园,登基木格时,经过依山亭前的阶梯,驻足拍的照片。",
        ),
    ]


def test_extract_place_candidates_prefers_answer_window() -> None:
    cands = extract_place_candidates(_segs(), answer_timestamp=200.0)
    names = [c.query for c in cands]
    assert any("黄河文化公园" in n or "郑州黄河文化公园" in n for n in names)


def test_lookup_groundtruth_uses_map_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _Client:
        def query(self, query: str | None, latlng: list[float] | None) -> dict[str, Any]:
            assert latlng is None
            assert query is not None
            calls.append(query)
            return {
                "status": "success",
                "error_message": None,
                "formatted_address": "Zhengzhou Yellow River Scenic Area",
                "resolved_latlng": [34.946, 113.512],
                "place_type": "park",
            }

    map_query.set_client(_Client())
    try:
        sug = lookup_groundtruth(_segs(), query=None)
        assert sug.status == "success"
        assert sug.latitude == pytest.approx(34.946)
        assert sug.longitude == pytest.approx(113.512)
        assert sug.gt_cli() == "34.946,113.512"
        assert calls
    finally:
        map_query.set_client(None)


def test_lookup_respects_manual_query(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Client:
        def query(self, query: str | None, latlng: list[float] | None) -> dict[str, Any]:
            assert query == "依山亭"
            return {
                "status": "success",
                "error_message": None,
                "formatted_address": "Yishan Pavilion",
                "resolved_latlng": [34.95, 113.52],
                "place_type": "attraction",
            }

    map_query.set_client(_Client())
    try:
        sug = lookup_groundtruth(_segs(), query="依山亭")
        assert sug.query == "依山亭"
        assert sug.status == "success"
    finally:
        map_query.set_client(None)
