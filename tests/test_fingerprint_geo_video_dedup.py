"""Unit tests for Chromaprint compare helpers (no Hub download)."""

from __future__ import annotations

from pytest import MonkeyPatch

from scripts.fingerprint_geo_video_dedup import (
    clip_window,
    compare_fingerprints,
    greedy_fp_pairs,
    is_hub_access_error,
    resolve_http_url,
)
from scripts.dedup_domestic_tutorials import make_item


def test_clip_window_skips_intro_on_long_videos() -> None:
    start, length = clip_window(200)
    assert start == 20.0
    assert length == 120.0


def test_clip_window_keeps_short_videos_whole() -> None:
    start, length = clip_window(25)
    assert start == 0.0
    assert length == 25


def test_clip_window_unknown_duration_starts_at_zero() -> None:
    start, length = clip_window(None)
    assert start == 0.0
    assert length == 120.0


def test_resolve_http_url_uses_endpoint(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://example.test")
    url = resolve_http_url("douyin/videos/high_quality/1.mp4")
    assert url == "https://example.test/datasets/Geo-Club/Geo-Video/resolve/main/douyin/videos/high_quality/1.mp4"


def test_is_hub_access_error_detects_403() -> None:
    assert is_hub_access_error(RuntimeError("403 Forbidden: AccessDenied"))
    assert not is_hub_access_error(RuntimeError("connection reset"))


def test_identical_fingerprints_score_near_one() -> None:
    fp = [0xAAAA5555, 0x12345678, 0xFFFFFFFF, 0x0] * 8
    assert compare_fingerprints(fp, fp) >= 0.99


def test_shifted_fingerprint_still_matches() -> None:
    fp = list(range(40, 80))
    assert compare_fingerprints(fp, [0, 1, *fp]) >= 0.90


def test_unrelated_fingerprints_score_low() -> None:
    left = [0x11111111] * 32
    right = [0xEEEEEEEE] * 32
    assert compare_fingerprints(left, right) < 0.40


def test_greedy_fp_pairs_are_one_to_one() -> None:
    a1 = make_item(source="bili", author="夜点", title="a", item_id="a", path="b/a", size=1)
    a2 = make_item(source="bili", author="夜点", title="b", item_id="b", path="b/b", size=1)
    d1 = make_item(source="douyin", author="夜点", title="a", item_id="1", path="d/1", size=1)
    d2 = make_item(source="douyin", author="夜点", title="b", item_id="2", path="d/2", size=1)
    fp_a = [10] * 32
    fp_b = [200] * 32
    pairs = greedy_fp_pairs(
        [(a1, fp_a), (a2, fp_b)],
        [(d1, fp_a), (d2, fp_b)],
        min_score=0.70,
    )
    assert len(pairs) == 2
    mapped = {p[0].item_id: p[1].item_id for p in pairs}
    assert mapped == {"a": "1", "b": "2"}
