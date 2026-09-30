"""Unit tests for Hub duplicate-delete planning (no Hub / no paid API)."""

from __future__ import annotations

from scripts.dedup_domestic_tutorials import make_item
from scripts.delete_geo_video_hub_dupes import collect_delete_targets, delete_on_hub


def _cluster(keep: object, drops: list[object], *, author: str = "夜点") -> dict:
    from dataclasses import asdict

    return {
        "author": author,
        "keep": asdict(keep) if hasattr(keep, "path") else keep,  # type: ignore[arg-type]
        "drop": [asdict(x) if hasattr(x, "path") else x for x in drops],
    }


def test_collect_deletes_cross_platform_drops_only() -> None:
    keep = make_item(
        source="bili",
        author="夜点",
        title="桥",
        item_id="BV1keepxxxxx",
        path="bilibili/yedian/001_BV1keepxxxxx/video.mp4",
        size=1,
        subset="main",
    )
    douyin = make_item(
        source="douyin",
        author="夜点",
        title="桥",
        item_id="111",
        path="douyin/videos/high_quality/111.mp4",
        size=1,
    )
    zhihu = make_item(
        source="zhihu",
        author="地球百科君",
        title="夜景",
        item_id="222",
        path="zhihu/videos/zhihu_zvideo_222.mp4",
        size=1,
    )
    bili_keep = make_item(
        source="bili",
        author="地球百科君",
        title="夜景",
        item_id="BV1earthxxxx",
        path="bilibili/earth_baike/001_BV1earthxxxx/video.mp4",
        size=1,
        subset="main",
    )
    title = [_cluster(keep, [douyin]), _cluster(bili_keep, [zhihu], author="地球百科君")]
    deletes, skipped = collect_delete_targets(title, [])
    paths = {x["path"] for x in deletes}
    assert paths == {douyin.path, zhihu.path}
    assert keep.path not in paths
    assert bili_keep.path not in paths
    assert skipped == []


def test_same_title_different_bv_bili_is_skipped() -> None:
    keep = make_item(
        source="bili",
        author="夜点",
        title="看街景图猜国家",
        item_id="BV1keepxxxxx",
        path="bilibili/yedian/067_BV1keepxxxxx/video.mp4",
        size=1,
        subset="main",
    )
    series = make_item(
        source="bili",
        author="夜点",
        title="看街景图猜国家",
        item_id="BV1otherxxxx",
        path="bilibili/yedian/100_BV1otherxxxx/video.mp4",
        size=1,
        subset="main",
    )
    deletes, skipped = collect_delete_targets([_cluster(keep, [series])], [])
    assert deletes == []
    assert skipped[0]["path"] == series.path
    assert "不同期" in skipped[0]["skip_reason"]


def test_same_bv_tutorial_allows_main_copy_delete() -> None:
    tutorial = make_item(
        source="bili",
        author="冯柯南",
        title="智利",
        item_id="BV1GtfhYwEvw",
        path="domestic_tutorials/bili/冯柯南up/智利_BV1GtfhYwEvw/智利.mp4",
        size=10,
        subset="tutorial",
    )
    main = make_item(
        source="bili",
        author="冯柯南",
        title="智利",
        item_id="BV1GtfhYwEvw",
        path="bilibili/fengkenan/021_BV1GtfhYwEvw/video.mp4",
        size=9,
        subset="main",
    )
    douyin = make_item(
        source="douyin",
        author="冯柯南",
        title="智利",
        item_id="333",
        path="douyin/videos/high_quality/333.mp4",
        size=8,
    )
    deletes, skipped = collect_delete_targets([_cluster(tutorial, [main, douyin], author="冯柯南")], [])
    paths = {x["path"] for x in deletes}
    assert paths == {main.path, douyin.path}
    assert tutorial.path not in paths
    assert skipped == []


def test_fingerprint_keep_does_not_delete_different_bv_bili() -> None:
    title_keep = make_item(
        source="bili",
        author="夜点",
        title="高空下的城市",
        item_id="BV1Kr421x78q",
        path="bilibili/yedian/068_BV1Kr421x78q/video.mp4",
        size=1,
        subset="main",
    )
    title_drop = make_item(
        source="bili",
        author="夜点",
        title="高空下的城市",
        item_id="BV1em41167Ek",
        path="bilibili/yedian/070_BV1em41167Ek/video.mp4",
        size=1,
        subset="main",
    )
    fp_drop = make_item(
        source="douyin",
        author="夜点",
        title="高空下的城市抖音",
        item_id="444",
        path="douyin/videos/high_quality/444.mp4",
        size=1,
    )
    title = [_cluster(title_keep, [title_drop])]
    fingerprint = [_cluster(title_drop, [fp_drop])]
    deletes, skipped = collect_delete_targets(title, fingerprint)
    paths = {x["path"] for x in deletes}
    assert paths == {fp_drop.path}
    assert title_drop.path not in paths
    assert any(x["path"] == title_drop.path for x in skipped)


def test_delete_on_hub_batches_injected_commit() -> None:
    seen: list[list[str]] = []

    def commit_fn(**kwargs: object) -> object:
        ops = kwargs["operations"]  # type: ignore[index]
        seen.append([op.path_in_repo for op in ops])  # type: ignore[attr-defined]
        return type("R", (), {"oid": f"sha{len(seen)}"})()

    shas, pr_url = delete_on_hub(["a.mp4", "b.mp4", "c.mp4"], commit_fn=commit_fn, batch_size=2)
    assert shas == ["sha1", "sha2"]
    assert pr_url is None
    assert seen == [["a.mp4", "b.mp4"], ["c.mp4"]]


def test_delete_on_hub_forwards_create_pr() -> None:
    seen: list[bool] = []

    def commit_fn(**kwargs: object) -> object:
        seen.append(bool(kwargs.get("create_pr")))
        return type("R", (), {"oid": "sha", "pr_url": "https://hf.co/pr/1", "pr_revision": "refs/pr/1"})()

    shas, pr_url = delete_on_hub(["a.mp4"], commit_fn=commit_fn, commit_kwargs={"create_pr": True})
    assert shas == ["sha"]
    assert pr_url == "https://hf.co/pr/1"
    assert seen == [True]
