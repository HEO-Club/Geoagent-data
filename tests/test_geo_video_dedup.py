"""Unit tests for Geo-Video metadata dedup helpers (no Hub / no paid API)."""

from __future__ import annotations

from scripts.dedup_domestic_tutorials import (
    canon_author,
    cluster_items,
    greedy_cross_pairs,
    learn_affixes,
    make_item,
    normalize_title,
    number_conflict,
    pick_keep,
    should_merge,
    strip_affixes,
    title_sim,
)


def test_author_aliases_cover_main_site_folders() -> None:
    assert canon_author("冯柯南up") == "冯柯南"
    assert canon_author("fengkenan") == "冯柯南"
    assert canon_author("earth_baike") == "地球百科君"
    assert canon_author("地球百科君") == "地球百科君"
    assert canon_author("yedian") == "夜点"
    assert canon_author("map_detective") == "地图侦探"
    assert canon_author("eyeopener") == "EyeOpener"
    assert canon_author("宇宙百科君") == "宇科君"


def test_normalize_title_strips_tags_and_punct() -> None:
    assert "图寻" not in normalize_title("智利地貌 #图寻 #地理")
    assert normalize_title("学习智利的地貌环境！") == "学习智利的地貌环境"


def test_series_affix_learning_is_author_level_not_single_video() -> None:
    titles = [f"河谷第{i}期，网络迷踪，地理推理侦探挑战赛" for i in range(1, 9)]
    titles.append("完全不同的单独标题abcdefgh")
    affixes = learn_affixes(titles)
    stripped = [strip_affixes(normalize_title(t), affixes) for t in titles[:8]]
    assert len({s for s in stripped if s}) >= 6


def test_same_bvid_merges_tutorial_and_main() -> None:
    a = make_item(
        source="bili",
        author="冯柯南",
        title="学习智利的地貌环境",
        item_id="BV1GtfhYwEvw",
        path="domestic_tutorials/bili/冯柯南/智利_BV1GtfhYwEvw/a.mp4",
        size=10,
        subset="tutorial",
    )
    b = make_item(
        source="bili",
        author="冯柯南up",
        title="学习智利的地貌环境，随机降落",
        item_id="BV1GtfhYwEvw",
        path="bilibili/fengkenan/021_BV1GtfhYwEvw/video.mp4",
        size=12,
        subset="main",
    )
    assert should_merge(a, b, min_sim=0.98)
    assert pick_keep([a, b]).subset == "tutorial"


def test_keep_prefers_bili_over_douyin() -> None:
    bili = make_item(source="bili", author="夜点", title="桥", item_id="BV1", path="bilibili/a", size=1, subset="main")
    dy = make_item(source="douyin", author="夜点", title="桥", item_id="1", path="douyin/a", size=99)
    assert pick_keep([dy, bili]).source == "bili"


def test_greedy_pairs_are_one_to_one() -> None:
    left = [
        make_item(source="bili", author="夜点", title="红桥夜景推理", item_id="a", path="b/a", size=1),
        make_item(source="bili", author="夜点", title="海边风力发电", item_id="b", path="b/b", size=1),
    ]
    right = [
        make_item(source="douyin", author="夜点", title="红桥夜景推理挑战", item_id="1", path="d/1", size=1),
        make_item(source="douyin", author="夜点", title="红桥夜景推理挑战加长版不像", item_id="2", path="d/2", size=1),
        make_item(source="douyin", author="夜点", title="海边风力发电场", item_id="3", path="d/3", size=1),
    ]
    pairs = greedy_cross_pairs(left, right, affixes=[], min_sim=0.80)
    assert len(pairs) == 2
    used_right = {p[1].item_id for p in pairs}
    assert len(used_right) == 2


def test_title_sim_substring_boost() -> None:
    assert title_sim("学习各个国家的元素符号对照", "各个国家的元素符号对照") >= 0.90
    assert title_sim("学习各个国家的元素", "各个国家的元素") >= 0.84


def test_number_conflict_keeps_different_episodes_apart() -> None:
    assert number_conflict("1分钟破解城市夜景", "2分钟猜出城市夜景图")
    assert not number_conflict("悬赏99猜河谷", "悬赏99猜有三座桥的河谷")


def test_series_template_does_not_glue_different_places() -> None:
    titles = [
        "网络谜踪照片推理地理分析挑战，未知的城市",
        "网络谜踪照片推理地理分析挑战，未知的大山城市",
        "网络谜踪照片推理地理分析挑战，未知的县城",
        "网络谜踪照片推理地理分析挑战，未知的平原村落",
    ]
    affixes = learn_affixes(titles + titles)
    bili = [
        make_item(source="bili", author="夜点", title=t, item_id=f"BV{i:0>10}", path=f"b/{i}", size=1, subset="main")
        for i, t in enumerate(titles)
    ]
    douyin = [
        make_item(source="douyin", author="夜点", title=t, item_id=str(i), path=f"d/{i}", size=1)
        for i, t in enumerate(titles)
    ]
    pairs = greedy_cross_pairs(bili, douyin, affixes=affixes, min_sim=0.84)
    assert len(pairs) == 4
    for left, right, _sim in pairs:
        assert left.title == right.title
    clusters = cluster_items(bili + douyin)
    assert len(clusters) == 4
    for group in clusters:
        assert len({x.title for x in group}) == 1
