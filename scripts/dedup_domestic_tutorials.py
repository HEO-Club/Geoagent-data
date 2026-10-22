"""Metadata-dedup Geo-Club/Geo-Video across B 站教程、B 站主站、顶层抖音与知乎.

Reads Hub trees / jsonl and Bilibili public view API only (no video download).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

REPO_ID = "Geo-Club/Geo-Video"
HUB_HOSTS = (
    "https://huggingface.co",
    "https://hf-mirror.com",
)
HUB_FILE = "https://huggingface.co/datasets/{repo}/blob/main/{path}"

BILI_TUTORIAL_DIR = "domestic_tutorials/bili"
BILI_MAIN_DIR = "bilibili"
DOUYIN_JSONL = "douyin/metadata/annotations.jsonl"
ZHIHU_JSONL = "zhihu/videos_metadata.jsonl"
BILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view?bvid={bvid}"

RAW_DIR = Path("data/runs/geo_video_dedup/_raw")
OUT_DIR = Path("data/runs/geo_video_dedup")

MEDIA_EXT = {".mp4", ".webm", ".mkv", ".mov"}

# 文件夹 slug → 规范化作者。别名只覆盖同人多号，不是单条视频特判。
FOLDER_AUTHOR: dict[str, str] = {
    "earth_baike": "地球百科君",
    "fengkenan": "冯柯南",
    "yedian": "夜点",
    "map_detective": "地图侦探",
    "eyeopener": "EyeOpener",
}

AUTHOR_ALIASES: dict[str, str] = {
    "冯柯南up": "冯柯南",
    "冯柯南": "冯柯南",
    "fengkenan": "冯柯南",
    "宇科君": "宇科君",
    "宇宙百科君": "宇科君",
    "地球百科君": "地球百科君",
    "earth_baike": "地球百科君",
    "夜点": "夜点",
    "yedian": "夜点",
    "地图侦探": "地图侦探",
    "map_detective": "地图侦探",
    "eyeopener": "EyeOpener",
    "eye opener": "EyeOpener",
    "彗星_comettt": "彗星",
    "红双喜（网络谜踪）": "红双喜",
    "麦格芬 MacGuffin": "麦格芬",
    "米小禾(韭菜盒子)": "米小禾",
}

HASHTAG_RE = re.compile(r"#\S+")
PUNCT_RE = re.compile(r"[\s_\-—–｜丨;；,，.。!！?？:：·~～'\"“”‘’()（）\[\]【】{}<>《》#@]+")
BVID_RE = re.compile(r"(BV[0-9A-Za-z]{10})")
INT_RE = re.compile(r"\d+")
EPISODE_RE = re.compile(
    r"(第[0-9一二三四五六七八九十]+[集期弹]|"
    r"[（(][0-9一二三四五六七八九十]+[)）]|"
    r"[上下续]$|"
    r"vol\.?\s*[0-9]+|"
    r"[0-9]+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Item:
    source: str
    author_raw: str
    author_canon: str
    title: str
    title_norm: str
    item_id: str
    path: str
    size: int
    kind: str
    hub_url: str
    duration_s: float | None = None
    sha256: str = ""
    extra_authors: tuple[str, ...] = ()
    subset: str = ""


def fetch_bytes(url: str) -> bytes:
    """GET bytes with short retries. Prefer the given URL as-is."""
    last_error: Exception | None = None
    req = urllib.request.Request(url, headers={"User-Agent": "Geoagent-data-dedup/1.0"})
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 — Hub SSL can flake
            last_error = exc
            time.sleep(0.4)
    assert last_error is not None
    raise last_error


def fetch_hub_bytes(rel_url_suffix: str) -> bytes:
    """Fetch a Hub path, falling back to the public mirror."""
    last_error: Exception | None = None
    for host in HUB_HOSTS:
        try:
            return fetch_bytes(f"{host}{rel_url_suffix}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


def fetch_tree(rel: str) -> list[dict[str, Any]]:
    quoted = urllib.parse.quote(rel, safe="/")
    payload: list[dict[str, Any]] = json.loads(
        fetch_hub_bytes(f"/api/datasets/{REPO_ID}/tree/main/{quoted}?recursive=1").decode("utf-8")
    )
    return payload


def fetch_text_file(rel: str, cache_name: str) -> str:
    cache = RAW_DIR / cache_name
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text(encoding="utf-8")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    quoted = urllib.parse.quote(rel, safe="/")
    data = fetch_hub_bytes(f"/datasets/{REPO_ID}/resolve/main/{quoted}")
    cache.write_bytes(data)
    return data.decode("utf-8")


def load_cached_json(rel: str) -> Any:
    path = RAW_DIR / rel
    return json.loads(path.read_text(encoding="utf-8"))


def canon_author(raw: str) -> str:
    s = raw.strip()
    if s.endswith("的作品"):
        s = s[: -len("的作品")]
    if s.endswith("up") and s not in {"Where_Are_U"}:
        s = s[:-2]
    key = AUTHOR_ALIASES.get(s) or AUTHOR_ALIASES.get(s.lower())
    return key or s


def normalize_title(title: str) -> str:
    """Strip hashtags, GeoGuessr boilerplate and punctuation."""
    t = HASHTAG_RE.sub(" ", title)
    t = t.replace("图寻geoguessr", " ").replace("geoguessr", " ")
    t = PUNCT_RE.sub("", t)
    return t.lower()


def episode_key(title: str) -> str:
    compact = title.lower().replace(" ", "")
    m = EPISODE_RE.search(compact)
    return m.group(1) if m else ""


def title_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 12 and shorter in longer:
        return 0.94
    return SequenceMatcher(None, a, b).ratio()


def learn_affixes(titles: Iterable[str], *, min_len: int = 8, min_ratio: float = 0.20) -> list[str]:
    """Learn repeated title prefixes/suffixes for one author (series templates)."""
    norms = [normalize_title(t) for t in titles if normalize_title(t)]
    n = len(norms)
    if n < 5:
        return []
    counts: Counter[str] = Counter()
    lengths = (8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40)
    for text in norms:
        if len(text) < min_len + 4:
            continue
        for length in lengths:
            if len(text) > length + 3:
                counts[text[:length]] += 1
                counts[text[-length:]] += 1
    threshold = max(4, int(min_ratio * n))
    affixes = [affix for affix, cnt in counts.items() if cnt >= threshold and len(affix) >= min_len]
    affixes.sort(key=len, reverse=True)
    return affixes[:12]


def strip_affixes(title_norm: str, affixes: Iterable[str]) -> str:
    text = title_norm
    for affix in affixes:
        if text.startswith(affix) and len(text) - len(affix) >= 4:
            text = text[len(affix) :]
        if text.endswith(affix) and len(text) - len(affix) >= 4:
            text = text[: -len(affix)]
    return text


def duration_ok(a: Item, b: Item) -> bool:
    if a.duration_s is None or b.duration_s is None:
        return True
    longer = max(a.duration_s, b.duration_s)
    if longer <= 0:
        return True
    return abs(a.duration_s - b.duration_s) / longer <= 0.20


def comparable_title(item: Item, affixes: Iterable[str]) -> str:
    base = normalize_title(EPISODE_RE.sub("", item.title))
    return strip_affixes(base, affixes)


def number_conflict(a: str, b: str) -> bool:
    """Different explicit numerals usually mean different episodes, not the same upload."""
    left = set(INT_RE.findall(a))
    right = set(INT_RE.findall(b))
    return bool(left and right and left.isdisjoint(right))


def source_key(item: Item) -> str:
    if item.source == "bili":
        return f"bili:{item.subset or 'main'}"
    return item.source


def should_merge(a: Item, b: Item, *, min_sim: float, affixes: Iterable[str] = ()) -> bool:
    """Decide whether two items are the same upload under the metadata contract."""
    if a.path == b.path:
        return False
    if a.sha256 and b.sha256 and a.sha256 == b.sha256:
        return True
    if BVID_RE.fullmatch(a.item_id or "") and a.item_id == b.item_id:
        return True
    ep_a, ep_b = episode_key(a.title), episode_key(b.title)
    if ep_a and ep_b and ep_a != ep_b:
        return False
    if number_conflict(a.title, b.title):
        return False
    if not duration_ok(a, b):
        return False
    same_author = (
        a.author_canon == b.author_canon
        or a.author_canon in b.extra_authors
        or b.author_canon in a.extra_authors
    )
    left = comparable_title(a, affixes)
    right = comparable_title(b, affixes)
    sim = title_sim(left, right)
    if a.source == b.source and a.subset == b.subset:
        return sim >= 0.96
    if same_author:
        return sim >= 0.84
    return sim >= min_sim


def cluster_items(items: list[Item], *, min_sim: float = 0.98) -> list[list[Item]]:
    """Hard-id unions plus one-to-one title matches inside each author.

    Cross-author merges only happen via sha256 or exact BVID. Series templates
    are stripped first so union-find cannot glue different episodes together.
    """
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    by_author: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        by_author[item.author_canon].append(i)
        for extra in item.extra_authors:
            by_author[extra].append(i)

    sha_groups: dict[str, list[int]] = defaultdict(list)
    bvid_groups: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        if item.sha256:
            sha_groups[item.sha256].append(i)
        if BVID_RE.fullmatch(item.item_id or ""):
            bvid_groups[item.item_id].append(i)
    for group in list(sha_groups.values()) + list(bvid_groups.values()):
        for j in group[1:]:
            union(group[0], j)

    for idxs in by_author.values():
        unique = list(dict.fromkeys(idxs))
        affixes = learn_affixes(items[i].title for i in unique)
        by_src: dict[str, list[int]] = defaultdict(list)
        for i in unique:
            by_src[source_key(items[i])].append(i)
        src_names = list(by_src)
        for a, src_a in enumerate(src_names):
            left_items = [items[i] for i in by_src[src_a]]
            for src_b in src_names[a + 1 :]:
                right_items = [items[i] for i in by_src[src_b]]
                pairs = greedy_cross_pairs(left_items, right_items, affixes=affixes, min_sim=0.84)
                index_left = {id(x): i for i, x in zip(by_src[src_a], left_items)}
                index_right = {id(x): i for i, x in zip(by_src[src_b], right_items)}
                for left, right, _sim in pairs:
                    union(index_left[id(left)], index_right[id(right)])
        for src_idxs in by_src.values():
            for a, i in enumerate(src_idxs):
                for j in src_idxs[a + 1 :]:
                    if should_merge(items[i], items[j], min_sim=0.96, affixes=affixes):
                        union(i, j)

    buckets: dict[int, list[Item]] = defaultdict(list)
    for i, item in enumerate(items):
        buckets[find(i)].append(item)
    return [g for g in buckets.values() if len(g) > 1]


def greedy_cross_pairs(
    left: list[Item],
    right: list[Item],
    *,
    affixes: Iterable[str],
    min_sim: float = 0.84,
) -> list[tuple[Item, Item, float]]:
    """One-to-one title matches across two platform lists of the same author."""
    scored: list[tuple[float, int, int]] = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if not duration_ok(a, b):
                continue
            ep_a, ep_b = episode_key(a.title), episode_key(b.title)
            if ep_a and ep_b and ep_a != ep_b:
                continue
            if number_conflict(a.title, b.title):
                continue
            sim = title_sim(comparable_title(a, affixes), comparable_title(b, affixes))
            if sim >= min_sim:
                scored.append((sim, i, j))
    scored.sort(reverse=True)
    used_l: set[int] = set()
    used_r: set[int] = set()
    pairs: list[tuple[Item, Item, float]] = []
    for sim, i, j in scored:
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        pairs.append((left[i], right[j], sim))
    return pairs


def pick_keep(group: list[Item]) -> Item:
    rank = {"bili": 2, "zhihu": 1, "douyin": 0}
    subset_rank = {"tutorial": 1, "main": 0}
    ranked = sorted(
        group,
        key=lambda x: (rank.get(x.source, 0), subset_rank.get(x.subset, 0), x.size),
        reverse=True,
    )
    return ranked[0]


def make_item(
    *,
    source: str,
    author: str,
    title: str,
    item_id: str,
    path: str,
    size: int,
    duration_s: float | None = None,
    sha256: str = "",
    extra_authors: Iterable[str] = (),
    subset: str = "",
) -> Item:
    extras = tuple(sorted({canon_author(x) for x in extra_authors if canon_author(x) != canon_author(author)}))
    return Item(
        source=source,
        author_raw=author,
        author_canon=canon_author(author),
        title=title,
        title_norm=normalize_title(title),
        item_id=item_id,
        path=path,
        size=size,
        kind="video",
        hub_url=HUB_FILE.format(repo=REPO_ID, path=urllib.parse.quote(path, safe="/")),
        duration_s=duration_s,
        sha256=sha256,
        extra_authors=extras,
        subset=subset,
    )


def _bvid_from_folder(folder: str) -> str:
    match = BVID_RE.search(folder)
    return match.group(1) if match else folder


def fetch_bili_view(bvid: str) -> dict[str, Any]:
    """Fetch public view metadata for one BVID; cache under _raw/bili_view/."""
    cache = RAW_DIR / "bili_view" / f"{bvid}.json"
    if cache.exists() and cache.stat().st_size > 2:
        return json.loads(cache.read_text(encoding="utf-8"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = BILI_VIEW_API.format(bvid=bvid)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Geoagent-data-dedup/1.0)",
            "Referer": "https://www.bilibili.com",
        },
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"bili view failed for {bvid}: {last_error}")


def bili_view_fields(bvid: str) -> tuple[str, float | None, str]:
    """Return (title, duration_s, owner_name); title falls back to BVID on failure."""
    try:
        payload = fetch_bili_view(bvid)
    except Exception:  # noqa: BLE001 — keep inventory even if one view call fails
        return bvid, None, ""
    if int(payload.get("code") or 0) != 0:
        return bvid, None, ""
    data = payload.get("data") or {}
    title = str(data.get("title") or bvid)
    duration_raw = data.get("duration")
    duration: float | None
    try:
        duration = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration = None
    owner = str((data.get("owner") or {}).get("name") or "")
    return title, duration, owner


def prefetch_bili_views(bvids: list[str]) -> None:
    unique = list(dict.fromkeys(bvids))
    missing = [b for b in unique if not (RAW_DIR / "bili_view" / f"{b}.json").exists()]
    if not missing:
        return
    RAW_DIR.joinpath("bili_view").mkdir(parents=True, exist_ok=True)
    print(f"prefetch {len(missing)} Bilibili view records")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(fetch_bili_view, bvid): bvid for bvid in missing}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  view fail {futs[fut]}: {exc}")
            if done % 50 == 0:
                print(f"  view {done}/{len(missing)}")


def load_bili_tutorials() -> list[Item]:
    tree_path = RAW_DIR / "trees" / "bili_tutorials.json"
    tree: list[dict[str, Any]]
    if tree_path.exists():
        tree = load_cached_json("trees/bili_tutorials.json")
    else:
        tree = fetch_tree(BILI_TUTORIAL_DIR)
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_path.write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")
    out: list[Item] = []
    seen: set[str] = set()
    for node in tree:
        if node.get("type") != "file":
            continue
        path = str(node["path"])
        if Path(path).suffix.lower() not in MEDIA_EXT:
            continue
        parts = path.split("/")
        if len(parts) < 5:
            continue
        parent = str(Path(path).parent)
        if parent in seen:
            continue
        seen.add(parent)
        author = parts[2]
        folder = parts[3]
        item_id = _bvid_from_folder(folder)
        title = BVID_RE.sub("", folder).rstrip("_") or item_id
        out.append(
            make_item(
                source="bili",
                author=author,
                title=title,
                item_id=item_id,
                path=path,
                size=int(node.get("size") or 0),
                subset="tutorial",
            )
        )
    return out


def load_bili_main() -> list[Item]:
    """Load top-level `bilibili/<creator>/<NNN_BVxxx>/` folders and fill titles via view API."""
    creators = ["earth_baike", "fengkenan", "yedian", "map_detective", "eyeopener", "sajia"]
    folders: list[tuple[str, str]] = []
    for creator in creators:
        cache = RAW_DIR / "trees" / f"{creator}.json"
        if cache.exists():
            tree = load_cached_json(f"trees/{creator}.json")
        else:
            tree = fetch_tree(f"{BILI_MAIN_DIR}/{creator}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")
        for node in tree:
            path = str(node.get("path") or "")
            folder = path.rsplit("/", 1)[-1]
            bvid = _bvid_from_folder(folder)
            if not BVID_RE.fullmatch(bvid):
                continue
            folders.append((creator, path))

    prefetch_bili_views([_bvid_from_folder(p.rsplit("/", 1)[-1]) for _, p in folders])

    out: list[Item] = []
    owner_votes: dict[str, Counter[str]] = defaultdict(Counter)
    pending: list[tuple[str, str, str, float | None, str]] = []
    for creator, path in folders:
        bvid = _bvid_from_folder(path.rsplit("/", 1)[-1])
        title, duration, owner = bili_view_fields(bvid)
        if owner:
            owner_votes[creator][canon_author(owner)] += 1
        pending.append((creator, path, title, duration, owner))

    creator_author: dict[str, str] = dict(FOLDER_AUTHOR)
    for creator, votes in owner_votes.items():
        if creator in creator_author:
            continue
        if votes:
            creator_author[creator] = votes.most_common(1)[0][0]

    for creator, path, title, duration, owner in pending:
        author = creator_author.get(creator) or owner or creator
        media_path = f"{path}/video.mp4"
        out.append(
            make_item(
                source="bili",
                author=author,
                title=title,
                item_id=_bvid_from_folder(path.rsplit("/", 1)[-1]),
                path=media_path,
                size=0,
                duration_s=duration,
                subset="main",
            )
        )
    return out


def load_douyin() -> list[Item]:
    text = fetch_text_file(DOUYIN_JSONL, "douyin_annotations.jsonl")
    out: list[Item] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        creators = [str(x) for x in (row.get("source_creators") or []) if str(x).strip()]
        selected = ""
        for var in row.get("duplicate_variants") or []:
            if var.get("selected") and var.get("creator_name"):
                selected = str(var["creator_name"])
                break
        author = selected or (creators[0] if creators else str(meta.get("nickname") or "unknown"))
        title = str(meta.get("title") or meta.get("desc") or row.get("aweme_id") or "")
        path = str(row.get("media_path") or f"douyin/videos/{row.get('aweme_id')}.mp4")
        extras = [c for c in creators if c != author]
        out.append(
            make_item(
                source="douyin",
                author=author,
                title=title,
                item_id=str(row.get("aweme_id") or Path(path).stem),
                path=path,
                size=int(row.get("size_bytes") or 0),
                sha256=str(row.get("sha256") or ""),
                extra_authors=extras,
                subset="top",
            )
        )
    return out


def load_zhihu() -> list[Item]:
    text = fetch_text_file(ZHIHU_JSONL, "zhihu_videos_metadata.jsonl")
    out: list[Item] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        zid = str(row.get("zvideo_id") or row.get("id") or "")
        path = str(row.get("video_file") or f"zhihu/videos/zhihu_zvideo_{zid}.mp4")
        if not path.startswith("zhihu/"):
            path = f"zhihu/videos/zhihu_zvideo_{zid}.mp4"
        dur = row.get("duration_seconds")
        duration: float | None
        try:
            duration = float(dur) if dur is not None else None
        except (TypeError, ValueError):
            duration = None
        out.append(
            make_item(
                source="zhihu",
                author=str(row.get("author_name") or "unknown"),
                title=str(row.get("title") or zid),
                item_id=zid,
                path=path,
                size=int(row.get("video_bytes") or 0),
                duration_s=duration,
                sha256=str(row.get("sha256") or ""),
                subset="top",
            )
        )
    return out


def author_overlap(inventory: dict[str, list[Item]]) -> list[dict[str, Any]]:
    by_src: dict[str, dict[str, list[Item]]] = {}
    for src, items in inventory.items():
        bucket: dict[str, list[Item]] = defaultdict(list)
        for it in items:
            bucket[it.author_canon].append(it)
        by_src[src] = bucket
    authors: set[str] = set()
    for bucket in by_src.values():
        authors |= set(bucket)
    rows: list[dict[str, Any]] = []
    for author in sorted(authors):
        counts = {src: len(by_src[src].get(author, [])) for src in inventory}
        bili_items = by_src.get("bili", {}).get(author, [])
        counts["bili_tutorial"] = sum(1 for x in bili_items if x.subset == "tutorial")
        counts["bili_main"] = sum(1 for x in bili_items if x.subset == "main")
        present = [src for src, n in counts.items() if src in inventory and n]
        if len(present) < 2:
            continue
        rows.append({"author": author, "counts": counts, "platforms": present})
    return rows


def unmatched_fingerprint_candidates(
    inventory: dict[str, list[Item]],
    clustered_paths: set[str],
) -> list[dict[str, Any]]:
    """Same-author leftovers that still sit on ≥2 platforms after title matching."""
    by_author: dict[str, dict[str, list[Item]]] = defaultdict(lambda: defaultdict(list))
    for src, items in inventory.items():
        for it in items:
            if it.path in clustered_paths:
                continue
            by_author[it.author_canon][src].append(it)
    rows: list[dict[str, Any]] = []
    for author, by_src in sorted(by_author.items()):
        present = [src for src, xs in by_src.items() if xs]
        if len(present) < 2:
            continue
        rows.append(
            {
                "author": author,
                "counts": {src: len(xs) for src, xs in by_src.items()},
                "note": "title/duration did not pair; next step is audio fingerprint, not ASR",
                "samples": {
                    src: [{"title": x.title[:80], "item_id": x.item_id, "path": x.path} for x in xs[:8]]
                    for src, xs in by_src.items()
                },
            }
        )
    return rows


def dump_items(items: list[Item]) -> list[dict[str, Any]]:
    return [asdict(x) for x in items]


def write_report(
    summary: dict[str, Any],
    cross: list[dict[str, Any]],
    same_platform: list[dict[str, Any]],
    shared: list[dict[str, Any]],
    unmatched: list[dict[str, Any]],
) -> None:
    lines = [
        "# Geo-Video 去重报告（B 站教程 + B 站主站 × douyin × zhihu）",
        "",
        f"仓库：`{REPO_ID}`",
        "范围：`domestic_tutorials/bili` · 顶层 `bilibili/` · 顶层 `douyin/` · 顶层 `zhihu/`",
        "方法：教程用文件夹标题；主站用 B 站公开 view API 补标题/时长；抖音 `annotations.jsonl`；知乎 `videos_metadata.jsonl`。**未下载视频。**",
        "",
        "## 规模",
        "",
    ]
    for source, stats in summary["counts"].items():
        authors = "、".join(stats["authors"][:14])
        extra = "" if len(stats["authors"]) <= 14 else f" 等 {len(stats['authors'])} 人"
        lines.append(f"- **{source}**：{stats['video']} 条视频；作者：{authors}{extra}")
    lines += [
        "",
        f"- 跨平台视频重复簇：**{len(cross)}**",
        f"- 同平台近标题簇：**{len(same_platform)}**（系列，默认不删）",
        f"- 去掉跨平台拷贝后约剩：**{summary['videos_if_drop_cross']}**",
        "",
        "## 跨平台同作者覆盖",
        "",
    ]
    if not shared:
        lines.append("四源之间没有同名作者重叠。")
    for row in shared:
        counts = row["counts"]
        bits = []
        if counts.get("bili_tutorial"):
            bits.append(f"B站教程 {counts['bili_tutorial']}")
        if counts.get("bili_main"):
            bits.append(f"B站主站 {counts['bili_main']}")
        if counts.get("douyin"):
            bits.append(f"抖音 {counts['douyin']}")
        if counts.get("zhihu"):
            bits.append(f"知乎 {counts['zhihu']}")
        lines.append(f"- **{row['author']}**：{' / '.join(bits)}")

    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cross:
        by_author[c["keep"]["author_canon"]].append(c)
    lines += ["", "## 按博主的标题实锤簇", ""]
    for author, clusters in sorted(by_author.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"### {author} · {len(clusters)} 簇")
        for i, c in enumerate(clusters[:12], 1):
            keep = c["keep"]
            plats = " × ".join(c["platforms"])
            lines.append(f"- {i}. 保留 [{keep['source']}/{keep.get('subset') or '-'}] {keep['title'][:70]}")
            for d in c["drop"]:
                lines.append(f"  - drop [{d['source']}] {d['title'][:70]} (`{d['item_id']}`)")
            lines.append(f"  - 平台：{plats}")
        if len(clusters) > 12:
            lines.append(f"  - 其余 {len(clusters) - 12} 簇见 `cross_platform_clusters.json`")
        lines.append("")

    lines += ["", "## 标题未配对、建议音频指纹", ""]
    if not unmatched:
        lines.append("同作者跨平台条目都已标题配对。")
    for row in unmatched:
        bits = " / ".join(f"{src} {n}" for src, n in row["counts"].items() if n)
        lines.append(f"- **{row['author']}**：剩余 {bits}")
        for src, samples in (row.get("samples") or {}).items():
            for s in samples[:3]:
                lines.append(f"  - [{src}] {s['title']}")

    lines += [
        "",
        "## 说明",
        "",
        "- 保留优先级：B 站（教程优先于主站同 BV）> 知乎 > 抖音。未改 Hub 文件。",
        "- 地球百科君主站 Hub 当前只有 204 条目录，同事表是 660，按已上传部分匹配。",
        "- 系列模板前缀/后缀按作者频次学习后剥离，再按平台做一对一标题匹配；不同数字期号不合并。",
        "- 标题完全相同的系列（如夜点「看街景图猜国家」）无法分期，不要按标题删，留给音频指纹。",
        "- 标题对不上的同作者剩余对不要上全库 ASR，下一步用 Chromaprint 音频指纹。",
    ]
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    tutorials = load_bili_tutorials()
    main_site = load_bili_main()
    inventory = {
        "bili": tutorials + main_site,
        "douyin": load_douyin(),
        "zhihu": load_zhihu(),
    }
    videos = [it for src in inventory for it in inventory[src]]
    clusters = cluster_items(videos)
    cross = []
    same_platform = []
    clustered_paths: set[str] = set()
    for g in clusters:
        sources = {x.source for x in g}
        keep = pick_keep(g)
        rec = {
            "keep": asdict(keep),
            "drop": [asdict(x) for x in g if x.path != keep.path],
            "members": [asdict(x) for x in g],
            "platforms": sorted(sources),
        }
        clustered_paths.update(x.path for x in g)
        if len(sources) >= 2:
            rec["reason"] = "same_or_near_title across platforms"
            cross.append(rec)
        else:
            rec["reason"] = "near-duplicate titles on the same platform"
            same_platform.append(rec)

    shared = author_overlap(inventory)
    unmatched = unmatched_fingerprint_candidates(inventory, clustered_paths)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "repo": REPO_ID,
        "scope": [BILI_TUTORIAL_DIR, BILI_MAIN_DIR, "douyin/", "zhihu/"],
        "counts": {
            source: {
                "total": len(inventory[source]),
                "video": len(inventory[source]),
                "authors": sorted({x.author_canon for x in inventory[source]}),
            }
            for source in inventory
        },
        "bili_breakdown": {
            "tutorial": len(tutorials),
            "main": len(main_site),
        },
        "cross_platform_video_clusters": len(cross),
        "same_platform_video_clusters": len(same_platform),
        "shared_authors": shared,
        "unmatched_authors_for_fingerprint": len(unmatched),
        "videos_if_drop_cross": len(videos) - sum(len(c["drop"]) for c in cross),
    }

    (OUT_DIR / "inventory.json").write_text(
        json.dumps({k: dump_items(v) for k, v in inventory.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "cross_platform_clusters.json").write_text(
        json.dumps(cross, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "same_platform_clusters.json").write_text(
        json.dumps(same_platform, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "shared_authors.json").write_text(
        json.dumps(shared, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "unmatched_for_fingerprint.json").write_text(
        json.dumps(unmatched, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, cross, same_platform, shared, unmatched)
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}, ensure_ascii=True, indent=2))
    print("counts", {s: summary["counts"][s]["total"] for s in summary["counts"]})
    print(f"Wrote {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
