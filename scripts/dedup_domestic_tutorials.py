"""Metadata-dedup Geo-Club/Geo-Video: bili tutorials vs top-level douyin / zhihu.

Reads Hub file trees and jsonl metadata only (no video download).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

REPO_ID = "Geo-Club/Geo-Video"
API_TREE = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}?recursive=1"
HUB_FILE = "https://huggingface.co/datasets/{repo}/blob/main/{path}"
HUB_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

BILI_DIR = "domestic_tutorials/bili"
DOUYIN_JSONL = "douyin/metadata/annotations.jsonl"
ZHIHU_JSONL = "zhihu/videos_metadata.jsonl"
RAW_DIR = Path("data/runs/geo_video_dedup/_raw")
OUT_DIR = Path("data/runs/geo_video_dedup")

SKIP_EXT = {".ass", ".jpg", ".jpeg", ".png", ".webp"}
MEDIA_EXT = {".mp4", ".webm", ".mkv", ".mov"}

AUTHOR_ALIASES: dict[str, str] = {
    "冯柯南up": "冯柯南",
    "冯柯南": "冯柯南",
    "宇科君": "宇科君",
    "宇宙百科君": "宇科君",
    "彗星_comettt": "彗星",
    "红双喜（网络谜踪）": "红双喜",
    "麦格芬 MacGuffin": "麦格芬",
    "米小禾(韭菜盒子)": "米小禾",
}

HASHTAG_RE = re.compile(r"#\S+")
PUNCT_RE = re.compile(r"[\s_\-—–｜丨;；,，.。!！?？:：·~～'\"“”‘’()（）\[\]【】{}<>《》#@]+")
BVID_RE = re.compile(r"(BV[0-9A-Za-z]{10})")
EPISODE_RE = re.compile(
    r"(第[0-9一二三四五六七八九十]+[集期弹]|"
    r"[（(][0-9一二三四五六七八九十]+[)）]|"
    r"[上下续]$|"
    r"vol\.?\s*[0-9]+)$",
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


def fetch_bytes(url: str) -> bytes:
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 — Hub SSL can flake
            last_error = exc
    assert last_error is not None
    raise last_error


def fetch_tree(rel: str) -> list[dict[str, Any]]:
    quoted = urllib.parse.quote(rel, safe="/")
    url = API_TREE.format(repo=REPO_ID, path=quoted)
    payload: list[dict[str, Any]] = json.loads(fetch_bytes(url).decode("utf-8"))
    return payload


def fetch_text_file(rel: str, cache_name: str) -> str:
    cache = RAW_DIR / cache_name
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text(encoding="utf-8")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    quoted = urllib.parse.quote(rel, safe="/")
    url = HUB_RESOLVE.format(repo=REPO_ID, path=quoted)
    data = fetch_bytes(url)
    cache.write_bytes(data)
    return data.decode("utf-8")


def canon_author(raw: str) -> str:
    s = raw.strip()
    if s.endswith("的作品"):
        s = s[: -len("的作品")]
    if s.endswith("up") and s not in {"Where_Are_U"}:
        s = s[:-2]
    return AUTHOR_ALIASES.get(s, AUTHOR_ALIASES.get(s.lower(), s))


def normalize_title(title: str) -> str:
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


def duration_ok(a: Item, b: Item) -> bool:
    if a.duration_s is None or b.duration_s is None:
        return True
    longer = max(a.duration_s, b.duration_s)
    if longer <= 0:
        return True
    return abs(a.duration_s - b.duration_s) / longer <= 0.20


def should_merge(a: Item, b: Item, *, min_sim: float) -> bool:
    if a.path == b.path:
        return False
    if a.sha256 and b.sha256 and a.sha256 == b.sha256:
        return True
    ep_a, ep_b = episode_key(a.title), episode_key(b.title)
    if ep_a and ep_b and ep_a != ep_b:
        return False
    if not duration_ok(a, b):
        return False
    same_author = a.author_canon == b.author_canon or a.author_canon in b.extra_authors or b.author_canon in a.extra_authors
    sim = title_sim(
        normalize_title(EPISODE_RE.sub("", a.title)),
        normalize_title(EPISODE_RE.sub("", b.title)),
    )
    if a.source == b.source:
        return sim >= 0.96
    if same_author:
        return sim >= 0.84
    return sim >= min_sim


def cluster_items(items: list[Item], *, min_sim: float = 0.90) -> list[list[Item]]:
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

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(items[i], items[j], min_sim=min_sim):
                union(i, j)

    buckets: dict[int, list[Item]] = defaultdict(list)
    for i, item in enumerate(items):
        buckets[find(i)].append(item)
    return [g for g in buckets.values() if len(g) > 1]


def pick_keep(group: list[Item]) -> Item:
    rank = {"bili": 2, "zhihu": 1, "douyin": 0}
    ranked = sorted(
        group,
        key=lambda x: (rank.get(x.source, 0), x.size),
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
    )


def load_bili() -> list[Item]:
    tree = fetch_tree(BILI_DIR)
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
        bvid_m = BVID_RE.search(folder)
        item_id = bvid_m.group(1) if bvid_m else folder
        title = BVID_RE.sub("", folder).rstrip("_")
        out.append(
            make_item(
                source="bili",
                author=author,
                title=title,
                item_id=item_id,
                path=path,
                size=int(node.get("size") or 0),
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
    authors = set(by_src["bili"]) | set(by_src["douyin"]) | set(by_src["zhihu"])
    rows: list[dict[str, Any]] = []
    for author in sorted(authors):
        counts = {src: len(by_src[src].get(author, [])) for src in inventory}
        present = [src for src, n in counts.items() if n]
        if len(present) < 2:
            continue
        rows.append({"author": author, "counts": counts, "platforms": present})
    return rows


def main() -> None:
    inventory = {
        "bili": load_bili(),
        "douyin": load_douyin(),
        "zhihu": load_zhihu(),
    }
    videos = [it for src in inventory for it in inventory[src]]
    clusters = cluster_items(videos)
    cross = []
    same_platform = []
    for g in clusters:
        sources = {x.source for x in g}
        keep = pick_keep(g)
        rec = {
            "keep": asdict(keep),
            "drop": [asdict(x) for x in g if x.path != keep.path],
            "members": [asdict(x) for x in g],
            "platforms": sorted(sources),
        }
        if len(sources) >= 2:
            rec["reason"] = "same_or_near_title across platforms"
            cross.append(rec)
        else:
            rec["reason"] = "near-duplicate titles on the same platform"
            same_platform.append(rec)

    shared = author_overlap(inventory)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def dump_items(items: list[Item]) -> list[dict[str, Any]]:
        return [asdict(x) for x in items]

    summary = {
        "repo": REPO_ID,
        "scope": [BILI_DIR, "douyin/", "zhihu/"],
        "counts": {
            source: {
                "total": len(inventory[source]),
                "video": len(inventory[source]),
                "authors": sorted({x.author_canon for x in inventory[source]}),
            }
            for source in inventory
        },
        "cross_platform_video_clusters": len(cross),
        "same_platform_video_clusters": len(same_platform),
        "shared_authors": shared,
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
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Geo-Video 去重报告（bili × douyin × zhihu）",
        "",
        f"仓库：`{REPO_ID}`",
        "范围：`domestic_tutorials/bili` · 顶层 `douyin/` · 顶层 `zhihu/`",
        "方法：bili 用文件树标题；抖音用 `annotations.jsonl`；知乎用 `videos_metadata.jsonl`。**未下载视频。**",
        "",
        "## 规模",
        "",
    ]
    for source, stats in summary["counts"].items():
        authors = "、".join(stats["authors"][:12])
        extra = "" if len(stats["authors"]) <= 12 else f" 等 {len(stats['authors'])} 人"
        lines.append(f"- **{source}**：{stats['video']} 条视频；作者：{authors}{extra}")
    lines += [
        "",
        f"- 跨平台视频重复簇：**{len(cross)}**",
        f"- 同平台近标题簇：**{len(same_platform)}**（系列，默认不删）",
        "",
        "## 跨平台同作者覆盖",
        "",
    ]
    if not shared:
        lines.append("三源之间没有同名作者重叠。")
    for row in shared:
        bits = " / ".join(f"{src} {n}" for src, n in row["counts"].items() if n)
        lines.append(f"- **{row['author']}**：{bits}")

    lines += ["", "## 跨平台重复（建议保留 bili，其次知乎，去掉抖音拷贝）", ""]
    if not cross:
        lines.append("未发现跨平台高置信标题重复。")
    shown = 0
    for i, c in enumerate(cross, 1):
        if shown >= 40:
            lines.append(f"其余 {len(cross) - 40} 簇见 `cross_platform_clusters.json`。")
            break
        keep = c["keep"]
        lines.append(f"### 簇 {i} · 保留 `{keep['source']}` / {keep['author_canon']}")
        lines.append(f"- keep: {keep['title'][:80]}  (`{keep['item_id']}`)")
        lines.append(f"  `{keep['path']}`")
        for d in c["drop"]:
            lines.append(
                f"- drop: [{d['source']}] {d['author_canon']} · {d['title'][:80]} (`{d['item_id']}`)"
            )
            lines.append(f"  `{d['path']}`")
        lines.append("")
        shown += 1

    lines += [
        "## 说明",
        "",
        "- 顶层抖音 `annotations.jsonl` 已把同文件多博主搬运收进 `duplicate_variants`，本报告按一条视频计。",
        "- 知乎视频元数据里 144 条全部是「地球百科君」，与抖音同作者交叉是重点。",
        "- 标题对不上的同作者对（尤其冯柯南 B 站 3 条 vs 抖音 100+ 条）需要音频指纹才能继续。",
    ]
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}, ensure_ascii=True, indent=2))
    print("counts", {s: summary["counts"][s]["total"] for s in summary["counts"]})
    print(f"Wrote {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
