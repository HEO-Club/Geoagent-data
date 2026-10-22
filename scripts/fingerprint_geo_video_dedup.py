"""Audio-fingerprint leftover Geo-Video copies (Chromaprint, no full-library ASR).

Downloads only unmatched same-author leftovers (plus generic-title clusters),
fingerprints a mid-clip, then one-to-one matches across platforms.
Does not modify Hub files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dedup_domestic_tutorials import (  # noqa: E402
    OUT_DIR,
    RAW_DIR,
    Item,
    duration_ok,
    pick_keep,
)

REPO_ID = "Geo-Club/Geo-Video"
TARGET_AUTHORS = ("EyeOpener", "冯柯南", "地图侦探", "地球百科君", "夜点", "宇科君")
FP_DIR = RAW_DIR / "fingerprints"
TMP_DIR = RAW_DIR / "tmp_audio"
BIN_DIR = RAW_DIR / "bin"
MATCH_MIN = 0.70
REVIEW_MIN = 0.55
SKIP_INTRO_S = 20.0
CLIP_LEN_S = 120.0


def find_ffmpeg() -> str:
    """Locate ffmpeg on PATH or the common WinGet install."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    hits = list(winget.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"))
    if hits:
        return str(hits[0])
    raise RuntimeError("ffmpeg not found")


def find_fpcalc() -> str:
    """Locate fpcalc from PATH or the cached Chromaprint zip."""
    found = shutil.which("fpcalc")
    if found:
        return found
    hits = list(BIN_DIR.glob("**/fpcalc.exe")) + list(BIN_DIR.glob("**/fpcalc"))
    if hits:
        return str(hits[0])
    raise RuntimeError("fpcalc not found; download Chromaprint into data/runs/geo_video_dedup/_raw/bin")


def clip_window(duration_s: float | None) -> tuple[float, float]:
    """Skip a shared intro when the video is long enough; return (start, length).

    Unknown duration starts at 0: skipping a fixed intro would seek past short
    clips and make Chromaprint fail on an empty wav.
    """
    if duration_s is None or duration_s <= 0:
        return 0.0, CLIP_LEN_S
    if duration_s < 40:
        return 0.0, duration_s
    start = min(SKIP_INTRO_S, duration_s * 0.12)
    length = min(CLIP_LEN_S, max(30.0, duration_s - start - 3.0))
    return start, length


def find_ffprobe() -> str | None:
    """Locate ffprobe next to ffmpeg, or on PATH."""
    found = shutil.which("ffprobe")
    if found:
        return found
    try:
        ffmpeg = Path(find_ffmpeg())
    except RuntimeError:
        return None
    candidate = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    return str(candidate) if candidate.exists() else None


def probe_duration_s(video: Path) -> float | None:
    """Read container duration via ffprobe. Returns None if unavailable."""
    probe = find_ffprobe()
    if probe is None:
        return None
    raw = subprocess.run(
        [probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        value = float((raw.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return None
    return value if value > 0 else None


def resolve_http_url(rel_path: str) -> str:
    """Build a Hub resolve URL from HF_ENDPOINT (no hardcoded host)."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/{REPO_ID}/resolve/main/{rel_path}"


def is_hub_access_error(exc: BaseException) -> bool:
    """True when huggingface_hub failed with a forbidden / access-denied payload."""
    text = str(exc).lower()
    return any(token in text for token in ("403", "forbidden", "access denied", "accessdenied"))


def download_via_http(rel_path: str, dest_dir: Path) -> Path:
    """Download one Hub file over HTTP. Used when the Xet/cas-bridge path is 403."""
    dest = dest_dir / "file" / Path(rel_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(resolve_http_url(rel_path), headers={"User-Agent": "geoagent-fingerprint"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as handle:
        shutil.copyfileobj(resp, handle)
    if dest.stat().st_size <= 0:
        raise RuntimeError(f"empty HTTP download: {rel_path}")
    return dest


def _popcount(value: int) -> int:
    return int(value).bit_count()


def compare_fingerprints(left: list[int], right: list[int], *, max_offset: int = 12) -> float:
    """Bit-agreement of two raw Chromaprint hashes after a small time alignment.

    Returns 0..1. Same re-encode typically stays well above 0.70; different
    episodes of one series with shared BGM stay lower if the clip skips the intro.
    """
    if not left or not right:
        return 0.0
    best = 0.0
    for offset in range(-max_offset, max_offset + 1):
        if offset >= 0:
            a, b = left[offset:], right
        else:
            a, b = left, right[-offset:]
        n = min(len(a), len(b))
        if n < 16:
            continue
        errors = sum(_popcount(a[i] ^ b[i]) for i in range(n))
        score = 1.0 - errors / (n * 32.0)
        if score > best:
            best = score
    return best


def cache_key(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def item_from_dict(row: dict[str, Any]) -> Item:
    extras = row.get("extra_authors") or []
    return Item(
        source=str(row["source"]),
        author_raw=str(row["author_raw"]),
        author_canon=str(row["author_canon"]),
        title=str(row["title"]),
        title_norm=str(row.get("title_norm") or ""),
        item_id=str(row["item_id"]),
        path=str(row["path"]),
        size=int(row.get("size") or 0),
        kind=str(row.get("kind") or "video"),
        hub_url=str(row.get("hub_url") or ""),
        duration_s=row.get("duration_s"),
        sha256=str(row.get("sha256") or ""),
        extra_authors=tuple(extras),
        subset=str(row.get("subset") or ""),
    )


def load_inventory() -> dict[str, list[Item]]:
    raw = json.loads((OUT_DIR / "inventory.json").read_text(encoding="utf-8"))
    return {src: [item_from_dict(x) for x in rows] for src, rows in raw.items()}


def collect_targets(inventory: dict[str, list[Item]]) -> list[Item]:
    """Leftovers of shared authors, plus members of generic-title cross clusters."""
    cross = json.loads((OUT_DIR / "cross_platform_clusters.json").read_text(encoding="utf-8"))
    clustered: set[str] = set()
    extra: list[Item] = []
    for cluster in cross:
        members = [item_from_dict(x) for x in cluster["members"]]
        clustered.update(x.path for x in members)
        if members[0].author_canon not in TARGET_AUTHORS:
            continue
        by_src: dict[str, int] = defaultdict(int)
        for x in members:
            by_src[x.source] += 1
        # 只复核「同平台多条」或「4 条以上」的宽簇，已一对一标题配对的不再重下。
        ambiguous = max(by_src.values()) >= 2 or len(members) >= 4
        if ambiguous:
            extra.extend(members)
    leftovers: list[Item] = []
    for items in inventory.values():
        for item in items:
            if item.author_canon in TARGET_AUTHORS and item.path not in clustered:
                leftovers.append(item)
    by_path = {x.path: x for x in leftovers}
    for item in extra:
        by_path.setdefault(item.path, item)
    return list(by_path.values())


def load_cached_fp(path: str) -> dict[str, Any] | None:
    cache = FP_DIR / f"{cache_key(path)}.json"
    if not cache.exists():
        return None
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("ok") and payload.get("fingerprint"):
        return payload
    return None


def save_cached_fp(path: str, payload: dict[str, Any]) -> None:
    FP_DIR.mkdir(parents=True, exist_ok=True)
    (FP_DIR / f"{cache_key(path)}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def download_video(rel_path: str, dest_dir: Path) -> Path:
    """Fetch one video; fall back to HTTP when the Hub/Xet path is forbidden."""
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=rel_path,
            cache_dir=str(dest_dir / "hub"),
            local_dir=str(dest_dir / "file"),
        )
        return Path(downloaded)
    except Exception as exc:  # noqa: BLE001 — Hub errors vary by backend
        if not is_hub_access_error(exc):
            raise
        try:
            return download_via_http(rel_path, dest_dir)
        except (urllib.error.URLError, OSError, RuntimeError) as http_exc:
            raise RuntimeError(f"hub 403 and HTTP fallback failed: {http_exc}") from exc


def fingerprint_local_video(video: Path, *, duration_s: float | None) -> list[int]:
    ffmpeg = find_ffmpeg()
    fpcalc = find_fpcalc()
    if duration_s is None or duration_s <= 0:
        duration_s = probe_duration_s(video)
    start, length = clip_window(duration_s)
    wav = video.with_suffix(".clip.wav")
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.2f}",
        "-t",
        f"{length:.2f}",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "11025",
        "-c:a",
        "pcm_s16le",
        str(wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    raw = subprocess.run([fpcalc, "-raw", "-json", str(wav)], check=True, capture_output=True, text=True)
    wav.unlink(missing_ok=True)
    payload = json.loads(raw.stdout)
    fingerprint = [int(x) for x in payload.get("fingerprint") or []]
    if len(fingerprint) < 16:
        raise RuntimeError(f"fingerprint too short: {video}")
    return fingerprint


def ensure_fingerprint(item: Item) -> dict[str, Any]:
    cached = load_cached_fp(item.path)
    if cached:
        return cached
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    work = TMP_DIR / cache_key(item.path)
    work.mkdir(parents=True, exist_ok=True)
    video: Path | None = None
    try:
        video = download_video(item.path, work)
        fingerprint = fingerprint_local_video(video, duration_s=item.duration_s)
        payload = {
            "ok": True,
            "path": item.path,
            "item_id": item.item_id,
            "author": item.author_canon,
            "source": item.source,
            "fingerprint": fingerprint,
            "n": len(fingerprint),
        }
        save_cached_fp(item.path, payload)
        return payload
    except Exception as exc:  # noqa: BLE001 — keep going; one video must not stop the batch
        payload = {"ok": False, "path": item.path, "error": str(exc)}
        save_cached_fp(item.path, payload)
        return payload
    finally:
        if video is not None:
            video.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)


def greedy_fp_pairs(
    left: list[tuple[Item, list[int]]],
    right: list[tuple[Item, list[int]]],
    *,
    min_score: float,
) -> list[tuple[Item, Item, float]]:
    scored: list[tuple[float, int, int]] = []
    for i, (a, fa) in enumerate(left):
        for j, (b, fb) in enumerate(right):
            if not duration_ok(a, b):
                continue
            score = compare_fingerprints(fa, fb)
            if score >= min_score:
                scored.append((score, i, j))
    scored.sort(reverse=True)
    used_l: set[int] = set()
    used_r: set[int] = set()
    pairs: list[tuple[Item, Item, float]] = []
    for score, i, j in scored:
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        pairs.append((left[i][0], right[j][0], score))
    return pairs


def source_bucket(item: Item) -> str:
    """Match across platforms only; tutorial/main are the same B 站 side."""
    return item.source


def match_author(items: list[Item], fps: dict[str, list[int]]) -> list[dict[str, Any]]:
    ready = [(it, fps[it.path]) for it in items if it.path in fps]
    by_src: dict[str, list[tuple[Item, list[int]]]] = defaultdict(list)
    for it, fp in ready:
        by_src[source_bucket(it)].append((it, fp))
    names = list(by_src)
    clusters: list[dict[str, Any]] = []
    used: set[str] = set()
    for i, src_a in enumerate(names):
        for src_b in names[i + 1 :]:
            for a, b, score in greedy_fp_pairs(by_src[src_a], by_src[src_b], min_score=MATCH_MIN):
                group = [a, b]
                keep = pick_keep(group)
                clusters.append(
                    {
                        "author": a.author_canon,
                        "score": round(score, 4),
                        "keep": asdict(keep),
                        "drop": [asdict(x) for x in group if x.path != keep.path],
                        "platforms": sorted({x.source for x in group}),
                        "reason": "chromaprint mid-clip match",
                    }
                )
                used.update(x.path for x in group)
    return clusters


def write_fingerprint_report(
    summary: dict[str, Any],
    pairs: list[dict[str, Any]],
    remaining: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    lines = [
        "# Geo-Video 音频指纹去重",
        "",
        "范围：标题未配对的同作者剩余条目，外加标题完全相同/过宽的跨平台簇。",
        f"方法：跳过约 {int(SKIP_INTRO_S)}s 片头，取中间最多 {int(CLIP_LEN_S)}s，Chromaprint 一对一匹配，阈值 {MATCH_MIN}。",
        "**未改 Hub 文件。**",
        "",
        "## 规模",
        "",
        f"- 计划指纹：**{summary['planned']}**",
        f"- 成功：**{summary['ok']}**",
        f"- 失败：**{summary['failed']}**",
        f"- 新增跨平台实锤簇：**{len(pairs)}**",
        f"- 指纹后仍未配对：**{summary['still_unmatched']}**",
        "",
        "## 按博主",
        "",
    ]
    for author, stats in summary["by_author"].items():
        lines.append(
            f"- **{author}**：成功 {stats['ok']} / 计划 {stats['planned']}；新配对 {stats['pairs']}；仍未配对 {stats['unmatched']}"
        )
    lines += ["", "## 新配对（保留 B 站，其次知乎）", ""]
    if not pairs:
        lines.append("没有过阈值的指纹配对。")
    for i, rec in enumerate(pairs[:40], 1):
        keep = rec["keep"]
        lines.append(f"{i}. score={rec['score']} 保留 [{keep['source']}] {keep['title'][:70]}")
        for d in rec["drop"]:
            lines.append(f"   - drop [{d['source']}] {d['title'][:70]} (`{d['item_id']}`)")
    if len(pairs) > 40:
        lines.append(f"其余 {len(pairs) - 40} 对见 `fingerprint_pairs.json`。")
    lines += ["", "## 下载或指纹失败", ""]
    if not errors:
        lines.append("无。")
    for row in errors[:20]:
        lines.append(f"- `{row['path']}`：{row.get('error', '')[:160]}")
    (OUT_DIR / "FINGERPRINT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    inventory = load_inventory()
    targets = collect_targets(inventory)
    planned_by = defaultdict(int)
    for item in targets:
        planned_by[item.author_canon] += 1
    print(f"targets {len(targets)} {dict(planned_by)}", flush=True)

    fps: dict[str, list[int]] = {}
    errors: list[dict[str, Any]] = []
    ok_by: dict[str, int] = defaultdict(int)
    fail_by: dict[str, int] = defaultdict(int)

    pending = [it for it in targets if load_cached_fp(it.path) is None]
    cached_ok = [it for it in targets if load_cached_fp(it.path) is not None]
    print(f"cached {len(cached_ok)} pending {len(pending)}", flush=True)
    for item in cached_ok:
        payload = load_cached_fp(item.path)
        assert payload is not None
        fps[item.path] = [int(x) for x in payload["fingerprint"]]
        ok_by[item.author_canon] += 1

    workers = 3
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(ensure_fingerprint, item): item for item in pending}
            done = 0
            for fut in as_completed(futs):
                item = futs[fut]
                payload = fut.result()
                done += 1
                if payload.get("ok"):
                    fps[item.path] = [int(x) for x in payload["fingerprint"]]
                    ok_by[item.author_canon] += 1
                else:
                    errors.append({"path": item.path, "error": payload.get("error")})
                    fail_by[item.author_canon] += 1
                if done % 10 == 0 or done == len(pending):
                    print(f"fingerprint {done}/{len(pending)} ok={len(fps)} fail={len(errors)}", flush=True)

    pairs: list[dict[str, Any]] = []
    remaining_rows: list[dict[str, Any]] = []
    pair_by: dict[str, int] = defaultdict(int)
    unmatched_by: dict[str, int] = defaultdict(int)
    by_author: dict[str, list[Item]] = defaultdict(list)
    for item in targets:
        by_author[item.author_canon].append(item)
    paired_paths: set[str] = set()
    for author, items in by_author.items():
        found = match_author(items, fps)
        pairs.extend(found)
        pair_by[author] = len(found)
        for rec in found:
            paired_paths.add(rec["keep"]["path"])
            paired_paths.update(x["path"] for x in rec["drop"])
        leftover = [x for x in items if x.path not in paired_paths and x.path in fps]
        unmatched_by[author] = len(leftover)
        if leftover:
            remaining_rows.append(
                {
                    "author": author,
                    "counts": {
                        src: sum(1 for x in leftover if x.source == src)
                        for src in ("bili", "douyin", "zhihu")
                    },
                    "samples": [
                        {"title": x.title[:80], "item_id": x.item_id, "path": x.path, "source": x.source}
                        for x in leftover[:8]
                    ],
                }
            )

    summary = {
        "planned": len(targets),
        "ok": len(fps),
        "failed": len(errors),
        "new_pairs": len(pairs),
        "still_unmatched": sum(unmatched_by.values()),
        "by_author": {
            author: {
                "planned": planned_by[author],
                "ok": ok_by[author],
                "failed": fail_by[author],
                "pairs": pair_by[author],
                "unmatched": unmatched_by[author],
            }
            for author in TARGET_AUTHORS
        },
        "match_min": MATCH_MIN,
        "clip": {"skip_intro_s": SKIP_INTRO_S, "length_s": CLIP_LEN_S},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fingerprint_pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "fingerprint_unmatched.json").write_text(
        json.dumps(remaining_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "fingerprint_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "fingerprint_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_fingerprint_report(summary, pairs, remaining_rows, errors)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    print(f"Wrote {OUT_DIR / 'FINGERPRINT_REPORT.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
