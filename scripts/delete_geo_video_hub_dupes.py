"""Delete Hub copies marked drop by title clusters + audio fingerprints.

Safety: never delete a title-cluster keep; never delete unmatched leftovers;
never delete same-title different-BV B 站系列. B 站主站只在教程已按同 BV 保留时删除。
Does not rewrite Hub metadata jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dedup_domestic_tutorials import BVID_RE, OUT_DIR, REPO_ID  # noqa: E402

BATCH_SIZE = 80
ALLOWED_PREFIXES = ("douyin/", "zhihu/", "bilibili/")


def extract_bvid(path: str, item_id: str = "") -> str | None:
    """Return a BV id from item_id or path, if present."""
    for text in (item_id, path):
        hit = BVID_RE.search(text or "")
        if hit:
            return hit.group(1)
    return None


def path_kind(path: str) -> str:
    if path.startswith("domestic_tutorials/"):
        return "bili_tutorial"
    if path.startswith("bilibili/"):
        return "bili_main"
    if path.startswith("douyin/"):
        return "douyin"
    if path.startswith("zhihu/"):
        return "zhihu"
    return "other"


def _rows_from_clusters(clusters: list[dict[str, Any]], origin: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in clusters:
        keep = rec["keep"]
        for drop in rec.get("drop") or []:
            rows.append(
                {
                    "origin": origin,
                    "author": rec.get("author") or keep.get("author_canon"),
                    "keep": keep,
                    "drop": drop,
                }
            )
    return rows


def collect_delete_targets(
    title_clusters: list[dict[str, Any]],
    fingerprint_pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split drop rows into Hub deletes vs skipped, using generic keep/series rules."""
    title_keep_by_path = {rec["keep"]["path"]: rec["keep"] for rec in title_clusters}
    title_keep_by_bvid: dict[str, dict[str, Any]] = {}
    for keep in title_keep_by_path.values():
        bvid = extract_bvid(str(keep.get("path") or ""), str(keep.get("item_id") or ""))
        if bvid and keep.get("source") == "bili" and keep.get("subset") == "tutorial":
            title_keep_by_bvid[bvid] = keep

    chosen: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    for row in _rows_from_clusters(title_clusters, "title") + _rows_from_clusters(
        fingerprint_pairs, "fingerprint"
    ):
        drop = row["drop"]
        path = str(drop["path"])
        kind = path_kind(path)
        reason = ""
        if kind == "other" or not path.startswith(ALLOWED_PREFIXES):
            reason = "path outside allowed prefixes"
        elif path in title_keep_by_path:
            reason = "title-cluster keep"
        elif kind == "bili_tutorial":
            reason = "never delete B 站教程"
        elif kind == "bili_main":
            bvid = extract_bvid(path, str(drop.get("item_id") or ""))
            tutorial = title_keep_by_bvid.get(bvid or "")
            if tutorial is None:
                reason = "B 站同标题不同期，不当重复删"
            else:
                reason = ""
        if reason:
            skipped.append({**row, "path": path, "kind": kind, "skip_reason": reason})
            continue
        prev = chosen.get(path)
        origins = {row["origin"]}
        if prev:
            origins.update(prev.get("origins") or ())
        chosen[path] = {
            "path": path,
            "kind": kind,
            "origin": "+".join(sorted(origins)),
            "origins": sorted(origins),
            "author": row.get("author"),
            "item_id": drop.get("item_id"),
            "title": drop.get("title"),
            "keep_path": row["keep"]["path"],
            "keep_source": row["keep"].get("source"),
            "keep_subset": row["keep"].get("subset"),
            "keep_id": row["keep"].get("item_id"),
        }
    deletes = sorted(chosen.values(), key=lambda x: x["path"])
    return deletes, skipped


def load_lists() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    title = json.loads((OUT_DIR / "cross_platform_clusters.json").read_text(encoding="utf-8"))
    fingerprint = json.loads((OUT_DIR / "fingerprint_pairs.json").read_text(encoding="utf-8"))
    return title, fingerprint


def write_manifest(deletes: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> Path:
    """Write the local delete plan used by the Hub commit."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_id": REPO_ID,
        "delete_count": len(deletes),
        "skip_count": len(skipped),
        "by_kind": dict(Counter(x["kind"] for x in deletes)),
        "skip_by_reason": dict(Counter(x["skip_reason"] for x in skipped)),
        "deletes": deletes,
        "skipped": skipped,
    }
    dest = OUT_DIR / "hub_delete_manifest.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Geo-Video Hub 删除清单",
        "",
        f"仓库：`{REPO_ID}`",
        "来源：标题跨平台簇 + 音频指纹配对的 drop。",
        "规则：不删标题 keep、不删未配对剩余、不删同标题不同 BV 的 B 站系列；B 站主站仅当教程已按同 BV 保留时删除。",
        "**未改元数据 jsonl。**",
        "",
        f"- 将删除：**{len(deletes)}**",
        f"- 跳过（清单里出现但不删）：**{len(skipped)}**",
        f"- 分类：{payload['by_kind']}",
        "",
    ]
    (OUT_DIR / "HUB_DELETE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def delete_on_hub(
    paths: list[str],
    *,
    commit_fn: Callable[..., Any],
    batch_size: int = BATCH_SIZE,
    commit_kwargs: dict[str, Any] | None = None,
) -> tuple[list[str], str | None]:
    """Commit DeleteOperations in batches. `commit_fn` is injected for tests."""
    from huggingface_hub import CommitOperationDelete

    extra = dict(commit_kwargs or {})
    shas: list[str] = []
    pr_url: str | None = None
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        ops = [CommitOperationDelete(path_in_repo=path) for path in batch]
        result = commit_fn(
            repo_id=REPO_ID,
            repo_type="dataset",
            operations=ops,
            commit_message=(
                f"Remove {len(batch)} cross-platform duplicate videos "
                f"({start + 1}-{start + len(batch)}/{len(paths)})"
            ),
            **extra,
        )
        sha = getattr(result, "oid", None) or getattr(result, "commit_id", None) or str(result)
        shas.append(str(sha))
        found_pr = getattr(result, "pr_url", None)
        if found_pr:
            pr_url = str(found_pr)
            extra.pop("create_pr", None)
            revision = getattr(result, "pr_revision", None)
            if revision:
                extra["revision"] = revision
            print(f"opened PR {pr_url}", flush=True)
        print(f"deleted {start + len(batch)}/{len(paths)} commit={sha}", flush=True)
    return shas, pr_url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually delete on Hub (needs write token)")
    args = parser.parse_args(argv)

    title, fingerprint = load_lists()
    deletes, skipped = collect_delete_targets(title, fingerprint)
    write_manifest(deletes, skipped)
    print(
        json.dumps(
            {
                "delete_count": len(deletes),
                "skip_count": len(skipped),
                "by_kind": dict(Counter(x["kind"] for x in deletes)),
                "skip_by_reason": dict(Counter(x["skip_reason"] for x in skipped)),
                "apply": args.apply,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    if not args.apply:
        print(f"Dry-run only. Manifest: {OUT_DIR / 'hub_delete_manifest.json'}", flush=True)
        return 0

    # Writes must go to the official Hub, not the read-only mirror used for downloads.
    os.environ.pop("HF_ENDPOINT", None)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:  # noqa: BLE001
            token = None
    if not token:
        print("No Hugging Face write token. Run `hf auth login` and retry --apply.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token, endpoint="https://huggingface.co")
    who = api.whoami()
    print(f"whoami {who.get('name')}", flush=True)
    paths = [row["path"] for row in deletes]
    try:
        shas, pr_url = delete_on_hub(paths, commit_fn=api.create_commit)
    except Exception as exc:  # noqa: BLE001 — Hub may require a PR instead of writing main
        if "create_pr=1" not in str(exc) and "403" not in str(exc):
            raise
        print("main is protected; opening a Pull Request instead", flush=True)
        shas, pr_url = delete_on_hub(
            paths,
            commit_fn=api.create_commit,
            commit_kwargs={"create_pr": True},
        )
    result_path = OUT_DIR / "hub_delete_result.json"
    result_path.write_text(
        json.dumps(
            {"ok": True, "commits": shas, "deleted": len(deletes), "pr": pr_url},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
