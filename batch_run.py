"""批量跑流水线；分片后单 writer 合并。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.config import get_settings
from pipeline.orchestrator import merge_jsonl_shards, run_one_video

logger = logging.getLogger(__name__)


def _load_jobs(path: str) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("jobs 文件须为 JSON 数组")
    return raw


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _run_job(job: dict[str, Any]) -> str:
    video = job["video_path"]
    entry = run_one_video(
        video,
        video_id=job.get("video_id"),
        anchor_transcript_path=(
            job.get("anchor_transcript_path") or job.get("transcript_path")
        ),
        image_path=job.get("image_path") or "",
        stage3_matcher=lambda _n, _f: None,
    )
    return entry.id


async def _bounded_run(jobs: list[dict[str, Any]], concurrency: int) -> list[str]:
    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[str] = []

    async def one(job: dict[str, Any]) -> None:
        async with sem:
            try:
                eid = await asyncio.to_thread(_run_job, job)
                results.append(eid)
            except Exception:
                logger.exception("job failed %s", job.get("video_path"))

    await asyncio.gather(*(one(j) for j in jobs))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="批量跑三阶段流水线")
    parser.add_argument("--jobs", required=True, help="JSON 任务清单")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    jobs = _load_jobs(args.jobs)
    conc = args.concurrency or settings.MAX_CONCURRENT_VIDEOS
    ids = asyncio.run(_bounded_run(jobs, conc))
    n = merge_jsonl_shards()
    print(json.dumps({"ok": len(ids), "merged_lines": n}, ensure_ascii=False))


if __name__ == "__main__":
    main()
