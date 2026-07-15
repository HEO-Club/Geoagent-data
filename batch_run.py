"""批处理入口：asyncio 并发 + tenacity 重试 + 分片合并。

错误隔离：单视频失败不影响其他视频；全部结束后由单 writer 合并 JSONL。
禁止多个协程直接追加同一个最终 JSONL。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pipeline.config import Settings, get_settings
from pipeline.schemas import TranscriptSegment, VideoInput
from pipeline.stage7_format import merge_jsonl_shards
from run_one_video import is_video_fully_completed, run_one_video, video_id_from_path

logger = logging.getLogger(__name__)


def load_video_jobs(manifest_path: str | Path) -> list[VideoInput]:
    """从批处理清单加载 VideoInput 列表。

    清单 JSON 格式::

        [
          {
            "video_path": "...",
            "transcript_path": "..." | "transcript": [...],
            "groundtruth": [lat, lng],
            "source_platform": "..."
          },
          ...
        ]
    """
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("批处理清单必须是 JSON 数组")

    jobs: list[VideoInput] = []
    for item in raw:
        if "transcript" in item:
            segments = item["transcript"]
        elif "transcript_path" in item:
            segments = json.loads(
                Path(item["transcript_path"]).read_text(encoding="utf-8")
            )
        else:
            raise ValueError(f"条目缺少 transcript / transcript_path: {item}")

        gt = item["groundtruth"]
        jobs.append(
            VideoInput(
                video_path=item["video_path"],
                transcript=[TranscriptSegment.model_validate(s) for s in segments],
                groundtruth=(float(gt[0]), float(gt[1])),
                source_platform=str(item.get("source_platform", "unknown")),
            )
        )
    return jobs


async def _run_one_with_retry(
    video_input: VideoInput,
    *,
    settings: Settings,
    max_attempts: int,
    video_id: Optional[str] = None,
    hooks: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """在线程池中执行 run_one_video，带 tenacity 重试。"""
    vid = video_id or video_id_from_path(video_input.video_path)
    last_exc: Optional[BaseException] = None
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            try:
                result = await asyncio.to_thread(
                    run_one_video,
                    video_input,
                    video_id=vid,
                    settings=settings,
                    hooks=hooks,
                )
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[%s] attempt %s failed: %s",
                    vid,
                    attempt.retry_state.attempt_number,
                    exc,
                )
                raise
    # 理论上不可达
    raise RuntimeError(f"{vid} 重试耗尽: {last_exc}")


async def batch_run_async(
    videos: list[VideoInput],
    *,
    settings: Optional[Settings] = None,
    max_attempts: int = 3,
    skip_completed: bool = True,
    merge: bool = True,
    hooks: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """并发处理多视频；跳过已完成；结束时单 writer 合并分片。

    单视频异常被捕获并记入 failures，不中断整批。
    hooks 仅用于测试注入，生产勿传。
    """
    cfg = settings or get_settings()
    sem = asyncio.Semaphore(max(1, cfg.MAX_CONCURRENT_VIDEOS))

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[str] = []

    async def _guarded(vin: VideoInput) -> None:
        vid = video_id_from_path(vin.video_path)
        if skip_completed and is_video_fully_completed(vid, cfg):
            logger.info("[%s] already completed, skip", vid)
            skipped.append(vid)
            return
        async with sem:
            try:
                out = await _run_one_with_retry(
                    vin,
                    settings=cfg,
                    max_attempts=max_attempts,
                    video_id=vid,
                    hooks=hooks,
                )
                results.append(out)
            except Exception as exc:  # noqa: BLE001 — 错误隔离
                logger.exception("[%s] failed after retries", vid)
                failures.append({"video_id": vid, "error": str(exc)})

    await asyncio.gather(*[_guarded(v) for v in videos])

    merge_counts: dict[str, int] = {}
    if merge:
        # 全部协程结束后单 writer 合并（无论成败，合并已有分片）
        merge_counts = merge_jsonl_shards(cfg.OUTPUT_DIR)

    return {
        "succeeded": results,
        "failed": failures,
        "skipped": skipped,
        "merge_counts": merge_counts,
        "total": len(videos),
        "success_count": len(results),
        "failure_count": len(failures),
        "skip_count": len(skipped),
    }


def batch_run(
    videos: list[VideoInput],
    *,
    settings: Optional[Settings] = None,
    max_attempts: int = 3,
    skip_completed: bool = True,
    merge: bool = True,
    hooks: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """同步封装 :func:`batch_run_async`。"""
    return asyncio.run(
        batch_run_async(
            videos,
            settings=settings,
            max_attempts=max_attempts,
            skip_completed=skip_completed,
            merge=merge,
            hooks=hooks,
        )
    )


def main() -> None:
    """CLI：``python batch_run.py --jobs jobs.json``。"""
    parser = argparse.ArgumentParser(description="批量运行 stage0–7")
    parser.add_argument(
        "--jobs",
        required=True,
        help="批处理清单 JSON（VideoInput 数组）",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="单视频 tenacity 最大尝试次数",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="不跳过已 completed 的视频",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="结束后不合并分片",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    jobs = load_video_jobs(args.jobs)
    summary = batch_run(
        jobs,
        max_attempts=args.max_attempts,
        skip_completed=not args.no_skip_completed,
        merge=not args.no_merge,
    )
    # 精简打印
    printable = {
        "total": summary["total"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "skip_count": summary["skip_count"],
        "failed": summary["failed"],
        "skipped": summary["skipped"],
        "merge_counts": summary["merge_counts"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
