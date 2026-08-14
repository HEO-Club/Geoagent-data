"""选图廉价预过滤：通用视觉特征，禁止词表/单视频特判。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrefilterVerdict:
    """廉价预过滤结果。"""

    keep: bool
    skip_reason: str = ""
    ui_or_map_penalty: bool = False


def _image_dhash(image_path: str) -> int | None:
    """计算轻量视觉哈希；读取失败时返回 None。"""
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            pixels = list(image.convert("L").resize((9, 8)).getdata())
    except Exception:  # noqa: BLE001
        return None
    value = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value


def is_near_duplicate(
    image_path: str,
    seen_hashes: list[int],
    *,
    max_distance: int,
) -> tuple[bool, int | None]:
    """与已见帧是否近重复。"""
    current = _image_dhash(image_path)
    if current is None:
        return False, None
    duplicate = any(
        (current ^ old).bit_count() <= max(0, int(max_distance))
        for old in seen_hashes
    )
    return duplicate, current


def prefilter_frame(image_path: str) -> PrefilterVerdict:
    """近黑/极低方差丢弃；明显地图瓦片或大面积侧栏降权（不硬杀）。"""
    path = Path(image_path)
    if not path.is_file():
        return PrefilterVerdict(keep=False, skip_reason="missing_file")
    try:
        from PIL import Image
        import statistics

        with Image.open(path) as image:
            gray = image.convert("L")
            small = gray.resize((64, 64))
            pixels = list(small.getdata())
            rgb = image.convert("RGB").resize((48, 48))
            rgb_pixels = list(rgb.getdata())
    except Exception as exc:  # noqa: BLE001
        return PrefilterVerdict(keep=True, skip_reason=f"read_error:{type(exc).__name__}")

    if not pixels:
        return PrefilterVerdict(keep=False, skip_reason="empty")
    mean = sum(pixels) / len(pixels)
    try:
        variance = statistics.pvariance(pixels)
    except statistics.StatisticsError:
        variance = 0.0
    if mean < 12.0 or variance < 8.0:
        return PrefilterVerdict(keep=False, skip_reason="near_black_or_flat")

    # 大面积暗侧栏：左右 12% 列明显更暗且中部较亮 → UI 降权
    ui_penalty = False
    cols = 48
    rows = 48
    left = [rgb_pixels[r * cols + c] for r in range(rows) for c in range(max(1, cols // 8))]
    right = [
        rgb_pixels[r * cols + c]
        for r in range(rows)
        for c in range(cols - max(1, cols // 8), cols)
    ]
    mid = [
        rgb_pixels[r * cols + c]
        for r in range(rows)
        for c in range(cols // 4, (3 * cols) // 4)
    ]
    if left and right and mid:
        left_luma = sum(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in left) / len(left)
        right_luma = sum(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in right) / len(
            right
        )
        mid_luma = sum(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in mid) / len(mid)
        side_dark = min(left_luma, right_luma) < mid_luma - 35
        side_contrast = abs(left_luma - mid_luma) > 40 or abs(right_luma - mid_luma) > 40
        if side_dark and side_contrast and mid_luma > 60:
            ui_penalty = True

    # 地图瓦片启发式：中等亮度 + 局部块状低方差网格感
    map_penalty = False
    block = 8
    block_means: list[float] = []
    for by in range(0, 64, block):
        for bx in range(0, 64, block):
            chunk = [
                pixels[y * 64 + x]
                for y in range(by, by + block)
                for x in range(bx, bx + block)
            ]
            block_means.append(sum(chunk) / len(chunk))
    if len(block_means) >= 16:
        try:
            bm_var = statistics.pvariance(block_means)
        except statistics.StatisticsError:
            bm_var = 0.0
        # 瓦片地图：整体方差不极端，但块均值呈中等离散、色偏偏冷灰绿
        cool = sum(1 for p in rgb_pixels if p[2] >= p[0] and p[1] >= p[0] - 8)
        cool_ratio = cool / max(1, len(rgb_pixels))
        if 20.0 < variance < 1800.0 and 40.0 < bm_var < 900.0 and cool_ratio > 0.55:
            map_penalty = True

    return PrefilterVerdict(
        keep=True,
        ui_or_map_penalty=ui_penalty or map_penalty,
    )


def subsample_timestamps(stamps: list[float], max_n: int) -> list[float]:
    """均匀抽稀到最多 max_n 个时间戳，保持端点覆盖。"""
    if max_n <= 0 or not stamps:
        return []
    if len(stamps) <= max_n:
        return list(stamps)
    if max_n == 1:
        return [stamps[len(stamps) // 2]]
    out: list[float] = []
    last_idx = len(stamps) - 1
    for i in range(max_n):
        idx = round(i * last_idx / (max_n - 1))
        t = stamps[idx]
        if not out or abs(out[-1] - t) > 1e-6:
            out.append(t)
    return out
