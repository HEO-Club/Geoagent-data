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


def _to_gray_array(image_path: str, size: tuple[int, int] | None = None) -> list[list[float]] | None:
    """读取灰度矩阵；size 给定则缩放。"""
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            gray = image.convert("L")
            if size is not None:
                gray = gray.resize(size)
            w, h = gray.size
            pixels = list(gray.getdata())
    except Exception:  # noqa: BLE001
        return None
    return [[float(pixels[y * w + x]) for x in range(w)] for y in range(h)]


def _ncc(a: list[list[float]], b: list[list[float]]) -> float:
    """归一化互相关；尺寸不一致或方差过小返回 0。"""
    if not a or not b or len(a) != len(b) or len(a[0]) != len(b[0]):
        return 0.0
    flat_a = [v for row in a for v in row]
    flat_b = [v for row in b for v in row]
    n = len(flat_a)
    if n == 0:
        return 0.0
    mean_a = sum(flat_a) / n
    mean_b = sum(flat_b) / n
    num = 0.0
    den_a = 0.0
    den_b = 0.0
    for va, vb in zip(flat_a, flat_b, strict=True):
        da = va - mean_a
        db = vb - mean_b
        num += da * db
        den_a += da * da
        den_b += db * db
    if den_a < 1e-6 or den_b < 1e-6:
        return 0.0
    return num / ((den_a * den_b) ** 0.5)


def _crop_panel(
    grid: list[list[float]],
    *,
    y0: float,
    y1: float,
    x0: float,
    x1: float,
) -> list[list[float]]:
    """按相对比例裁切面板。"""
    h = len(grid)
    w = len(grid[0]) if grid else 0
    r0 = max(0, min(h - 1, int(h * y0)))
    r1 = max(r0 + 1, min(h, int(h * y1)))
    c0 = max(0, min(w - 1, int(w * x0)))
    c1 = max(c0 + 1, min(w, int(w * x1)))
    return [row[c0:c1] for row in grid[r0:r1]]


def _resize_grid(grid: list[list[float]], size: tuple[int, int]) -> list[list[float]]:
    """最近邻缩放到目标尺寸。"""
    tw, th = size
    h = len(grid)
    w = len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return []
    out: list[list[float]] = []
    for y in range(th):
        sy = min(h - 1, int(y * h / th))
        row: list[float] = []
        for x in range(tw):
            sx = min(w - 1, int(x * w / tw))
            row.append(grid[sy][sx])
        out.append(row)
    return out


def containment_precheck_score(
    path_a: str,
    path_b: str,
    *,
    min_score: float = 0.82,
) -> tuple[str, float]:
    """廉价检测 B 是否为 A 的面板/放大裁切，或反之。

    Returns:
        (kind, score)：kind 为 ``none`` / ``a_contains_b`` / ``b_contains_a``。
        仅当 score >= min_score 时才应强制合并；否则交给 VLM。
    """
    # 保留较多细节，避免先缩到同尺寸再切半导致信号过弱
    grid_a = _to_gray_array(path_a, (160, 96))
    grid_b = _to_gray_array(path_b, (160, 96))
    if grid_a is None or grid_b is None:
        return "none", 0.0

    probe_size = (48, 48)
    probe_b = _resize_grid(grid_b, probe_size)
    probe_a = _resize_grid(grid_a, probe_size)
    panels = (
        (0.0, 1.0, 0.0, 0.5),  # left
        (0.0, 1.0, 0.5, 1.0),  # right
        (0.0, 0.5, 0.0, 1.0),  # top
        (0.5, 1.0, 0.0, 1.0),  # bottom
        (0.12, 0.88, 0.12, 0.88),  # center
        (0.0, 1.0, 0.0, 1.0),  # full
    )

    def best_contains(
        container: list[list[float]], probe: list[list[float]]
    ) -> float:
        best = 0.0
        for y0, y1, x0, x1 in panels:
            crop = _crop_panel(container, y0=y0, y1=y1, x0=x0, x1=x1)
            crop = _resize_grid(crop, probe_size)
            best = max(best, _ncc(crop, probe))
        return best

    score_ab = best_contains(grid_a, probe_b)
    score_ba = best_contains(grid_b, probe_a)
    if score_ab >= score_ba and score_ab >= float(min_score):
        return "a_contains_b", float(score_ab)
    if score_ba > score_ab and score_ba >= float(min_score):
        return "b_contains_a", float(score_ba)
    return "none", float(max(score_ab, score_ba))


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
