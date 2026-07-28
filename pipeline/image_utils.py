"""图像辅助：按 Move 选帧、归一化 bbox 裁图。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional, Sequence

_KEYFRAME_TIME_RE = re.compile(r"t(\d+(?:\.\d+)?)\.(?:jpg|jpeg|png|webp)$", re.I)


def parse_keyframe_time(path: str) -> Optional[float]:
    """从 ``t12.000.jpg`` 文件名解析时间戳；失败返回 None。"""
    name = Path(path).name
    m = _KEYFRAME_TIME_RE.search(name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def resolve_keyframe_for_time(
    keyframes: Sequence[str],
    start_time: float,
    end_time: float,
    *,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """在 keyframes 中选与 Move 时间窗最近的一帧。

    优先落在 ``[start_time, end_time]`` 内的帧；否则取时间差最小者。
    """
    if not keyframes:
        return fallback
    timed: list[tuple[float, str]] = []
    for path in keyframes:
        t = parse_keyframe_time(path)
        if t is None:
            continue
        timed.append((t, path))
    if not timed:
        return keyframes[0] if keyframes else fallback

    in_window = [(t, p) for t, p in timed if start_time - 1e-9 <= t <= end_time + 1e-9]
    if in_window:
        mid = (start_time + end_time) / 2.0
        in_window.sort(key=lambda x: abs(x[0] - mid))
        return in_window[0][1]

    target = (start_time + end_time) / 2.0
    timed.sort(key=lambda x: abs(x[0] - target))
    return timed[0][1]


def _normalize_bbox_xyxy(bbox: list[float]) -> Optional[tuple[float, float, float, float]]:
    """将 ``[x,y,w,h]`` 或 ``[x1,y1,x2,y2]`` 归一化框转为像素比例 xyxy。"""
    if len(bbox) != 4 or not all(isinstance(x, (int, float)) for x in bbox):
        return None
    a, b, c, d = (float(x) for x in bbox)
    # 明显非归一化（经纬度等）→ 拒绝裁剪
    if any(abs(x) > 1.5 for x in (a, b, c, d)):
        return None
    # [x,y,w,h]：w/h 通常更小；若 c>a 且 d>b 且 (c-a)<1.1 也可能是 xyxy
    # 约定：若 c<=1 且 d<=1 且 (c<a or d<b 不可能为 xyxy 终点) → 优先 xywh
    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and c >= 0.0 and d >= 0.0:
        # 若第三、四项更像宽高（c+a<=1.05 且 d+b<=1.05）按 xywh
        if a + c <= 1.05 and b + d <= 1.05 and c > 0 and d > 0:
            x1, y1, x2, y2 = a, b, a + c, b + d
        elif c >= a and d >= b:
            x1, y1, x2, y2 = a, b, c, d
        else:
            x1, y1, x2, y2 = a, b, a + max(c, 0.01), b + max(d, 0.01)
    else:
        return None
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
        return None
    return x1, y1, x2, y2


def candidate_keyframes_near_move(
    keyframes: Sequence[str],
    start_time: float,
    end_time: float,
    *,
    max_candidates: int = 5,
) -> list[str]:
    """取 Move 起点/中点/终点附近若干关键帧候选。"""
    if not keyframes:
        return []
    mid = (start_time + end_time) / 2.0
    targets = [start_time, mid, end_time]
    timed: list[tuple[float, str]] = []
    for path in keyframes:
        t = parse_keyframe_time(path)
        if t is None:
            continue
        timed.append((t, path))
    if not timed:
        return list(keyframes[:max_candidates])
    scored: list[tuple[float, str]] = []
    for t, p in timed:
        dist = min(abs(t - tg) for tg in targets)
        scored.append((dist, p))
    scored.sort(key=lambda x: x[0])
    out: list[str] = []
    seen: set[str] = set()
    for _, p in scored:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= max_candidates:
            break
    return out


def compose_content_relative_bbox(
    content_bbox: list[float],
    action_bbox: list[float],
) -> list[float]:
    """内容区 bbox 与 Action 相对 bbox → 全图归一化 xywh。"""
    from pipeline.evidence_routing import combine_bboxes

    return combine_bboxes(content_bbox, action_bbox)


def expand_bbox_xywh(
    bbox: list[float],
    *,
    margin: float = 0.08,
) -> list[float]:
    """将归一化 ``[x,y,w,h]``（或可解析为 xywh 的框）向外扩展 ``margin`` 并 clamp 到 ``[0,1]``。

    非法输入原样返回。用于 zoom/ocr 降低 stage3 框偏导致的机械 empty。
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return bbox
    if not all(isinstance(x, (int, float)) for x in bbox):
        return bbox
    m = max(0.0, float(margin))
    if m <= 0.0:
        return [float(x) for x in bbox]
    norm = _normalize_bbox_xyxy([float(x) for x in bbox])
    if norm is None:
        return [float(x) for x in bbox]
    x1, y1, x2, y2 = norm
    x1 = max(0.0, x1 - m)
    y1 = max(0.0, y1 - m)
    x2 = min(1.0, x2 + m)
    y2 = min(1.0, y2 + m)
    w = max(0.02, x2 - x1)
    h = max(0.02, y2 - y1)
    if x1 + w > 1.0:
        x1 = max(0.0, 1.0 - w)
    if y1 + h > 1.0:
        y1 = max(0.0, 1.0 - h)
    return [x1, y1, w, h]


def crop_image_by_bbox(
    image_path: str,
    bbox: list[float],
    *,
    cache_dir: Optional[str] = None,
) -> str:
    """按归一化 bbox 裁图，返回裁剪图路径；非法框或失败则返回原图。"""
    norm = _normalize_bbox_xyxy(bbox)
    if norm is None:
        return image_path
    src = Path(image_path)
    if not src.is_file():
        return image_path

    import cv2  # 延迟导入

    img = cv2.imread(str(src))
    if img is None:
        return image_path
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return image_path
    x1, y1, x2, y2 = norm
    left = int(round(x1 * w))
    top = int(round(y1 * h))
    right = int(round(x2 * w))
    bottom = int(round(y2 * h))
    left = max(0, min(w - 1, left))
    top = max(0, min(h - 1, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))
    crop = img[top:bottom, left:right]
    if crop.size == 0:
        return image_path

    if cache_dir:
        out_dir = Path(cache_dir) / "cropped"
    else:
        out_dir = src.parent / "_cropped"
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        f"{src.resolve()}|{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}".encode("utf-8")
    ).hexdigest()[:16]
    out_path = out_dir / f"{src.stem}_{digest}.jpg"
    if not out_path.is_file():
        if not cv2.imwrite(str(out_path), crop):
            return image_path
    return str(out_path.resolve())
