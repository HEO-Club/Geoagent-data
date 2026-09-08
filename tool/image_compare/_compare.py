"""image_compare 的本地比较：ORB 特征、像素差、直方图与跨图几何。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from tool.contract import Observation, RuntimeContext
from tool.image_edit._transform import RegionError, _parse_region, _try_json
from tool.runtime.image_store import ImageResolveError, put_image, resolve_image_ref

CURRENT_IMAGES_REF = "$current_images"
_METHODS = frozenset({"feature", "pixel", "histogram", "geometry", "auto"})
_DEFAULT_METHOD = "auto"
_ORB_NFEATURES = 2000
_LOWE_RATIO = 0.75
_RANSAC_REPROJ = 5.0
_AUTO_MIN_INLIERS = 4
_AUTO_MIN_INLIER_RATIO = 0.3
_PIXEL_CHANGE_THRESHOLD = 25
_RESIDUAL_THRESHOLD = 30
_MAX_UNMATCHED_REGIONS = 8
_MIN_UNMATCHED_AREA = 16
_MAX_DRAWN_MATCHES = 80
_ASSUMPTIONS = [
    "匹配分数是证据，不表示地点相同",
    "桥梁、电塔、建筑等相似结构仍需其他证据交叉核验",
    "本实现未使用跨视角/跨年代深度学习匹配",
]


class CompareInputError(Exception):
    """images / method 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class AlignError(Exception):
    """pixel 比较要求 ROI 同尺寸。"""

    def __init__(self, message: str, error_code: str = "images_not_aligned") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass
class LoadedImage:
    """已解析的源图及其比较用 ROI。"""

    image_id: str
    image: Image.Image
    region: tuple[int, int, int, int]

    def roi_rgb(self) -> Image.Image:
        """返回 ROI 的 RGB 拷贝。"""

        return self.image.crop(self.region).convert("RGB")


def execute_compare(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """对两张或多张图片做程序化比较，返回可核验匹配证据而非地点结论。"""

    del purpose
    try:
        refs = _parse_image_refs(inputs.get("images"), ctx)
        method = _parse_method(inputs.get("method"))
        loaded = _load_images(refs, ctx)
        regions = _parse_region_list(
            inputs.get("region"),
            [(item.image.size[0], item.image.size[1]) for item in loaded],
            ctx,
        )
    except ImageResolveError as exc:
        return _fail(str(exc), exc.error_code)
    except RegionError as exc:
        return _fail(str(exc), exc.error_code)
    except CompareInputError as exc:
        return _fail(str(exc), exc.error_code)

    for item, box in zip(loaded, regions, strict=True):
        item.region = box

    applied: dict[str, Any] = {
        "method": method,
        "regions": [list(item.region) for item in loaded],
    }
    reference = loaded[0]
    pairs: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    used_methods: list[str] = []

    try:
        for other in loaded[1:]:
            pair, pair_arts = _compare_pair(
                reference,
                other,
                method=method,
                ctx=ctx,
            )
            pairs.append(pair)
            used_methods.append(str(pair["method_used"]))
            _merge_artifacts(artifacts, pair_arts)
    except AlignError as exc:
        return _fail(str(exc), exc.error_code)

    if method == "auto":
        unique = list(dict.fromkeys(used_methods))
        applied["auto_selected"] = unique[0] if len(unique) == 1 else unique
    if any(item == "feature" for item in used_methods):
        applied["detector"] = "ORB"

    scores = [float(pair["score"]) for pair in pairs]
    summary: dict[str, Any] = {
        "pair_count": len(pairs),
        "mean_score": _mean(scores),
        "min_score": min(scores) if scores else 0.0,
    }
    inlier_ratios = [
        float(pair["inlier_ratio"])
        for pair in pairs
        if "inlier_ratio" in pair
    ]
    if inlier_ratios:
        summary["mean_inlier_ratio"] = _mean(inlier_ratios)
        summary["min_inlier_ratio"] = min(inlier_ratios)

    return Observation(
        ok=True,
        result={
            "method": method,
            "applied": applied,
            "pairs": pairs,
            "summary": summary,
            "assumptions": list(_ASSUMPTIONS),
        },
        artifacts=artifacts,
    )


def _parse_image_refs(raw: Any, ctx: RuntimeContext | None) -> list[str]:
    value = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        loaded = _try_json(stripped)
        if isinstance(loaded, list):
            value = loaded
        elif stripped == CURRENT_IMAGES_REF or stripped == "":
            value = None
        else:
            raise CompareInputError("images 必须是至少两张图片的列表", "missing_input")

    if value is None:
        current = list(ctx.current_images) if ctx is not None else []
        if not current:
            raise CompareInputError("缺少必填输入 images", "missing_input")
        value = current

    if not isinstance(value, list):
        raise CompareInputError("缺少必填输入 images", "missing_input")
    refs: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise CompareInputError("缺少必填输入 images", "missing_input")
        refs.append(item.strip())
    if len(refs) < 2:
        raise CompareInputError("images 至少需要两张图片", "too_few_images")
    return refs


def _parse_method(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_METHOD
    if not isinstance(raw, str) or raw.strip() not in _METHODS:
        raise CompareInputError(
            "method 必须是 feature、pixel、histogram、geometry 或 auto",
            "invalid_method",
        )
    return raw.strip()


def _load_images(refs: list[str], ctx: RuntimeContext | None) -> list[LoadedImage]:
    loaded: list[LoadedImage] = []
    for ref in refs:
        image_id, path = resolve_image_ref(ref, ctx)
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except OSError as exc:
            raise ImageResolveError(f"无法读取图片: {exc}", "image_not_found") from exc
        width, height = image.size
        loaded.append(
            LoadedImage(image_id=image_id, image=image, region=(0, 0, width, height)),
        )
    return loaded


def _parse_region_list(
    raw: Any,
    sizes: list[tuple[int, int]],
    ctx: RuntimeContext | None,
) -> list[tuple[int, int, int, int]]:
    if raw is None:
        return [(0, 0, width, height) for width, height in sizes]

    value: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        loaded = _try_json(stripped)
        if loaded is not None:
            value = loaded
        else:
            return [_parse_region(stripped, width, height, ctx) for width, height in sizes]

    if isinstance(value, (list, tuple)) and value and not _looks_like_box(value):
        if len(value) != len(sizes):
            raise RegionError("region 列表长度必须与 images 相同", "invalid_region")
        return [
            _parse_region(item, width, height, ctx)
            for item, (width, height) in zip(value, sizes, strict=True)
        ]
    return [_parse_region(value, width, height, ctx) for width, height in sizes]


def _looks_like_box(value: Any) -> bool:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    if isinstance(value, dict):
        return {"x1", "y1", "x2", "y2"} <= set(value)
    return False


def _compare_pair(
    left: LoadedImage,
    right: LoadedImage,
    *,
    method: str,
    ctx: RuntimeContext | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if method == "pixel":
        return _pixel_pair(left, right, ctx)
    if method == "histogram":
        return _histogram_pair(left, right), {}
    if method == "geometry":
        return _geometry_pair(left, right), {}
    if method == "feature":
        return _feature_pair(left, right, ctx)

    feature_pair, feature_arts = _feature_pair(left, right, ctx)
    inlier_count = int(feature_pair.get("inlier_count") or 0)
    inlier_ratio = float(feature_pair.get("inlier_ratio") or 0.0)
    if inlier_count >= _AUTO_MIN_INLIERS and inlier_ratio >= _AUTO_MIN_INLIER_RATIO:
        feature_pair["method_used"] = "feature"
        return feature_pair, feature_arts
    histogram_pair = _histogram_pair(left, right)
    histogram_pair["method_used"] = "histogram"
    return histogram_pair, {}


def _feature_pair(
    left: LoadedImage,
    right: LoadedImage,
    ctx: RuntimeContext | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    roi_a = left.roi_rgb()
    roi_b = right.roi_rgb()
    gray_a = _gray(roi_a)
    gray_b = _gray(roi_b)
    orb = cv2.ORB_create(
        nfeatures=_ORB_NFEATURES,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=15,
    )
    keypoints_a, desc_a = orb.detectAndCompute(gray_a, None)
    keypoints_b, desc_b = orb.detectAndCompute(gray_b, None)
    matches = _ratio_matches(desc_a, desc_b)

    homography: list[list[float]] | None = None
    inlier_mask: np.ndarray | None = None
    inlier_count = 0
    if len(matches) >= 4:
        src_pts = np.float32([keypoints_b[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_a[item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, _RANSAC_REPROJ)
        if matrix is not None and mask is not None and _finite_matrix(matrix):
            inlier_count = int(mask.ravel().sum())
            if inlier_count >= 4:
                homography = [[float(value) for value in row] for row in matrix.tolist()]
                inlier_mask = mask.ravel().astype(bool)

    match_count = len(matches)
    inlier_ratio = (inlier_count / match_count) if match_count else 0.0
    pair: dict[str, Any] = {
        "image_a": left.image_id,
        "image_b": right.image_id,
        "method_used": "feature",
        "match_count": match_count,
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "score": float(inlier_ratio),
        "homography": homography,
        "unmatched_regions": [],
    }
    artifacts: dict[str, Any] = {}
    if homography is None or inlier_mask is None:
        return pair, artifacts

    bgr_a = _pil_to_bgr(roi_a)
    bgr_b = _pil_to_bgr(roi_b)
    matrix_np = np.asarray(homography, dtype=np.float64)
    height, width = bgr_a.shape[:2]
    warped = cv2.warpPerspective(bgr_b, matrix_np, (width, height))
    pair["unmatched_regions"] = _unmatched_regions(bgr_a, warped, matrix_np, bgr_b.shape[:2])
    overlay = cv2.addWeighted(bgr_a, 0.5, warped, 0.5, 0.0)
    valid = cv2.warpPerspective(
        np.full(bgr_b.shape[:2], 255, dtype=np.uint8),
        matrix_np,
        (width, height),
    )
    overlay[valid == 0] = bgr_a[valid == 0]
    reg_id, reg_path = put_image(
        _bgr_to_pil(overlay),
        source_id=left.image_id,
        suffix="png",
        ctx=ctx,
    )
    pair["registration_image_id"] = reg_id
    artifacts["registration_image_id"] = artifacts.get("registration_image_id") or reg_id
    artifacts["image_path"] = artifacts.get("image_path") or str(reg_path)

    inlier_matches = [item for item, keep in zip(matches, inlier_mask, strict=True) if keep]
    drawn = inlier_matches[:_MAX_DRAWN_MATCHES]
    if drawn:
        vis = cv2.drawMatches(
            bgr_a,
            keypoints_a,
            bgr_b,
            keypoints_b,
            drawn,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        vis_id, vis_path = put_image(
            _bgr_to_pil(vis),
            source_id=left.image_id,
            suffix="png",
            ctx=ctx,
        )
        pair["match_vis_image_id"] = vis_id
        artifacts["match_vis_image_id"] = artifacts.get("match_vis_image_id") or vis_id
        artifacts["match_vis_image_path"] = artifacts.get("match_vis_image_path") or str(vis_path)
    return pair, artifacts


def _ratio_matches(desc_a: np.ndarray | None, desc_b: np.ndarray | None) -> list[cv2.DMatch]:
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < _LOWE_RATIO * second.distance:
            good.append(first)
    return good


def _unmatched_regions(
    base: np.ndarray,
    warped: np.ndarray,
    homography: np.ndarray,
    source_shape: tuple[int, int],
) -> list[list[int]]:
    height, width = base.shape[:2]
    valid = cv2.warpPerspective(
        np.full(source_shape, 255, dtype=np.uint8),
        homography,
        (width, height),
    )
    residual = cv2.absdiff(base, warped)
    residual_gray = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(residual_gray, _RESIDUAL_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask[valid == 0] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        if area < _MIN_UNMATCHED_AREA:
            continue
        boxes.append((area, (x, y, x + box_w, y + box_h)))
    boxes.sort(key=lambda item: item[0], reverse=True)
    return [list(box) for _, box in boxes[:_MAX_UNMATCHED_REGIONS]]


def _pixel_pair(
    left: LoadedImage,
    right: LoadedImage,
    ctx: RuntimeContext | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arr_a = np.asarray(left.roi_rgb(), dtype=np.int16)
    arr_b = np.asarray(right.roi_rgb(), dtype=np.int16)
    if arr_a.shape != arr_b.shape:
        raise AlignError("pixel 比较要求两张图的 ROI 宽高相同，禁止缩放对齐")
    diff = np.abs(arr_a - arr_b)
    mae = float(diff.mean())
    channel_max = diff.max(axis=2)
    changed_ratio = float((channel_max > _PIXEL_CHANGE_THRESHOLD).mean())
    score = max(0.0, 1.0 - min(1.0, mae / 255.0))
    vis = np.clip(diff, 0, 255).astype(np.uint8)
    image_id, path = put_image(
        Image.fromarray(vis, mode="RGB"),
        source_id=left.image_id,
        suffix="png",
        ctx=ctx,
    )
    pair = {
        "image_a": left.image_id,
        "image_b": right.image_id,
        "method_used": "pixel",
        "mae": mae,
        "changed_ratio": changed_ratio,
        "score": score,
        "diff_image_id": image_id,
    }
    artifacts = {
        "diff_image_id": image_id,
        "image_path": str(path),
    }
    return pair, artifacts


def _histogram_pair(left: LoadedImage, right: LoadedImage) -> dict[str, Any]:
    hist_a = _hsv_hist(left.roi_rgb())
    hist_b = _hsv_hist(right.roi_rgb())
    correlation = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    bhattacharyya = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))
    return {
        "image_a": left.image_id,
        "image_b": right.image_id,
        "method_used": "histogram",
        "histogram": {
            "correlation": correlation,
            "bhattacharyya": bhattacharyya,
        },
        "score": correlation,
    }


def _geometry_pair(left: LoadedImage, right: LoadedImage) -> dict[str, Any]:
    width_a, height_a = _roi_size(left)
    width_b, height_b = _roi_size(right)
    aspect_a = width_a / height_a
    aspect_b = width_b / height_b
    area_a = float(width_a * height_a)
    area_b = float(width_b * height_b)
    aspect_ratio = aspect_a / aspect_b
    area_ratio = area_a / area_b
    score = min(aspect_a, aspect_b) / max(aspect_a, aspect_b)
    return {
        "image_a": left.image_id,
        "image_b": right.image_id,
        "method_used": "geometry",
        "aspect_ratio": float(aspect_ratio),
        "geometry": {
            "width_a": width_a,
            "height_a": height_a,
            "width_b": width_b,
            "height_b": height_b,
            "aspect_a": float(aspect_a),
            "aspect_b": float(aspect_b),
            "aspect_ratio": float(aspect_ratio),
            "area_a": area_a,
            "area_b": area_b,
            "area_ratio": float(area_ratio),
        },
        "score": float(score),
    }


def _roi_size(item: LoadedImage) -> tuple[int, int]:
    x1, y1, x2, y2 = item.region
    return (max(1, x2 - x1), max(1, y2 - y1))


def _hsv_hist(image: Image.Image) -> np.ndarray:
    hsv = cv2.cvtColor(_pil_to_bgr(image), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _gray(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(_pil_to_bgr(image), cv2.COLOR_BGR2GRAY)


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _finite_matrix(matrix: np.ndarray) -> bool:
    return bool(np.isfinite(matrix).all())


def _merge_artifacts(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if key not in target:
            target[key] = value


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _fail(error: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=error, error_code=error_code)
