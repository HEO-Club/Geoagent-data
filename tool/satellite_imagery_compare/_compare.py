"""satellite_imagery_compare 共享执行器：取景后按地理网格做时相差或候选相似度。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from tool.contract import Observation, RuntimeContext, declared_inputs
from tool.runtime.image_store import ImageResolveError, put_image, resolve_image_ref
from tool.satellite_imagery_query._query import (
    EngineUnavailableError,
    SatelliteInputError,
    execute_retrieve,
    _parse_time_spec,
    _reject_unsupported_provider,
    _strip_forbidden,
)

_OP_TIME = "compare_time"
_OP_CANDIDATES = "compare_candidates"
_PREVIOUS_CANDIDATES_REF = "$previous_candidates"
_CURRENT_IMAGE_REF = "$current_image"
_PIXEL_CHANGE_THRESHOLD = 25
_MAX_CHANGE_REGIONS = 8
_MIN_CHANGE_AREA = 16
_CLOUD_COVER_HIGH = 30.0
_GSD_RATIO_MIN = 2.0
_SEASON_SPREAD_MONTHS = 3
_ASSUMPTIONS = [
    "像素差和相似度是证据，不能直接写成建筑拆除、水位上涨或地点相同",
    "云层、阴影、季节物候和水体颜色可造成伪变化，本实现未做 SCL/NDVI 掩膜",
    "当前按 Copernicus 真彩预览的同一输出网格对齐，未使用 Rasterio/GDAL 重投影",
    "未调用视觉模型解释变化区域",
]
_TEXT_TEMPLATE_ASSUMPTION = "文本/结构化模板未做语义匹配，只返回影像与视觉相似度"


class CompareInputError(Exception):
    """times / candidates / template 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass
class CapturedScene:
    """一次 retrieve 得到的可比较影像。"""

    image_id: str
    path: Path | None
    captured_at: str | None
    resolution_m: float | None
    cloud_cover: float | None
    collection: str | None
    scene_id: str | None
    coverage: bool
    label: Any
    label_key: str


@dataclass(frozen=True)
class ParsedTemplate:
    """比对模板：图片引用、文本或结构化对象。"""

    kind: str
    applied: Any
    image_ref: str | None = None


def execute_compare_time(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """同一区域多时相取景后按像素网格做差，不输出地物变化结论。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "area", "times", "comparison")
        time_items = _parse_times(inputs.get("times"))
        area = inputs.get("area")
        if area in (None, ""):
            raise CompareInputError("缺少必填输入 area", "missing_input")
        captures: list[CapturedScene] = []
        for original, spec in time_items:
            payload: dict[str, Any] = {
                "area": area,
                "time_range": original,
            }
            capture = _retrieve_scene(payload, ctx, label=spec.applied, label_key="time")
            if capture is not None:
                captures.append(capture)
        if len(captures) < 2:
            return _fail("可比较影像不足两个时相", "insufficient_imagery")
        captures.sort(key=_capture_sort_key)
        pairs, pair_artifacts, any_resampled = _diff_pairs(captures, ctx)
        conditions = _compare_conditions(captures)
        applied: dict[str, Any] = {
            "area": area,
            "times": [spec.applied for _, spec in time_items],
        }
        if inputs.get("comparison") not in (None, ""):
            applied["comparison"] = inputs.get("comparison")
        if any_resampled:
            applied["resampled"] = True
        return _ok(
            _OP_TIME,
            sources=captures,
            extra={
                "comparable": conditions["comparable"],
                "pairs": pairs,
                "conditions": conditions,
            },
            applied=applied,
            artifacts=pair_artifacts,
            assumptions=list(_ASSUMPTIONS),
        )
    except CompareInputError as exc:
        return _fail(str(exc), exc.error_code)
    except SatelliteInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)
    except ImageResolveError as exc:
        return _fail(str(exc), exc.error_code)


def execute_compare_candidates(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """多候选取景后按统一模板做外观相似度，不输出地点结论。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "candidates", "template", "time_range", "provider")
        _reject_unsupported_provider(inputs.get("provider"))
        candidates = _parse_candidates(inputs.get("candidates"), ctx)
        template = _parse_template(inputs.get("template"), ctx)
        captures: list[CapturedScene] = []
        for index, candidate in enumerate(candidates):
            payload: dict[str, Any] = {"area": candidate}
            if inputs.get("time_range") not in (None, ""):
                payload["time_range"] = inputs.get("time_range")
            _copy_optional_retrieve_fields(inputs, payload)
            capture = _retrieve_scene(
                payload,
                ctx,
                label=index,
                label_key="candidate",
            )
            if capture is None:
                continue
            captures.append(capture)
        if not captures:
            return _fail("可比较影像不足，候选均无覆盖", "insufficient_imagery")
        ranked, pairwise, template_arts = _score_candidates(captures, template, ctx)
        conditions = _compare_conditions(captures)
        extra: dict[str, Any] = {
            "template_kind": template.kind,
            "conditions_consistent": conditions["comparable"],
            "ranked": ranked,
            "conditions": conditions,
        }
        if pairwise:
            extra["pairwise"] = pairwise
        applied: dict[str, Any] = {
            "candidates": candidates,
            "template": template.applied,
        }
        if inputs.get("time_range") not in (None, ""):
            applied["time_range"] = inputs.get("time_range")
        if inputs.get("provider") not in (None, ""):
            applied["provider"] = inputs.get("provider")
        assumptions = list(_ASSUMPTIONS)
        if template.kind in {"text", "object"}:
            assumptions.append(_TEXT_TEMPLATE_ASSUMPTION)
        artifacts = _source_artifacts(captures)
        artifacts.update(template_arts)
        return _ok(
            _OP_CANDIDATES,
            sources=captures,
            extra=extra,
            applied=applied,
            artifacts=artifacts,
            assumptions=assumptions,
        )
    except CompareInputError as exc:
        return _fail(str(exc), exc.error_code)
    except SatelliteInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)
    except ImageResolveError as exc:
        return _fail(str(exc), exc.error_code)


def _copy_optional_retrieve_fields(inputs: dict[str, Any], payload: dict[str, Any]) -> None:
    for key in ("provider", "time_range"):
        if key in payload:
            continue
        value = inputs.get(key)
        if value not in (None, ""):
            payload[key] = value


def _parse_times(raw: Any) -> list[tuple[Any, Any]]:
    value: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        loaded = _try_json(stripped)
        if isinstance(loaded, list):
            value = loaded
        elif stripped:
            raise CompareInputError("times 至少需要两个时间点", "missing_input")
    if not isinstance(value, list) or len(value) < 2:
        raise CompareInputError("缺少必填输入 times，至少两个时间点", "missing_input")
    items: list[tuple[Any, Any]] = []
    for item in value:
        if item in (None, ""):
            raise CompareInputError("times 含有空时间点", "missing_input")
        spec = _parse_time_spec(item)
        items.append((item, spec))
    return items


def _parse_candidates(raw: Any, ctx: RuntimeContext | None) -> list[Any]:
    value = raw
    if raw in (None, "", _PREVIOUS_CANDIDATES_REF, "$previous_tool_result"):
        value = _candidates_from_ctx(ctx)
    elif isinstance(raw, str):
        stripped = raw.strip()
        if stripped in {_PREVIOUS_CANDIDATES_REF, "$previous_tool_result"}:
            value = _candidates_from_ctx(ctx)
        else:
            loaded = _try_json(stripped)
            if isinstance(loaded, list):
                value = loaded
            else:
                raise CompareInputError(
                    "candidates 必须是真实地点或区域列表，不能只给候选数量",
                    "missing_input",
                )
    if isinstance(value, bool) or isinstance(value, int):
        raise CompareInputError(
            "candidates 必须是真实地点或区域列表，不能只给候选数量",
            "missing_input",
        )
    if isinstance(value, dict):
        nested = value.get("candidates")
        value = nested if isinstance(nested, list) else [value]
    if not isinstance(value, list) or len(value) < 2:
        raise CompareInputError("缺少必填输入 candidates，至少两个真实候选", "missing_input")
    cleaned: list[Any] = []
    for item in value:
        if item in (None, ""):
            raise CompareInputError("candidates 含有空候选", "missing_input")
        if isinstance(item, bool) or isinstance(item, int):
            raise CompareInputError(
                "candidates 必须是真实地点或区域列表，不能只给候选数量",
                "missing_input",
            )
        cleaned.append(item)
    return cleaned


def _candidates_from_ctx(ctx: RuntimeContext | None) -> Any:
    if ctx is None:
        return None
    extras = ctx.extras
    if extras.get("previous_candidates") not in (None, ""):
        return extras["previous_candidates"]
    previous = ctx.previous_tool_result
    if previous is None:
        return None
    if isinstance(previous, list):
        return previous
    if isinstance(previous, dict):
        for key in ("candidates", "locations", "previous_candidates"):
            nested = previous.get(key)
            if nested not in (None, ""):
                return nested
        result = previous.get("result")
        if isinstance(result, dict):
            for key in ("candidates", "locations"):
                nested = result.get(key)
                if nested not in (None, ""):
                    return nested
    return None


def _parse_template(raw: Any, ctx: RuntimeContext | None) -> ParsedTemplate:
    if raw in (None, ""):
        raise CompareInputError("缺少必填输入 template", "missing_input")
    if isinstance(raw, dict):
        image_ref = raw.get("image", raw.get("image_id", raw.get("path")))
        if isinstance(image_ref, str) and image_ref.strip():
            resolved = _as_image_ref(image_ref.strip(), ctx)
            if resolved is None:
                raise CompareInputError("template 图片引用无法解析", "image_not_found")
            return ParsedTemplate(kind="image", applied=raw, image_ref=resolved)
        return ParsedTemplate(kind="object", applied=raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            raise CompareInputError("缺少必填输入 template", "missing_input")
        resolved = _as_image_ref(stripped, ctx)
        if resolved is not None:
            return ParsedTemplate(kind="image", applied=stripped, image_ref=resolved)
        return ParsedTemplate(kind="text", applied=stripped)
    raise CompareInputError("template 必须是文本、对象或图片引用", "missing_input")


def _as_image_ref(token: str, ctx: RuntimeContext | None) -> str | None:
    if token == _CURRENT_IMAGE_REF:
        current = ctx.current_image if ctx is not None else None
        if not current:
            raise CompareInputError("未设置 $current_image", "image_not_found")
        return current
    try:
        image_id, _path = resolve_image_ref(token, ctx)
    except ImageResolveError:
        return None
    return image_id


def _retrieve_scene(
    payload: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    label: Any,
    label_key: str,
) -> CapturedScene | None:
    observation = execute_retrieve(purpose="compare_fetch", inputs=payload, ctx=ctx)
    if not observation.ok:
        raise CompareInputError(
            observation.error or "影像检索失败",
            observation.error_code or "engine_unavailable",
        )
    result = observation.result or {}
    if not result.get("coverage") or not result.get("image_id"):
        return None
    image_id = str(result["image_id"])
    images = observation.artifacts.get("images")
    path: Path | None = None
    if isinstance(images, dict) and image_id in images:
        path = Path(str(images[image_id]))
    return CapturedScene(
        image_id=image_id,
        path=path if path is not None and path.is_file() else None,
        captured_at=_optional_str(result.get("captured_at")),
        resolution_m=_as_float(result.get("resolution_m")),
        cloud_cover=_as_float(result.get("cloud_cover")),
        collection=_optional_str(result.get("collection")),
        scene_id=_optional_str(result.get("scene_id")),
        coverage=True,
        label=label,
        label_key=label_key,
    )


def _open_rgb(capture: CapturedScene, ctx: RuntimeContext | None) -> Image.Image:
    if capture.path is not None and capture.path.is_file():
        with Image.open(capture.path) as opened:
            return opened.convert("RGB")
    _image_id, path = resolve_image_ref(capture.image_id, ctx)
    with Image.open(path) as opened:
        return opened.convert("RGB")


def _diff_pairs(
    captures: list[CapturedScene],
    ctx: RuntimeContext | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    reference = captures[0]
    ref_image = _open_rgb(reference, ctx)
    pairs: list[dict[str, Any]] = []
    artifacts = _source_artifacts(captures)
    any_resampled = False
    for other in captures[1:]:
        other_image = _open_rgb(other, ctx)
        pair, pair_arts, resampled = _pixel_diff(
            ref_image,
            other_image,
            image_id_a=reference.image_id,
            image_id_b=other.image_id,
            ctx=ctx,
        )
        pairs.append(pair)
        any_resampled = any_resampled or resampled
        _merge_artifacts(artifacts, pair_arts)
    return pairs, artifacts, any_resampled


def _pixel_diff(
    image_a: Image.Image,
    image_b: Image.Image,
    *,
    image_id_a: str,
    image_id_b: str,
    ctx: RuntimeContext | None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    rgb_a = image_a.convert("RGB")
    rgb_b = image_b.convert("RGB")
    resampled = False
    if rgb_a.size != rgb_b.size:
        rgb_b = rgb_b.resize(rgb_a.size, Image.Resampling.BILINEAR)
        resampled = True
    arr_a = np.asarray(rgb_a, dtype=np.int16)
    arr_b = np.asarray(rgb_b, dtype=np.int16)
    diff = np.abs(arr_a - arr_b)
    mae = float(diff.mean())
    channel_max = diff.max(axis=2)
    changed_ratio = float((channel_max > _PIXEL_CHANGE_THRESHOLD).mean())
    bgr_a = cv2.cvtColor(np.asarray(rgb_a), cv2.COLOR_RGB2BGR)
    bgr_b = cv2.cvtColor(np.asarray(rgb_b), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(bgr_a, 0.5, bgr_b, 0.5, 0.0)
    overlay_id, overlay_path = put_image(
        Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
        source_id=image_id_a,
        suffix="png",
        ctx=ctx,
    )
    diff_id, diff_path = put_image(
        Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8), mode="RGB"),
        source_id=image_id_a,
        suffix="png",
        ctx=ctx,
    )
    pair = {
        "image_a": image_id_a,
        "image_b": image_id_b,
        "mae": mae,
        "changed_ratio": changed_ratio,
        "change_regions": _change_regions(channel_max),
        "overlay_image_id": overlay_id,
        "diff_image_id": diff_id,
        "resampled": resampled,
    }
    artifacts = {
        "overlay_image_id": overlay_id,
        "image_path": str(overlay_path),
        "diff_image_id": diff_id,
        "diff_image_path": str(diff_path),
        overlay_id: str(overlay_path),
        diff_id: str(diff_path),
    }
    return pair, artifacts, resampled


def _change_regions(channel_max: np.ndarray) -> list[list[int]]:
    mask = (channel_max > _PIXEL_CHANGE_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        if area < _MIN_CHANGE_AREA:
            continue
        boxes.append((area, (x, y, x + box_w, y + box_h)))
    boxes.sort(key=lambda item: item[0], reverse=True)
    return [list(box) for _, box in boxes[:_MAX_CHANGE_REGIONS]]


def _score_candidates(
    captures: list[CapturedScene],
    template: ParsedTemplate,
    ctx: RuntimeContext | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    images = [_open_rgb(item, ctx) for item in captures]
    artifacts: dict[str, Any] = {}
    scores: list[float]
    pairwise: list[dict[str, Any]] = []
    if template.kind == "image" and template.image_ref:
        template_id, template_path = resolve_image_ref(template.image_ref, ctx)
        with Image.open(template_path) as opened:
            template_image = opened.convert("RGB")
        artifacts["template_image_id"] = template_id
        scores = [_image_template_score(template_image, item) for item in images]
    else:
        scores = []
        for index, image in enumerate(images):
            others = [other for j, other in enumerate(images) if j != index]
            if not others:
                scores.append(0.0)
                continue
            scores.append(
                float(sum(_histogram_correlation(image, other) for other in others) / len(others))
            )
        for i, left in enumerate(captures):
            for j in range(i + 1, len(captures)):
                pairwise.append(
                    {
                        "image_a": left.image_id,
                        "image_b": captures[j].image_id,
                        "score": _histogram_correlation(images[i], images[j]),
                    }
                )
    ranked = []
    for capture, score in zip(captures, scores, strict=True):
        item = _source_payload(capture)
        item["score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranked, pairwise, artifacts


def _image_template_score(template: Image.Image, candidate: Image.Image) -> float:
    hist = _histogram_correlation(template, candidate)
    gray_t = _gray(template)
    gray_c = _gray(candidate)
    if gray_t.shape[0] > gray_c.shape[0] or gray_t.shape[1] > gray_c.shape[1]:
        gray_t = cv2.resize(gray_t, (gray_c.shape[1], gray_c.shape[0]), interpolation=cv2.INTER_AREA)
    match = cv2.matchTemplate(gray_c, gray_t, cv2.TM_CCOEFF_NORMED)
    match_score = float(match.max()) if match.size else 0.0
    return float((hist + match_score) / 2.0)


def _histogram_correlation(left: Image.Image, right: Image.Image) -> float:
    hist_a = _hsv_hist(left)
    hist_b = _hsv_hist(right)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def _hsv_hist(image: Image.Image) -> np.ndarray:
    hsv = cv2.cvtColor(_pil_to_bgr(image), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _gray(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(_pil_to_bgr(image), cv2.COLOR_BGR2GRAY)


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _compare_conditions(captures: list[CapturedScene]) -> dict[str, Any]:
    gsds = [
        item.resolution_m
        for item in captures
        if item.resolution_m is not None and item.resolution_m > 0
    ]
    resolution_consistent = True
    notes: list[str] = []
    if len(gsds) >= 2:
        min_gsd = min(gsds)
        max_gsd = max(gsds)
        if max_gsd / min_gsd >= _GSD_RATIO_MIN:
            resolution_consistent = False
            notes.append(f"原生分辨率 {min_gsd:g}m vs {max_gsd:g}m，比较条件不一致")
    clouds = [item.cloud_cover for item in captures if item.cloud_cover is not None]
    cloud_ok = not any(value > _CLOUD_COVER_HIGH for value in clouds)
    if not cloud_ok:
        notes.append("云量高，差异可能是云/影")
    months = [_capture_month(item.captured_at) for item in captures]
    known_months = [item for item in months if item is not None]
    season_spread = 0
    if len(known_months) >= 2:
        season_spread = max(
            _month_circular_delta(left, right)
            for i, left in enumerate(known_months)
            for right in known_months[i + 1 :]
        )
    if season_spread >= _SEASON_SPREAD_MONTHS:
        notes.append("采集月份相距较大，差异可能含季节或物候伪变化")
    comparable = resolution_consistent and cloud_ok and season_spread < _SEASON_SPREAD_MONTHS
    return {
        "resolution_consistent": resolution_consistent,
        "cloud_ok": cloud_ok,
        "season_spread_months": season_spread,
        "comparable": comparable,
        "notes": notes,
    }


def _capture_month(captured_at: str | None) -> int | None:
    if captured_at is None or len(captured_at) < 7:
        return None
    try:
        return date.fromisoformat(captured_at[:10]).month
    except ValueError:
        return None


def _month_circular_delta(left: int, right: int) -> int:
    delta = abs(left - right)
    return min(delta, 12 - delta)


def _capture_sort_key(item: CapturedScene) -> tuple[str, str]:
    return (item.captured_at or "", item.image_id)


def _source_payload(capture: CapturedScene) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_id": capture.image_id,
        "captured_at": capture.captured_at,
        "resolution_m": capture.resolution_m,
        "cloud_cover": capture.cloud_cover,
        "collection": capture.collection,
        "scene_id": capture.scene_id,
        "coverage": capture.coverage,
        capture.label_key: capture.label,
    }
    return payload


def _source_artifacts(captures: list[CapturedScene]) -> dict[str, Any]:
    image_ids = [item.image_id for item in captures]
    images = {
        item.image_id: str(item.path)
        for item in captures
        if item.path is not None
    }
    artifacts: dict[str, Any] = {"image_ids": image_ids}
    if images:
        artifacts["images"] = images
    return artifacts


def _ok(
    operation: str,
    *,
    sources: list[CapturedScene],
    extra: dict[str, Any],
    applied: dict[str, Any],
    artifacts: dict[str, Any],
    assumptions: list[str],
) -> Observation:
    result: dict[str, Any] = {
        "operation": operation,
        "sources": [_source_payload(item) for item in sources],
        "applied": applied,
        "assumptions": assumptions,
    }
    result.update(extra)
    _merge_source_ids(artifacts, sources)
    return Observation(
        ok=True,
        result=_strip_forbidden(result),
        artifacts=artifacts,
    )


def _merge_source_ids(artifacts: dict[str, Any], sources: list[CapturedScene]) -> None:
    existing = artifacts.get("image_ids")
    ids = [item.image_id for item in sources]
    if isinstance(existing, list):
        for image_id in ids:
            if image_id not in existing:
                existing.append(image_id)
    else:
        artifacts["image_ids"] = ids
    images = artifacts.setdefault("images", {})
    if isinstance(images, dict):
        for item in sources:
            if item.path is not None:
                images[item.image_id] = str(item.path)


def _merge_artifacts(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    incoming_images = incoming.get("images")
    if isinstance(incoming_images, dict):
        images = target.setdefault("images", {})
        if isinstance(images, dict):
            images.update(incoming_images)
    incoming_ids = incoming.get("image_ids")
    if isinstance(incoming_ids, list):
        ids = target.setdefault("image_ids", [])
        if isinstance(ids, list):
            for image_id in incoming_ids:
                if image_id not in ids:
                    ids.append(image_id)
    for key, value in incoming.items():
        if key in {"images", "image_ids"}:
            continue
        if key not in target:
            target[key] = value
        if isinstance(target.get("images"), dict) and key not in {"overlay_image_id", "diff_image_id", "image_path", "diff_image_path"}:
            if isinstance(value, str) and key.startswith("img_"):
                target["images"][key] = value


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _optional_str(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _as_float(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return float(raw.strip())
        except ValueError:
            return None
    return None


def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)
