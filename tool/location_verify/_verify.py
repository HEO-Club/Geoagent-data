"""location_verify 验证：街景比对 + 文字一致性
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tool.contract import Observation, RuntimeContext, declared_inputs
from tool.geocode.geocode import execute as geocode_exec
from tool.image_compare.compare import execute as image_compare_exec
from tool.ocr_read.recognize import execute as ocr_recognize_exec
from tool.runtime.image_store import ImageResolveError, resolve_image_ref
from tool.streetview_query.capture import execute as sv_capture_exec
from tool.streetview_query.open import execute as sv_open_exec

# --- 默认参数 ---
_DEFAULT_HEADINGS = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
_DEFAULT_PITCH = 0.0
_DEFAULT_FOV = 90.0
_INLIER_THRESHOLD = 0.30
_MIN_MATCH_COUNT = 20
_MEANINGFUL_MIN_LEN = 2

# --- 判定值 ---
_VERDICT_MATCHED = "matched"
_VERDICT_NO_MATCH = "no_positive_match"
_VERDICT_INCONCLUSIVE = "inconclusive"

_CONCLUSION_MATCH = "match"
_CONCLUSION_WEAK = "weak"
_CONCLUSION_NONE = "none"
_CONCLUSION_PARTIAL = "partial"
_CONCLUSION_NO_TEXT = "no_text"

_ASSUMPTIONS = (
    "弱图像匹配或文字缺失仅表示「未确认」，不等于答案错误",
    "image_compare 特征相似度是证据，不自动等于地点相同",
    "ocr_read 只提取原始文字，不改写、不纠错；误识别文字只算 partial",
    "街景与待定位图视角不同会造成弱匹配",
)


class LocationVerifyInputError(Exception):
    """location 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def execute_verify(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 location_verify.verify：两路证据融合输出答案可信度判定。"""

    del purpose
    ctx = _retain_streetview_images(ctx)
    raw = dict(inputs or {})
    if "location" not in raw:
        return _fail("缺少必填输入 location", "missing_input")
    declared = declared_inputs(
        raw,
        "location",
        "reference_image",
        "headings",
        "provider",
        "compare_method",
    )
    try:
        area, coords = _split_location(declared["location"])
        headings = _parse_headings(declared.get("headings"))
        provider = declared.get("provider")
        compare_method = declared.get("compare_method") or "auto"
    except LocationVerifyInputError as exc:
        return _fail(str(exc), exc.error_code)

    reference = declared.get("reference_image") or "$current_image"
    try:
        ref_id, _ = resolve_image_ref(reference, ctx)
    except ImageResolveError as exc:
        return _fail(str(exc), exc.error_code)

    applied = {
        "provider": provider if isinstance(provider, str) else None,
        "headings": headings,
        "compare_method": compare_method,
        "reference_image": ref_id,
    }

    # 地点元信息（可选）：地名时尝试 geocode 记录坐标与粒度
    resolved = None
    match_level = None
    if area is not None:
        resolved, match_level = _try_geocode(area, ctx)

    # Part 1：打开街景
    open_obs = _open_streetview(area=area, coords=coords, provider=provider, ctx=ctx)
    if not open_obs.ok:
        return _verdict(
            verdict=_VERDICT_INCONCLUSIVE,
            confidence=0.10,
            location_input=declared["location"],
            resolved=resolved,
            match_level=match_level,
            coverage=False,
            applied=applied,
            limitations=[f"街景无法打开：{open_obs.error or open_obs.error_code or '未知错误'}"],
        )
    open_result = open_obs.result if isinstance(open_obs.result, dict) else {}
    coverage = bool(open_result.get("coverage"))
    session_id = open_obs.session if isinstance(open_obs.session, str) else open_result.get("session")
    if not coverage:
        return _verdict(
            verdict=_VERDICT_INCONCLUSIVE,
            confidence=0.10,
            location_input=declared["location"],
            resolved=resolved,
            match_level=match_level,
            coverage=False,
            applied=applied,
            limitations=["答案点无街景覆盖，无法用街景验证"],
        )

    # Part 1：多朝向截图 + image_compare，找最相似方位
    sweep: list[dict[str, Any]] = []
    for heading in headings:
        item = _sweep_heading(
            session_id=session_id,
            ref_id=ref_id,
            heading=heading,
            compare_method=compare_method,
            ctx=ctx,
        )
        if item is not None:
            sweep.append(item)
    best = _best_sweep(sweep)
    image_signal = _build_image_signal(best, sweep)

    # Part 2：OCR 文字一致性
    query_texts = _ocr_texts(
        ocr_recognize_exec(purpose="提取待定位原图文字", inputs={"image": ref_id}, ctx=ctx)
    )
    street_texts: list[str] = []
    for item in sweep:
        obs = ocr_recognize_exec(
            purpose="提取街景截图文字", inputs={"image": item["image_id"]}, ctx=ctx
        )
        street_texts.extend(_ocr_texts(obs))
    ocr_signal = _build_ocr_signal(query_texts, _dedupe(street_texts))

    # 用完删除街景截图，遵守街景「用后删除」的许可约束
    _cleanup_images(ctx, [item["image_id"] for item in sweep])

    # 融合
    verdict, confidence = _fusion(image_signal["conclusion"], ocr_signal["conclusion"])

    return _verdict(
        verdict=verdict,
        confidence=confidence,
        location_input=declared["location"],
        resolved=resolved,
        match_level=match_level,
        coverage=coverage,
        applied=applied,
        image_signal=image_signal,
        ocr_signal=ocr_signal,
    )


def _split_location(raw: Any) -> tuple[str | None, list[float] | None]:
    """把 location 拆成 (地名字符串, 坐标)；二者必有其一非 None。"""

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise LocationVerifyInputError("location 不能为空", "invalid_input")
        return text, None
    if isinstance(raw, (list, tuple)):
        nums = [_as_float(value) for value in raw]
        if len(nums) < 2 or None in nums[:2]:
            raise LocationVerifyInputError("location 坐标数组至少需要 [lon, lat]", "invalid_input")
        return None, [float(nums[0]), float(nums[1])]
    if isinstance(raw, dict):
        lon = _as_float(raw.get("lon", raw.get("lng")))
        lat = _as_float(raw.get("lat"))
        if lon is None or lat is None:
            raise LocationVerifyInputError("location 坐标对象需要 lon 与 lat", "invalid_input")
        return None, [lon, lat]
    raise LocationVerifyInputError("location 必须是地名、[lon,lat] 或 {lon,lat}", "invalid_input")


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_headings(raw: Any) -> list[float]:
    if isinstance(raw, (list, tuple)):
        nums = [_as_float(value) for value in raw]
        nums = [value for value in nums if value is not None]
        return nums or list(_DEFAULT_HEADINGS)
    single = _as_float(raw)
    if single is not None:
        return [single]
    return list(_DEFAULT_HEADINGS)


def _try_geocode(query: str, ctx: RuntimeContext | None) -> tuple[dict[str, Any] | None, str | None]:
    """尽力解析地名坐标与粒度；失败返回 (None, None)，不中断验证。"""

    obs = geocode_exec(
        purpose="解析答案地点坐标与粒度",
        inputs={"query": query, "direction": "forward"},
        ctx=ctx,
    )
    if not obs.ok or not isinstance(obs.result, dict):
        return None, None
    results = obs.result.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return None, None
    first = results[0]
    loc = first.get("location")
    resolved = loc if isinstance(loc, dict) and "lon" in loc and "lat" in loc else None
    match_level = first.get("match_level")
    return resolved, (match_level if isinstance(match_level, str) else None)


def _open_streetview(
    *,
    area: str | None,
    coords: list[float] | None,
    provider: Any,
    ctx: RuntimeContext | None,
) -> Observation:
    open_inputs: dict[str, Any] = {}
    if coords is not None:
        open_inputs["coordinates"] = coords
    elif area is not None:
        open_inputs["area"] = area
    if isinstance(provider, str) and provider.strip():
        open_inputs["provider"] = provider.strip()
    return sv_open_exec(
        purpose="在答案地点建立街景会话用于比对",
        inputs=open_inputs,
        ctx=ctx,
    )


def _sweep_heading(
    *,
    session_id: str,
    ref_id: str,
    heading: float,
    compare_method: str,
    ctx: RuntimeContext | None,
) -> dict[str, Any] | None:
    """截取一个朝向并与其比对；任一子步失败返回 None。"""

    cap = sv_capture_exec(
        purpose="截取指定朝向街景用于比对",
        inputs={"session": session_id, "heading": heading, "pitch": _DEFAULT_PITCH, "fov": _DEFAULT_FOV},
        ctx=ctx,
    )
    if not cap.ok or not isinstance(cap.result, dict):
        return None
    cap_image_id = cap.result.get("image_id")
    if not isinstance(cap_image_id, str) or not cap_image_id:
        return None
    cmp = image_compare_exec(
        purpose="比较街景截图与待定位原图",
        inputs={"images": [ref_id, cap_image_id], "method": compare_method},
        ctx=ctx,
    )
    if not cmp.ok or not isinstance(cmp.result, dict):
        return None
    metrics = _first_pair_metrics(cmp.result)
    return {"heading": heading, "image_id": cap_image_id, **metrics}


def _first_pair_metrics(result: dict[str, Any]) -> dict[str, Any]:
    pairs = result.get("pairs")
    first = pairs[0] if isinstance(pairs, list) and pairs and isinstance(pairs[0], dict) else {}
    return {
        "inlier_ratio": _as_float(first.get("inlier_ratio")),
        "match_count": _as_int(first.get("match_count")),
        "score": _as_float(first.get("score")),
        "method": first.get("method_used") if isinstance(first.get("method_used"), str) else None,
    }


def _best_sweep(sweep: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not sweep:
        return None
    return max(sweep, key=_sweep_key)


def _sweep_key(item: dict[str, Any]) -> tuple[float, int, float]:
    inlier = item.get("inlier_ratio")
    count = item.get("match_count")
    score = item.get("score")
    return (
        inlier if isinstance(inlier, (int, float)) else -1.0,
        count if isinstance(count, int) else 0,
        score if isinstance(score, (int, float)) else -1.0,
    )


def _build_image_signal(best: dict[str, Any] | None, sweep: list[dict[str, Any]]) -> dict[str, Any]:
    conclusion = _CONCLUSION_NONE
    if best is not None:
        inlier = best.get("inlier_ratio")
        count = best.get("match_count")
        if (
            isinstance(inlier, (int, float))
            and isinstance(count, int)
            and inlier >= _INLIER_THRESHOLD
            and count >= _MIN_MATCH_COUNT
        ):
            conclusion = _CONCLUSION_MATCH
        elif isinstance(count, int) and count > 0:
            conclusion = _CONCLUSION_WEAK
    return {
        "best_heading": best.get("heading") if best else None,
        "capture_image_id": best.get("image_id") if best else None,
        "inlier_ratio": best.get("inlier_ratio") if best else None,
        "match_count": best.get("match_count") if best else None,
        "score": best.get("score") if best else None,
        "method": best.get("method") if best else None,
        "conclusion": conclusion,
        "compared_headings": [item.get("heading") for item in sweep],
    }


def _ocr_texts(obs: Observation) -> list[str]:
    if not obs.ok or not isinstance(obs.result, dict):
        return []
    items = obs.result.get("items")
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
    return out


def _dedupe(texts: list[str]) -> list[str]:
    return list(dict.fromkeys(texts))


def _build_ocr_signal(query_texts: list[str], street_texts: list[str]) -> dict[str, Any]:
    a = _dedupe([text for text in query_texts if _meaningful(text)])
    b = _dedupe([text for text in street_texts if _meaningful(text)])
    common = [text for text in a if text in b]
    partial: list[dict[str, str]] = []
    if not common and a and b:
        for x in a:
            for y in b:
                if x != y and (x in y or y in x):
                    partial.append({"query": x, "streetview": y})
    if not a or not b:
        conclusion = _CONCLUSION_NO_TEXT
    elif common:
        conclusion = _CONCLUSION_MATCH
    elif partial:
        conclusion = _CONCLUSION_PARTIAL
    else:
        conclusion = _CONCLUSION_NONE
    return {
        "query_text": a,
        "streetview_text": b,
        "common": common,
        "partial": partial,
        "conclusion": conclusion,
    }


def _meaningful(text: str) -> bool:
    return len(text) >= _MEANINGFUL_MIN_LEN


class _RetainImageStore:
    """包装 image_store，仅把 delete 置空。
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def delete(self, image_id: str) -> None:
        del image_id
        return None


def _retain_streetview_images(ctx: RuntimeContext | None) -> RuntimeContext | None:
    """让本工具期间产生的街景截图不被 capture 立即删除。"""

    if ctx is None or ctx.image_store is None:
        return ctx
    return replace(ctx, image_store=_RetainImageStore(ctx.image_store))


def _cleanup_images(ctx: RuntimeContext | None, image_ids: list[str]) -> None:
    """删除本工具期间保留的街景截图，避免把街景影像留在会话目录。"""

    store = ctx.image_store if ctx is not None else None
    if store is None:
        return
    underlying = getattr(store, "_store", store)
    for image_id in image_ids:
        try:
            underlying.delete(image_id)
        except Exception:
            continue


def _fusion(image_conclusion: str, ocr_conclusion: str) -> tuple[str, float]:
    """两路证据融合为最终 verdict + confidence。"""

    if image_conclusion == _CONCLUSION_MATCH or ocr_conclusion == _CONCLUSION_MATCH:
        if image_conclusion == _CONCLUSION_MATCH and ocr_conclusion == _CONCLUSION_MATCH:
            return _VERDICT_MATCHED, 0.85
        return _VERDICT_MATCHED, 0.70
    if image_conclusion == _CONCLUSION_WEAK or ocr_conclusion == _CONCLUSION_PARTIAL:
        return _VERDICT_NO_MATCH, 0.30
    return _VERDICT_NO_MATCH, 0.20


def _verdict(
    *,
    verdict: str,
    confidence: float,
    location_input: Any,
    resolved: dict[str, Any] | None,
    match_level: str | None,
    coverage: bool,
    applied: dict[str, Any],
    image_signal: dict[str, Any] | None = None,
    ocr_signal: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> Observation:
    result: dict[str, Any] = {
        "operation": "verify",
        "verdict": verdict,
        "confidence": confidence,
        "location": {
            "input": location_input,
            "resolved": resolved,
            "match_level": match_level,
        },
        "coverage": coverage,
        "signals": {
            "image": image_signal if image_signal is not None else _empty_image_signal(),
            "ocr": ocr_signal if ocr_signal is not None else _empty_ocr_signal(),
        },
        "applied": applied,
        "assumptions": list(_ASSUMPTIONS),
    }
    if limitations:
        result["limitations"] = limitations
    return Observation(ok=True, result=result)


def _empty_image_signal() -> dict[str, Any]:
    return {
        "best_heading": None,
        "capture_image_id": None,
        "inlier_ratio": None,
        "match_count": None,
        "score": None,
        "method": None,
        "conclusion": _CONCLUSION_NONE,
        "compared_headings": [],
    }


def _empty_ocr_signal() -> dict[str, Any]:
    return {
        "query_text": [],
        "streetview_text": [],
        "common": [],
        "partial": [],
        "conclusion": _CONCLUSION_NO_TEXT,
    }


def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)
