"""stage6：使用 groundtruth 做验证、泄漏检查与质量评分。

groundtruth 仅在本阶段使用。
轨迹质量（TAO 形态、递进链、流畅度）已由 stage5 judge 拒绝采样负责，
本阶段只做 GT 相关检查：泄漏、覆盖、距离、一致性、禁止 Tool。
quality_score 基分 = traj.stage5_judge_score（缺省 1.0）。
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Callable, Optional, Union

from pydantic import BaseModel, Field

from pipeline.coarse_tool_policy import COARSE_FORBIDDEN_SEED_TOOLS
from pipeline.config import Settings, get_settings
from pipeline.llm import call_structured
from pipeline.schemas import (
    AgentRole,
    LocationHypothesis,
    SubmitAnswerResult,
    Trajectory,
    TrajectoryVerificationReport,
    VerificationResult,
)


class PlaceHints(BaseModel):
    """逆地理得到的地点提示（供覆盖检查与泄漏 LLM）。"""

    country: str
    region: Optional[str] = None
    city: Optional[str] = None
    display_name: Optional[str] = None


# 可注入：groundtruth (lat,lng) → PlaceHints；亦兼容旧式 (country, region) 元组
ReverseGeocodeFn = Callable[
    [tuple[float, float]],
    Union[PlaceHints, tuple[str, Optional[str]]],
]


class LeakageJudgeResult(BaseModel):
    """泄漏 LLM-as-judge：是否直接使用 GT / 后见之明。"""

    leaked: bool
    reasons: list[str] = Field(default_factory=list)


# Agent1 训练轨迹禁止出现的种子 Tool（与 stage5 / SPEC 一致）
_COARSE_FORBIDDEN_TRAJECTORY_TOOLS: frozenset[str] = COARSE_FORBIDDEN_SEED_TOOLS

# 粗略坐标模式：程序化泄漏兜底
_COORD_RE = re.compile(
    r"(?<!\d)(-?\d{1,2}\.\d{2,})\s*[,，]\s*(-?\d{1,3}\.\d{2,})(?!\d)"
)


def _normalize_place_name(name: str) -> str:
    """通用地点名规范化，不维护国家/地区枚举。"""
    text = unicodedata.normalize("NFKD", name).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(
        r"(?:province|state|region|prefecture|county|city|省|市|州|区|县)$",
        "",
        text.strip(),
    )
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


# 仅用于 stage6 对 GT 反向地理编码结果的覆盖检查（中英别名），
# 绝不参与 Agent1 视频事实抽取 / Observation 闭包。
_GT_COUNTRY_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"china", "中国", "prc", "中华人民共和国"}),
    frozenset({"france", "法国", "république française", "republique francaise"}),
    frozenset({"japan", "日本", "nippon"}),
    frozenset(
        {
            "united states",
            "united states of america",
            "usa",
            "u.s.",
            "u.s.a.",
            "us",
            "美国",
        }
    ),
    frozenset({"united kingdom", "uk", "u.k.", "britain", "great britain", "英国"}),
)
_GT_REGION_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"henan", "河南", "河南省"}),
    frozenset({"île-de-france", "ile-de-france", "ile de france"}),
)


def _gt_alias_cluster(
    name: str, groups: tuple[frozenset[str], ...]
) -> set[str]:
    """GT 覆盖检查用别名簇。"""
    key = name.strip().casefold()
    if not key:
        return set()
    for group in groups:
        if key in group or any(key in g or g in key for g in group if len(g) >= 2):
            return set(group) | {key}
    return {key}


def _names_cover(
    candidates: list[str],
    target: str,
    *,
    alias_groups: tuple[frozenset[str], ...] = (),
) -> bool:
    """候选列表是否覆盖规范化后的目标地名。"""
    if alias_groups:
        target_set = _gt_alias_cluster(target, alias_groups)
        if not target_set:
            return True
        for cand in candidates:
            if _gt_alias_cluster(cand, alias_groups) & target_set:
                return True
    normalized_target = _normalize_place_name(target)
    if not normalized_target:
        return True
    for cand in candidates:
        normalized_candidate = _normalize_place_name(cand)
        if normalized_candidate and (
            normalized_candidate in normalized_target
            or normalized_target in normalized_candidate
        ):
            return True
    return False


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """球面距离（km）；优先 geopy，不可用时回退 haversine。"""
    try:
        from geopy.distance import geodesic

        return float(geodesic(a, b).kilometers)
    except Exception:
        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        return 6371.0 * 2 * math.asin(math.sqrt(h))


def _normalize_place_hints(
    raw: Union[PlaceHints, tuple[str, Optional[str]]],
) -> PlaceHints:
    """将注入的逆地理结果归一为 PlaceHints。"""
    if isinstance(raw, PlaceHints):
        return raw
    country, region = raw
    return PlaceHints(country=country, region=region)


def _reverse_geocode_nominatim(coords: tuple[float, float]) -> PlaceHints:
    """Nominatim 反向地理编码 → PlaceHints。"""
    from geopy.geocoders import Nominatim

    settings = get_settings()
    geolocator = Nominatim(
        user_agent=settings.NOMINATIM_USER_AGENT,
        timeout=settings.NOMINATIM_TIMEOUT_SEC,
    )
    loc = geolocator.reverse(coords, language="zh", exactly_one=True)
    if loc is None or not getattr(loc, "raw", None):
        # 中文失败时回退英文
        loc = geolocator.reverse(coords, language="en", exactly_one=True)
    if loc is None or not getattr(loc, "raw", None):
        raise ValueError(f"无法反向解析坐标: {coords}")
    address = loc.raw.get("address", {})
    if not isinstance(address, dict):
        address = {}
    country = address.get("country")
    if not country:
        raise ValueError(f"反向解析缺少 country: {coords}")
    region = (
        address.get("state")
        or address.get("region")
        or address.get("province")
        or address.get("county")
    )
    city = (
        address.get("city")
        or address.get("town")
        or address.get("municipality")
        or address.get("city_district")
    )
    display = getattr(loc, "address", None) or loc.raw.get("display_name")
    return PlaceHints(
        country=str(country),
        region=str(region) if region else None,
        city=str(city) if city else None,
        display_name=str(display) if display else None,
    )


def default_reverse_geocode(coords: tuple[float, float]) -> PlaceHints:
    """默认反向地理编码（仅 Nominatim）。测试应 mock 或注入 reverse_geocode。"""
    return _reverse_geocode_nominatim(coords)


def _collect_thought_text(traj: Trajectory) -> str:
    return "\n".join(s.thought for s in traj.steps)


def _collect_trajectory_visible_text(traj: Trajectory) -> str:
    """组装泄漏 judge 可见全文：user_query / Thought / Action / Obs / 角色输出。"""
    parts: list[str] = [
        f"agent_role: {traj.agent_role.value}",
        f"user_query:\n{traj.user_query}",
    ]
    for i, step in enumerate(traj.steps, start=1):
        parts.append(f"--- Step {i} Thought ---\n{step.thought}")
        parts.append(
            f"--- Step {i} Action ---\n"
            f"tool={step.action.tool} params={json.dumps(step.action.params, ensure_ascii=False)}"
        )
        if step.observation is not None:
            parts.append(
                f"--- Step {i} Observation ---\n"
                f"{json.dumps(step.observation, ensure_ascii=False)}"
            )
    if traj.coarse_output is not None:
        parts.append(
            "--- coarse_output ---\n"
            f"{traj.coarse_output.model_dump_json()}"
        )
    if traj.fine_output is not None:
        parts.append(
            "--- fine_output ---\n"
            f"{traj.fine_output.model_dump_json()}"
        )
    if traj.verifier_output is not None:
        parts.append(
            "--- verifier_output ---\n"
            f"{traj.verifier_output.model_dump_json()}"
        )
    return "\n".join(parts)


def _coord_near(
    lat: float,
    lng: float,
    target: tuple[float, float],
    *,
    tol_deg: float = 0.05,
) -> bool:
    return abs(lat - target[0]) <= tol_deg and abs(lng - target[1]) <= tol_deg


def _detect_coord_leakage(
    traj: Trajectory,
    groundtruth: tuple[float, float],
) -> list[str]:
    """窄化程序化坐标泄漏兜底（不再因 FINE 早步近 GT 坐标 hard-fail）。"""
    reasons: list[str] = []
    text = _collect_thought_text(traj)
    role = traj.agent_role

    if role == AgentRole.COARSE:
        if traj.coarse_output is not None:
            summary = traj.coarse_output.reasoning_summary
            if _COORD_RE.search(summary):
                reasons.append("COARSE coarse_output 出现坐标")
        if _COORD_RE.search(text):
            reasons.append("COARSE Thought 出现坐标")
        return reasons

    if role == AgentRole.FINE:
        # FINE 允许在证据支持下尽早写出精确坐标；交给整链 LLM judge
        return reasons

    # VERIFIER：仅当「把 GT 当已知正确答案」话术 + 近 GT 且非 handoff
    gt_in_thought = False
    for match in _COORD_RE.finditer(text):
        try:
            lat, lng = float(match.group(1)), float(match.group(2))
        except ValueError:
            continue
        if not _coord_near(lat, lng, groundtruth):
            continue
        cand = traj.fine_handoff
        if cand is not None and _coord_near(
            cand.latitude, cand.longitude, groundtruth, tol_deg=0.01
        ):
            # handoff 本身就接近 GT：复述候选坐标不算程序化泄漏
            continue
        if cand is not None and _coord_near(
            lat, lng, (cand.latitude, cand.longitude), tol_deg=0.05
        ):
            continue
        gt_in_thought = True
        break
    if gt_in_thought and re.search(
        r"真实|正确答案|ground\s*truth|实际坐标|真值", text, re.IGNORECASE
    ):
        reasons.append("VERIFIER Thought 将 groundtruth 当作已知答案")
    return reasons


def _run_leakage_llm_judge(
    traj: Trajectory,
    groundtruth: tuple[float, float],
    place: PlaceHints,
) -> LeakageJudgeResult:
    """整链泄漏判定：直接使用 GT / 后见之明 → leaked；定位准本身不算泄漏。"""
    visible = _collect_trajectory_visible_text(traj)
    gt_hints = {
        "latitude": groundtruth[0],
        "longitude": groundtruth[1],
        "country": place.country,
        "region": place.region,
        "city": place.city,
        "display_name": place.display_name,
    }
    handoff_hint = ""
    if traj.agent_role == AgentRole.VERIFIER and traj.fine_handoff is not None:
        handoff_hint = (
            "\nfine_handoff（合法验证候选，复述不算泄露）: "
            f"{traj.fine_handoff.model_dump_json()}"
        )
    prompt = (
        "你是地理定位训练数据的答案泄漏审查员。\n"
        "从整条推理链（user_query + Thought/Action/Observation + 角色输出）判断"
        "是否「直接使用 groundtruth / 后见之明」。\n"
        "规则：\n"
        "1. leaked=true 仅当：无图像/Obs/user_query 依据地粘贴最终答案；"
        "或明确把 GT 当作「正确答案/真值/官方答案」；"
        "或 COARSE 以最终精准 POI/坐标作结论；"
        "或 VERIFIER 把 GT 当已知正确答案（复述 fine_handoff 除外）。\n"
        "2. leaked=false：前向推理自然收束到与 GT 一致的地点/坐标；"
        "FINE 非终端步已写出精确地点但与 Obs/user_query 线索连贯；"
        "终端 submit_answer / fine_output 命中 GT；"
        "复用 user_query 中的「已知线索」；"
        "COARSE 出现非最终答案的候选地区（策略 B）。\n"
        "3. 重要：定位到准确地点本身 ≠ 泄漏。\n"
        f"agent_role: {traj.agent_role.value}\n"
        f"groundtruth: {json.dumps(gt_hints, ensure_ascii=False)}\n"
        f"{handoff_hint}\n"
        f"trajectory:\n{visible}\n"
    )
    return call_structured(prompt, LeakageJudgeResult)


def _detect_leakage(
    traj: Trajectory,
    groundtruth: tuple[float, float],
    *,
    enabled: bool,
    place_hints: Optional[PlaceHints] = None,
    run_llm: bool = True,
) -> tuple[bool, list[str], list[str]]:
    """泄漏检测。返回 (detected, hard_reasons, soft_warnings)。"""
    if not enabled:
        return False, [], []

    hard: list[str] = []
    soft: list[str] = []

    hard.extend(_detect_coord_leakage(traj, groundtruth))

    if run_llm:
        place = place_hints
        if place is None:
            try:
                place = default_reverse_geocode(groundtruth)
            except Exception as exc:  # noqa: BLE001
                soft.append(f"泄漏 judge 逆地理失败: {exc}")
                place = PlaceHints(country="unknown")
        try:
            result = _run_leakage_llm_judge(traj, groundtruth, place)
            if result.leaked:
                hard.extend(result.reasons or ["LLM 判定存在直接使用 GT / 后见之明"])
        except Exception as exc:  # noqa: BLE001
            # judge 失败不误杀：仅 soft
            soft.append(f"泄漏 LLM-as-judge 调用失败: {exc}")

    return (len(hard) > 0), hard, soft


def _validate_coarse_regions_format(hyp: LocationHypothesis) -> list[str]:
    """行政区语义由 stage5 judge 验证，此处不做程序化后缀表。"""
    _ = hyp
    return []


def _check_coarse_coverage(
    hyp: LocationHypothesis,
    groundtruth: tuple[float, float],
    reverse_geocode: ReverseGeocodeFn,
) -> tuple[bool, list[str], list[str]]:
    """检查 LocationHypothesis 是否覆盖真值国家/一级行政区。

    regions 为空表示仅收窄到国家，不因空列表 hard-fail。
    regions 非空时未覆盖 GT 一级行政区 → hard-fail。
    返回 (ok, hard_fails, soft_warnings)。
    """
    hard: list[str] = []
    soft: list[str] = []
    hard.extend(_validate_coarse_regions_format(hyp))
    try:
        place = _normalize_place_hints(reverse_geocode(groundtruth))
        country, region = place.country, place.region
    except Exception as exc:  # noqa: BLE001
        soft.append(f"反向地理编码失败: {exc}")
        return len(hard) == 0, hard, soft

    countries_norm = [c.strip() for c in hyp.possible_countries if c.strip()]
    if not _names_cover(
        countries_norm, country, alias_groups=_GT_COUNTRY_ALIAS_GROUPS
    ):
        hard.append(
            f"COARSE possible_countries 未覆盖真值国家 {country!r}"
        )

    if region:
        regions_norm = [r.strip() for r in hyp.possible_regions if r.strip()]
        if regions_norm and not _names_cover(
            regions_norm, region, alias_groups=_GT_REGION_ALIAS_GROUPS
        ):
            hard.append(
                f"COARSE possible_regions 未覆盖真值一级行政区 {region!r}"
            )

    return len(hard) == 0, hard, soft


def _check_fine_distance(
    fine: SubmitAnswerResult,
    groundtruth: tuple[float, float],
    threshold_km: float,
) -> tuple[float, list[str]]:
    """计算距离误差；超过阈值 → hard fail。"""
    dist = _haversine_km((fine.latitude, fine.longitude), groundtruth)
    hard: list[str] = []
    if dist > threshold_km:
        hard.append(
            f"FINE 距离误差 {dist:.2f}km 超过阈值 {threshold_km}km"
        )
    return dist, hard


def _check_verifier_consistency(
    verifier: VerificationResult,
    distance_error_km: Optional[float],
    threshold_km: float,
) -> tuple[list[str], list[str]]:
    """Agent3 verdict 是否与 Agent2 误差一致性匹配。"""
    hard: list[str] = []
    soft: list[str] = []
    if distance_error_km is None:
        soft.append("缺少 FINE 距离误差，无法校验 VERIFIER 一致性")
        return hard, soft

    within = distance_error_km <= threshold_km
    if within and verifier.verdict == "fail":
        soft.append(
            "FINE 距离在阈值内但 VERIFIER 判 fail（可能过严）"
        )
    if (not within) and verifier.verdict == "pass":
        hard.append(
            "FINE 距离超出阈值但 VERIFIER 判 pass（与误差不一致）"
        )
    return hard, soft


def _detect_coarse_forbidden_tools(traj: Trajectory) -> list[str]:
    """程序化兜底：Agent1 轨迹不得含 web_search/map_query/RIS/submit。"""
    hard: list[str] = []
    for i, step in enumerate(traj.steps, start=1):
        name = step.action.tool
        if name in _COARSE_FORBIDDEN_TRAJECTORY_TOOLS:
            hard.append(f"COARSE 轨迹禁止 Tool: step{i}={name}")
    return hard


def verify_and_score(
    traj: Trajectory,
    groundtruth: tuple[float, float],
    *,
    settings: Optional[Settings] = None,
    reverse_geocode: Optional[ReverseGeocodeFn] = None,
    run_leakage_llm: bool = True,
) -> TrajectoryVerificationReport:
    """使用 groundtruth 验证轨迹并给出质量分。

    groundtruth 仅在本阶段使用。stage6 只做 GT 相关检查；
    轨迹质量已由 stage5 judge 拒绝采样负责，本阶段不再做形态裁判 /
    递进链裁判 / 合理性 soft judge。
    判定顺序：泄漏（LLM+坐标兜底）→ 角色专项。
    quality_score：基分 = traj.stage5_judge_score（缺省 1.0）；
    有 hard-fail → min(quality, 0.3)；FINE 按距离误差混合；clamp [0,1]。
    """
    cfg = settings or get_settings()
    geocode = reverse_geocode or default_reverse_geocode
    threshold = cfg.DISTANCE_ERROR_THRESHOLD_KM

    hard_fail_reasons: list[str] = []
    soft_warnings: list[str] = []
    distance_error_km: Optional[float] = None

    # 逆地理一次，供覆盖检查与泄漏 LLM 共用
    place_for_leak: Optional[PlaceHints] = None
    try:
        place_for_leak = _normalize_place_hints(geocode(groundtruth))
    except Exception as exc:  # noqa: BLE001
        soft_warnings.append(f"反向地理编码失败: {exc}")

    # --- 泄漏检测 ---
    leaked, leak_hard, leak_soft = _detect_leakage(
        traj,
        groundtruth,
        enabled=cfg.ANSWER_LEAK_CHECK_ENABLED,
        place_hints=place_for_leak,
        run_llm=run_leakage_llm and cfg.ANSWER_LEAK_CHECK_ENABLED,
    )
    hard_fail_reasons.extend(leak_hard)
    soft_warnings.extend(leak_soft)

    # --- 角色专项（仅 GT 相关）---
    if traj.agent_role == AgentRole.COARSE:
        if traj.coarse_output is None:
            hard_fail_reasons.append("COARSE 缺少 coarse_output")
        else:
            _ok, hard, soft = _check_coarse_coverage(
                traj.coarse_output, groundtruth, geocode
            )
            hard_fail_reasons.extend(hard)
            soft_warnings.extend(soft)
        hard_fail_reasons.extend(_detect_coarse_forbidden_tools(traj))

    elif traj.agent_role == AgentRole.FINE:
        if traj.fine_output is None:
            hard_fail_reasons.append("FINE 缺少 fine_output")
        else:
            distance_error_km, hard = _check_fine_distance(
                traj.fine_output, groundtruth, threshold
            )
            hard_fail_reasons.extend(hard)

    elif traj.agent_role == AgentRole.VERIFIER:
        if traj.verifier_output is None:
            hard_fail_reasons.append("VERIFIER 缺少 verifier_output")
        else:
            if traj.fine_handoff is not None:
                distance_error_km = _haversine_km(
                    (traj.fine_handoff.latitude, traj.fine_handoff.longitude),
                    groundtruth,
                )
            hard, soft = _check_verifier_consistency(
                traj.verifier_output, distance_error_km, threshold
            )
            hard_fail_reasons.extend(hard)
            soft_warnings.extend(soft)

    # --- 质量分：基分来自 stage5 judge ---
    quality = (
        float(traj.stage5_judge_score)
        if traj.stage5_judge_score is not None
        else 1.0
    )
    if hard_fail_reasons:
        quality = min(quality, 0.3)
    if soft_warnings:
        quality = max(0.0, quality - 0.05 * min(len(soft_warnings), 4))
    if distance_error_km is not None and traj.agent_role == AgentRole.FINE:
        if distance_error_km <= threshold:
            dist_factor = 1.0 - 0.5 * (distance_error_km / max(threshold, 1e-6))
            quality = min(1.0, 0.5 * quality + 0.5 * dist_factor)
        else:
            quality = min(quality, 0.2)
    quality = float(max(0.0, min(1.0, quality)))

    passed = len(hard_fail_reasons) == 0 and not leaked

    return TrajectoryVerificationReport(
        passed=passed,
        quality_score=quality,
        distance_error_km=distance_error_km,
        hard_fail_reasons=hard_fail_reasons,
        soft_warnings=soft_warnings,
        leakage_detected=leaked,
    )
