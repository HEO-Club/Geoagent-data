"""stage6：使用 groundtruth 做验证、泄漏检查与质量评分。

groundtruth 仅在本阶段使用；LLM-as-judge prompt 不得把 raw groundtruth
坐标作为“应输出答案”写入。
"""

from __future__ import annotations

import math
import re
from typing import Callable, Optional

from pydantic import BaseModel, Field

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

# 可注入的反向地理编码：groundtruth (lat,lng) → (country, region/admin1)
ReverseGeocodeFn = Callable[[tuple[float, float]], tuple[str, Optional[str]]]


class _JudgeResult(BaseModel):
    """LLM-as-judge 结构化输出（合理性评估）。"""

    reasonable: bool
    issues: list[str] = Field(default_factory=list)
    score: float = Field(ge=0, le=1)


# 粗略坐标模式：用于泄漏检测（不含把候选坐标误判为真值泄漏时的辅助）
_COORD_RE = re.compile(
    r"(?<!\d)(-?\d{1,2}\.\d{2,})\s*[,，]\s*(-?\d{1,3}\.\d{2,})(?!\d)"
)

# 国家别名：GT 反向地理编码名与模型输出（中/英）应对齐
_COUNTRY_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "china",
            "中国",
            "prc",
            "p.r.c.",
            "people's republic of china",
            "中华人民共和国",
        }
    ),
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
    frozenset({"france", "法国", "république française", "republique francaise"}),
    frozenset({"japan", "日本", "nippon"}),
    frozenset({"united kingdom", "uk", "u.k.", "britain", "great britain", "英国"}),
)

# 一级行政区常见别名（中英）
_REGION_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"henan", "河南", "河南省"}),
    frozenset({"beijing", "北京", "北京市"}),
    frozenset({"shanghai", "上海", "上海市"}),
    frozenset({"guangdong", "广东", "广东省"}),
    frozenset({"sichuan", "四川", "四川省"}),
    frozenset({"zhejiang", "浙江", "浙江省"}),
    frozenset({"jiangsu", "江苏", "江苏省"}),
    frozenset({"shandong", "山东", "山东省"}),
    frozenset({"hubei", "湖北", "湖北省"}),
    frozenset({"hunan", "湖南", "湖南省"}),
    frozenset({"île-de-france", "ile-de-france", "ile de france"}),
)


def _alias_cluster(
    name: str, groups: tuple[frozenset[str], ...]
) -> set[str]:
    """将地名归一到别名簇（小写）；未知名则仅含自身。"""
    key = name.strip().lower()
    if not key:
        return set()
    for group in groups:
        if key in group:
            return set(group)
        # 允许「河南省」对「河南」等包含关系落入同簇
        if any(key in g or g in key for g in group if len(g) >= 2):
            return set(group) | {key}
    return {key}


def _names_cover(
    candidates: list[str],
    target: str,
    groups: tuple[frozenset[str], ...],
) -> bool:
    """候选列表是否覆盖目标地名（含中英别名）。"""
    target_set = _alias_cluster(target, groups)
    if not target_set:
        return True
    for cand in candidates:
        cand_set = _alias_cluster(cand, groups)
        if cand_set & target_set:
            return True
        # 回退：子串包含（无别名表时）
        c = cand.strip().lower()
        t = target.strip().lower()
        if c and t and (c in t or t in c):
            return True
    return False

_CITY_HINT_RE = re.compile(
    r"\b(Paris|London|Tokyo|Beijing|Shanghai|New York|Berlin|Rome|Madrid|"
    r"Sydney|Moscow|Cairo|Dubai|Seoul|Singapore|Bangkok|Toronto|Chicago|"
    r"巴黎|伦敦|东京|北京|上海|纽约|柏林|罗马|马德里)\b",
    re.IGNORECASE,
)


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """球面距离（km）；优先 geopy，不可用时回退 haversine。"""
    try:
        from geopy.distance import geodesic

        return float(geodesic(a, b).kilometers)
    except Exception:
        # 离线回退，保证测试环境无 geopy 时仍可算距离
        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        return 6371.0 * 2 * math.asin(math.sqrt(h))


def _reverse_geocode_nominatim(
    coords: tuple[float, float],
) -> tuple[str, Optional[str]]:
    """Nominatim 反向地理编码。"""
    from geopy.geocoders import Nominatim

    from pipeline.config import get_settings

    settings = get_settings()
    geolocator = Nominatim(
        user_agent=settings.NOMINATIM_USER_AGENT,
        timeout=settings.NOMINATIM_TIMEOUT_SEC,
    )
    loc = geolocator.reverse(coords, language="en", exactly_one=True)
    if loc is None or not getattr(loc, "raw", None):
        raise ValueError(f"无法反向解析坐标: {coords}")
    address = loc.raw.get("address", {})
    country = address.get("country")
    if not country:
        raise ValueError(f"反向解析缺少 country: {coords}")
    region = (
        address.get("state")
        or address.get("region")
        or address.get("province")
        or address.get("county")
    )
    return str(country), (str(region) if region else None)


def _reverse_geocode_amap(coords: tuple[float, float]) -> tuple[str, Optional[str]]:
    """高德逆地理编码 → (country, region)。"""
    import json
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    from pipeline.config import get_settings

    settings = get_settings()
    api_key = (settings.AMAP_API_KEY or "").strip()
    if not api_key:
        raise ValueError("未配置 AMAP_API_KEY（MAP_PROVIDER=amap 时必填）")
    base = (settings.AMAP_BASE_URL or "").rstrip("/")
    lat, lng = float(coords[0]), float(coords[1])
    params = {
        "key": api_key,
        "location": f"{lng:.6f},{lat:.6f}",
        "extensions": "base",
        "output": "JSON",
    }
    url = f"{base}/v3/geocode/regeo?{urlencode(params)}"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=float(settings.AMAP_TIMEOUT_SEC)) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict) or str(payload.get("status")) != "1":
        raise ValueError(f"AMap reverse 失败: {payload!r}")
    regeo = payload.get("regeocode")
    if not isinstance(regeo, dict):
        raise ValueError(f"无法反向解析坐标: {coords}")
    component = regeo.get("addressComponent")
    if not isinstance(component, dict):
        raise ValueError(f"反向解析缺少 addressComponent: {coords}")
    country = component.get("country")
    if not country:
        raise ValueError(f"反向解析缺少 country: {coords}")
    region = component.get("province") or component.get("city")
    return str(country), (str(region) if region else None)


def default_reverse_geocode(coords: tuple[float, float]) -> tuple[str, Optional[str]]:
    """默认反向地理编码；按 MAP_PROVIDER 选择高德或 Nominatim。

    测试应 mock 本函数或注入 reverse_geocode。
    """
    from pipeline.config import get_settings

    provider = get_settings().MAP_PROVIDER.strip().lower()
    if provider in {"amap", "gaode", "高德"}:
        return _reverse_geocode_amap(coords)
    if provider in {"nominatim", "osm"}:
        return _reverse_geocode_nominatim(coords)
    raise ValueError(
        f"不支持的 MAP_PROVIDER={get_settings().MAP_PROVIDER!r}；可选: amap / nominatim"
    )


def _collect_thought_text(traj: Trajectory) -> str:
    return "\n".join(s.thought for s in traj.steps)


def _coord_near(
    lat: float,
    lng: float,
    target: tuple[float, float],
    *,
    tol_deg: float = 0.05,
) -> bool:
    return abs(lat - target[0]) <= tol_deg and abs(lng - target[1]) <= tol_deg


def _detect_leakage(
    traj: Trajectory,
    groundtruth: tuple[float, float],
    *,
    enabled: bool,
) -> tuple[bool, list[str]]:
    """程序化泄漏检测（1.6）。返回 (detected, reasons)。"""
    if not enabled:
        return False, []

    reasons: list[str] = []
    text = _collect_thought_text(traj)
    role = traj.agent_role

    # Thought 中写出与 groundtruth 几乎相同的坐标 → 泄漏
    # VERIFIER：允许引用 fine_handoff 候选坐标（即使候选碰巧靠近 GT）
    for match in _COORD_RE.finditer(text):
        lat_s, lng_s = match.group(1), match.group(2)
        try:
            lat, lng = float(lat_s), float(lng_s)
        except ValueError:
            continue
        if not _coord_near(lat, lng, groundtruth):
            continue
        if role == AgentRole.VERIFIER:
            cand = traj.fine_handoff
            if cand is not None and _coord_near(
                lat, lng, (cand.latitude, cand.longitude), tol_deg=0.05
            ):
                continue  # 候选坐标，非 GT 泄漏
        # Agent2 最后一步允许在 Action 中提交坐标；Thought 仍不应提前写出真值
        reasons.append("thought 中出现与 groundtruth 接近的坐标")
        break

    if role == AgentRole.COARSE:
        # Agent1 不得出现最终城市、精确地点或坐标
        if _CITY_HINT_RE.search(text):
            reasons.append("COARSE Thought 出现具体城市名")
        if traj.coarse_output is not None:
            summary = traj.coarse_output.reasoning_summary
            if _CITY_HINT_RE.search(summary):
                reasons.append("COARSE coarse_output 出现具体城市名")
            if _COORD_RE.search(summary):
                reasons.append("COARSE coarse_output 出现坐标")
        if _COORD_RE.search(text):
            reasons.append("COARSE Thought 出现坐标")

    elif role == AgentRole.FINE:
        # Agent2：非最后一步 Thought 不得出现与真值接近的坐标（后见之明）
        for i, step in enumerate(traj.steps[:-1]):
            for match in _COORD_RE.finditer(step.thought):
                try:
                    lat, lng = float(match.group(1)), float(match.group(2))
                except ValueError:
                    continue
                if _coord_near(lat, lng, groundtruth):
                    reasons.append(
                        f"FINE 非终端步 Thought[{i}] 过早出现真值坐标"
                    )
                    break

    elif role == AgentRole.VERIFIER:
        # Agent3 不得把 groundtruth 当作已知答案引用
        # （候选 fine_handoff 坐标允许出现；仅当同时出现“真实/正确答案”措辞+真值坐标时记泄漏）
        gt_in_thought = False
        for match in _COORD_RE.finditer(text):
            try:
                lat, lng = float(match.group(1)), float(match.group(2))
            except ValueError:
                continue
            if _coord_near(lat, lng, groundtruth):
                cand = traj.fine_handoff
                if cand is None or not _coord_near(
                    cand.latitude, cand.longitude, groundtruth, tol_deg=0.01
                ):
                    # 真值与候选不同，却写出了真值坐标
                    gt_in_thought = True
                    break
        if gt_in_thought and re.search(
            r"真实|正确答案|ground\s*truth|实际坐标", text, re.IGNORECASE
        ):
            reasons.append("VERIFIER Thought 将 groundtruth 当作已知答案")

    return (len(reasons) > 0), reasons


def _check_coarse_coverage(
    hyp: LocationHypothesis,
    groundtruth: tuple[float, float],
    reverse_geocode: ReverseGeocodeFn,
) -> tuple[bool, list[str], list[str]]:
    """检查 LocationHypothesis 是否覆盖真值国家/地区。

    返回 (ok, hard_fails, soft_warnings)。
    """
    hard: list[str] = []
    soft: list[str] = []
    try:
        country, region = reverse_geocode(groundtruth)
    except Exception as exc:  # noqa: BLE001 — 外部编码失败记 soft，避免整批崩溃
        soft.append(f"反向地理编码失败: {exc}")
        return True, hard, soft

    countries_norm = [c.strip() for c in hyp.possible_countries if c.strip()]
    if not _names_cover(countries_norm, country, _COUNTRY_ALIAS_GROUPS):
        hard.append(
            f"COARSE possible_countries 未覆盖真值国家 {country!r}"
        )

    if region:
        regions_norm = [r.strip() for r in hyp.possible_regions if r.strip()]
        if regions_norm and not _names_cover(
            regions_norm, region, _REGION_ALIAS_GROUPS
        ):
            soft.append(
                f"COARSE possible_regions 可能未覆盖真值地区 {region!r}"
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


def _run_llm_judge(traj: Trajectory) -> _JudgeResult:
    """合理性 judge；prompt 不含 raw groundtruth 坐标作为应输出答案。"""
    steps_brief = []
    for i, step in enumerate(traj.steps, start=1):
        steps_brief.append(
            f"Step {i}: thought={step.thought!r}; tool={step.action.tool}"
        )
    prompt = (
        "评估下列地理定位 Agent 轨迹的推理是否合理、前后自洽。\n"
        "不要假设存在唯一正确坐标；不要把任何坐标当作必须输出的标准答案。\n"
        f"agent_role: {traj.agent_role.value}\n"
        f"steps:\n" + "\n".join(steps_brief)
    )
    return call_structured(prompt, _JudgeResult)


def verify_and_score(
    traj: Trajectory,
    groundtruth: tuple[float, float],
    *,
    settings: Optional[Settings] = None,
    reverse_geocode: Optional[ReverseGeocodeFn] = None,
    run_judge: bool = True,
) -> TrajectoryVerificationReport:
    """使用 groundtruth 验证轨迹并给出质量分。

    groundtruth 仅在本阶段使用。
    Agent1：LocationHypothesis 是否覆盖真值国家/地区（模型外地理编码）
    Agent2：geopy 距离误差；超过 DISTANCE_ERROR_THRESHOLD_KM → hard fail
    Agent3：verdict 是否与 Agent2 误差一致性匹配
    全员：泄漏检测 + LLM-as-judge（合理性；prompt 仍不得含 raw groundtruth
    坐标作为“应输出答案”）
    """
    cfg = settings or get_settings()
    geocode = reverse_geocode or default_reverse_geocode
    threshold = cfg.DISTANCE_ERROR_THRESHOLD_KM

    hard_fail_reasons: list[str] = []
    soft_warnings: list[str] = []
    distance_error_km: Optional[float] = None

    # --- 泄漏检测 ---
    leaked, leak_reasons = _detect_leakage(
        traj,
        groundtruth,
        enabled=cfg.ANSWER_LEAK_CHECK_ENABLED,
    )
    hard_fail_reasons.extend(leak_reasons)

    # --- 角色专项 ---
    if traj.agent_role == AgentRole.COARSE:
        if traj.coarse_output is None:
            hard_fail_reasons.append("COARSE 缺少 coarse_output")
        else:
            _ok, hard, soft = _check_coarse_coverage(
                traj.coarse_output, groundtruth, geocode
            )
            hard_fail_reasons.extend(hard)
            soft_warnings.extend(soft)

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
            # 用 fine_handoff 相对真值的距离作为一致性参照
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

    # --- LLM-as-judge ---
    judge_score = 1.0
    if run_judge:
        try:
            judge = _run_llm_judge(traj)
            judge_score = judge.score
            if not judge.reasonable:
                soft_warnings.append("LLM-as-judge 认为推理不合理")
                soft_warnings.extend(judge.issues[:5])
        except Exception as exc:  # noqa: BLE001
            soft_warnings.append(f"LLM-as-judge 调用失败: {exc}")
            judge_score = 0.5

    # --- 质量分 ---
    quality = judge_score
    if hard_fail_reasons:
        quality = min(quality, 0.3)
    if soft_warnings:
        quality = max(0.0, quality - 0.05 * min(len(soft_warnings), 4))
    if distance_error_km is not None and traj.agent_role == AgentRole.FINE:
        # 距离越近分越高（阈值内线性映射到 0.5~1 加权）
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
