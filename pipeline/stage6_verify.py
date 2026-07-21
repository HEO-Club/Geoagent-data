"""stage6：使用 groundtruth 做验证、泄漏检查与质量评分。

groundtruth 仅在本阶段使用。
TAO 形态：LLM-as-judge（不含 GT）。
泄漏判定：整链 LLM-as-judge（直接用 GT/后见之明）+ 窄化坐标程序化兜底。
合理性 LLM-as-judge prompt 不得把 raw groundtruth 坐标当作应输出答案。
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Optional, Union

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
from pipeline.tao_style_examples import fewshot_block_for_role, tao_judge_checklist


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


class _JudgeResult(BaseModel):
    """合理性 LLM-as-judge 结构化输出。"""

    reasonable: bool
    issues: list[str] = Field(default_factory=list)
    score: float = Field(ge=0, le=1)


class LeakageJudgeResult(BaseModel):
    """泄漏 LLM-as-judge：是否直接使用 GT / 后见之明。"""

    leaked: bool
    reasons: list[str] = Field(default_factory=list)


class TaoStyleJudgeResult(BaseModel):
    """TAO 形态 LLM-as-judge：是否为标准图片地理定位推理链。"""

    is_standard_geo_tao: bool
    issues: list[str] = Field(default_factory=list)


class CoarseReasoningChainJudgeResult(BaseModel):
    """COARSE 递进推理链 LLM-as-judge（不含 GT）。"""

    identifies_geo_human_features: bool
    narrows_scope_progressively: bool
    has_reasoning_gap: bool
    thought_action_aligned: bool
    coarse_scope_within_role: bool
    issues: list[str] = Field(default_factory=list)


# Agent1 训练轨迹禁止出现的种子 Tool（与 stage5 投影一致）
_COARSE_FORBIDDEN_TRAJECTORY_TOOLS: frozenset[str] = frozenset(
    {"web_search", "map_query", "reverse_image_search", "submit_answer"}
)

# 粗略坐标模式：程序化泄漏兜底
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
        c = cand.strip().lower()
        t = target.strip().lower()
        if c and t and (c in t or t in c):
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
    country, region = raw[0], raw[1] if len(raw) > 1 else None
    return PlaceHints(country=str(country), region=region)


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


def _run_tao_style_judge(traj: Trajectory) -> TaoStyleJudgeResult:
    """LLM 判定是否为标准图片地理定位 TAO；prompt 不含 groundtruth。"""
    steps_brief: list[str] = []
    for i, step in enumerate(traj.steps, start=1):
        steps_brief.append(
            f"Step {i}: tool={step.action.tool}; thought={step.thought!r}"
        )
    extra = ""
    if traj.coarse_output is not None:
        extra += f"\ncoarse_output.reasoning_summary: {traj.coarse_output.reasoning_summary!r}"
    prompt = (
        "你是地理定位 SFT 数据的 TAO 形态审查员。\n"
        "判断该轨迹的 Thought 是否为标准图片地理定位推理链。\n"
        f"{tao_judge_checklist()}\n"
        "若存在旁白叙事体、视频元叙事、非地理推理、或本步 Obs 时序倒置 → "
        "is_standard_geo_tao=false。\n"
        "VERIFIER 复述候选定位用于验证是允许的。\n"
        f"agent_role: {traj.agent_role.value}\n"
        f"steps:\n" + "\n".join(steps_brief) + extra
    )
    return call_structured(prompt, TaoStyleJudgeResult)


def _detect_tao_style_failures(
    traj: Trajectory,
    *,
    run_llm: bool = True,
) -> tuple[list[str], list[str]]:
    """返回 (hard_fails, soft_warnings)。"""
    hard: list[str] = []
    soft: list[str] = []
    if not run_llm:
        return hard, soft
    try:
        result = _run_tao_style_judge(traj)
        if not result.is_standard_geo_tao:
            hard.append("非标准地理定位 TAO / 旁白叙事体")
            hard.extend(result.issues[:5])
    except Exception as exc:  # noqa: BLE001
        soft.append(f"TAO 形态 LLM-as-judge 调用失败: {exc}")
    return hard, soft


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
        place = _normalize_place_hints(reverse_geocode(groundtruth))
        country, region = place.country, place.region
    except Exception as exc:  # noqa: BLE001
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


def _detect_coarse_forbidden_tools(traj: Trajectory) -> list[str]:
    """程序化兜底：Agent1 轨迹不得含投影禁止 Tool。"""
    hard: list[str] = []
    for i, step in enumerate(traj.steps, start=1):
        name = step.action.tool
        if name in _COARSE_FORBIDDEN_TRAJECTORY_TOOLS:
            hard.append(f"COARSE 轨迹禁止 Tool: step{i}={name}")
    return hard


def _run_coarse_reasoning_chain_judge(
    traj: Trajectory,
) -> CoarseReasoningChainJudgeResult:
    """COARSE 递进链裁判；prompt 不含 groundtruth。"""
    steps_brief: list[str] = []
    for i, step in enumerate(traj.steps, start=1):
        steps_brief.append(
            f"Step {i}: tool={step.action.tool}; thought={step.thought!r}; "
            f"observation={step.observation!r}"
        )
    extra = ""
    if traj.coarse_output is not None:
        extra = f"\ncoarse_output: {traj.coarse_output.model_dump_json()}"
    prompt = (
        "你是 Agent1（粗定位）递进推理审查员。\n"
        "判断轨迹是否为严密的「特征识别 → 排除/收窄 → 下一步验证」链。\n"
        "规则：\n"
        "1. identifies_geo_human_features：Thought 须指出具体地理/人文特征；\n"
        "2. narrows_scope_progressively：须逐步收窄到国家/地区，禁止跳步；\n"
        "3. has_reasoning_gap：无依据跳步、本步 Obs 时序倒置、单一弱特征直接结论 → true；\n"
        "4. thought_action_aligned：每步 Thought 须解释为何调用该 Action；\n"
        "5. coarse_scope_within_role：结论仅国家/地区级，不得最终精准 POI/坐标；\n"
        "6. 允许 zoom_inspect/ocr/sun_position_calc 及适配的动态特征观察 Tool；"
        "禁止 web_search/map_query/reverse_image_search/submit_answer；\n"
        "7. prompt 不含真值坐标。\n"
        f"{fewshot_block_for_role(AgentRole.COARSE)}\n"
        f"user_query: {traj.user_query!r}\n"
        f"steps:\n" + "\n".join(steps_brief) + extra
    )
    return call_structured(prompt, CoarseReasoningChainJudgeResult)


def _detect_coarse_reasoning_failures(
    traj: Trajectory,
    *,
    run_llm: bool = True,
) -> tuple[list[str], list[str]]:
    """COARSE 专项：禁止 Tool + 递进链裁判。返回 (hard, soft)。"""
    hard = _detect_coarse_forbidden_tools(traj)
    soft: list[str] = []
    if not run_llm:
        return hard, soft
    try:
        result = _run_coarse_reasoning_chain_judge(traj)
        if result.has_reasoning_gap:
            hard.append("COARSE 推理跳步 / 递进链缺口")
        if not result.narrows_scope_progressively:
            hard.append("COARSE 未体现逐步缩小范围")
        if not result.thought_action_aligned:
            hard.append("COARSE Thought 与 Action 不对齐")
        if not result.identifies_geo_human_features:
            hard.append("COARSE 缺少具体地理/人文特征识别")
        if not result.coarse_scope_within_role:
            hard.append("COARSE 结论超出国家/地区边界")
        hard.extend(result.issues[:5])
    except Exception as exc:  # noqa: BLE001
        soft.append(f"COARSE 递进链 LLM-as-judge 调用失败: {exc}")
    return hard, soft


def verify_and_score(
    traj: Trajectory,
    groundtruth: tuple[float, float],
    *,
    settings: Optional[Settings] = None,
    reverse_geocode: Optional[ReverseGeocodeFn] = None,
    run_judge: bool = True,
    run_leakage_llm: bool = True,
    run_tao_style_llm: bool = True,
    run_coarse_reasoning_llm: bool = True,
) -> TrajectoryVerificationReport:
    """使用 groundtruth 验证轨迹并给出质量分。

    groundtruth 仅在本阶段使用。
    顺序：TAO 形态 → 泄漏 → 角色专项（含 COARSE 递进链）→ 合理性 soft judge。
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

    # --- TAO 形态（替代旁白词表）---
    tao_hard, tao_soft = _detect_tao_style_failures(
        traj, run_llm=run_tao_style_llm
    )
    hard_fail_reasons.extend(tao_hard)
    soft_warnings.extend(tao_soft)

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
        chain_hard, chain_soft = _detect_coarse_reasoning_failures(
            traj, run_llm=run_coarse_reasoning_llm
        )
        hard_fail_reasons.extend(chain_hard)
        soft_warnings.extend(chain_soft)

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

    # --- 合理性 LLM-as-judge ---
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
