"""准备步骤：从带时间戳文字稿推断答案地名，经 Nominatim 解析坐标。

本模块位于主流水线之外：产出供 ``run_one_video.py --gt`` 使用的 groundtruth 建议，
不把 groundtruth 注入 stage5；也不走 Observation 合成管线。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from pipeline.schemas import TranscriptSegment
from pipeline.stage0_preprocess import locate_answer_timestamp

# Nominatim 公共实例（仅 prep_groundtruth 离线辅助，非 Observation）
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_USER_AGENT = "geoagent-dataset/1.0 (prep_groundtruth; local)"
_NOMINATIM_TIMEOUT_SEC = 10.0
_MIN_REQUEST_INTERVAL_SEC = 1.05
_last_request_monotonic: float = 0.0

# 从答案附近口述中抓取常见地名后缀（控制长度，避免吞进整句）
_PLACE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"(?:在|于|位于|来到|登上|经过)([\u4e00-\u9fff]{2,20}"
        r"(?:文化公园|风景名胜区|风景区|公园|大桥|铁路桥|公路桥|广场|"
        r"车站|码头|寺庙|古镇|古城|大学|博物馆|塔|山|亭))",
        r"([\u4e00-\u9fff]{2,16}(?:黄河文化公园|文化公园|风景名胜区))",
        r"((?:郑州|许昌|洛阳|开封|北京|上海|南京|武汉|西安)"
        r"[\u4e00-\u9fff]{0,12}(?:黄河文化公园|文化公园|风景区|公园|大桥|山|亭))",
        r"([\u4e00-\u9fff]{2,12}(?:铁路桥|公路桥|大桥|依山亭|吉木格))",
    )
)

_NOISE_PREFIX = re.compile(
    r"^(这|那|该|此|其|的|了|着|过|是|就|把|被|反而|不远处|经过|下游|中国|首座)+"
)
_BAD_SUBSTRINGS = (
    "我国",
    "始建",
    "新建了",
    "父亲",
    "照片",
    "时候",
    "一段",
    "一条",
    "建成",
    "正交",
    "特大桥",  # 过长描述性桥名易误抽，保留「郑州黄河大桥」等短名
)


class PlaceCandidate(BaseModel):
    """从文字稿抽出的候选地名。"""

    query: str
    source_start: float
    source_text: str


class GroundtruthSuggestion(BaseModel):
    """地理编码得到的 groundtruth 建议。"""

    query: str
    latitude: float
    longitude: float
    formatted_address: Optional[str] = None
    place_type: Optional[str] = None
    status: str
    answer_timestamp: Optional[float] = None
    candidates: list[PlaceCandidate] = Field(default_factory=list)
    observation: dict[str, Any] = Field(default_factory=dict)

    def gt_cli(self) -> str:
        """``run_one_video.py --gt`` 可用的 LAT,LNG 字符串。"""
        return f"{self.latitude},{self.longitude}"


def _throttle() -> None:
    global _last_request_monotonic
    now = time.monotonic()
    wait = _MIN_REQUEST_INTERVAL_SEC - (now - _last_request_monotonic)
    if wait > 0:
        time.sleep(wait)
    _last_request_monotonic = time.monotonic()


def geocode_place(query: str) -> dict[str, Any]:
    """用 Nominatim 将地名解析为坐标；返回 observation 风格 dict（非 Tool 执行）。"""
    q = (query or "").strip()
    if not q:
        return {
            "status": "empty",
            "error_message": None,
            "formatted_address": None,
            "resolved_latlng": None,
            "place_type": None,
        }
    try:
        _throttle()
        params = urlencode(
            {"q": q, "format": "json", "limit": 1, "addressdetails": 1}
        )
        req = Request(
            f"{_NOMINATIM_URL}?{params}",
            headers={"User-Agent": _NOMINATIM_USER_AGENT},
        )
        with urlopen(req, timeout=_NOMINATIM_TIMEOUT_SEC) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        if not raw:
            return {
                "status": "empty",
                "error_message": None,
                "formatted_address": None,
                "resolved_latlng": None,
                "place_type": None,
            }
        hit = raw[0]
        lat = float(hit["lat"])
        lng = float(hit["lon"])
        place_type = hit.get("type") if isinstance(hit.get("type"), str) else None
        return {
            "status": "success",
            "error_message": None,
            "formatted_address": hit.get("display_name"),
            "resolved_latlng": [lat, lng],
            "place_type": place_type,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": str(exc),
            "formatted_address": None,
            "resolved_latlng": None,
            "place_type": None,
        }


def load_transcript(path: str | Path) -> list[TranscriptSegment]:
    """读取 TranscriptSegment 列表，或兼容 whisper 风格 ``{segments:[...]}``。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "segments" in raw:
        items = raw["segments"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("文字稿 JSON 须为列表，或含 segments 字段的对象")
    return [TranscriptSegment.model_validate(item) for item in items]


def _normalize_place(text: str) -> str:
    t = text.strip().strip("，。,.!！？?、；;：:\"'`")
    t = _NOISE_PREFIX.sub("", t)
    return t.strip()


def extract_place_candidates(
    transcript: list[TranscriptSegment],
    *,
    answer_timestamp: Optional[float] = None,
    window_sec: float = 90.0,
) -> list[PlaceCandidate]:
    """在答案宣布时刻附近抽取地名候选（靠后片段优先）。"""
    if not transcript:
        return []

    if answer_timestamp is not None:
        ts = float(answer_timestamp)
    else:
        try:
            ts = locate_answer_timestamp(transcript)
        except ValueError:
            # 找不到宣布句时，回退到后 1/3 时间轴
            last_end = max(float(s.end) for s in transcript)
            ts = last_end * 2.0 / 3.0

    window = [
        seg
        for seg in transcript
        if seg.start >= ts - 15.0 and seg.start <= ts + window_sec
    ]
    if not window:
        window = [seg for seg in transcript if seg.start >= ts][:8]
    if not window:
        window = transcript[-8:]
    # 靠后的答案句更可能含最终地名
    window = sorted(window, key=lambda s: s.start)

    found: list[PlaceCandidate] = []
    seen: set[str] = set()
    for seg in window:
        for pattern in _PLACE_PATTERNS:
            for match in pattern.finditer(seg.text):
                parts = [p for p in match.groups() if p]
                query = _normalize_place("".join(parts))
                if len(query) < 2 or len(query) > 24 or query in seen:
                    continue
                if any(bad in query for bad in _BAD_SUBSTRINGS):
                    continue
                seen.add(query)
                found.append(
                    PlaceCandidate(
                        query=query,
                        source_start=float(seg.start),
                        source_text=seg.text,
                    )
                )
    # 更长、更具体的地名优先
    found.sort(key=lambda c: (-len(c.query), -c.source_start))
    return found


def lookup_groundtruth(
    transcript: list[TranscriptSegment],
    *,
    query: Optional[str] = None,
    answer_timestamp: Optional[float] = None,
) -> GroundtruthSuggestion:
    """用 Nominatim 将答案地名解析为经纬度。

    Args:
        transcript: 带时间戳文字稿。
        query: 手动指定地名；为空则从文字稿自动抽取。
        answer_timestamp: 可选覆盖答案时刻；默认走 stage0 定位逻辑。

    Returns:
        GroundtruthSuggestion；失败时 status!=success 且不填坐标字段约束由调用方检查。
    """
    ts: Optional[float]
    try:
        ts = (
            float(answer_timestamp)
            if answer_timestamp is not None
            else locate_answer_timestamp(transcript)
        )
    except ValueError:
        ts = None

    candidates = extract_place_candidates(
        transcript, answer_timestamp=ts
    )
    queries: list[str] = []
    if query and query.strip():
        queries.append(query.strip())
    queries.extend(c.query for c in candidates)

    # 去重保序
    deduped: list[str] = []
    seen_q: set[str] = set()
    for q in queries:
        if q not in seen_q:
            seen_q.add(q)
            deduped.append(q)

    if not deduped:
        return GroundtruthSuggestion(
            query="",
            latitude=0.0,
            longitude=0.0,
            status="empty",
            answer_timestamp=ts,
            candidates=candidates,
            observation={
                "status": "empty",
                "error_message": None,
                "formatted_address": None,
                "resolved_latlng": None,
                "place_type": None,
            },
        )

    last_obs: dict[str, Any] = {}
    for q in deduped:
        obs = geocode_place(q)
        last_obs = obs
        if obs.get("status") == "success" and obs.get("resolved_latlng"):
            lat, lng = obs["resolved_latlng"]
            return GroundtruthSuggestion(
                query=q,
                latitude=float(lat),
                longitude=float(lng),
                formatted_address=obs.get("formatted_address"),
                place_type=obs.get("place_type"),
                status="success",
                answer_timestamp=ts,
                candidates=candidates,
                observation=obs,
            )

    # 全部失败：返回最后一次结果，坐标占位 0 并由 status 标明
    return GroundtruthSuggestion(
        query=deduped[0],
        latitude=0.0,
        longitude=0.0,
        formatted_address=last_obs.get("formatted_address"),
        place_type=last_obs.get("place_type"),
        status=str(last_obs.get("status") or "error"),
        answer_timestamp=ts,
        candidates=candidates,
        observation=last_obs,
    )


def lookup_groundtruth_from_file(
    transcript_path: str | Path,
    *,
    query: Optional[str] = None,
) -> GroundtruthSuggestion:
    """从文字稿文件路径解析 groundtruth 建议。"""
    return lookup_groundtruth(load_transcript(transcript_path), query=query)
