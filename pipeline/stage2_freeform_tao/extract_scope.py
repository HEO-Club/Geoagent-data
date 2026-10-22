"""从字幕抽取外部给定线索与 working_scope（禁止读 groundtruth）。"""

from __future__ import annotations

from pipeline.llm import call_structured
from pipeline.schemas.clues import (
    BoundKind,
    ClueExtractionResult,
    WorkingScope,
)
from pipeline.schemas.transcript import TranscriptSegment

EXTRACT_SYSTEM_HINT = (
    "你从地理定位讲解视频的字幕中，抽取非地理推理得到的地点范围和相关地点信息，"
    "并规范化为 working_scope。"
    "输入仅为带时间戳字幕；禁止使用或猜测 groundtruth / 官方真值。\n"
    "判定轴心：这条范围或地点信息是不是靠看图、方位、地物、日照或阴影推理出来的；"
    "不是谁转述、是否出现固定话术。\n"
    "渠道无关：片头设定、出题转述、对话转述、评论/私信、IP 属地等均可；"
    "只要不是地理推理定位得到的地点范围或相关地点信息，就应进入 raw_given_clues / working_scope；"
    "不得因未出现「求助者/评论/私信」等渠道词而丢弃。"
    "活动地、IP 属地、籍贯地和国家/地区陈述都应规范化为 working_scope；"
    "不得把推理过程中的方位收窄结论（如「西侧或西北侧」）升格为 scope。\n"
    "角色：\n"
    "- photo_location_constraint：直接约束拍摄地"
    "（如「在X内」「X附近」「拍摄城市/县/市就是X」「就在X县（市）」）；\n"
    "- person_or_social_attribute：人物籍贯/身份等；\n"
    "- other_non_location：其它非地点外部信息。\n"
    "working_scope：凡不是靠看图、方位、地物、日照或阴影等地理推理定位得到的范围和相关地点信息，"
    "一律写入 working_scope。包括出题给定的国家/地区/城市、IP 属地、籍贯地、活动地、"
    "评论或私信中的地点约束；不要求同时出现「离家不远」。\n"
    "- region 只写地点范围短语，不写渠道，禁止出现「IP」「网友」「字幕」「评论」。"
    "明确给出的行政区用 bound_kind=inside，region 保持原文粒度，不必强行加「内」。"
    "「附近」「住所一带」等软表达用 bound_kind=near，region 必须是「X附近」，禁止升格成「X内」。\n"
    "- 时间修饰（如「90年代的X县」）不否决地理边界，仍应保留对应 working_scope；\n"
    "- 粒度不得细于原文。只有完全没有非推理地点范围时，working_scope=null。\n"
    "candidate_hypotheses：由画面或地理推理得到的候选（如「太阳能板说明低纬度，所以是江西」"
    "「很可能是…」「我猜在…」），仅审计，不得写入 working_scope。"
)


def _format_transcript(transcript: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for seg in transcript:
        lines.append(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}")
    return "\n".join(lines)


def sanitize_working_scope(scope: WorkingScope | None) -> WorkingScope | None:
    """程序化兜底：空 region 丢弃；软先验禁止「内」升格写法。"""
    if scope is None:
        return None
    region = scope.region.strip()
    if not region:
        return None
    if scope.bound_kind == BoundKind.near:
        # 软先验不得写成硬边界「…内」
        if region.endswith("内") or "市内" in region or region.endswith("市里"):
            softened = region
            for suffix in ("市内", "市里", "内"):
                if softened.endswith(suffix):
                    softened = softened[: -len(suffix)].rstrip()
                    break
            if not softened:
                return None
            if not softened.endswith("附近"):
                softened = f"{softened}附近"
            return WorkingScope(region=softened, bound_kind=BoundKind.near)
    return WorkingScope(region=region, bound_kind=scope.bound_kind)


def normalize_extraction(result: ClueExtractionResult) -> ClueExtractionResult:
    """校验并清洗抽取结果。"""
    return ClueExtractionResult(
        raw_given_clues=list(result.raw_given_clues),
        working_scope=sanitize_working_scope(result.working_scope),
        candidate_hypotheses=list(result.candidate_hypotheses),
    )


def extract_working_scope(
    transcript: list[TranscriptSegment],
) -> ClueExtractionResult:
    """从字幕抽取 raw_given_clue / working_scope / candidate_hypotheses。

    Args:
        transcript: 阶段1 字幕（不得含 groundtruth）。

    Returns:
        经 Pydantic 与程序化兜底清洗后的抽取结果。
    """
    prompt = (
        f"{EXTRACT_SYSTEM_HINT}\n\n"
        "字幕：\n"
        f"{_format_transcript(transcript)}\n\n"
        "请输出 raw_given_clues、working_scope（可 null）、candidate_hypotheses。"
    )
    raw = call_structured(prompt, ClueExtractionResult, lane="llm")
    return normalize_extraction(raw)
