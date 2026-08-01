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
    "你从地理定位讲解视频的字幕中，抽取问题设置阶段的外部给定线索，"
    "并在有合法边界时规范化为 working_scope。"
    "输入仅为带时间戳字幕；禁止使用或猜测 groundtruth / 官方真值。"
    "角色：\n"
    "- photo_location_constraint：外部沟通直接约束拍摄地（如「在X内」「X附近」）；\n"
    "- person_or_social_attribute：人物籍贯/身份等；\n"
    "- other_non_location：其它非地点外部信息。\n"
    "working_scope：\n"
    "- bound_kind=inside：仅当原文直接说拍摄地在「X内 / 未出X」等硬边界；region 写「X内」类短语；\n"
    "- bound_kind=near：聊天「拍摄地为X附近」，或籍贯地名 +「离家不远/附近」等软距离话；"
    "region 必须写「X附近」类展示短语，禁止升格成「X内」或「就在X市」；\n"
    "- 粒度不得细于原文；无合法硬/软边界时 working_scope=null。\n"
    "candidate_hypotheses：博主演绎候选（如「很可能是…」「我猜在…」），仅审计，"
    "不得写入 working_scope。\n"
    "人物属性默认不单独构成已知线索段；仅当籍贯 ∧ 离家不远可推出软先验时写入 working_scope。"
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
