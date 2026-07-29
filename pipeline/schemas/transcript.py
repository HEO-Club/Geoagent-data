"""字幕相关 schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """带时间戳的文字稿片段。"""

    start: float
    end: float
    text: str


class Stage1Result(BaseModel):
    """阶段1 落盘包装。"""

    video_id: str
    video_path: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
