"""数据集与 manifest schema。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """训练用 chat message。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class DatasetEntry(BaseModel):
    """最终 JSONL 中的单条训练样本（单一 agent）。"""

    id: str
    source_video: str
    messages: list[ChatMessage]
    quality_score: Optional[float] = None


class ManifestV2(BaseModel):
    """断点续跑清单。"""

    video_id: str
    stages: dict[str, str] = Field(default_factory=dict)
    updated_at: str = ""
