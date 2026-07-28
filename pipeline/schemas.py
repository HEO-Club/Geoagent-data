"""全项目 Pydantic v2 数据契约（字段名与结构不可擅自更改）。"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal, Optional, TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# 常量（A 规则 / 命名提示）
# ---------------------------------------------------------------------------

SEED_TOOL_NAMES: set[str] = {
    "web_search",
    "reverse_image_search",
    "map_query",
    "ocr",
    "zoom_inspect",
    "sun_position_calc",
    "submit_answer",
}

ALLOWED_VERBS_HINT: set[str] = {
    "get",
    "search",
    "query",
    "detect",
    "extract",
    "calculate",
    "lookup",
    "compare",
    "estimate",
}

_FORBIDDEN_TOOL_NAMES: set[str] = {
    "tool",
    "helper",
    "utility",
    "new_tool",
    "custom_tool",
}

_FORBIDDEN_PARAM_NAMES: set[str] = {
    "image",
    "image_path",
    "frame",
    "video",
    "data",
    "input",
    "output",
    "value",
    "result",
    "info",
}

_TOOL_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

ParamValue: TypeAlias = str | int | float | bool | list[float] | list[str] | tuple[float, float]


# ---------------------------------------------------------------------------
# 4.1 输入资源
# ---------------------------------------------------------------------------


class TranscriptSegment(BaseModel):
    """带时间戳的文字稿片段。"""

    start: float
    end: float
    text: str


class VideoInput(BaseModel):
    """单条视频的流水线输入。"""

    video_path: str
    transcript: list[TranscriptSegment]
    groundtruth: tuple[float, float]  # (lat, lng)；仅 stage6 消费
    source_platform: str


# ---------------------------------------------------------------------------
# 4.2 Agent 角色
# ---------------------------------------------------------------------------


class AgentRole(str, Enum):
    COARSE = "coarse_locator"
    FINE = "fine_locator"
    VERIFIER = "verifier"


# ---------------------------------------------------------------------------
# 4.3 Tool 体系
# ---------------------------------------------------------------------------


def _value_matches_param_type(value: Any, type_name: str) -> bool:
    """检查 example/default 是否符合 ParamField 声明类型。"""
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "float":
        # JSON/配置中整型字面量也视为合法 float 示例
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "bbox":
        return (
            isinstance(value, list)
            and len(value) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
        )
    if type_name in ("latlng", "lat_range"):
        if isinstance(value, tuple) and len(value) == 2:
            return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
        if isinstance(value, list) and len(value) == 2:
            return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
        return False
    if type_name == "string_list":
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    return False


class ParamField(BaseModel):
    """Tool 输入参数字段定义。"""

    name: str
    type: Literal[
        "string",
        "float",
        "int",
        "bool",
        "bbox",
        "latlng",
        "lat_range",
        "string_list",
    ]
    required: bool
    description: str
    example: ParamValue
    default: ParamValue | None = None
    enum_values: list[str] | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _FIELD_NAME_RE.fullmatch(v):
            raise ValueError(f"params 字段名必须为小写下划线风格: {v!r}")
        if v in _FORBIDDEN_PARAM_NAMES:
            raise ValueError(f"禁止的 params 字段名: {v!r}")
        return v

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: str) -> str:
        if not (10 <= len(v) <= 60):
            raise ValueError(f"description 长度须为 10~60 字符，当前 {len(v)}")
        return v

    @model_validator(mode="after")
    def _validate_param_constraints(self) -> ParamField:
        if self.required and self.default is not None:
            raise ValueError("required=True 时 default 必须为 None")
        if self.enum_values is not None:
            if self.type != "string":
                raise ValueError("enum_values 仅允许 type=string")
            if len(self.enum_values) == 0:
                raise ValueError("enum_values 若提供则不得为空列表")
        if not _value_matches_param_type(self.example, self.type):
            raise ValueError(f"example 不符合声明类型 {self.type}")
        if self.default is not None and not _value_matches_param_type(self.default, self.type):
            raise ValueError(f"default 不符合声明类型 {self.type}")
        if self.enum_values:
            if isinstance(self.example, str) and self.example not in self.enum_values:
                raise ValueError("example 必须落在 enum_values 内")
            if (
                self.default is not None
                and isinstance(self.default, str)
                and self.default not in self.enum_values
            ):
                raise ValueError("default 必须落在 enum_values 内")
        return self


class ObservationField(BaseModel):
    """Tool Observation 字段定义。"""

    name: str
    type: Literal[
        "string",
        "float",
        "int",
        "bool",
        "string_list",
        "result_list",
        "latlng",
        "lat_range",
        "bbox",
    ]
    nullable: bool
    description: str
    item_fields: list[ObservationField] | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _FIELD_NAME_RE.fullmatch(v):
            raise ValueError(f"observation 字段名必须为小写下划线风格: {v!r}")
        return v

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: str) -> str:
        if not (10 <= len(v) <= 80):
            raise ValueError(f"description 长度须为 10~80 字符，当前 {len(v)}")
        return v

    @model_validator(mode="after")
    def _validate_item_fields(self) -> ObservationField:
        if self.type == "result_list":
            if self.item_fields is None:
                raise ValueError("type=result_list 时 item_fields 必填")
            if not (2 <= len(self.item_fields) <= 5):
                raise ValueError("item_fields 长度须为 2~5")
            if any(f.type == "result_list" for f in self.item_fields):
                raise ValueError("禁止嵌套 result_list")
        elif self.item_fields is not None:
            raise ValueError("非 result_list 类型时 item_fields 必须为 None")
        if self.nullable and "null" not in self.description.lower():
            # nullable=true 时须说明何时为 null（中英文均可）
            if "空" not in self.description and "无" not in self.description:
                raise ValueError("nullable=true 时 description 须说明何时为 null")
        return self


ObservationField.model_rebuild()


def validate_tool_name(name: str, *, is_seed: bool | None = None) -> str:
    """校验 Tool 名称是否符合 A1–A4 硬规则。

    Args:
        name: 待校验名称。
        is_seed: 是否种子 Tool；None 时按 SEED_TOOL_NAMES 自动判断。
    """
    if is_seed is None:
        is_seed = name in SEED_TOOL_NAMES
    if not (3 <= len(name) <= 64):
        raise ValueError(f"Tool 名称长度须为 3~64: {name!r}")
    if not _TOOL_NAME_RE.fullmatch(name):
        raise ValueError(f"Tool 名称仅允许 [a-z0-9_]: {name!r}")
    if name.startswith("_") or name.endswith("_"):
        raise ValueError(f"Tool 名称不得以下划线开头或结尾: {name!r}")
    if "__" in name:
        raise ValueError(f"Tool 名称不得包含连续下划线: {name!r}")
    if name in _FORBIDDEN_TOOL_NAMES:
        raise ValueError(f"禁止无意义 Tool 名称: {name!r}")
    # A4: 非种子 Tool 名称至少两个语义 token
    if not is_seed and "_" not in name:
        raise ValueError(f"非种子 Tool 名称至少包含两个语义 token: {name!r}")
    return name


class ToolDefinition(BaseModel):
    """Registry 中的单条 Tool 定义（仅 schema，无真实 executor）。"""

    name: str
    description: str
    params: list[ParamField]
    observation_fields: list[ObservationField]
    allowed_agents: list[AgentRole]
    is_terminal: bool = False
    created_at: str
    source_video_timestamp: Optional[float] = None
    source_narration: Optional[str] = None
    derived_from_existing_tools: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        return validate_tool_name(v)

    @model_validator(mode="after")
    def _validate_tool_rules(self) -> ToolDefinition:
        # F7 / params≤5 / observation_fields≤8
        if len(self.params) > 5:
            raise ValueError("params 数量不得超过 5")
        if not self.is_terminal and len(self.observation_fields) > 8:
            raise ValueError("observation_fields 数量不得超过 8")

        param_names = [p.name for p in self.params]
        obs_names = [o.name for o in self.observation_fields]
        if len(param_names) != len(set(param_names)):
            raise ValueError("params 字段名不得重复")
        if len(obs_names) != len(set(obs_names)):
            raise ValueError("observation_fields 字段名不得重复")

        # F1: 名称集合不得有交集
        overlap = set(param_names) & set(obs_names)
        if overlap:
            raise ValueError(f"params 与 observation_fields 字段名不得有交集: {overlap}")

        # F5: params 禁止图像相关名（ParamField 已拦；此处双保险）
        for p in self.params:
            if p.name in {"image", "image_path", "frame", "video"}:
                raise ValueError(f"params 禁止字段名: {p.name}")

        # F3: params 有 bbox 时 observation 不得再有 bbox
        if any(p.type == "bbox" for p in self.params):
            if any(o.type == "bbox" for o in self.observation_fields):
                raise ValueError("params 含 bbox 时 observation_fields 不得再含 bbox")

        # F2 / terminal 规则
        if self.is_terminal:
            if self.observation_fields:
                raise ValueError("is_terminal=True 时 observation_fields 必须为空列表")
        else:
            if not self.observation_fields:
                raise ValueError("非 terminal Tool 的 observation_fields 不得为空")
            status = next((o for o in self.observation_fields if o.name == "status"), None)
            err = next((o for o in self.observation_fields if o.name == "error_message"), None)
            if status is None or status.type != "string" or status.nullable:
                raise ValueError("非 terminal Tool 必须包含非空 string 字段 status")
            if err is None or err.type != "string" or not err.nullable:
                raise ValueError("非 terminal Tool 必须包含 nullable string 字段 error_message")

        return self


# ---------------------------------------------------------------------------
# 4.9 Agent 交接物与输出
# ---------------------------------------------------------------------------


class LocationHypothesis(BaseModel):
    """Agent1 粗定位输出 / 向 Agent2 的交接物。"""

    possible_countries: list[str]
    possible_regions: list[str]
    reasoning_summary: str
    confidence: float = Field(ge=0, le=1)
    key_clues_remaining: list[str]


class SubmitAnswerResult(BaseModel):
    """与 submit_answer params 对齐；Agent2 → Agent3 交接物。"""

    latitude: float
    longitude: float
    location_name: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class VerificationResult(BaseModel):
    """Agent3 验证输出。"""

    verdict: Literal["pass", "fail"]
    failed_checks: list[str]
    suggested_recheck: str
    return_to_agent: Optional[Literal[1, 2]] = None

    @model_validator(mode="after")
    def _validate_verdict_rules(self) -> VerificationResult:
        if self.verdict == "pass" and self.return_to_agent is not None:
            raise ValueError("verdict=pass 时 return_to_agent 必须为 None")
        if self.verdict == "fail" and self.return_to_agent not in (1, 2):
            raise ValueError("verdict=fail 时 return_to_agent 必须为 1 或 2")
        return self


# ---------------------------------------------------------------------------
# 4.10 阶段中间产物
# ---------------------------------------------------------------------------


class AgentTimeSegment(BaseModel):
    """单个 Agent 的时间区间（不得用 AgentRole 作 JSON key）。"""

    agent_role: AgentRole
    start_time: float
    end_time: float


class PreprocessResult(BaseModel):
    """stage0 输出。"""

    answer_timestamp: float
    agent_segments: list[AgentTimeSegment]
    revision_segments: list[tuple[float, float]]
    post_answer_evidence_windows: list[tuple[float, float]] = Field(
        default_factory=list
    )


class TimedScreenAction(BaseModel):
    """stage1 识别的带时间戳屏幕操作。"""

    start_time: float
    end_time: float
    description: str
    visible_clues: list[str] = Field(default_factory=list)


class Move(BaseModel):
    """stage2 对齐后的推理动作单元（通常对应一次屏幕操作会话）。"""

    start_time: float
    end_time: float
    narration: str  # 来自 asr_raw 或 vlm_transcript
    screen_action: Optional[str] = None
    visible_clues: list[str] = Field(default_factory=list)
    agent_role: AgentRole


class Action(BaseModel):
    """规范化后的 Tool 调用。"""

    tool: str
    params: dict[str, Any]


class NormalizationMode(str, Enum):
    MATCHED = "matched"
    COMPOSED = "composed"
    TOOL_REGISTERED = "tool_registered"
    FALLBACK = "fallback"
    THOUGHT_ONLY = "thought_only"


class NormalizedStep(BaseModel):
    """stage3 输出的规范化步骤。"""

    move: Move
    thought_draft: str
    actions: list[Action]
    normalization_mode: NormalizationMode
    matched_tool_confidence: Optional[float] = None
    fallback_reason: Optional[str] = None


class ObservationSource(str, Enum):
    LLM_SYNTHESIZED = "llm_synthesized"


class ObservationExecutionResult(BaseModel):
    """stage4 单次 Action 执行结果。"""

    action: Action
    observation: Optional[dict[str, Any]] = None
    source: Optional[ObservationSource] = None
    status: Literal["success", "empty", "error", "skipped"]
    error_message: Optional[str] = None
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# 4.11 轨迹
# ---------------------------------------------------------------------------


class TrajectoryStep(BaseModel):
    """轨迹中的单步 T→A→O。"""

    thought: str
    action: Action
    observation: Optional[dict[str, Any]] = None
    observation_source: Optional[ObservationSource] = None


class RevisionSource(str, Enum):
    VIDEO_OBSERVED = "video_observed"
    SYSTEM_FEEDBACK = "system_feedback"


class RevisionContext(BaseModel):
    """返工上下文。"""

    source: RevisionSource
    parent_trajectory_id: str
    target_agent: AgentRole
    revision_round: int
    verification_result: Optional[VerificationResult] = None
    video_segment: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def _validate_revision_fields(self) -> RevisionContext:
        if self.source == RevisionSource.SYSTEM_FEEDBACK and self.verification_result is None:
            raise ValueError("system_feedback 时 verification_result 必填")
        if self.source == RevisionSource.VIDEO_OBSERVED and self.video_segment is None:
            raise ValueError("video_observed 时 video_segment 必填")
        return self


class Trajectory(BaseModel):
    """单条 Agent 轨迹。"""

    id: str
    agent_role: AgentRole
    system_prompt: str
    user_query: str
    image_path: str
    steps: list[TrajectoryStep]

    coarse_handoff: Optional[LocationHypothesis] = None
    fine_handoff: Optional[SubmitAnswerResult] = None

    coarse_output: Optional[LocationHypothesis] = None
    fine_output: Optional[SubmitAnswerResult] = None
    verifier_output: Optional[VerificationResult] = None

    is_revision: bool = False
    parent_trajectory_id: Optional[str] = None
    revision_round: int = 0
    revision_source: Optional[RevisionSource] = None
    revision_input: Optional[VerificationResult] = None

    # stage5 judge best-of-k 入选候选的 rubric 得分（0~1）；stage6 以其为质量基分
    stage5_judge_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_handoffs(self) -> Trajectory:
        role = self.agent_role
        if role == AgentRole.COARSE:
            if self.coarse_handoff is not None or self.fine_handoff is not None:
                raise ValueError("COARSE 轨迹的 coarse_handoff/fine_handoff 必须为 None")
        elif role == AgentRole.FINE:
            if self.coarse_handoff is None:
                raise ValueError("FINE 轨迹的 coarse_handoff 必填")
            if self.fine_handoff is not None:
                raise ValueError("FINE 轨迹的 fine_handoff 必须为 None")
        elif role == AgentRole.VERIFIER:
            if self.fine_handoff is None:
                raise ValueError("VERIFIER 轨迹的 fine_handoff 必填")
        return self


# ---------------------------------------------------------------------------
# 4.12 验证报告与数据集条目
# ---------------------------------------------------------------------------


class TrajectoryVerificationReport(BaseModel):
    """stage6 验证报告。"""

    passed: bool
    quality_score: float
    distance_error_km: Optional[float] = None
    hard_fail_reasons: list[str] = Field(default_factory=list)
    soft_warnings: list[str] = Field(default_factory=list)
    leakage_detected: bool = False


class ChatMessage(BaseModel):
    """训练用 chat message。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class DatasetEntry(BaseModel):
    """最终 JSONL 中的单条训练样本。"""

    id: str
    source_video: str
    agent_role: AgentRole
    groundtruth: tuple[float, float]
    messages: list[ChatMessage]
    coarse_handoff: Optional[LocationHypothesis] = None
    fine_handoff: Optional[SubmitAnswerResult] = None
    is_revision: bool = False
    parent_trajectory_id: Optional[str] = None
    revision_round: int = 0
    revision_source: Optional[RevisionSource] = None
    revision_input: Optional[VerificationResult] = None
    quality_score: float
    verified: bool
    distance_error_km: Optional[float] = None


# ---------------------------------------------------------------------------
# 4.13 Checkpoint / Manifest
# ---------------------------------------------------------------------------


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class StageManifestEntry(BaseModel):
    """单阶段 checkpoint 状态。"""

    stage: str
    status: StageStatus
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None


class VideoManifest(BaseModel):
    """单视频各阶段 manifest。"""

    video_id: str
    stages: list[StageManifestEntry]
