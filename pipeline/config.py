"""配置加载：敏感项与模型名仅从环境变量 / .env 读取。"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """pipeline 运行时配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_ENV: str = "test"
    ALLOW_REAL_API: bool = False

    GOOGLE_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    DASHSCOPE_API_KEY: str = ""
    MOONSHOT_API_KEY: str = ""

    LLM_IMAGE_MAX_SIDE: int = 768
    LLM_IMAGE_JPEG_QUALITY: int = 75

    ASR_PROVIDER: str = "qwen"
    ASR_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ASR_MODEL: str = "qwen3-asr-flash"
    ASR_LANGUAGE: str = "zh"
    STAGE1_ALLOW_VLM_FALLBACK: bool = True

    VLM_PROVIDER: str = "qwen"
    VLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    VLM_MODEL: str = "qwen3.7-plus"

    LLM_PROVIDER: str = "kimi"
    LLM_BASE_URL: str = "https://api.moonshot.cn/v1"
    LLM_MODEL: str = "kimi-k3"
    LLM_ANTHROPIC_BASE_URLS: str = "https://api.anthropic.com"
    LLM_ANTHROPIC_MODEL: str = "claude-sonnet-5"
    LLM_ANTHROPIC_API_KEY: str = ""
    LLM_ANTHROPIC_STREAM: bool = True
    LLM_MAX_OUTPUT_TOKENS: int = 8192
    ALLOW_INSECURE_LLM_ENDPOINTS: bool = False
    GEMINI_MODEL: str = "gemini-2.0-flash"
    KIMI_REASONING_EFFORT: str = "low"
    LLM_TIMEOUT_SEC: float = 300.0

    MAX_CONCURRENT_VIDEOS: int = 1
    STAGE2_BEST_OF_K: int = 1

    # 阶段1.5 审核切分
    AUDIT_SPARSE_FRAME_COUNT: int = 8
    AUDIT_MAX_KEYFRAMES_PER_TASK: int = 8
    AUDIT_DISPLAY_SAMPLE_INTERVAL_SEC: float = 1.0
    AUDIT_DISPLAY_WINDOW_MAX_SEC: float = 90.0
    AUDIT_MAX_SAMPLED_FRAMES: int = 120
    AUDIT_MAX_VLM_FRAME_VERIFIES: int = 24
    AUDIT_TASK_BOUNDARY_TOLERANCE_SEC: float = 20.0
    AUDIT_MIN_FRAME_QUALITY: float = 0.65
    AUDIT_MIN_CHAIN_SUPPORT: float = 0.5
    AUDIT_VISUAL_HASH_DISTANCE: int = 6
    AUDIT_TRAJECTORY_IMAGE_CHECK: bool = True

    # 阶段4 置信度评分（权重运行时归一化）
    STAGE4_W_EVIDENCE_GROUNDING: float = 0.30
    STAGE4_W_FINAL_ANSWER_SUPPORT: float = 0.20
    STAGE4_W_TOOL_PARAM_CORRECTNESS: float = 0.20
    STAGE4_W_LOGICAL_CONSISTENCY: float = 0.15
    STAGE4_W_INPUT_QUALITY_ALIGNMENT: float = 0.10
    STAGE4_W_SFT_FORMAT_COMPLETENESS: float = 0.05
    STAGE4_HARD_GATE_CAP: float = 0.30
    # 配合苛评刻度：硬门槛/核心缺口优先看；0.93+ 才标低优先级
    STAGE4_PRIORITY_HIGH_BELOW: float = 0.70
    STAGE4_PRIORITY_MEDIUM_BELOW: float = 0.93
    STAGE4_JUDGE_NEUTRAL_SCORE: float = 0.50

    TOOL_CATALOG_PATH: str = "canonical_tool_catalog.json"
    TOOL_TREES_PATH: str = "tool_trees.json"
    INTERMEDIATE_DIR: str = "data/intermediate"
    SELECTED_DIR: str = "data/selected"
    RUNS_DIR: str = "data/runs"
    OUTPUT_DIR: str = "data/output"
    TRANSCRIPTS_DIR: str = "data/transcripts"
    CACHE_DIR: str = ".cache"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内缓存的 Settings 单例。"""
    return Settings()


def clear_settings_cache() -> None:
    """测试用：清除 get_settings 缓存。"""
    get_settings.cache_clear()
