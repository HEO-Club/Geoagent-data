"""配置加载：全部敏感项与模型名仅从环境变量 / .env 读取。"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """流水线运行时配置（pydantic-settings）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_ENV: str = "test"
    # 是否允许真实 LLM 调用（测试默认 false）
    ALLOW_REAL_API: bool = False

    GOOGLE_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    # 通义千问 / 百炼（DashScope）— 主要用于 VLM 通道
    DASHSCOPE_API_KEY: str = ""
    # Moonshot / Kimi — 主要用于 stage3+ 主通道
    MOONSHOT_API_KEY: str = ""

    # stage1：抽帧可较密，但送入 VLM 的关键帧需抽样，避免请求体过大
    STAGE1_VLM_MAX_FRAMES: int = 6
    # 多模态上传前最长边像素（缩小以降低超时/断连）
    LLM_IMAGE_MAX_SIDE: int = 768
    LLM_IMAGE_JPEG_QUALITY: int = 75

    # --- VLM 通道（stage3 之前：prep_transcript_vlm / stage1）---
    VLM_PROVIDER: str = "qwen"
    VLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    VLM_MODEL: str = "qwen3.7-plus"

    # --- LLM 主通道（stage3+：normalize / Obs / stage5–6；亦支持多模态）---
    LLM_PROVIDER: str = "kimi"
    LLM_BASE_URL: str = "https://api.moonshot.cn/v1"
    LLM_MODEL: str = "kimi-k3"
    # 兼容旧配置；仅 *_PROVIDER=gemini 时作为回退
    GEMINI_MODEL: str = "gemini-2.0-flash"
    # kimi-k3：low|high|max（流水线默认 low 以控延迟/费用）
    KIMI_REASONING_EFFORT: str = "low"
    LLM_TIMEOUT_SEC: float = 300.0

    MAX_CONCURRENT_VIDEOS: int = 1
    ANSWER_LEAK_CHECK_ENABLED: bool = True
    # Observation LLM 合成校验失败重试次数
    OBS_SYNTH_MAX_RETRY: int = 3
    MAX_REVISION_ROUNDS: int = 2
    # stage5 逐步因果生成：每条轨迹候选数（judge 择优）与最低入选分
    STAGE5_BEST_OF_K: int = 2
    STAGE5_JUDGE_THRESHOLD: float = 0.6

    # stage6 / prep_groundtruth 逆地理（仅 Nominatim；不用于 Observation 合成）
    NOMINATIM_USER_AGENT: str = "geoagent-dataset/1.0 (stage6; local)"
    NOMINATIM_TIMEOUT_SEC: float = 10.0

    TOOL_REGISTRY_PATH: str = "tool_registry.json"
    INTERMEDIATE_DIR: str = "data/intermediate"
    OUTPUT_DIR: str = "data/output"
    CACHE_DIR: str = ".cache"

    DISTANCE_ERROR_THRESHOLD_KM: float = 25.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内缓存的 Settings 单例。"""
    return Settings()


def clear_settings_cache() -> None:
    """测试用：清除 get_settings 缓存。"""
    get_settings.cache_clear()
