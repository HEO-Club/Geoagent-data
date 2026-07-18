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
    ALLOW_REAL_API: bool = False

    GOOGLE_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    # 通义千问 / 百炼（DashScope）
    DASHSCOPE_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    SERPAPI_KEY: str = ""
    # 保留字段以兼容旧 .env；map_query 已改用 Nominatim，不再读取此 key
    GOOGLE_MAPS_KEY: str = ""

    # 地图后端：amap（高德，推荐国内）| nominatim（OSM）
    MAP_PROVIDER: str = "amap"
    # 高德 Web 服务 Key（仅 .env；勿提交仓库）
    AMAP_API_KEY: str = ""
    AMAP_BASE_URL: str = "https://restapi.amap.com"
    AMAP_TIMEOUT_SEC: float = 10.0
    # Nominatim（OpenStreetMap）— MAP_PROVIDER=nominatim 时使用
    NOMINATIM_USER_AGENT: str = "geoagent-dataset/1.0 (local; contact via project repo)"
    NOMINATIM_TIMEOUT_SEC: float = 10.0

    # stage1：抽帧可较密，但送入 VLM 的关键帧需抽样，避免请求体过大
    STAGE1_VLM_MAX_FRAMES: int = 6
    # 多模态上传前最长边像素（缩小以降低超时/断连）
    LLM_IMAGE_MAX_SIDE: int = 768
    LLM_IMAGE_JPEG_QUALITY: int = 75

    # LLM 提供方：qwen（默认，DashScope）| gemini
    LLM_PROVIDER: str = "qwen"
    # OpenAI 兼容 base_url；国内北京区默认如下
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 主模型名（百炼 Model ID，如 qwen3.7-plus）
    LLM_MODEL: str = "qwen3.7-plus"
    # 兼容旧配置；仅 LLM_PROVIDER=gemini 时作为回退
    GEMINI_MODEL: str = "gemini-2.0-flash"
    MAX_CONCURRENT_VIDEOS: int = 1
    ANSWER_LEAK_CHECK_ENABLED: bool = True
    DRAFT_TOOL_MAX_RETRY: int = 3
    MAX_REVISION_ROUNDS: int = 2

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
