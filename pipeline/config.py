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
    TAVILY_API_KEY: str = ""
    SERPAPI_KEY: str = ""
    GOOGLE_MAPS_KEY: str = ""

    # 默认值只写在配置层，不散落在业务代码
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
