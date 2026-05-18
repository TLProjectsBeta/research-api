from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # LLM API Keys
    openai_api_key: str = ""
    google_api_key: str = ""

    # LangSmith
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "research-api"

    # Application
    app_name: str = "Research Assistant API"
    app_version: str = "1.0.0"
    debug: bool = False
    api_keys: list[str] = []

    # Redis
    redis_url: str = "redis://localhost:6379"

    # LLM Settings
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2048

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()