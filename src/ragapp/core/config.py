import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_env_file():
    """
    Dynamically determine the path to the .env file based on the environment.
    Usage:
    Available values:
    dev, test, prod
    """
    env = os.getenv("ENVIRONMENT", "dev")
    return f".env.{env}"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables take precedence over values defined here.

    A .env file can be used for local development.
    """

    model_config = SettingsConfigDict(
        env_file=get_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_DEBUG: bool = True
    APP_NAME: str = "RAGApp"
    ENVIRONMENT: str

    # Logging
    LOGGING_LEVEL: str

    # Redis / Cache — MLOPS_TODO #8
    #
    # One Redis instance, four SEPARATE logical databases.
    #   cache     (db 0) — ephemeral lookups
    #   broker    (db 1) — Celery broker           (M4 #83)
    #   result    (db 2) — Celery result backend   (M4 #83)
    #   ratelimit (db 3) — distributed token bucket (M0 #9)
    REDIS_CACHE_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND_URL: str = "redis://localhost:6379/2"
    REDIS_RATELIMIT_URL: str = "redis://localhost:6379/3"

    # Generic healthcheck/ops ping target. Defaults to the cache db.
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_NAMESPACE: str
    CACHE_TTL: int = 60

    # CORS
    CORS_ORIGINS: List[str]

    # Static API key. Flag-gated, off by default in local dev.
    API_KEY: str | None = None
    API_KEY_ENABLED: bool = False

    # LLM Generator
    #   fake    (default) — deterministic, offline; CI + unit tests + GPU-free dev
    #   openai           — any OpenAI-compatible /v1/chat/completions server:
    #                       vLLM in prod (Qwen2.5-7B-AWQ), Ollama in dev,
    #                       TGI / llama.cpp / Azure — only the base_url+model

    GENERATOR_BACKEND: str = "fake"
    GENERATOR_SYSTEM_PROMPT: str = (
        "You are a careful assistant that answers questions about the Egyptian "
        "Civil Code. Answer only from the context provided. If the context does "
        "not contain the answer, say so explicitly instead of guessing."
    )
    OPENAI_BASE_URL: str | None = None
    OPENAI_API_KEY: str | None = None
    GENERATOR_MODEL: str = ""
    GENERATOR_TEMPERATURE: float = 0.0
    GENERATOR_MAX_TOKENS: int = 1000
    GENERATOR_TIMEOUT: float = 120.0

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = False
    # Tokens added to each key's bucket per REFILL_PERIOD seconds.
    RATE_LIMIT_CAPACITY: int = 60
    RATE_LIMIT_REFILL_PERIOD: int = 60
    # Bucket TTL so idle keys don't pile up in Redis (>= 2 x REFILL_PERIOD).
    RATE_LIMIT_BUCKET_TTL: int = 120


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    """
    return Settings()
