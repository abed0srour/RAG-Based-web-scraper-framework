"""Central configuration for the Distributed RAG Web Scraper Framework.

Every component (API, Celery workers, RAG pipeline) imports the same
``settings`` object, so the whole distributed system is configured from one
place: environment variables, or a local ``.env`` file during development.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed, validated settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- LLM ----------
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="claude-sonnet-5-5", alias="LLM_MODEL")

    # ---------- Celery / Redis ----------
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/1", alias="CELERY_RESULT_BACKEND"
    )

    # ---------- Vector store ----------
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection: str = Field(default="scraped_docs", alias="CHROMA_COLLECTION")

    # ---------- API ----------
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT")

    # ---------- Scraper behaviour ----------
    scraper_user_agent: str = Field(
        default="DistributedRAGScraper/1.0", alias="SCRAPER_USER_AGENT"
    )
    scraper_timeout: int = Field(default=20, alias="SCRAPER_TIMEOUT")
    scraper_max_retries: int = Field(default=3, alias="SCRAPER_MAX_RETRIES")
    scraper_rate_limit_seconds: float = Field(
        default=1.0, alias="SCRAPER_RATE_LIMIT_SECONDS"
    )

    # ---------- Paths ----------
    data_dir: Path = PROJECT_ROOT / "data"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
