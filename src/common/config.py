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

    # ---------- RabbitMQ (dynamic URL frontier) ----------
    rabbitmq_host: str = Field(default="localhost", alias="RABBITMQ_HOST")
    rabbitmq_port: int = Field(default=5672, alias="RABBITMQ_PORT")
    rabbitmq_user: str = Field(default="scraper", alias="RABBITMQ_USER")
    rabbitmq_password: str = Field(default="scraper_pass", alias="RABBITMQ_PASSWORD")
    rabbitmq_vhost: str = Field(default="/", alias="RABBITMQ_VHOST")

    # Queue / exchange topology
    frontier_exchange: str = Field(default="scraper.frontier", alias="FRONTIER_EXCHANGE")
    frontier_queue: str = Field(default="urls.frontier", alias="FRONTIER_QUEUE")
    frontier_dlq: str = Field(default="urls.dead-letter", alias="FRONTIER_DLQ")
    frontier_prefetch: int = Field(default=1, alias="FRONTIER_PREFETCH")

    # ---------- PostgreSQL (raw + parsed data) ----------
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5433, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="scraper", alias="POSTGRES_USER")
    postgres_password: str = Field(default="scraper_pass", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="scraped_data", alias="POSTGRES_DB")

    # ---------- Celery / Redis (used from Task 4 onwards) ----------
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

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
