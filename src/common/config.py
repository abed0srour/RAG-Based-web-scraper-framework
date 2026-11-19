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
    llm_model: str = Field(default="claude-opus-5-5", alias="LLM_MODEL")
    # "auto" uses Claude when ANTHROPIC_API_KEY is set and falls back to the
    # free offline extractive summariser otherwise; "anthropic" and
    # "extractive" pin one provider. See src/rag/llm.py.
    llm_provider: str = Field(default="auto", alias="LLM_PROVIDER")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_effort: str = Field(default="low", alias="LLM_EFFORT")

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

    # ---------- MongoDB (NoSQL store for the processed documents) ----------
    mongo_host: str = Field(default="localhost", alias="MONGO_HOST")
    mongo_port: int = Field(default=27018, alias="MONGO_PORT")
    mongo_user: str = Field(default="scraper", alias="MONGO_USER")
    mongo_password: str = Field(default="scraper_pass", alias="MONGO_PASSWORD")
    mongo_db: str = Field(default="processed_data", alias="MONGO_DB")
    mongo_collection: str = Field(default="documents", alias="MONGO_COLLECTION")
    mongo_auth_source: str = Field(default="admin", alias="MONGO_AUTH_SOURCE")
    mongo_timeout_ms: int = Field(default=5000, alias="MONGO_TIMEOUT_MS")

    # ---------- Data processing (Task 3) ----------
    processing_batch_size: int = Field(default=50, alias="PROCESSING_BATCH_SIZE")
    processing_min_words: int = Field(default=3, alias="PROCESSING_MIN_WORDS")

    # ---------- Celery / Redis (used from Task 4 onwards) ----------
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # ---------- Vector store (Task 4) ----------
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection: str = Field(default="scraped_docs", alias="CHROMA_COLLECTION")
    chroma_timeout: int = Field(default=30, alias="CHROMA_TIMEOUT")

    # ---------- RAG (Task 4) ----------
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_cache_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "models", alias="EMBEDDING_CACHE_DIR"
    )
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    chunk_words: int = Field(default=180, alias="CHUNK_WORDS")
    chunk_overlap_words: int = Field(default=40, alias="CHUNK_OVERLAP_WORDS")
    chunk_min_words: int = Field(default=20, alias="CHUNK_MIN_WORDS")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_candidates: int = Field(default=20, alias="RAG_CANDIDATES")
    rag_index_batch_size: int = Field(default=128, alias="RAG_INDEX_BATCH_SIZE")
    rag_context_chars: int = Field(default=6000, alias="RAG_CONTEXT_CHARS")

    # ---------- Full-text search : Elasticsearch (Task 6) ----------
    # The fourth store. MongoDB's text index answers "which documents contain
    # these words"; Elasticsearch answers the question a search *interface*
    # asks - ranked hits with highlighted snippets, facet counts and
    # as-you-type suggestions - which is why the UI searches here.
    elastic_host: str = Field(default="localhost", alias="ELASTIC_HOST")
    elastic_port: int = Field(default=9200, alias="ELASTIC_PORT")
    elastic_scheme: str = Field(default="http", alias="ELASTIC_SCHEME")
    # Security is off in the development cluster, so these are empty by
    # default; a real deployment sets both and switches the scheme to https.
    elastic_user: str = Field(default="", alias="ELASTIC_USER")
    elastic_password: str = Field(default="", alias="ELASTIC_PASSWORD")
    elastic_index: str = Field(default="scraped_documents", alias="ELASTIC_INDEX")
    elastic_timeout: int = Field(default=30, alias="ELASTIC_TIMEOUT")
    elastic_index_batch_size: int = Field(
        default=200, alias="ELASTIC_INDEX_BATCH_SIZE"
    )
    # Characters of each document's text sent to the index. The stored text is
    # unbounded; a highlighted snippet never needs the whole of a long page.
    elastic_max_text_chars: int = Field(default=60_000, alias="ELASTIC_MAX_TEXT_CHARS")
    # Snippet count and width for the highlighter.
    elastic_highlight_fragments: int = Field(
        default=3, alias="ELASTIC_HIGHLIGHT_FRAGMENTS"
    )
    elastic_highlight_size: int = Field(default=180, alias="ELASTIC_HIGHLIGHT_SIZE")

    # ---------- API (Task 5) ----------
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT")
    api_version: str = Field(default="1.0.0", alias="API_VERSION")
    api_docs_enabled: bool = Field(default=True, alias="API_DOCS_ENABLED")
    api_cors_origins: str = Field(default="*", alias="API_CORS_ORIGINS")

    # Authentication. Keys are declared as
    #   name:key:scope,scope;name:key:scope
    # and parsed by src/api/security.py. The defaults below are development
    # keys only - they are printed in the repository, so a real deployment
    # must set API_KEYS to its own values.
    api_auth_enabled: bool = Field(default=True, alias="API_AUTH_ENABLED")
    api_keys: str = Field(
        default=(
            "reader:dev-reader-key:raw,documents;"
            "analyst:dev-analyst-key:raw,documents,search,ask;"
            "admin:dev-admin-key:*"
        ),
        alias="API_KEYS",
    )

    # Rate limiting. One fixed window per key, plus tighter windows for the
    # two expensive endpoint groups (retrieval and generation).
    api_rate_limit: int = Field(default=120, alias="API_RATE_LIMIT")
    api_rate_window_seconds: int = Field(default=60, alias="API_RATE_WINDOW_SECONDS")
    api_search_rate_limit: int = Field(default=30, alias="API_SEARCH_RATE_LIMIT")
    api_ask_rate_limit: int = Field(default=10, alias="API_ASK_RATE_LIMIT")
    # auto -> Redis when reachable, in-process counters otherwise.
    api_rate_limit_backend: str = Field(default="auto", alias="API_RATE_LIMIT_BACKEND")

    # Paging and payload caps.
    api_page_size: int = Field(default=20, alias="API_PAGE_SIZE")
    api_max_page_size: int = Field(default=100, alias="API_MAX_PAGE_SIZE")
    api_max_html_chars: int = Field(default=200_000, alias="API_MAX_HTML_CHARS")
    # Load the ONNX embedding model at start-up instead of on the first
    # search. Costs about a second of start-up and removes it from the first
    # client request - worth it behind a readiness probe, which will simply
    # hold the replica out of the load balancer until the model is in memory.
    api_warm_embedder: bool = Field(default=False, alias="API_WARM_EMBEDDER")

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

    @property
    def mongo_uri(self) -> str:
        """Connection string for the MongoDB document store.

        Credentials are left out when no user is configured, so the same
        settings object works both against the authenticated container and
        against a bare local ``mongod`` started without a root user.
        """
        if self.mongo_user:
            credentials = f"{self.mongo_user}:{self.mongo_password}@"
            auth = f"?authSource={self.mongo_auth_source}"
        else:
            credentials = ""
            auth = ""
        return f"mongodb://{credentials}{self.mongo_host}:{self.mongo_port}/{auth}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
