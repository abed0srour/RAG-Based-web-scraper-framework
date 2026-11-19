"""Response and request models for the REST API.

These Pydantic models are the contract of the service. They exist for three
reasons beyond tidiness:

1. **The response shape stops depending on the storage layer.** A MongoDB
   document or a SQLAlchemy row can grow a field without that field silently
   appearing in a public response, and a renamed column breaks a test here
   rather than a client in production.
2. **Validation of what comes in.** Query parameters and request bodies are
   coerced and bounds-checked by FastAPI before any handler runs, so an
   endpoint never has to defend itself against ``limit=-1`` or
   ``mode=sideways``; the client gets a 422 naming the offending field.
3. **The documentation writes itself.** Every model and field description
   below appears in the OpenAPI schema served at ``/docs``, which is the
   API documentation deliverable for this task.

Every listing endpoint returns the same envelope - :class:`Page`, with the
items and a ``meta`` block carrying the total count and the next offset - so
paging works identically on raw pages, documents and links.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Paging envelope
# ---------------------------------------------------------------------------


class PageMeta(BaseModel):
    """How much data exists and how to ask for the next slice of it."""

    limit: int = Field(description="Maximum number of items requested")
    offset: int = Field(description="Number of items skipped")
    returned: int = Field(description="Number of items in this response")
    total: int = Field(description="Total number of items matching the filters")
    next_offset: int | None = Field(
        default=None,
        description="Offset of the next page, or null when this is the last one",
    )


class Page(BaseModel, Generic[T]):
    """A page of results plus its metadata."""

    meta: PageMeta
    items: list[T]


def page_of(items: list[Any], *, total: int, limit: int, offset: int) -> dict[str, Any]:
    """Build the envelope for a listing response."""
    consumed = offset + len(items)
    return {
        "meta": {
            "limit": limit,
            "offset": offset,
            "returned": len(items),
            "total": total,
            "next_offset": consumed if consumed < total else None,
        },
        "items": items,
    }


# ---------------------------------------------------------------------------
# Raw crawl data (PostgreSQL)
# ---------------------------------------------------------------------------


class CrawlJobModel(BaseModel):
    """One crawl run."""

    id: int
    job_name: str
    site: str
    seed_urls: list[str] = Field(default_factory=list)
    status: str = Field(description="running, completed or failed")
    pages_scraped: int
    pages_failed: int
    started_at: datetime
    finished_at: datetime | None = None


class RawPageSummary(BaseModel):
    """Metadata of one fetched page, without the HTML body."""

    id: int
    job_id: int | None = None
    url: str
    site: str
    depth: int
    http_status: int | None = None
    content_type: str | None = None
    content_length: int = Field(description="Size of the stored HTML in bytes")
    content_hash: str = Field(description="SHA-256 of the body, for change detection")
    fetched_at: datetime
    fetched_by: str | None = Field(
        default=None, description="host:pid of the worker that fetched the page"
    )


class RawPageDetail(RawPageSummary):
    """One fetched page including its response headers and HTML body."""

    headers: dict[str, Any] = Field(default_factory=dict)
    raw_html: str | None = Field(
        default=None,
        description="The unmodified HTML, omitted when include_html=false",
    )
    html_truncated: bool = Field(
        default=False,
        description="True when the body was cut at API_MAX_HTML_CHARS",
    )


class ParsedPageSummary(BaseModel):
    """The extracted view of one page, without the clean text."""

    id: int
    raw_page_id: int
    url: str
    site: str
    title: str | None = None
    description: str | None = None
    language: str | None = None
    headings: list[str] = Field(default_factory=list)
    word_count: int
    structured_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Site-specific fields extracted by the parser",
    )
    parser_used: str
    parsed_at: datetime


class ParsedPageDetail(ParsedPageSummary):
    """The extracted view including the cleaned main text."""

    clean_text: str = ""


class LinkModel(BaseModel):
    """One link discovered on a crawled page (the frontier audit trail)."""

    id: int
    source_url: str
    target_url: str
    site: str
    anchor_text: str | None = None
    depth: int
    enqueued: bool = Field(
        description="True once the URL was published to the frontier queue"
    )
    discovered_at: datetime


# ---------------------------------------------------------------------------
# Processed / enhanced documents (MongoDB)
# ---------------------------------------------------------------------------


class DocumentStats(BaseModel):
    """Size statistics computed by the normaliser."""

    model_config = ConfigDict(extra="allow")

    word_count: int = 0
    char_count: int = 0


class DocumentSummary(BaseModel):
    """A processed document without its body text.

    ``extra="allow"`` is deliberate: the ``fields`` sub-document has a
    different shape on every website, and the API must not flatten that
    variation away.
    """

    model_config = ConfigDict(extra="allow")

    doc_id: str
    url: str
    site: str
    title: str | None = None
    description: str | None = None
    language: str | None = None
    keywords: list[str] = Field(default_factory=list)
    stats: DocumentStats | None = None
    enriched: bool = Field(
        default=False,
        description="True when the RAG augmentation stage has written a summary",
    )
    summary: str | None = Field(
        default=None, description="The generated summary, when the document is enriched"
    )


class DocumentDetail(DocumentSummary):
    """A processed document including its cleaned text and enrichment."""

    text: str | None = Field(
        default=None, description="Cleaned body text, omitted when include_text=false"
    )
    headings: list[str] = Field(default_factory=list)
    fields: dict[str, Any] = Field(
        default_factory=dict, description="Site-specific structured fields"
    )
    source: dict[str, Any] = Field(
        default_factory=dict, description="Provenance: raw page id, status, fetch time"
    )
    processing: dict[str, Any] = Field(
        default_factory=dict, description="Which pipeline version produced this document"
    )
    enrichment: dict[str, Any] | None = Field(
        default=None, description="Generated summary, topics, citations and quality flags"
    )


class EnrichmentResponse(BaseModel):
    """The RAG enrichment of one document, on its own."""

    doc_id: str
    url: str
    enrichment: dict[str, Any]


class ChunkModel(BaseModel):
    """One indexed passage of a document, as stored in the vector index."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    doc_id: str = ""
    chunk_index: int = 0
    heading: str = ""
    text: str = ""


# ---------------------------------------------------------------------------
# Search and RAG
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    """One retrieved passage and the evidence for why it ranked here."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    doc_id: str
    url: str
    site: str
    title: str
    heading: str = ""
    chunk_index: int = 0
    score: float = Field(description="Fused score (reciprocal rank fusion in hybrid mode)")
    vector_score: float | None = Field(default=None, description="Cosine relevance, 0..1")
    vector_rank: int | None = None
    keyword_score: float | None = Field(default=None, description="MongoDB text score")
    keyword_rank: int | None = None
    retrievers: list[str] = Field(
        default_factory=list, description="Which retriever(s) produced this passage"
    )
    words: int = 0
    text: str | None = None


class SearchTimings(BaseModel):
    embed: float = 0.0
    vector_search: float = 0.0
    keyword_search: float = 0.0
    total: float = 0.0


class SearchSource(BaseModel):
    doc_id: str
    url: str
    site: str
    title: str
    chunks: int
    best_score: float


class SearchResponse(BaseModel):
    """The result of one retrieval call."""

    query: str
    mode: str = Field(description="vector, keyword or hybrid")
    site: str | None = None
    results: int
    top_score: float
    vector_candidates: int = 0
    keyword_candidates: int = 0
    timings_ms: SearchTimings
    sources: list[SearchSource] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=list,
        description="Degradations, e.g. one retriever being unavailable",
    )
    chunks: list[SearchHit] = Field(default_factory=list)


class AskRequest(BaseModel):
    """A question for the RAG pipeline."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What do the quotes say about life?",
                "mode": "hybrid",
                "top_k": 5,
                "max_sentences": 4,
            }
        }
    )

    question: str = Field(min_length=3, max_length=1000)
    mode: str = Field(default="hybrid", pattern="^(vector|keyword|hybrid)$")
    top_k: int | None = Field(default=None, ge=1, le=20)
    site: str | None = Field(default=None, max_length=255)
    max_sentences: int = Field(default=5, ge=1, le=15)
    include_context: bool = Field(
        default=False, description="Include the passage texts behind the answer"
    )


class CitationModel(BaseModel):
    index: int
    url: str
    title: str
    site: str
    doc_id: str
    chunk_id: str
    score: float


class AskTimings(BaseModel):
    retrieve: float = 0.0
    generate: float = 0.0
    total: float = 0.0


class AskResponse(BaseModel):
    """A generated answer with the citations that support it."""

    question: str
    answer: str
    provider: str
    model: str
    grounded_by_construction: bool = Field(
        description="True when every sentence was taken from the retrieved passages"
    )
    citations: list[CitationModel] = Field(default_factory=list)
    timings_ms: AskTimings
    notes: list[str] = Field(default_factory=list)
    retrieval: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness: the process is up and serving."""

    status: str = "ok"
    service: str
    version: str
    uptime_seconds: float


class ComponentHealth(BaseModel):
    """One dependency of the service."""

    status: str = Field(description="up, down or disabled")
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Readiness: which backing services are reachable right now.

    The endpoint answers 200 when the API can serve *something* and 503 when
    no data store is reachable, which is what a Kubernetes readiness probe
    needs in order to take one replica out of the load balancer instead of
    restarting it.
    """

    status: str = Field(description="ready or degraded")
    checks: dict[str, ComponentHealth]


class RateLimitStatus(BaseModel):
    backend: str
    degraded: bool
    window_seconds: int
    limits: dict[str, int]


class IdentityResponse(BaseModel):
    """What the service knows about the caller - useful for debugging keys."""

    name: str
    key_id: str
    scopes: list[str]
    authenticated: bool
    rate_limit: RateLimitStatus


class ErrorDetail(BaseModel):
    status: int
    message: str
    type: str
    path: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """The single error shape every failing request returns."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "status": 429,
                    "message": "rate limit exceeded: at most 30 search request(s) ...",
                    "type": "rate_limited",
                    "path": "/v1/search",
                    "request_id": "7f3c1d9a2b",
                }
            }
        }
    )

    error: ErrorDetail
