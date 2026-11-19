"""Shared FastAPI dependencies: paging, and the handles to the data layer.

The data-layer accessors here exist so that the routers never construct a
``DocumentStore`` or a ``Retriever`` themselves. Two things follow from that:

* **The expensive objects are built once.** The retriever holds the loaded
  ONNX embedding model (tens of megabytes and a second of start-up), so one
  instance is shared by every request instead of one per call.
* **The tests can replace them.** Every accessor is used through
  ``Depends(...)``, which means a test can inject a fake store or a fake
  retriever with ``app.dependency_overrides`` and exercise the full HTTP
  path - routing, auth, rate limiting, validation, serialisation - without
  PostgreSQL, MongoDB, Chroma or an LLM being up. That is what makes the
  response tests for this task runnable on a laptop with nothing started.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Query

from src.common.config import settings
from src.processing.store import DocumentStore
from src.storage.database import ScrapeRepository

# ---------------------------------------------------------------------------
# Paging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pagination:
    """Validated ``limit``/``offset`` pair."""

    limit: int
    offset: int


def pagination(
    limit: Annotated[
        int | None,
        Query(ge=1, description="Items per page (defaults to API_PAGE_SIZE)"),
    ] = None,
    offset: Annotated[int, Query(ge=0, description="Items to skip")] = 0,
) -> Pagination:
    """Clamp paging parameters to the configured maximum.

    The cap is enforced here rather than per endpoint so that no listing can
    be turned into a full table dump by one large ``limit``, which is the
    other half of rate limiting: bounding the size of a request as well as
    their number.
    """
    requested = limit or settings.api_page_size
    return Pagination(limit=min(requested, settings.api_max_page_size), offset=offset)


PaginationDep = Annotated[Pagination, Depends(pagination)]


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

_store: DocumentStore | None = None


def get_repository() -> type[ScrapeRepository]:
    """The PostgreSQL repository (static methods, nothing to construct)."""
    return ScrapeRepository


def get_document_store() -> DocumentStore:
    """The MongoDB document store, created on first use."""
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store


def get_retriever_dep() -> Any:
    """The hybrid retriever, with its embedding model loaded once.

    Imported lazily: ``src.rag.retriever`` pulls in the embedding stack, and
    an API that only serves the crawl endpoints should not pay that cost at
    import time.
    """
    from src.rag.retriever import get_retriever

    return get_retriever()


def get_vector_store() -> Any:
    """The Chroma vector store (used by the chunk and readiness endpoints)."""
    from src.rag.vector_store import VectorStore

    return VectorStore()


def get_search_index_dep() -> Any:
    """The Elasticsearch full-text index (Task 6).

    Shared like the others: the client owns a connection pool, and building
    one per request would discard it. Imported lazily so that an API serving
    only the crawl endpoints does not need the search client at import time.
    """
    from src.search.index import get_search_index

    return get_search_index()


RepositoryDep = Annotated[type[ScrapeRepository], Depends(get_repository)]
StoreDep = Annotated[DocumentStore, Depends(get_document_store)]
RetrieverDep = Annotated[Any, Depends(get_retriever_dep)]
VectorStoreDep = Annotated[Any, Depends(get_vector_store)]
SearchIndexDep = Annotated[Any, Depends(get_search_index_dep)]
