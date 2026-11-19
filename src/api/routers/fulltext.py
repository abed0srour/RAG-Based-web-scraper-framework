"""Endpoints for **full-text search**, answered by Elasticsearch (Task 6).

Task 5 already exposed two kinds of search: passage retrieval over the
vector index, and document-level keyword search over MongoDB's text index.
This router adds the third, and it is the one the web UI uses, because it is
the only one that returns what a search *interface* needs:

``GET /v1/fulltext``
    Ranked documents for a query, each with the fragments that matched and
    the terms marked, plus the facet counts over the whole matching set so
    the UI can offer "311 from books.toscrape.com" as a filter. Supports
    filtering by site, language, length and enrichment state, five sort
    orders, and offset paging over an exact total.

``GET /v1/fulltext/suggest``
    As-you-type suggestions, matched against edge n-grams built at index
    time. Cheap enough to call on every keystroke.

``GET /v1/fulltext/similar/{doc_id}``
    Documents sharing significant terms with this one - the lexical
    counterpart of ``/v1/search/similar``, which compares embeddings.

``GET /v1/fulltext/sites``
    The indexed sites and their counts, which is what populates the UI's
    site filter before any query has been typed.

All of them need the ``search`` scope and share the search rate-limit
window with the retrieval endpoints: they are cheap individually, but the
UI calls them on every keystroke, so they belong in the same budget.

An empty ``q`` is allowed here and is not an error: it means "browse", and
with a sort and a filter that is exactly what the UI's initial view needs.
That is why the minimum length applies to the suggest endpoint only.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from src.api.deps import PaginationDep, SearchIndexDep
from src.api.rate_limit import enforce_search_limit
from src.api.schemas import FullTextHit, FullTextResponse, SuggestionModel
from src.api.security import require_scope
from src.search.index import SORTS

router = APIRouter(
    prefix="/fulltext",
    tags=["full-text search"],
    dependencies=[Depends(require_scope("search")), Depends(enforce_search_limit)],
)


@router.get(
    "",
    response_model=FullTextResponse,
    summary="Full-text search with highlights and facets (Elasticsearch)",
)
def fulltext_search(
    search: SearchIndexDep,
    page: PaginationDep,
    q: Annotated[
        str,
        Query(
            max_length=500,
            description="The query text; empty browses the index by the chosen sort",
        ),
    ] = "",
    site: Annotated[str | None, Query(description="Restrict to one site")] = None,
    language: Annotated[str | None, Query(description="Restrict to one language")] = None,
    min_words: Annotated[
        int | None, Query(ge=0, description="Drop documents shorter than this")
    ] = None,
    enriched: Annotated[
        bool | None,
        Query(description="Only documents that have (or have not) a generated summary"),
    ] = None,
    sort: Annotated[
        str,
        Query(
            pattern="^(relevance|recent|oldest|words|title)$",
            description="relevance, recent, oldest, words or title",
        ),
    ] = "relevance",
    highlight: Annotated[
        bool, Query(description="Return the matched fragments")
    ] = True,
    facets: Annotated[bool, Query(description="Return the facet counts")] = True,
) -> Any:
    """Search the full-text index.

    The handler does no reshaping: ``SearchIndex.search`` already returns the
    response shape, so the query logic lives in one place and the tests can
    assert on the query a set of parameters produces without a cluster.
    """
    if sort not in SORTS:  # pragma: no cover - the pattern above rejects these
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown sort {sort!r}",
        )
    return search.search(
        q,
        limit=page.limit,
        offset=page.offset,
        site=site,
        language=language,
        min_words=min_words,
        enriched=enriched,
        sort=sort,
        highlight=highlight,
        facets=facets,
    )


@router.get(
    "/suggest",
    response_model=list[SuggestionModel],
    summary="As-you-type suggestions for the search box",
)
def suggest(
    search: SearchIndexDep,
    q: Annotated[
        str, Query(min_length=2, max_length=100, description="What has been typed")
    ],
    limit: Annotated[int, Query(ge=1, le=20, description="Suggestions to return")] = 8,
) -> Any:
    """Titles and keywords beginning with ``q``."""
    return search.suggest(q, limit=limit)


@router.get(
    "/sites",
    summary="Indexed sites and their document counts",
)
def sites(search: SearchIndexDep) -> Any:
    """Populates the UI's site filter."""
    return search.sites()


@router.get(
    "/similar/{doc_id}",
    response_model=list[FullTextHit],
    summary="Documents sharing significant terms with this one",
)
def similar(
    search: SearchIndexDep,
    doc_id: Annotated[str, Path(description="The document to compare against")],
    limit: Annotated[int, Query(ge=1, le=20, description="Documents to return")] = 5,
) -> Any:
    """Lexically similar documents, by shared high-TF-IDF terms."""
    if search.get(doc_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"document {doc_id} is not in the full-text index; it may not "
                "exist or may not be indexed yet"
            ),
        )
    return search.similar(doc_id, limit=limit)
