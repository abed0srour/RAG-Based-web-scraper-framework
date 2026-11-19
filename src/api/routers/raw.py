"""Endpoints for fetching the **raw scraped data** (Task 5, deliverable 1).

"Raw" here means the crawl as it was recorded in PostgreSQL by Phase 2,
before any cleaning: the HTTP status, the response headers and the unmodified
HTML body, plus the parser's extracted view of the same page and the link
graph that drove the frontier.

Why the raw data is exposed at all, given that the processed documents are
nicer to consume: it is the evidence. A client that disagrees with a cleaned
document can fetch the exact bytes the decision was made from, and a change
to the parsing rules can be checked against stored pages without re-crawling
the target site. The split between this router and
``src/api/routers/documents.py`` mirrors the split between the two stores.

Every route in this module requires the ``raw`` scope.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status

from src.api.deps import PaginationDep, RepositoryDep
from src.api.schemas import (
    CrawlJobModel,
    LinkModel,
    Page,
    ParsedPageDetail,
    ParsedPageSummary,
    RawPageDetail,
    RawPageSummary,
    page_of,
)
from src.api.security import require_scope
from src.common.config import settings

router = APIRouter(
    prefix="/raw",
    tags=["raw data"],
    dependencies=[Depends(require_scope("raw"))],
)


@router.get(
    "/jobs",
    response_model=Page[CrawlJobModel],
    summary="List crawl jobs",
)
def list_jobs(
    repository: RepositoryDep,
    page: PaginationDep,
    site: Annotated[str | None, Query(description="Filter by site")] = None,
    status_filter: Annotated[
        str | None,
        Query(alias="status", description="Filter by job status"),
    ] = None,
) -> Any:
    """Every crawl run, newest first - the provenance of every stored page."""
    items, total = repository.list_jobs(
        site=site, status=status_filter, limit=page.limit, offset=page.offset
    )
    return page_of(items, total=total, limit=page.limit, offset=page.offset)


@router.get(
    "/pages",
    response_model=Page[RawPageSummary],
    summary="List fetched pages (metadata only)",
)
def list_raw_pages(
    repository: RepositoryDep,
    page: PaginationDep,
    site: Annotated[str | None, Query(description="Filter by site")] = None,
    job_id: Annotated[int | None, Query(ge=1, description="Filter by crawl job")] = None,
    http_status: Annotated[
        int | None, Query(ge=100, le=599, description="Filter by HTTP status")
    ] = None,
    url_contains: Annotated[
        str | None,
        Query(min_length=2, max_length=500, description="Case-insensitive URL filter"),
    ] = None,
) -> Any:
    """Metadata of the fetched pages, without the HTML bodies.

    The bodies are left out on purpose: one page can be hundreds of
    kilobytes, so a 20-item listing with bodies would be megabytes. Fetch
    them one at a time from ``/raw/pages/{page_id}``.
    """
    items, total = repository.list_raw_pages(
        site=site,
        job_id=job_id,
        http_status=http_status,
        url_contains=url_contains,
        limit=page.limit,
        offset=page.offset,
    )
    return page_of(items, total=total, limit=page.limit, offset=page.offset)


@router.get(
    "/pages/{page_id}",
    response_model=RawPageDetail,
    summary="Fetch one page with its headers and HTML",
    responses={404: {"description": "No page with this id"}},
)
def get_raw_page(
    repository: RepositoryDep,
    page_id: Annotated[int, Path(ge=1)],
    include_html: Annotated[
        bool, Query(description="Include the HTML body in the response")
    ] = True,
) -> Any:
    """One fetched page.

    The body is truncated at ``API_MAX_HTML_CHARS`` and the response says so
    in ``html_truncated``, so a single pathological page cannot produce a
    response large enough to matter to the client or the proxy in front of
    the service.
    """
    record = repository.get_raw_page(
        page_id,
        include_html=include_html,
        max_html_chars=settings.api_max_html_chars,
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no raw page with id {page_id}",
        )
    return record


@router.get(
    "/pages/{page_id}/html",
    response_class=Response,
    summary="Fetch one page's HTML as-is",
    responses={
        200: {"content": {"text/html": {}}, "description": "The stored HTML body"},
        404: {"description": "No page with this id"},
    },
)
def get_raw_page_html(
    repository: RepositoryDep,
    page_id: Annotated[int, Path(ge=1)],
) -> Response:
    """The stored HTML body, served with its original content type.

    A JSON string containing HTML is awkward to work with - it has to be
    unescaped before it can be opened in a browser or fed to a parser - so
    this endpoint returns the bytes themselves instead.
    """
    found = repository.get_raw_page_html(page_id)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no raw page with id {page_id}",
        )
    html, content_type = found
    media_type = content_type.split(";")[0].strip() or "text/html"
    return Response(content=html, media_type=media_type)


@router.get(
    "/pages/{page_id}/parsed",
    response_model=ParsedPageDetail,
    summary="Fetch the parser's view of one page",
    responses={404: {"description": "This page has not been parsed"}},
)
def get_parsed_page(
    repository: RepositoryDep,
    page_id: Annotated[int, Path(ge=1, description="The *raw page* id")],
) -> Any:
    """Title, clean text, headings and the site-specific structured fields."""
    record = repository.get_parsed_page(page_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"raw page {page_id} has no parsed record",
        )
    return record


@router.get(
    "/parsed",
    response_model=Page[ParsedPageSummary],
    summary="List parsed pages",
)
def list_parsed_pages(
    repository: RepositoryDep,
    page: PaginationDep,
    site: Annotated[str | None, Query(description="Filter by site")] = None,
    min_words: Annotated[
        int | None, Query(ge=0, description="Only pages with at least this many words")
    ] = None,
) -> Any:
    """The extracted view of the crawl, without the clean text bodies."""
    items, total = repository.list_parsed_pages(
        site=site, min_words=min_words, limit=page.limit, offset=page.offset
    )
    return page_of(items, total=total, limit=page.limit, offset=page.offset)


@router.get(
    "/links",
    response_model=Page[LinkModel],
    summary="List discovered links",
)
def list_links(
    repository: RepositoryDep,
    page: PaginationDep,
    site: Annotated[str | None, Query(description="Filter by site")] = None,
    source_url: Annotated[
        str | None, Query(description="Only links found on this page")
    ] = None,
    enqueued: Annotated[
        bool | None, Query(description="Filter by whether the URL reached the frontier")
    ] = None,
) -> Any:
    """The frontier audit trail: which page produced which URL, and when."""
    items, total = repository.list_links(
        site=site,
        source_url=source_url,
        enqueued=enqueued,
        limit=page.limit,
        offset=page.offset,
    )
    return page_of(items, total=total, limit=page.limit, offset=page.offset)


@router.get(
    "/stats",
    summary="Crawl statistics per site",
    response_model=dict,
)
def raw_stats(repository: RepositoryDep) -> Any:
    """Per-site counters over the raw, parsed and link tables."""
    return repository.stats()
