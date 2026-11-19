"""Endpoints for querying the **processed and enhanced content** (deliverable 2).

These serve the MongoDB collection written by Phase 3 and enriched by Phase 4:
one JSON document per page, carrying the cleaned text, the normalised
metadata, the site-specific ``fields`` sub-document, and - once the
augmentation stage has run - an ``enrichment`` block with the generated
summary, the topics, the related pages and the quality flags.

Two design points worth noting:

* **The body text is opt-in.** A listing never returns ``text``, and the
  detail endpoint takes ``include_text``. Documents average a few thousand
  words, so returning them by default would make every listing an order of
  magnitude larger than the metadata the caller usually wants.
* **The enrichment is addressable on its own.** ``/{doc_id}/enrichment``
  exists because the generated content is what distinguishes this corpus
  from the raw crawl, and a client that only wants the summaries should not
  have to download the documents to get them. It answers 404 when a document
  has not been enriched yet, which keeps "not generated" distinguishable
  from "generated and empty".

Every route in this module requires the ``documents`` scope.
"""

from __future__ import annotations

from typing import Annotated, Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from src.api.deps import PaginationDep, StoreDep, VectorStoreDep
from src.api.schemas import (
    ChunkModel,
    DocumentDetail,
    DocumentSummary,
    EnrichmentResponse,
    Page,
    page_of,
)
from src.api.security import require_scope

router = APIRouter(
    prefix="/documents",
    tags=["processed content"],
    dependencies=[Depends(require_scope("documents"))],
)

#: Fields returned by a listing: everything except the body text and the
#: full enrichment block (only its summary is projected).
SUMMARY_FIELDS = (
    "doc_id",
    "url",
    "site",
    "title",
    "description",
    "language",
    "keywords",
    "stats",
    "enrichment.summary",
    "enrichment.topics",
)


def _as_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the enrichment into the two fields a listing shows."""
    record = dict(document)
    enrichment = record.pop("enrichment", None) or {}
    summary = str(enrichment.get("summary") or "").strip()
    record["enriched"] = bool(summary)
    record["summary"] = summary or None
    return record


@router.get(
    "",
    response_model=Page[DocumentSummary],
    summary="List processed documents",
)
def list_documents(
    store: StoreDep,
    page: PaginationDep,
    site: Annotated[str | None, Query(description="Filter by site")] = None,
    enriched: Annotated[
        bool | None,
        Query(description="true = only documents with a generated summary"),
    ] = None,
    min_words: Annotated[
        int | None, Query(ge=0, description="Only documents with at least this many words")
    ] = None,
) -> Any:
    """Processed documents, most recently processed first."""
    documents, total = store.list_documents(
        site=site,
        enriched=enriched,
        min_words=min_words,
        limit=page.limit,
        offset=page.offset,
        projection=SUMMARY_FIELDS,
    )
    items = [_as_summary(document) for document in documents]
    return page_of(items, total=total, limit=page.limit, offset=page.offset)


@router.get(
    "/stats",
    response_model=dict,
    summary="Corpus statistics per site",
)
def document_stats(store: StoreDep) -> Any:
    """Document counts, word counts, compression ratios and the indexes."""
    return store.stats()


@router.get(
    "/sites",
    response_model=list[str],
    summary="List the sites present in the corpus",
)
def document_sites(store: StoreDep) -> Any:
    """The values accepted by every ``site`` filter in this API."""
    return store.sites()


@router.get(
    "/duplicates",
    response_model=list[dict],
    summary="Groups of documents with identical text",
)
def duplicate_documents(
    store: StoreDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Any:
    """Near-duplicate detection by content hash - a data-quality view."""
    return store.duplicates(limit=limit)


@router.get(
    "/by-url",
    response_model=DocumentDetail,
    summary="Fetch a document by its URL",
    responses={404: {"description": "No document stored for this URL"}},
)
def get_document_by_url(
    store: StoreDep,
    url: Annotated[str, Query(min_length=8, description="The page URL")],
    include_text: Annotated[bool, Query()] = False,
) -> Any:
    """Look a document up by URL instead of by ``doc_id``.

    The identifier is derived from the canonicalised URL, so this is an
    indexed lookup and not a scan: the same normalisation the crawler applied
    is applied to the query.
    """
    document = store.get_by_url(url)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no processed document for url {url!r}",
        )
    return _detail(document, include_text=include_text)


@router.get(
    "/{doc_id}",
    response_model=DocumentDetail,
    summary="Fetch one processed document",
    responses={404: {"description": "No document with this id"}},
)
def get_document(
    store: StoreDep,
    doc_id: Annotated[str, Path(min_length=4, max_length=128)],
    include_text: Annotated[
        bool, Query(description="Include the cleaned body text")
    ] = True,
) -> Any:
    """One document, with its enrichment block when it has been generated."""
    document = store.get(doc_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no processed document with doc_id {doc_id!r}",
        )
    return _detail(document, include_text=include_text)


def _detail(document: Mapping[str, Any], *, include_text: bool) -> dict[str, Any]:
    record = dict(document)
    if not include_text:
        record.pop("text", None)
    enrichment = record.get("enrichment") or {}
    summary = str(enrichment.get("summary") or "").strip()
    record["enriched"] = bool(summary)
    record["summary"] = summary or None
    return record


@router.get(
    "/{doc_id}/enrichment",
    response_model=EnrichmentResponse,
    summary="Fetch the generated enrichment of one document",
    responses={404: {"description": "This document has not been enriched"}},
)
def get_document_enrichment(
    store: StoreDep,
    doc_id: Annotated[str, Path(min_length=4, max_length=128)],
) -> Any:
    """The summary, topics, related pages, quality flags and token counts."""
    record = store.get_enrichment(doc_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"document {doc_id!r} has no enrichment yet; run "
                "'python -m src.rag.augment' to generate it"
            ),
        )
    return record


@router.get(
    "/{doc_id}/chunks",
    response_model=list[ChunkModel],
    summary="Fetch the indexed passages of one document",
)
def get_document_chunks(
    vectors: VectorStoreDep,
    doc_id: Annotated[str, Path(min_length=4, max_length=128)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Any:
    """How the document is represented in the vector index.

    This is the view that explains a search result: the passages here are
    exactly the units that retrieval ranks, so a client can see why a page
    matched a query it does not obviously contain.
    """
    return vectors.get_document_chunks(doc_id, limit=limit)
