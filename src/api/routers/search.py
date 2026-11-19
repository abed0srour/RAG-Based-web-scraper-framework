"""Endpoints for **searching the indexed data** and asking questions of it.

This is deliverable 3 of Task 5, and it is where the API stops being a view
over a database and starts exposing the retrieval system built in Phase 4.
There are three shapes of query here, deliberately kept separate because
they cost different amounts and answer different questions:

``GET /v1/search``
    Passage retrieval over the chunk index, in the three modes the retriever
    supports: ``vector`` (embed the query, nearest neighbours in Chroma),
    ``keyword`` (MongoDB's weighted full-text index) and ``hybrid``
    (both, fused with reciprocal rank fusion). The response carries the
    per-retriever scores and ranks, so a client can see *why* a passage was
    returned rather than only that it was.

``GET /v1/search/documents``
    Document-level keyword search, answered entirely by the MongoDB text
    index. No embedding model is touched, so it stays fast and available even
    when the vector store is down - and it is the right endpoint when the
    caller wants pages, not passages.

``POST /v1/ask``
    The full RAG read path: retrieve, then generate an answer from the
    retrieved context with numbered citations. It is a POST because the
    question is content rather than an addressable resource, and because
    questions can be long enough to be awkward in a URL.

Retrieval requires the ``search`` scope and generation the ``ask`` scope, and
both groups carry their own rate-limit window on top of the global one:
retrieval loads an ONNX model and queries two indexes, and generation can
call a paid API, so these are the two endpoints a runaway client would hurt
most.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status

from src.api.deps import RetrieverDep, StoreDep
from src.api.rate_limit import enforce_ask_limit, enforce_search_limit
from src.api.schemas import AskRequest, AskResponse, SearchHit, SearchResponse
from src.api.security import require_scope
from src.common.config import settings

router = APIRouter(tags=["search"])


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search the indexed corpus (vector, keyword or hybrid)",
    dependencies=[Depends(require_scope("search")), Depends(enforce_search_limit)],
)
def search(
    retriever: RetrieverDep,
    q: Annotated[
        str,
        Query(min_length=2, max_length=500, description="The query text"),
    ],
    mode: Annotated[
        str,
        Query(pattern="^(vector|keyword|hybrid)$", description="Retrieval strategy"),
    ] = "hybrid",
    top_k: Annotated[
        int | None, Query(ge=1, le=50, description="Passages to return")
    ] = None,
    site: Annotated[str | None, Query(description="Restrict to one site")] = None,
    min_score: Annotated[
        float, Query(ge=0.0, le=1.0, description="Drop passages below this score")
    ] = 0.0,
    per_document: Annotated[
        int | None,
        Query(ge=1, le=20, description="Cap on passages contributed by one page"),
    ] = None,
    include_text: Annotated[
        bool, Query(description="Include the passage texts in the response")
    ] = True,
) -> Any:
    """Retrieve the passages most relevant to ``q``.

    In ``hybrid`` mode the two retrievers are independent: if the vector
    store or the embedding model is unavailable the keyword half still
    answers, and the degradation is reported in ``notes`` rather than as an
    error. That is the fault-tolerance property of the retrieval layer made
    visible over HTTP.
    """
    result = retriever.retrieve(
        q,
        top_k=top_k or settings.rag_top_k,
        mode=mode,
        site=site,
        min_score=min_score,
        per_document=per_document,
    )
    return result.as_dict(include_text=include_text)


@router.get(
    "/search/documents",
    response_model=list[dict],
    summary="Document-level keyword search (MongoDB text index)",
    dependencies=[Depends(require_scope("search")), Depends(enforce_search_limit)],
)
def search_documents(
    store: StoreDep,
    q: Annotated[str, Query(min_length=2, max_length=500)],
    site: Annotated[str | None, Query(description="Restrict to one site")] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> Any:
    """Whole pages ranked by MongoDB's weighted text score.

    The weights come from the compound text index created in Phase 3, which
    scores a title match above a body match. No embedding is computed, so
    this endpoint works with the vector store down.
    """
    return store.search(
        q,
        site=site,
        limit=limit,
        projection=("doc_id", "url", "site", "title", "description", "keywords"),
    )


@router.get(
    "/search/similar/{doc_id}",
    response_model=list[SearchHit],
    summary="Find pages similar to a given document",
    dependencies=[Depends(require_scope("search")), Depends(enforce_search_limit)],
)
def similar_documents(
    retriever: RetrieverDep,
    doc_id: Annotated[str, Path(min_length=4, max_length=128)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> Any:
    """Passages from *other* pages whose embeddings resemble this one.

    Answers 404 when the document has no passages in the vector index, which
    means either that it does not exist or that it has not been indexed yet -
    both of which a client needs to know about rather than see as an empty
    list.
    """
    chunks = retriever.related_to_document(doc_id, top_k=top_k)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"no indexed passages for doc_id {doc_id!r}; the document may "
                "not exist or may not be indexed yet"
            ),
        )
    return [chunk.as_dict() for chunk in chunks]


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question answered from the indexed corpus (RAG)",
    dependencies=[Depends(require_scope("ask")), Depends(enforce_ask_limit)],
)
def ask(
    retriever: RetrieverDep,
    payload: Annotated[AskRequest, Body()],
) -> Any:
    """Retrieve, then generate an answer with numbered citations.

    When nothing in the index matches, the response is still a 200 with an
    empty ``answer`` and a note saying so: "no grounded answer exists" is a
    successful outcome of a RAG query, not a client error, and the citations
    list makes the difference checkable.
    """
    from src.rag.augment import answer_question

    result = answer_question(
        payload.question,
        top_k=payload.top_k,
        mode=payload.mode,
        site=payload.site,
        max_sentences=payload.max_sentences,
        retriever=retriever,
    )
    return result.as_dict(include_context=payload.include_context)
