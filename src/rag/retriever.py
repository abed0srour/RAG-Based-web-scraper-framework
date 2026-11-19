"""The retrieval function: fetching the relevant content for a query.

This is the **R** in RAG, and the piece the rest of the module is built
around. Given a question, it returns the handful of passages from the crawled
corpus that are most likely to contain the answer, each one carrying the URL
it came from so that whatever is generated afterwards can be traced back to a
real page.

Retrieval runs in one of three modes.

``vector``
    Embed the query with the same model that embedded the chunks, then ask
    Chroma for the nearest neighbours. This finds passages that *mean* the
    same thing as the query even when they share no vocabulary with it.

``keyword``
    Full-text search in MongoDB, answered by the weighted text index built in
    Task 3, then narrowed to the best-matching chunks of the pages it returns.
    This is exact-term search: it finds a product code, a surname or a price
    that an embedding model has no reason to place anywhere in particular.

``hybrid`` (the default)
    Both of the above, fused. The two retrievers fail in opposite directions -
    semantic search misses rare literal strings, keyword search misses
    paraphrases - and a corpus of scraped pages contains plenty of both. Any
    passage found by both retrievers is almost certainly relevant, so fusing
    the two rankings puts those first.

**Fusion by rank, not by score.** The two retrievers produce numbers that are
not comparable: a cosine relevance in ``0 .. 1`` and an unbounded MongoDB
``textScore``. Normalising them against each other would mean inventing a
weighting with nothing to justify it. Reciprocal Rank Fusion avoids the
problem by discarding the magnitudes and using only the positions::

    score(chunk) = sum over retrievers of 1 / (k + rank)

with ``k = 60`` (the value from the original TREC experiments; it damps the
influence of the top rank just enough that two second-places can outrank one
first-place). The result is ordinal, robust, and needs no tuning per corpus.

The retrieved passages are returned as a :class:`RetrievalResult`, which can
hand back either the raw chunks or a ready-to-use, cited context block for the
LLM stage in :mod:`src.rag.augment`.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from src.common.config import settings
from src.processing.store import DocumentStore, StoreUnavailable
from src.rag.embedder import Embedder, EmbeddingUnavailable, get_embedder
from src.rag.vector_store import VectorStore, VectorStoreUnavailable

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant. 60 is the value from the original paper and
# is the usual default; it is kept explicit here because it is the only tuning
# knob in the fusion step.
RRF_K = 60

# Documents whose chunks are pulled back for keyword re-ranking. Each one
# costs a read from Chroma, so the keyword side looks at the strongest hits
# rather than at every match.
_KEYWORD_DOC_LIMIT = 8
_KEYWORD_CHUNKS_PER_DOC = 2

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how in into is it its of on
    or that the their there these this to was were what when where which who
    why will with about over under between do does did can could should would
    """.split()
)

MODES = ("hybrid", "vector", "keyword")


class RetrievalError(RuntimeError):
    """Raised when a retrieval mode cannot run at all."""


def query_terms(query: str) -> list[str]:
    """The content words of a query, used for keyword scoring and highlights."""
    return [
        word
        for word in _WORD_RE.findall((query or "").lower())
        if word not in _STOPWORDS and len(word) > 1
    ]


@dataclass
class RetrievedChunk:
    """One retrieved passage and the evidence for why it was retrieved."""

    chunk_id: str
    doc_id: str = ""
    text: str = ""
    url: str = ""
    site: str = ""
    title: str = ""
    heading: str = ""
    chunk_index: int = 0
    score: float = 0.0                      # fused score (RRF, or the single retriever's)
    vector_score: float | None = None       # cosine relevance, 0 .. 1
    vector_rank: int | None = None
    keyword_score: float | None = None      # MongoDB textScore
    keyword_rank: int | None = None
    retrievers: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """A short human label for the citation line."""
        name = self.title or self.url or self.doc_id
        return f"{name} ({self.site})" if self.site else name

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "url": self.url,
            "site": self.site,
            "title": self.title,
            "heading": self.heading,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 6),
            "vector_score": round(self.vector_score, 4) if self.vector_score is not None else None,
            "vector_rank": self.vector_rank,
            "keyword_score": round(self.keyword_score, 4) if self.keyword_score is not None else None,
            "keyword_rank": self.keyword_rank,
            "retrievers": self.retrievers,
            "words": len(self.text.split()),
            "text": self.text,
        }


@dataclass
class RetrievalResult:
    """Everything one retrieval call produced, including how it got there."""

    query: str
    mode: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    vector_candidates: int = 0
    keyword_candidates: int = 0
    embed_ms: float = 0.0
    vector_ms: float = 0.0
    keyword_ms: float = 0.0
    total_ms: float = 0.0
    site: str | None = None
    notes: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.chunks)

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0

    def sources(self) -> list[dict[str, Any]]:
        """The distinct pages behind the retrieved passages, in rank order."""
        seen: dict[str, dict[str, Any]] = {}
        for chunk in self.chunks:
            entry = seen.get(chunk.doc_id)
            if entry is None:
                seen[chunk.doc_id] = {
                    "doc_id": chunk.doc_id,
                    "url": chunk.url,
                    "site": chunk.site,
                    "title": chunk.title,
                    "chunks": 1,
                    "best_score": round(chunk.score, 6),
                }
            else:
                entry["chunks"] += 1
        return list(seen.values())

    def context(self, *, max_chars: int | None = None, numbered: bool = True) -> str:
        """Render the retrieved passages as a cited context block.

        The numbering is what makes the generation stage checkable: the
        prompt asks for ``[n]`` markers, so every sentence in the summary can
        be traced back to the passage - and therefore the URL - it came from.
        """
        budget = max_chars or settings.rag_context_chars
        blocks: list[str] = []
        used = 0

        for position, chunk in enumerate(self.chunks, start=1):
            header = f"[{position}] {chunk.title or chunk.url}" if numbered else (chunk.title or chunk.url)
            if chunk.heading and chunk.heading != chunk.title:
                header = f"{header} — {chunk.heading}"
            block = f"{header}\nURL: {chunk.url}\n{chunk.text.strip()}"

            if used + len(block) > budget:
                remaining = budget - used
                # A passage is only truncated when it is the reason the budget
                # ran out; anything shorter than a sentence is dropped whole.
                if remaining > 400:
                    blocks.append(block[:remaining].rstrip() + " …")
                break

            blocks.append(block)
            used += len(block) + 2

        return "\n\n".join(blocks)

    def as_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        chunks = [chunk.as_dict() for chunk in self.chunks]
        if not include_text:
            for chunk in chunks:
                chunk.pop("text", None)
        return {
            "query": self.query,
            "mode": self.mode,
            "site": self.site,
            "results": len(self.chunks),
            "top_score": round(self.top_score, 6),
            "vector_candidates": self.vector_candidates,
            "keyword_candidates": self.keyword_candidates,
            "timings_ms": {
                "embed": round(self.embed_ms, 2),
                "vector_search": round(self.vector_ms, 2),
                "keyword_search": round(self.keyword_ms, 2),
                "total": round(self.total_ms, 2),
            },
            "sources": self.sources(),
            "notes": self.notes,
            "chunks": chunks,
        }

    def print_report(self, *, snippet: int = 200) -> None:
        print()
        print("=" * 72)
        print(f"  Retrieval — {self.mode} — {self.query!r}")
        print("=" * 72)
        print(f"  passages returned : {len(self.chunks)}")
        print(f"  candidates        : {self.vector_candidates} vector, "
              f"{self.keyword_candidates} keyword")
        print(f"  timings (ms)      : embed {self.embed_ms:.1f}, "
              f"vector {self.vector_ms:.1f}, keyword {self.keyword_ms:.1f}, "
              f"total {self.total_ms:.1f}")
        for note in self.notes:
            print(f"  note              : {note}")
        print("-" * 72)
        for position, chunk in enumerate(self.chunks, start=1):
            found_by = "+".join(chunk.retrievers) or "?"
            print(f"  [{position}] score {chunk.score:.4f}  ({found_by})  {chunk.label}")
            detail = []
            if chunk.vector_score is not None:
                detail.append(f"cosine {chunk.vector_score:.3f} @ rank {chunk.vector_rank}")
            if chunk.keyword_score is not None:
                detail.append(f"textScore {chunk.keyword_score:.2f} @ rank {chunk.keyword_rank}")
            if detail:
                print(f"       {', '.join(detail)}")
            print(f"       {chunk.url}")
            text = " ".join(chunk.text.split())
            print(f"       \"{text[:snippet]}{'…' if len(text) > snippet else ''}\"")
        print("=" * 72)


def _chunk_from_hit(hit: Mapping[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(hit.get("chunk_id") or ""),
        doc_id=str(hit.get("doc_id") or ""),
        text=str(hit.get("text") or ""),
        url=str(hit.get("url") or ""),
        site=str(hit.get("site") or ""),
        title=str(hit.get("title") or ""),
        heading=str(hit.get("heading") or ""),
        chunk_index=int(hit.get("chunk_index") or 0),
    )


def _keyword_overlap(text: str, terms: Sequence[str]) -> int:
    """How many of the query's content words appear in a passage."""
    if not terms:
        return 0
    body = set(_WORD_RE.findall(text.lower()))
    return sum(1 for term in terms if term in body)


class Retriever:
    """The retrieval function, with its three modes and its fusion step."""

    def __init__(
        self,
        *,
        vectors: VectorStore | None = None,
        store: DocumentStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.vectors = vectors or VectorStore()
        self.store = store or DocumentStore()
        self.embedder = embedder or get_embedder()

    # ---------------- the two retrievers ----------------

    def vector_search(
        self,
        query: str,
        *,
        limit: int,
        site: str | None = None,
        result: RetrievalResult | None = None,
    ) -> list[RetrievedChunk]:
        """Semantic search: embed the query, then search Chroma."""
        started = time.perf_counter()
        vector = self.embedder.embed_query(query)
        embed_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        hits = self.vectors.query(vector, top_k=limit, site=site)
        search_ms = (time.perf_counter() - started) * 1000.0

        if result is not None:
            result.embed_ms = embed_ms
            result.vector_ms = search_ms
            result.vector_candidates = len(hits)

        chunks: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            chunk = _chunk_from_hit(hit)
            chunk.vector_score = float(hit.get("score") or 0.0)
            chunk.vector_rank = rank
            chunk.retrievers = ["vector"]
            chunks.append(chunk)
        return chunks

    def keyword_search(
        self,
        query: str,
        *,
        limit: int,
        site: str | None = None,
        result: RetrievalResult | None = None,
    ) -> list[RetrievedChunk]:
        """Exact-term search: MongoDB's text index, narrowed to chunks.

        The text index ranks *pages*, but retrieval has to return passages, so
        each page's stored chunks are pulled from Chroma and the ones that
        actually contain the query's words are kept. That keeps both
        retrievers working in the same unit, which is what makes the rankings
        fusible.
        """
        started = time.perf_counter()
        terms = query_terms(query)

        try:
            documents = self.store.search(
                query,
                site=site,
                limit=min(limit, _KEYWORD_DOC_LIMIT),
                projection=("doc_id", "url", "site", "title"),
            )
        except StoreUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad text query must not kill the call
            if result is not None:
                result.notes.append(f"keyword search skipped: {exc}")
            logger.debug("Keyword search failed for %r: %s", query, exc)
            return []

        chunks: list[RetrievedChunk] = []
        for document in documents:
            doc_id = str(document.get("doc_id") or "")
            if not doc_id:
                continue

            try:
                stored = self.vectors.get_document_chunks(doc_id, limit=50)
            except VectorStoreUnavailable as exc:
                # The chunk store is down, not this one document: retrying it
                # for every remaining page would multiply one connection
                # timeout by the number of matches, which turns a degraded
                # search into a request that never returns. Stop here and
                # report what was lost.
                if result is not None:
                    result.notes.append(
                        f"chunk lookup unavailable, keyword results truncated: {exc}"
                    )
                logger.warning("Chunk lookup unavailable during keyword search: %s", exc)
                break

            scored = sorted(
                stored,
                key=lambda row: (
                    _keyword_overlap(str(row.get("text") or ""), terms),
                    -int(row.get("chunk_index") or 0),
                ),
                reverse=True,
            )
            for row in scored[:_KEYWORD_CHUNKS_PER_DOC]:
                if terms and not _keyword_overlap(str(row.get("text") or ""), terms):
                    # The page matched on a field that is not in this passage
                    # (its title, say). Returning the passage anyway would be
                    # a false hit, so it is left out.
                    continue
                chunk = _chunk_from_hit({**row, **{k: v for k, v in document.items() if v}})
                chunk.keyword_score = float(document.get("score") or 0.0)
                chunk.retrievers = ["keyword"]
                chunks.append(chunk)

        # Rank by the page's textScore, then by position in the page.
        chunks.sort(key=lambda chunk: (-(chunk.keyword_score or 0.0), chunk.chunk_index))
        for rank, chunk in enumerate(chunks, start=1):
            chunk.keyword_rank = rank

        if result is not None:
            result.keyword_ms = (time.perf_counter() - started) * 1000.0
            result.keyword_candidates = len(chunks)

        return chunks[:limit]

    # ---------------- fusion ----------------

    @staticmethod
    def fuse(*rankings: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion over one or more ranked chunk lists.

        Chunks found by several retrievers are merged into a single entry that
        keeps every retriever's own score and rank, which is what makes a
        ranking explainable after the fact.
        """
        merged: dict[str, RetrievedChunk] = {}
        totals: dict[str, float] = {}

        for ranking in rankings:
            for rank, chunk in enumerate(ranking, start=1):
                key = chunk.chunk_id or f"{chunk.doc_id}:{chunk.chunk_index}"
                existing = merged.get(key)
                if existing is None:
                    merged[key] = chunk
                    totals[key] = 0.0
                    existing = chunk
                else:
                    # Keep whichever evidence each retriever contributed.
                    if chunk.vector_score is not None:
                        existing.vector_score = chunk.vector_score
                        existing.vector_rank = chunk.vector_rank
                    if chunk.keyword_score is not None:
                        existing.keyword_score = chunk.keyword_score
                        existing.keyword_rank = chunk.keyword_rank
                    for name in chunk.retrievers:
                        if name not in existing.retrievers:
                            existing.retrievers.append(name)
                    if not existing.text:
                        existing.text = chunk.text

                totals[key] += 1.0 / (RRF_K + rank)

        for key, chunk in merged.items():
            chunk.score = totals[key]

        return sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)

    # ---------------- the public retrieval function ----------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: str = "hybrid",
        site: str | None = None,
        candidates: int | None = None,
        min_score: float = 0.0,
        per_document: int | None = None,
    ) -> RetrievalResult:
        """Fetch the passages most relevant to ``query``.

        ``per_document`` caps how many passages one page may contribute, so a
        single long page cannot fill the whole context window and crowd out
        the corroboration a summary needs.
        """
        if mode not in MODES:
            raise RetrievalError(f"unknown retrieval mode {mode!r}; expected one of {MODES}")
        if not (query or "").strip():
            raise RetrievalError("the query is empty")

        wanted = top_k or settings.rag_top_k
        pool = candidates or max(settings.rag_candidates, wanted * 3)
        result = RetrievalResult(query=query, mode=mode, site=site)
        started = time.perf_counter()

        vector_chunks: list[RetrievedChunk] = []
        keyword_chunks: list[RetrievedChunk] = []

        if mode in ("vector", "hybrid"):
            try:
                vector_chunks = self.vector_search(query, limit=pool, site=site, result=result)
            except (VectorStoreUnavailable, EmbeddingUnavailable) as exc:
                if mode == "vector":
                    raise
                # In hybrid mode the keyword half can still answer, which is
                # the point of running two retrievers.
                result.notes.append(f"vector search unavailable: {exc}")
                logger.warning("Vector search unavailable, falling back to keyword: %s", exc)

        if mode in ("keyword", "hybrid"):
            try:
                keyword_chunks = self.keyword_search(query, limit=pool, site=site, result=result)
            except StoreUnavailable as exc:
                if mode == "keyword":
                    raise
                result.notes.append(f"keyword search unavailable: {exc}")
                logger.warning("Keyword search unavailable, using vectors only: %s", exc)

        if mode == "vector":
            ranked = vector_chunks
            for chunk in ranked:
                chunk.score = chunk.vector_score or 0.0
        elif mode == "keyword":
            ranked = keyword_chunks
            # A rank-based score keeps the scale comparable with the other modes.
            for rank, chunk in enumerate(ranked, start=1):
                chunk.score = 1.0 / (RRF_K + rank)
        else:
            ranked = self.fuse(vector_chunks, keyword_chunks)

        if min_score > 0:
            ranked = [chunk for chunk in ranked if chunk.score >= min_score]

        if per_document:
            seen: dict[str, int] = {}
            capped: list[RetrievedChunk] = []
            for chunk in ranked:
                count = seen.get(chunk.doc_id, 0)
                if count >= per_document:
                    continue
                seen[chunk.doc_id] = count + 1
                capped.append(chunk)
            ranked = capped

        result.chunks = ranked[:wanted]
        result.total_ms = (time.perf_counter() - started) * 1000.0

        logger.info(
            "Retrieved %s passage(s) for %r in %.1f ms (mode=%s)",
            len(result.chunks), query, result.total_ms, mode,
        )
        return result

    def related_to_document(
        self, doc_id: str, *, top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Passages from *other* pages that resemble this page's opening.

        Used by the augmentation stage: a summary is more useful when it can
        say how a page relates to the rest of the corpus, and this is the
        cheapest honest way to find that out.
        """
        own = self.vectors.get_document_chunks(doc_id, limit=1)
        if not own:
            return []

        vector = self.embedder.embed_texts([str(own[0].get("text") or "")])
        if not vector:
            return []

        hits = self.vectors.query(vector[0], top_k=top_k + 10)
        chunks: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            if str(hit.get("doc_id")) == doc_id:
                continue
            chunk = _chunk_from_hit(hit)
            chunk.vector_score = float(hit.get("score") or 0.0)
            chunk.vector_rank = rank
            chunk.score = chunk.vector_score
            chunk.retrievers = ["vector"]
            chunks.append(chunk)
            if len(chunks) >= top_k:
                break
        return chunks


_DEFAULT: Retriever | None = None


def get_retriever() -> Retriever:
    """The process-wide retriever (shares the loaded embedding model)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever()
    return _DEFAULT


def retrieve(query: str, **kwargs: Any) -> RetrievalResult:
    """Module-level retrieval function - the module's main entry point.

    ``from src.rag.retriever import retrieve`` is all a caller needs:

    >>> result = retrieve("who wrote about life", top_k=3)   # doctest: +SKIP
    >>> [chunk.url for chunk in result.chunks]               # doctest: +SKIP
    """
    return get_retriever().retrieve(query, **kwargs)
