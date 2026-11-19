"""The generation stage: combining retrieved text with AI-generated output.

Retrieval on its own returns passages. This module is what turns them into
something better than the passages were, and it does so in the two directions
the task asks for.

**1. Answering a question (read path).**
:func:`answer_question` retrieves the relevant passages, hands them to the
configured provider as a numbered, cited context block, and returns the
generated answer together with the citations behind it. Nothing reaches the
model that did not come out of the corpus, and every answer carries the URLs
it was built from, so a claim can always be checked against the page it came
from.

**2. Enriching the stored documents (write path).**
:class:`Augmenter` walks the processed documents and writes an ``enrichment``
sub-document back into MongoDB next to each one. That is where the data
quality actually improves, and it improves in four measurable ways:

*A contextual summary.*
    Scraped pages have no summary. The passages of the page itself are
    combined with semantically *related* passages from other pages - which is
    only possible because the vector index exists - so the summary describes
    the page in the context of the corpus rather than in isolation.

*Missing fields filled.*
    A page with no ``<meta name="description">`` ends Task 3 with
    ``description: null``. The generated summary supplies one, and
    ``enrichment.fields_filled`` records that it was derived rather than
    scraped, so a derived value is never mistaken for a scraped one.

*Topics.*
    Keywords extracted from the summary rather than from the raw body are
    about what the page is *for*, not about which words it repeats most.

*Related pages.*
    The nearest neighbours of the page, stored as links. A crawl produces a
    link graph; this produces a similarity graph, which the pages themselves
    do not contain.

Every enrichment records a before/after **quality score** - a six-point
checklist over the fields a document is expected to have - so the effect of
the stage is a number, not an impression. The provider and model that
produced each summary are stored alongside it, which keeps a free extractive
summary distinguishable from a Claude-generated one for ever afterwards.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Sequence

from src.common.config import settings
from src.common.logging_config import setup_logging
from src.processing.normalizer import extract_keywords
from src.processing.store import DocumentStore, StoreUnavailable
from src.rag.embedder import EmbeddingUnavailable
from src.rag.llm import (
    BaseLLM,
    LLMError,
    LLMResponse,
    content_words,
    get_llm,
    split_sentences,
)
from src.rag.retriever import RetrievalResult, Retriever, RetrievedChunk, get_retriever
from src.rag.vector_store import VectorStoreUnavailable

logger = logging.getLogger(__name__)

ENRICHMENT_VERSION = "1.0"

# The quality checklist. Each entry is a field a complete document is expected
# to have; the score is the fraction satisfied. Keeping it explicit (rather
# than a hand-tuned weighting) makes the before/after comparison auditable.
_QUALITY_CHECKS = (
    "has_title",
    "has_description",
    "has_summary",
    "has_keywords",
    "has_headings",
    "has_body",
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """One numbered source behind a generated answer."""

    index: int
    url: str
    title: str
    site: str
    doc_id: str
    chunk_id: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "url": self.url,
            "title": self.title,
            "site": self.site,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
        }


@dataclass
class RagAnswer:
    """A generated answer, its citations and the retrieval behind it."""

    question: str
    answer: str
    provider: str
    model: str
    citations: list[Citation] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    grounded: bool = False
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self, *, include_context: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "provider": self.provider,
            "model": self.model,
            "grounded_by_construction": self.grounded,
            "citations": [citation.as_dict() for citation in self.citations],
            "timings_ms": {
                "retrieve": round(self.retrieve_ms, 2),
                "generate": round(self.generate_ms, 2),
                "total": round(self.total_ms, 2),
            },
            "notes": self.notes,
        }
        if self.retrieval is not None:
            data["retrieval"] = self.retrieval.as_dict(include_text=include_context)
        return data

    def print_report(self) -> None:
        print()
        print("=" * 72)
        print(f"  Question : {self.question}")
        print(f"  Provider : {self.provider} ({self.model})")
        print("=" * 72)
        print(self.answer or "(the retrieved passages do not answer this question)")
        print("-" * 72)
        for citation in self.citations:
            print(f"  [{citation.index}] {citation.title or citation.url}")
            print(f"       {citation.url}")
        print("-" * 72)
        print(f"  retrieve {self.retrieve_ms:.0f} ms | generate "
              f"{self.generate_ms:.0f} ms | total {self.total_ms:.0f} ms")
        for note in self.notes:
            print(f"  note: {note}")
        print("=" * 72)


def _citations_from(chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            index=position,
            url=chunk.url,
            title=chunk.title,
            site=chunk.site,
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            score=chunk.score,
        )
        for position, chunk in enumerate(chunks, start=1)
    ]


def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    mode: str = "hybrid",
    site: str | None = None,
    max_sentences: int = 5,
    retriever: Retriever | None = None,
    llm: BaseLLM | None = None,
    instruction: str | None = None,
) -> RagAnswer:
    """Retrieve, then generate: the full RAG read path in one call.

    The answer is produced only from the retrieved context block, and the
    numbering in that block matches the returned citations, so ``[2]`` in the
    answer is the second citation's URL.
    """
    engine = retriever or get_retriever()
    model = llm or get_llm()
    started = time.perf_counter()

    result = engine.retrieve(question, top_k=top_k, mode=mode, site=site, per_document=2)
    retrieve_ms = (time.perf_counter() - started) * 1000.0

    if not result.chunks:
        return RagAnswer(
            question=question,
            answer="",
            provider=model.name,
            model=model.model,
            retrieval=result,
            retrieve_ms=retrieve_ms,
            total_ms=retrieve_ms,
            notes=["nothing in the index matched this question"] + result.notes,
        )

    context = result.context()
    generation_started = time.perf_counter()
    response = model.summarise(
        question,
        context,
        passages=result.chunks,
        max_sentences=max_sentences,
        instruction=instruction,
    )
    generate_ms = (time.perf_counter() - generation_started) * 1000.0

    return RagAnswer(
        question=question,
        answer=response.text,
        provider=response.provider,
        model=response.model,
        citations=_citations_from(result.chunks),
        retrieval=result,
        grounded=response.grounded,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        total_ms=(time.perf_counter() - started) * 1000.0,
        notes=result.notes + response.notes,
    )


# ---------------------------------------------------------------------------
# Document enrichment
# ---------------------------------------------------------------------------


def quality_flags(document: Mapping[str, Any], *, summary: str | None = None) -> dict[str, bool]:
    """Evaluate the quality checklist for one document."""
    enrichment = document.get("enrichment") or {}
    stored_summary = summary if summary is not None else (enrichment.get("summary") or "")
    stats = document.get("stats") or {}
    return {
        "has_title": bool((document.get("title") or "").strip()),
        "has_description": bool((document.get("description") or "").strip()),
        "has_summary": bool((stored_summary or "").strip()),
        "has_keywords": bool(document.get("keywords")),
        "has_headings": bool(document.get("headings")),
        "has_body": int(stats.get("word_count") or 0) >= 20,
    }


def quality_score(flags: Mapping[str, bool]) -> float:
    """The checklist as a ``0 .. 1`` score."""
    return round(sum(1 for key in _QUALITY_CHECKS if flags.get(key)) / len(_QUALITY_CHECKS), 4)


@dataclass
class AugmentStats:
    """Counters for one enrichment run, including the quality delta."""

    documents_read: int = 0
    documents_augmented: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    fields_filled: int = 0
    related_links: int = 0
    llm_calls: int = 0
    llm_ms: float = 0.0
    retrieve_ms: float = 0.0
    wall_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    quality_before: list[float] = field(default_factory=list)
    quality_after: list[float] = field(default_factory=list)
    providers: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def avg_quality_before(self) -> float:
        return (sum(self.quality_before) / len(self.quality_before)) if self.quality_before else 0.0

    @property
    def avg_quality_after(self) -> float:
        return (sum(self.quality_after) / len(self.quality_after)) if self.quality_after else 0.0

    @property
    def quality_gain(self) -> float:
        return self.avg_quality_after - self.avg_quality_before

    @property
    def ms_per_document(self) -> float:
        return (self.wall_ms / self.documents_augmented) if self.documents_augmented else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents_read": self.documents_read,
            "documents_augmented": self.documents_augmented,
            "documents_skipped": self.documents_skipped,
            "documents_failed": self.documents_failed,
            "fields_filled": self.fields_filled,
            "related_links": self.related_links,
            "llm_calls": self.llm_calls,
            "llm_ms": round(self.llm_ms, 2),
            "retrieve_ms": round(self.retrieve_ms, 2),
            "wall_ms": round(self.wall_ms, 2),
            "ms_per_document": round(self.ms_per_document, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "providers": self.providers,
            "quality": {
                "before": round(self.avg_quality_before, 4),
                "after": round(self.avg_quality_after, 4),
                "gain": round(self.quality_gain, 4),
                "gain_percent": round(self.quality_gain * 100, 1),
            },
            "errors": self.errors[:10],
        }

    def print_report(self) -> None:
        data = self.as_dict()
        print()
        print("=" * 68)
        print("  RAG augmentation run")
        print("=" * 68)
        print(f"  documents read       : {data['documents_read']:>10,}")
        print(f"  documents augmented  : {data['documents_augmented']:>10,}")
        print(f"  documents skipped    : {data['documents_skipped']:>10,}  (no text to summarise)")
        print(f"  documents failed     : {data['documents_failed']:>10,}")
        print("-" * 68)
        print(f"  summaries generated  : {data['llm_calls']:>10,}")
        print(f"  providers used       : {data['providers']}")
        print(f"  missing fields filled: {data['fields_filled']:>10,}")
        print(f"  related links added  : {data['related_links']:>10,}")
        if data["input_tokens"] or data["output_tokens"]:
            print(f"  tokens in / out      : {data['input_tokens']:,} / {data['output_tokens']:,}")
        print("-" * 68)
        print(f"  retrieval time       : {data['retrieve_ms'] / 1000.0:>10.2f} s")
        print(f"  generation time      : {data['llm_ms'] / 1000.0:>10.2f} s")
        print(f"  wall clock           : {data['wall_ms'] / 1000.0:>10.2f} s "
              f"({data['ms_per_document']:.0f} ms/document)")
        print("-" * 68)
        print("  data quality (6-point checklist, averaged)")
        print(f"    before             : {data['quality']['before']:>10.3f}")
        print(f"    after              : {data['quality']['after']:>10.3f}")
        print(f"    improvement        : {data['quality']['gain_percent']:>9.1f}%")
        if self.errors:
            print("-" * 68)
            print(f"  first {min(5, len(self.errors))} error(s):")
            for message in self.errors[:5]:
                print(f"    - {message}")
        print("=" * 68)


class Augmenter:
    """Writes contextual summaries and derived fields back into MongoDB."""

    def __init__(
        self,
        *,
        store: DocumentStore | None = None,
        retriever: Retriever | None = None,
        llm: BaseLLM | None = None,
        max_sentences: int = 4,
        related_count: int = 3,
        dry_run: bool = False,
    ) -> None:
        self.store = store or DocumentStore()
        self.retriever = retriever or get_retriever()
        self.llm = llm or get_llm()
        self.max_sentences = max_sentences
        self.related_count = related_count
        self.dry_run = dry_run

    # ---------------- reading ----------------

    def iter_documents(
        self,
        *,
        site: str | None = None,
        limit: int | None = None,
        only_missing: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Stream documents to enrich, newest schema first.

        ``only_missing`` is what makes a second run cheap: a document that
        already carries an enrichment of the current version is not sent to
        the LLM again.
        """
        self.store.require()

        filters: dict[str, Any] = {}
        if site:
            filters["site"] = site
        if only_missing:
            filters["enrichment.version"] = {"$ne": ENRICHMENT_VERSION}

        projection = {
            "_id": 0, "doc_id": 1, "url": 1, "site": 1, "title": 1,
            "description": 1, "headings": 1, "keywords": 1, "text": 1,
            "stats": 1, "enrichment": 1,
        }
        cursor = self.store.collection.find(filters, projection).sort("site", 1)
        if limit:
            cursor = cursor.limit(int(limit))
        for document in cursor:
            yield document

    # ---------------- the context for one document ----------------

    def build_context(
        self, document: Mapping[str, Any]
    ) -> tuple[str, str, int, list[RetrievedChunk], RetrievalResult | None]:
        """Assemble the passages a document's summary is generated from.

        Two sources are combined, which is the whole point of doing this after
        the index exists:

        * the document's **own** chunks, read straight out of the vector store
          (no search needed - they are addressed by ``doc_id``);
        * the **nearest neighbours** from other pages, which give the model
          the corpus-level context that the page itself cannot provide.
        """
        doc_id = str(document.get("doc_id") or "")
        query = _summary_query(document)

        own_chunks: list[RetrievedChunk] = []
        try:
            for row in self.retriever.vectors.get_document_chunks(doc_id, limit=3):
                own_chunks.append(
                    RetrievedChunk(
                        chunk_id=str(row.get("chunk_id") or ""),
                        doc_id=doc_id,
                        text=str(row.get("text") or ""),
                        url=str(row.get("url") or document.get("url") or ""),
                        site=str(row.get("site") or document.get("site") or ""),
                        title=str(row.get("title") or document.get("title") or ""),
                        heading=str(row.get("heading") or ""),
                        chunk_index=int(row.get("chunk_index") or 0),
                        score=1.0,
                        retrievers=["self"],
                    )
                )
        except VectorStoreUnavailable:
            raise

        if not own_chunks:
            # The page is not in the vector index (indexing skipped it, or it
            # was processed after the last index run). Its own clean text is
            # still a valid basis for a summary, so it is used directly.
            text = str(document.get("text") or "")
            if text.strip():
                own_chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{doc_id}:text",
                        doc_id=doc_id,
                        text=text[:4000],
                        url=str(document.get("url") or ""),
                        site=str(document.get("site") or ""),
                        title=str(document.get("title") or ""),
                        score=1.0,
                        retrievers=["mongodb"],
                    )
                )

        related = self.retriever.related_to_document(doc_id, top_k=self.related_count)

        result = RetrievalResult(
            query=str(query),
            mode="document+related",
            chunks=own_chunks + related,
        )
        return str(query), result.context(), len(own_chunks), related, result

    # ---------------- enriching one document ----------------

    def augment_document(
        self, document: Mapping[str, Any], *, write: bool = True
    ) -> dict[str, Any]:
        """Generate and (optionally) store the enrichment for one document."""
        doc_id = str(document.get("doc_id") or "")
        if not doc_id:
            raise ValueError("a document needs a doc_id before it can be augmented")

        before = quality_flags(document)

        started = time.perf_counter()
        query, context, own_passages, related, result = self.build_context(document)
        retrieve_ms = (time.perf_counter() - started) * 1000.0

        instruction = (
            f"Write at most {self.max_sentences} sentences describing what this "
            "page contains and what it is useful for. Passage [1] onwards are "
            "the page itself; any later passages are related pages from the "
            "same corpus and may be used only to place this page in context."
        )

        started = time.perf_counter()
        if context.strip():
            response = self.llm.summarise(
                query,
                context,
                passages=result.chunks if result else (),
                max_sentences=self.max_sentences,
                instruction=instruction,
                # The summary describes *this* page, so only the page's own
                # passages may be quoted from. The related passages stay in
                # the context as background (and are stored as links), but a
                # sentence from another page must never end up in this
                # page's description.
                max_passage_rank=own_passages,
            )
        else:
            response = LLMResponse(
                text="",
                provider=self.llm.name,
                model=self.llm.model,
                notes=["no text available to summarise"],
            )
        generate_ms = (time.perf_counter() - started) * 1000.0

        summary = response.text.strip()
        after = quality_flags(document, summary=summary)

        fields_filled: list[str] = []
        updates: dict[str, Any] = {}

        # Fill a missing description from the summary. The original value is
        # never overwritten: a scraped description is evidence, a derived one
        # is an inference, and the two must stay distinguishable.
        if summary and not before["has_description"]:
            derived = _first_sentences(summary, 2)
            if derived:
                updates["description"] = derived
                fields_filled.append("description")
                after["has_description"] = True

        topics = _topics_from(summary, document)

        enrichment = {
            "version": ENRICHMENT_VERSION,
            "summary": summary,
            "summary_query": query,
            "topics": topics,
            "provider": response.provider,
            "model": response.model,
            "grounded_by_construction": response.grounded,
            "generated_at": utcnow().isoformat(),
            "passages_used": len(result.chunks) if result else 0,
            "related": [
                {
                    "doc_id": chunk.doc_id,
                    "url": chunk.url,
                    "title": chunk.title,
                    "score": round(chunk.score, 4),
                }
                for chunk in related
            ],
            "quality": {
                "checks_before": before,
                "checks_after": after,
                "score_before": quality_score(before),
                "score_after": quality_score(after),
                "fields_filled": fields_filled,
            },
            "timings_ms": {
                "retrieve": round(retrieve_ms, 2),
                "generate": round(generate_ms, 2),
            },
            "tokens": {
                "input": response.input_tokens,
                "output": response.output_tokens,
            },
            "notes": response.notes,
        }
        updates["enrichment"] = enrichment

        if write and not self.dry_run:
            self.store.collection.update_one({"doc_id": doc_id}, {"$set": updates})

        return enrichment

    # ---------------- the run ----------------

    def run(
        self,
        *,
        site: str | None = None,
        limit: int | None = None,
        only_missing: bool = False,
    ) -> AugmentStats:
        """Enrich the corpus (or one site) and report the quality delta."""
        stats = AugmentStats()
        started = time.perf_counter()

        for document in self.iter_documents(site=site, limit=limit, only_missing=only_missing):
            stats.documents_read += 1

            try:
                enrichment = self.augment_document(document)
            except (VectorStoreUnavailable, EmbeddingUnavailable, LLMError) as exc:
                # Infrastructure failure: the next document would fail the
                # same way, so the run stops with what it has written.
                stats.errors.append(str(exc))
                stats.wall_ms = (time.perf_counter() - started) * 1000.0
                raise
            except Exception as exc:  # noqa: BLE001 - one bad document, not a lost run
                stats.documents_failed += 1
                stats.errors.append(f"{document.get('url')}: {exc}")
                logger.warning("Augmentation failed for %s: %s", document.get("url"), exc)
                continue

            if not enrichment.get("summary"):
                stats.documents_skipped += 1
                continue

            quality = enrichment["quality"]
            stats.documents_augmented += 1
            stats.llm_calls += 1
            stats.fields_filled += len(quality["fields_filled"])
            stats.related_links += len(enrichment["related"])
            stats.quality_before.append(float(quality["score_before"]))
            stats.quality_after.append(float(quality["score_after"]))
            stats.retrieve_ms += float(enrichment["timings_ms"]["retrieve"])
            stats.llm_ms += float(enrichment["timings_ms"]["generate"])
            stats.input_tokens += int(enrichment["tokens"]["input"] or 0)
            stats.output_tokens += int(enrichment["tokens"]["output"] or 0)
            provider = str(enrichment["provider"])
            stats.providers[provider] = stats.providers.get(provider, 0) + 1

        stats.wall_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Augmented %s document(s) in %.2f s (quality %.3f -> %.3f)",
            stats.documents_augmented, stats.wall_ms / 1000.0,
            stats.avg_quality_before, stats.avg_quality_after,
        )
        return stats


def _first_sentences(text: str, count: int) -> str:
    """The first ``count`` sentences of ``text``, with citation markers removed."""
    cleaned = re.sub(r"\s*\[\d+\]", "", text).strip() if text else ""
    sentences = split_sentences(cleaned)
    return " ".join(sentences[:count])[:2000]


def _topics_from(summary: str, document: Mapping[str, Any]) -> list[str]:
    """Topic words for a document, taken from the summary where possible.

    The summary states what the page is about; the body repeats whatever the
    site's template repeats. Words from the summary are therefore preferred,
    with the body's own keywords used to fill the list out.

    The frequency ranking and the stop-word list are the normaliser's
    (:func:`src.processing.normalizer.extract_keywords`), reused rather than
    reimplemented - a second, weaker stop-word list here produced topics like
    "actually" and "pretty" on real pages.
    """
    topics: list[str] = list(extract_keywords(summary, limit=10))

    for keyword in document.get("keywords") or []:
        if len(topics) >= 10:
            break
        if isinstance(keyword, str) and keyword not in topics:
            topics.append(keyword)
    return topics[:10]


def _summary_query(document: Mapping[str, Any]) -> str:
    """What the summary is asked to be *about*.

    The title alone is not enough: on a templated site every page is called
    the same thing ("Quotes to Scrape"), which gives the query-focused
    summariser nothing to focus on. The headings and the document's own
    keywords are therefore folded in, which is what makes the summary differ
    between two pages of the same site.
    """
    parts: list[str] = []
    for value in (document.get("title"), document.get("description")):
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    headings = [h for h in (document.get("headings") or []) if isinstance(h, str)][:3]
    parts.extend(headings)

    keywords = [k for k in (document.get("keywords") or []) if isinstance(k, str)][:8]
    if keywords:
        parts.append(" ".join(keywords))

    query = " — ".join(dict.fromkeys(part for part in parts if part))
    return query or f"the content of {document.get('url')}"


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate contextual summaries for the processed documents "
                    "and write them back into MongoDB."
    )
    parser.add_argument("--site", default=None, help="Only augment one site.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum documents.")
    parser.add_argument(
        "--only-missing", action="store_true",
        help="Skip documents that already carry a current enrichment.",
    )
    parser.add_argument(
        "--sentences", type=int, default=4,
        help="Maximum sentences per summary.",
    )
    parser.add_argument(
        "--provider", default=None, choices=("auto", "anthropic", "extractive"),
        help="Override LLM_PROVIDER for this run.",
    )
    parser.add_argument(
        "--ask", default=None,
        help="Ask one question instead of enriching documents.",
    )
    parser.add_argument(
        "--mode", default="hybrid", choices=("hybrid", "vector", "keyword"),
        help="Retrieval mode used by --ask.",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Passages retrieved by --ask."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate the summaries but write nothing.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Print the result as JSON.",
    )
    args = parser.parse_args(argv)

    setup_logging("rag-augment")

    try:
        llm = get_llm(args.provider)
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.ask:
        try:
            answer = answer_question(
                args.ask, top_k=args.top_k, mode=args.mode, site=args.site, llm=llm
            )
        except (StoreUnavailable, VectorStoreUnavailable, EmbeddingUnavailable, LLMError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        if args.as_json:
            print(json.dumps(answer.as_dict(), indent=2, ensure_ascii=False))
        else:
            answer.print_report()
        return 0 if answer.answer else 1

    augmenter = Augmenter(llm=llm, max_sentences=args.sentences, dry_run=args.dry_run)
    try:
        stats = augmenter.run(site=args.site, limit=args.limit, only_missing=args.only_missing)
    except (StoreUnavailable, VectorStoreUnavailable, EmbeddingUnavailable, LLMError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(stats.as_dict(), indent=2))
    else:
        stats.print_report()

    return 1 if stats.documents_failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
