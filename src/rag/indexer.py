"""The indexing pipeline: MongoDB documents in, vectors out.

This is the write path of the RAG module, and the counterpart of
:mod:`src.processing.pipeline` one stage further down the chain:

.. code-block:: text

    MongoDB (processed_data.documents)
        |  read in batches, newest schema only
        v
    chunker    - cut each document into overlapping passages
        |
        v
    embedder   - 384-dim vector per passage, locally, on the CPU
        |
        v
    Chroma     - upsert <doc_id>:<nnnn> -> (vector, text, metadata)

It is written as a batched streaming loop rather than "load everything, then
embed everything": the corpus is unbounded in principle, embedding is the slow
stage, and a crash halfway through a run must leave a usable index behind. One
batch is read, chunked, embedded and written before the next is read, so peak
memory is a function of the batch size and not of the corpus size.

Failure of a single document is contained the same way the Task 3 pipeline
contains it: the document is counted, the reason recorded, and the run
continues. A document that fails to embed is the one page missing from the
index, not a lost run.

Run it with::

    python -m src.rag.indexer --reset
    python -m src.rag.indexer --site quotes.toscrape.com --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from src.common.config import settings
from src.common.logging_config import setup_logging
from src.processing.store import DocumentStore, StoreUnavailable
from src.rag.chunker import Chunk, ChunkStats, chunk_document
from src.rag.embedder import Embedder, EmbeddingUnavailable, get_embedder
from src.rag.vector_store import VectorStore, VectorStoreUnavailable

logger = logging.getLogger(__name__)

# Only the fields the chunker actually reads are pulled out of MongoDB. The
# body text is the bulk of a document, so excluding anything unused keeps the
# read cost roughly at the text itself.
_PROJECTION = {
    "_id": 0,
    "doc_id": 1,
    "url": 1,
    "site": 1,
    "title": 1,
    "description": 1,
    "headings": 1,
    "text": 1,
    "stats.word_count": 1,
}


def node_id() -> str:
    """Host and PID, so a log line says which worker produced it."""
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass
class IndexStats:
    """Counters and timings for one indexing run."""

    documents_read: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_written: int = 0
    words_indexed: int = 0
    chunk_sizes: list[int] = field(default_factory=list)
    embed_ms: float = 0.0
    store_ms: float = 0.0
    wall_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)

    @property
    def chunks_per_document(self) -> float:
        return (self.chunks_written / self.documents_indexed) if self.documents_indexed else 0.0

    @property
    def avg_chunk_words(self) -> float:
        return (sum(self.chunk_sizes) / len(self.chunk_sizes)) if self.chunk_sizes else 0.0

    @property
    def chunks_per_second(self) -> float:
        return (self.chunks_written / (self.wall_ms / 1000.0)) if self.wall_ms else 0.0

    @property
    def embed_ms_per_chunk(self) -> float:
        return (self.embed_ms / self.chunks_written) if self.chunks_written else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": node_id(),
            "documents_read": self.documents_read,
            "documents_indexed": self.documents_indexed,
            "documents_skipped": self.documents_skipped,
            "documents_failed": self.documents_failed,
            "chunks_written": self.chunks_written,
            "words_indexed": self.words_indexed,
            "chunks_per_document": round(self.chunks_per_document, 2),
            "avg_chunk_words": round(self.avg_chunk_words, 1),
            "embed_ms": round(self.embed_ms, 2),
            "embed_ms_per_chunk": round(self.embed_ms_per_chunk, 3),
            "store_ms": round(self.store_ms, 2),
            "wall_ms": round(self.wall_ms, 2),
            "chunks_per_second": round(self.chunks_per_second, 1),
            "model": self.model,
            "errors": self.errors[:10],
        }

    def print_report(self) -> None:
        data = self.as_dict()
        print()
        print("=" * 68)
        print("  Vector indexing run")
        print("=" * 68)
        print(f"  node                 : {data['node']}")
        print(f"  embedding model      : {self.model.get('model', '?')} "
              f"({self.model.get('dimension', '?')} dims, {self.model.get('runtime', '?')})")
        print("-" * 68)
        print(f"  documents read       : {data['documents_read']:>10,}")
        print(f"  documents indexed    : {data['documents_indexed']:>10,}")
        print(f"  documents skipped    : {data['documents_skipped']:>10,}  (no usable text)")
        print(f"  documents failed     : {data['documents_failed']:>10,}")
        print("-" * 68)
        print(f"  chunks written       : {data['chunks_written']:>10,}")
        print(f"  chunks per document  : {data['chunks_per_document']:>10.2f}")
        print(f"  avg words per chunk  : {data['avg_chunk_words']:>10.1f}")
        print(f"  words indexed        : {data['words_indexed']:>10,}")
        print("-" * 68)
        print(f"  embedding time       : {data['embed_ms'] / 1000.0:>10.2f} s "
              f"({data['embed_ms_per_chunk']:.2f} ms/chunk)")
        print(f"  vector store time    : {data['store_ms'] / 1000.0:>10.2f} s")
        print(f"  wall clock           : {data['wall_ms'] / 1000.0:>10.2f} s")
        print(f"  throughput           : {data['chunks_per_second']:>10.1f} chunks/s")
        if self.errors:
            print("-" * 68)
            print(f"  first {min(5, len(self.errors))} error(s):")
            for message in self.errors[:5]:
                print(f"    - {message}")
        print("=" * 68)


class Indexer:
    """Reads processed documents and maintains the vector index."""

    def __init__(
        self,
        *,
        store: DocumentStore | None = None,
        vectors: VectorStore | None = None,
        embedder: Embedder | None = None,
        batch_size: int | None = None,
        dry_run: bool = False,
    ) -> None:
        self.store = store or DocumentStore()
        self.vectors = vectors or VectorStore()
        self.embedder = embedder or get_embedder()
        self.batch_size = batch_size or settings.rag_index_batch_size
        self.dry_run = dry_run

    # ---------------- reading ----------------

    def iter_documents(
        self,
        *,
        site: str | None = None,
        limit: int | None = None,
        min_words: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream processed documents out of MongoDB.

        A server-side cursor is used rather than ``list(find(...))`` so that
        the corpus never has to fit in memory at once.
        """
        self.store.require()

        filters: dict[str, Any] = {}
        if site:
            filters["site"] = site
        if min_words:
            filters["stats.word_count"] = {"$gte": int(min_words)}

        cursor = (
            self.store.collection.find(filters, _PROJECTION)
            .sort("processing.processed_at", 1)
            .batch_size(min(self.batch_size, 200))
        )
        if limit:
            cursor = cursor.limit(int(limit))

        for document in cursor:
            yield document

    # ---------------- writing ----------------

    def index_document(self, document: Mapping[str, Any], *, replace: bool = True) -> list[Chunk]:
        """Chunk, embed and store one document; returns the chunks written."""
        chunks = chunk_document(document)
        if not chunks:
            return []

        embeddings = self.embedder.embed_texts([chunk.embedding_text for chunk in chunks])

        if not self.dry_run:
            if replace:
                self.vectors.delete_document(chunks[0].doc_id)
            self.vectors.upsert_chunks(chunks, embeddings)
        return chunks

    def _flush(self, chunks: Sequence[Chunk], stats: IndexStats) -> None:
        """Embed and write one batch of chunks, recording the two timings."""
        if not chunks:
            return

        started = time.perf_counter()
        embeddings = self.embedder.embed_texts([chunk.embedding_text for chunk in chunks])
        stats.embed_ms += (time.perf_counter() - started) * 1000.0

        if self.dry_run:
            stats.chunks_written += len(chunks)
            return

        started = time.perf_counter()
        written = self.vectors.upsert_chunks(list(chunks), embeddings)
        stats.store_ms += (time.perf_counter() - started) * 1000.0
        stats.chunks_written += written

        logger.debug("Wrote %s chunk(s) to %s", written, self.vectors.collection_name)

    def run(
        self,
        *,
        site: str | None = None,
        limit: int | None = None,
        min_words: int | None = None,
        replace: bool = True,
    ) -> IndexStats:
        """Index the corpus (or one site) and return the run statistics."""
        stats = IndexStats(model=self.embedder.describe())
        chunk_counters = ChunkStats()

        if not self.dry_run:
            self.vectors.ensure_collection(
                {
                    "embedding_model": self.embedder.model_name,
                    "dimension": self.embedder.dimension,
                    "chunk_words": settings.chunk_words,
                    "chunk_overlap_words": settings.chunk_overlap_words,
                }
            )

        started = time.perf_counter()
        batch: list[Chunk] = []

        for document in self.iter_documents(site=site, limit=limit, min_words=min_words):
            stats.documents_read += 1
            doc_id = str(document.get("doc_id") or "")

            try:
                chunks = chunk_document(document)
            except Exception as exc:  # noqa: BLE001 - one bad document, not a lost run
                stats.documents_failed += 1
                stats.errors.append(f"{document.get('url')}: {exc}")
                logger.warning("Chunking failed for %s: %s", document.get("url"), exc)
                continue

            if not chunks:
                stats.documents_skipped += 1
                chunk_counters.documents_skipped += 1
                continue

            # Chunks of an earlier, longer version of this page would survive
            # an upsert, so they are deleted before the new ones are written.
            if replace and not self.dry_run and doc_id:
                self.vectors.delete_document(doc_id)

            stats.documents_indexed += 1
            chunk_counters.documents += 1
            for chunk in chunks:
                stats.words_indexed += chunk.word_count
                stats.chunk_sizes.append(chunk.word_count)
                batch.append(chunk)

            if len(batch) >= self.batch_size:
                try:
                    self._flush(batch, stats)
                except (EmbeddingUnavailable, VectorStoreUnavailable) as exc:
                    # These are infrastructure failures, not data failures:
                    # retrying the next batch would fail the same way.
                    stats.wall_ms = (time.perf_counter() - started) * 1000.0
                    stats.errors.append(str(exc))
                    raise
                batch = []

        if batch:
            self._flush(batch, stats)

        stats.wall_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Indexed %s document(s) as %s chunk(s) in %.2f s (%.1f chunks/s)",
            stats.documents_indexed, stats.chunks_written,
            stats.wall_ms / 1000.0, stats.chunks_per_second,
        )
        return stats


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Embed the processed documents and index them in Chroma."
    )
    parser.add_argument("--site", default=None, help="Only index one site.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum documents.")
    parser.add_argument(
        "--min-words", type=int, default=None,
        help="Skip documents with fewer clean words than this.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=settings.rag_index_batch_size,
        help="Chunks embedded and written per batch.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete the collection first, rebuilding the index from scratch.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chunk and embed, but write nothing (useful for timing).",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Print the run statistics as JSON.",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Print one indexed chunk to show what is stored.",
    )
    args = parser.parse_args(argv)

    setup_logging("rag-indexer")

    vectors = VectorStore()
    try:
        if args.reset and not args.dry_run:
            vectors.drop()
            print(f"Collection {vectors.collection_name!r} dropped.")
    except VectorStoreUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    indexer = Indexer(vectors=vectors, batch_size=args.batch_size, dry_run=args.dry_run)

    try:
        stats = indexer.run(site=args.site, limit=args.limit, min_words=args.min_words)
    except (StoreUnavailable, VectorStoreUnavailable, EmbeddingUnavailable) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(stats.as_dict(), indent=2))
    else:
        stats.print_report()

    if args.sample and not args.dry_run:
        for row in vectors.peek(1):
            print("--- one indexed chunk ---")
            print(json.dumps(row, indent=2, ensure_ascii=False)[:1500])

    return 1 if stats.documents_failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
