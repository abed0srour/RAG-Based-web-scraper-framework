"""Build the Elasticsearch index from the processed documents (Task 6).

    python -m src.search.indexer                # index everything
    python -m src.search.indexer --rebuild      # drop the index first
    python -m src.search.indexer --site books.toscrape.com --limit 50
    python -m src.search.indexer --json

MongoDB stays the system of record: this is a *projection* of it, built by
streaming the collection through a server-side cursor and bulk-writing
batches, so the corpus never has to fit in memory and the index can be
rebuilt from scratch at any time without touching the source data.

``--rebuild`` exists because an analyzer cannot be changed on a live index:
when the mapping in :mod:`src.search.index` changes, the index has to be
dropped and refilled, and that is a one-command operation rather than a
migration.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from src.common.config import settings
from src.common.logging_config import setup_logging
from src.processing.store import DocumentStore, StoreUnavailable
from src.search.index import SearchIndex, SearchUnavailable

logger = logging.getLogger(__name__)

# Only the fields the index actually needs are pulled out of MongoDB. The
# page text is the big one and is needed; everything not listed here (the
# chunk bookkeeping, the content hash, the cleaner's per-stage timings) is
# left in the document store.
_PROJECTION = {
    "_id": 0,
    "doc_id": 1,
    "url": 1,
    "site": 1,
    "path": 1,
    "title": 1,
    "description": 1,
    "language": 1,
    "headings": 1,
    "text": 1,
    "keywords": 1,
    "fields": 1,
    "stats": 1,
    "source": 1,
    "processing": 1,
    "enrichment": 1,
}


@dataclass
class IndexStats:
    """What one indexing run did, and how fast."""

    documents_read: int = 0
    documents_indexed: int = 0
    documents_failed: int = 0
    batches: int = 0
    index_created: bool = False
    rebuilt: bool = False
    read_ms: float = 0.0
    write_ms: float = 0.0
    total_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def documents_per_second(self) -> float:
        seconds = self.total_ms / 1000.0
        return round(self.documents_indexed / seconds, 1) if seconds > 0 else 0.0

    @property
    def ms_per_document(self) -> float:
        return (
            round(self.total_ms / self.documents_indexed, 2)
            if self.documents_indexed
            else 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents_read": self.documents_read,
            "documents_indexed": self.documents_indexed,
            "documents_failed": self.documents_failed,
            "batches": self.batches,
            "index_created": self.index_created,
            "rebuilt": self.rebuilt,
            "timings_ms": {
                "mongo_read": round(self.read_ms, 2),
                "elastic_write": round(self.write_ms, 2),
                "total": round(self.total_ms, 2),
            },
            "throughput": {
                "documents_per_second": self.documents_per_second,
                "ms_per_document": self.ms_per_document,
            },
            "errors": self.errors[:10],
        }

    def print_report(self) -> None:
        print("=" * 68)
        print("  Elasticsearch indexing run")
        print("=" * 68)
        print(f"  documents read       {self.documents_read}")
        print(f"  documents indexed    {self.documents_indexed}")
        if self.documents_failed:
            print(f"  documents failed     {self.documents_failed}")
        print(f"  bulk batches         {self.batches}")
        if self.rebuilt:
            print("  index               rebuilt from scratch")
        elif self.index_created:
            print("  index               created")
        print("-" * 68)
        print(f"  MongoDB read         {self.read_ms:9.1f} ms")
        print(f"  Elasticsearch write  {self.write_ms:9.1f} ms")
        print(f"  total                {self.total_ms:9.1f} ms")
        print("-" * 68)
        print(f"  throughput           {self.documents_per_second} docs/s "
              f"({self.ms_per_document} ms/doc)")
        if self.errors:
            print("  errors:")
            for message in self.errors[:5]:
                print(f"    - {message}")
        print("=" * 68)


class SearchIndexer:
    """Streams processed documents from MongoDB into Elasticsearch."""

    def __init__(
        self,
        *,
        store: DocumentStore | None = None,
        search: SearchIndex | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.store = store or DocumentStore()
        self.search = search or SearchIndex()
        self.batch_size = batch_size or settings.elastic_index_batch_size

    def iter_documents(
        self,
        *,
        site: str | None = None,
        limit: int | None = None,
        min_words: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream the documents to index out of MongoDB."""
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

    def run(
        self,
        *,
        site: str | None = None,
        limit: int | None = None,
        min_words: int | None = None,
        rebuild: bool = False,
    ) -> IndexStats:
        """Index every matching document, in batches."""
        stats = IndexStats(rebuilt=rebuild)
        started = time.perf_counter()

        if rebuild:
            self.search.delete_index()
        stats.index_created = self.search.ensure_index()

        batch: list[Mapping[str, Any]] = []
        read_started = time.perf_counter()

        for document in self.iter_documents(site=site, limit=limit, min_words=min_words):
            stats.documents_read += 1
            batch.append(document)
            if len(batch) >= self.batch_size:
                stats.read_ms += (time.perf_counter() - read_started) * 1000.0
                self._flush(batch, stats)
                batch = []
                read_started = time.perf_counter()

        stats.read_ms += (time.perf_counter() - read_started) * 1000.0
        if batch:
            self._flush(batch, stats)

        stats.total_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Indexed %d/%d document(s) into %s in %.0f ms",
            stats.documents_indexed,
            stats.documents_read,
            self.search.index_name,
            stats.total_ms,
        )
        return stats

    def _flush(self, batch: list[Mapping[str, Any]], stats: IndexStats) -> None:
        started = time.perf_counter()
        result = self.search.index_documents(batch, batch_size=self.batch_size)
        stats.write_ms += (time.perf_counter() - started) * 1000.0
        stats.batches += 1
        stats.documents_indexed += int(result.get("indexed") or 0)
        stats.documents_failed += int(result.get("failed") or 0)
        stats.errors.extend(result.get("errors") or [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index the processed MongoDB documents in Elasticsearch."
    )
    parser.add_argument("--site", default=None, help="Only index one site.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum documents.")
    parser.add_argument(
        "--min-words", type=int, default=None,
        help="Skip documents with fewer clean words than this.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=settings.elastic_index_batch_size,
        help="Documents per bulk request.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete the index first, so the mapping is reapplied.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Print the run statistics as JSON.",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Print one indexed document to show what is stored.",
    )
    args = parser.parse_args(argv)

    setup_logging("search-indexer")

    indexer = SearchIndexer(batch_size=args.batch_size)
    try:
        stats = indexer.run(
            site=args.site,
            limit=args.limit,
            min_words=args.min_words,
            rebuild=args.rebuild,
        )
    except (StoreUnavailable, SearchUnavailable) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(stats.as_dict(), indent=2))
    else:
        stats.print_report()

    if args.sample:
        hits = indexer.search.search("", limit=1, facets=False)["hits"]
        if hits:
            print("--- one indexed document ---")
            print(json.dumps(hits[0], indent=2, ensure_ascii=False)[:1500])

    return 1 if stats.documents_failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
