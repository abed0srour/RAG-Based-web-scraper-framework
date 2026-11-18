"""The data-processing pipeline: PostgreSQL -> clean -> normalise -> MongoDB.

This module is the orchestrator of Task 3. It reads the raw HTML that the
scraping module stored in PostgreSQL, runs it through the cleaner and the
normaliser, and writes the resulting JSON documents into MongoDB.

    python -m src.processing.pipeline                       # process everything
    python -m src.processing.pipeline --site quotes.toscrape.com
    python -m src.processing.pipeline --limit 100 --batch-size 25
    python -m src.processing.pipeline --reset               # rebuild from scratch
    python -m src.processing.pipeline --dry-run             # no database writes

Three properties of the implementation are worth pointing out.

**It is restartable.** The document key is derived from the URL, so processing
a page twice replaces its document instead of duplicating it. The pipeline can
therefore be killed half-way and simply run again - and ``--since-id`` lets it
continue from the last processed row when only new pages matter.

**It streams.** Rows are fetched in batches with a server-side cursor
(``yield_per``), so the memory it needs is proportional to the batch size, not
to the size of the corpus. A raw HTML page can be hundreds of kilobytes; a
naive ``SELECT *`` over a large crawl would not fit in memory.

**It measures itself.** Every stage is timed per page, and
:class:`ProcessingStats` reports throughput together with the p50/p95 latency
of each stage. Those are the numbers the efficiency test in
``src/processing/benchmark.py`` compares across data samples.
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

from sqlalchemy import select

from src.common.config import settings
from src.common.logging_config import setup_logging
from src.processing.cleaner import clean_html
from src.processing.normalizer import (
    NormalizationError,
    normalize_page,
    validate_document,
)
from src.processing.store import DocumentStore, StoreUnavailable
from src.storage.database import session_scope
from src.storage.models import ParsedPage, RawPage

logger = logging.getLogger(__name__)


def node_id() -> str:
    """Identify the process that produced a document (hostname:pid)."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Avoids a numpy dependency in the hot path."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 3)


@dataclass
class ProcessingStats:
    """Counters and timings for one pipeline run."""

    pages_read: int = 0
    documents_written: int = 0
    inserted: int = 0
    modified: int = 0
    failed: int = 0
    skipped_empty: int = 0
    raw_bytes: int = 0
    clean_bytes: int = 0
    total_words: int = 0
    clean_ms: list[float] = field(default_factory=list)
    normalize_ms: list[float] = field(default_factory=list)
    store_ms: list[float] = field(default_factory=list)
    wall_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    # ---- derived measures ----

    @property
    def pages_per_second(self) -> float:
        if self.wall_ms <= 0:
            return 0.0
        return round(self.pages_read / (self.wall_ms / 1000.0), 2)

    @property
    def mb_per_second(self) -> float:
        if self.wall_ms <= 0:
            return 0.0
        return round((self.raw_bytes / 1_048_576) / (self.wall_ms / 1000.0), 2)

    @property
    def reduction_ratio(self) -> float:
        """Share of the raw bytes removed as noise across the whole run."""
        if not self.raw_bytes:
            return 0.0
        return round(1.0 - (self.clean_bytes / self.raw_bytes), 4)

    def stage_timings(self) -> dict[str, dict[str, float]]:
        def describe(samples: list[float]) -> dict[str, float]:
            return {
                "mean": round(sum(samples) / len(samples), 3) if samples else 0.0,
                "p50": _percentile(samples, 0.50),
                "p95": _percentile(samples, 0.95),
                "max": round(max(samples), 3) if samples else 0.0,
                "total": round(sum(samples), 1),
            }

        return {
            "clean": describe(self.clean_ms),
            "normalize": describe(self.normalize_ms),
            "store": describe(self.store_ms),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "pages_read": self.pages_read,
            "documents_written": self.documents_written,
            "inserted": self.inserted,
            "modified": self.modified,
            "failed": self.failed,
            "skipped_empty": self.skipped_empty,
            "raw_bytes": self.raw_bytes,
            "clean_bytes": self.clean_bytes,
            "reduction_ratio": self.reduction_ratio,
            "total_words": self.total_words,
            "wall_ms": round(self.wall_ms, 1),
            "pages_per_second": self.pages_per_second,
            "mb_per_second": self.mb_per_second,
            "stage_ms": self.stage_timings(),
            "errors": self.errors[:10],
        }

    def print_report(self) -> None:
        d = self.as_dict()
        print()
        print("=" * 74)
        print(" PROCESSING RUN SUMMARY")
        print("=" * 74)
        print(f"  pages read          : {d['pages_read']}")
        print(f"  documents written   : {d['documents_written']} "
              f"({d['inserted']} new, {d['modified']} updated)")
        if d["skipped_empty"]:
            print(f"  skipped (no text)   : {d['skipped_empty']}")
        if d["failed"]:
            print(f"  failed              : {d['failed']}")
        print(f"  raw -> clean        : {d['raw_bytes']:,} B -> "
              f"{d['clean_bytes']:,} B "
              f"({d['reduction_ratio'] * 100:.1f}% removed)")
        print(f"  words extracted     : {d['total_words']:,}")
        print(f"  wall time           : {d['wall_ms'] / 1000:.2f} s")
        print(f"  throughput          : {d['pages_per_second']} pages/s, "
              f"{d['mb_per_second']} MB/s")
        print()
        print(f"  {'stage':<12} {'mean ms':>9} {'p50 ms':>9} {'p95 ms':>9} "
              f"{'max ms':>9} {'total ms':>10}")
        print("  " + "-" * 62)
        for stage, timings in d["stage_ms"].items():
            print(f"  {stage:<12} {timings['mean']:>9.2f} {timings['p50']:>9.2f} "
                  f"{timings['p95']:>9.2f} {timings['max']:>9.2f} "
                  f"{timings['total']:>10.1f}")
        for message in d["errors"]:
            print(f"  ! {message}")
        print()


class ProcessingPipeline:
    """Reads raw pages, produces documents, writes them to the document store."""

    def __init__(
        self,
        *,
        store: DocumentStore | None = None,
        batch_size: int | None = None,
        min_words: int | None = None,
        dry_run: bool = False,
    ) -> None:
        self.store = store or DocumentStore()
        self.batch_size = batch_size or settings.processing_batch_size
        self.min_words = (
            settings.processing_min_words if min_words is None else min_words
        )
        self.dry_run = dry_run
        self.node = node_id()

    # ---------------- reading ----------------

    @staticmethod
    def iter_raw_pages(
        *,
        site: str | None = None,
        limit: int | None = None,
        since_id: int | None = None,
        batch_size: int = 50,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream raw pages out of PostgreSQL, newest ``id`` last.

        The ``parsed_pages`` row is joined in so that the per-site structured
        fields extracted during the crawl (prices, authors, ratings) are
        carried into the document instead of being re-derived here. The
        length filters are what the benchmark uses to build samples of
        comparable page size.
        """
        statement = (
            select(
                RawPage.id,
                RawPage.job_id,
                RawPage.url,
                RawPage.site,
                RawPage.depth,
                RawPage.http_status,
                RawPage.content_type,
                RawPage.raw_html,
                RawPage.content_length,
                RawPage.fetched_at,
                RawPage.fetched_by,
                ParsedPage.title,
                ParsedPage.description,
                ParsedPage.language,
                ParsedPage.headings,
                ParsedPage.structured_data,
            )
            .join(ParsedPage, ParsedPage.raw_page_id == RawPage.id, isouter=True)
            .order_by(RawPage.id)
        )
        if site:
            statement = statement.where(RawPage.site == site)
        if since_id:
            statement = statement.where(RawPage.id > since_id)
        if min_length is not None:
            statement = statement.where(RawPage.content_length >= min_length)
        if max_length is not None:
            statement = statement.where(RawPage.content_length < max_length)
        if limit:
            statement = statement.limit(limit)

        with session_scope() as session:
            # yield_per keeps a server-side cursor open and buffers only one
            # batch of rows at a time.
            result = session.execute(statement).mappings().yield_per(batch_size)
            for row in result:
                yield dict(row)

    # ---------------- processing one page ----------------

    def process_page(self, row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, float, float]:
        """Clean and normalise one raw page.

        Returns ``(document, clean_ms, normalize_ms)``; the document is None
        when the page carried no usable text, which is reported as a skip
        rather than as an error - an empty page is a fact about the crawl, not
        a failure of the processing.
        """
        cleaned = clean_html(row.get("raw_html") or "")

        if cleaned.word_count < self.min_words:
            return None, cleaned.duration_ms, 0.0

        started = time.perf_counter()
        document = normalize_page(
            url=row["url"],
            cleaned=cleaned,
            site=row.get("site"),
            title=row.get("title"),
            description=row.get("description"),
            language=row.get("language"),
            headings=list(row.get("headings") or []),
            structured_data=row.get("structured_data") or {},
            source={
                "raw_page_id": row.get("id"),
                "job_id": row.get("job_id"),
                "depth": row.get("depth"),
                "http_status": row.get("http_status"),
                "content_type": row.get("content_type"),
                "fetched_at": row.get("fetched_at"),
                "fetched_by": row.get("fetched_by"),
            },
            processed_by=self.node,
        )
        validate_document(document)
        normalize_ms = (time.perf_counter() - started) * 1000.0
        return document, cleaned.duration_ms, normalize_ms

    # ---------------- the run ----------------

    def run(
        self,
        *,
        site: str | None = None,
        limit: int | None = None,
        since_id: int | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        ensure_indexes: bool = True,
    ) -> ProcessingStats:
        """Process pages and write the documents, in batches."""
        stats = ProcessingStats()

        if not self.dry_run:
            self.store.require()
            if ensure_indexes:
                self.store.ensure_indexes()

        started = time.perf_counter()
        batch: list[dict[str, Any]] = []

        for row in self.iter_raw_pages(
            site=site,
            limit=limit,
            since_id=since_id,
            batch_size=self.batch_size,
            min_length=min_length,
            max_length=max_length,
        ):
            stats.pages_read += 1
            stats.raw_bytes += row.get("content_length") or len(row.get("raw_html") or "")

            try:
                document, clean_ms, normalize_ms = self.process_page(row)
            except Exception as exc:  # noqa: BLE001
                # One unusable page must never stop a batch run; it is counted
                # and the URL is reported so it can be inspected afterwards.
                stats.failed += 1
                stats.errors.append(f"{row.get('url')}: {exc}")
                logger.warning("Processing failed for %s: %s", row.get("url"), exc)
                continue

            stats.clean_ms.append(clean_ms)
            if document is None:
                stats.skipped_empty += 1
                continue

            stats.normalize_ms.append(normalize_ms)
            stats.clean_bytes += document["stats"]["char_count"]
            stats.total_words += document["stats"]["word_count"]
            batch.append(document)

            if len(batch) >= self.batch_size:
                self._flush(batch, stats)
                batch = []

        if batch:
            self._flush(batch, stats)

        stats.wall_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Processed %s page(s) into %s document(s) in %.2f s (%.2f pages/s)",
            stats.pages_read, stats.documents_written,
            stats.wall_ms / 1000.0, stats.pages_per_second,
        )
        return stats

    def _flush(self, batch: list[dict[str, Any]], stats: ProcessingStats) -> None:
        """Write one batch of documents and record how long it took."""
        if self.dry_run:
            stats.documents_written += len(batch)
            return

        started = time.perf_counter()
        result = self.store.upsert_many(batch)
        elapsed = (time.perf_counter() - started) * 1000.0

        stats.store_ms.append(elapsed)
        stats.inserted += result["inserted"]
        stats.modified += result["modified"]
        stats.failed += result["failed"]
        stats.documents_written += len(batch) - result["failed"]
        logger.debug(
            "Flushed %s document(s) in %.1f ms (%s new, %s updated)",
            len(batch), elapsed, result["inserted"], result["modified"],
        )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean, normalise and index the scraped pages."
    )
    parser.add_argument("--site", default=None, help="Only process one site.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum pages.")
    parser.add_argument(
        "--since-id", type=int, default=None,
        help="Only pages with a raw_pages.id greater than this.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=settings.processing_batch_size,
        help="Documents written to MongoDB per round trip.",
    )
    parser.add_argument(
        "--min-words", type=int, default=settings.processing_min_words,
        help="Pages with fewer clean words than this are skipped.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Drop the collection first, rebuilding the index from scratch.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Clean and normalise, but write nothing (useful for timing).",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Print the run statistics as JSON.",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Print one finished document to show the JSON schema.",
    )
    args = parser.parse_args(argv)

    setup_logging("processing")

    store = DocumentStore()
    if args.reset and not args.dry_run:
        try:
            store.require()
            store.drop()
        except StoreUnavailable as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    pipeline = ProcessingPipeline(
        store=store,
        batch_size=args.batch_size,
        min_words=args.min_words,
        dry_run=args.dry_run,
    )

    try:
        stats = pipeline.run(
            site=args.site, limit=args.limit, since_id=args.since_id
        )
    except StoreUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(stats.as_dict(), indent=2))
    else:
        stats.print_report()

    if args.sample and not args.dry_run:
        first = store.collection.find_one({}, {"_id": 0, "text": 0})
        if first:
            print("--- one stored document (body text omitted) ---")
            print(json.dumps(first, indent=2, ensure_ascii=False, default=str))

    return 1 if stats.failed else 0


if __name__ == "__main__":
    sys.exit(main())
