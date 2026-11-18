"""Efficiency test for the data-processing module.

The requirement for this task is to *test processing efficiency by running
different data samples*. "Different samples" is interpreted along the three
axes that actually change the cost of processing a page:

1. **The source website.** ``books.toscrape.com`` serves large, table-heavy
   product pages; ``quotes.toscrape.com`` serves small, text-heavy listing
   pages. Comparing them separates the cost of the markup from the cost of the
   text.
2. **The document size.** The same code is run over small (< 10 KB), medium
   (10-40 KB) and large (>= 40 KB) pages, which shows how the per-page cost
   scales with the payload and whether any stage is worse than linear.
3. **The write batch size.** The store stage is measured at batch sizes 1, 10,
   50 and 200 over the same documents. This isolates the network round trip
   from the CPU work and is what justifies the batching in the pipeline.

Each stage is timed separately, because they are bounded by different
resources: cleaning and normalisation are CPU-bound and local, while storing
is dominated by the round trip to MongoDB. Reporting a single end-to-end
number would hide that.

    python -m src.processing.benchmark                     # full suite
    python -m src.processing.benchmark --repeat 3           # average 3 runs
    python -m src.processing.benchmark --no-store           # CPU stages only
    python -m src.processing.benchmark --json out.json      # machine-readable

The rows are loaded into memory **once, before timing starts**, so the
measurements do not include PostgreSQL read time - the subject here is the
processing, not the database.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.common.config import settings
from src.common.logging_config import quiet_noisy_loggers
from src.processing.cleaner import clean_html
from src.processing.normalizer import normalize_page, validate_document
from src.processing.pipeline import ProcessingPipeline, _percentile
from src.processing.store import DocumentStore, StoreUnavailable

logger = logging.getLogger(__name__)

# Size buckets, in bytes of raw HTML.
SIZE_BUCKETS: list[tuple[str, int, int | None]] = [
    ("small  (<10 KB)", 0, 10_240),
    ("medium (10-40 KB)", 10_240, 40_960),
    ("large  (>=40 KB)", 40_960, None),
]

BATCH_SIZES = (1, 10, 50, 200)

# A separate collection, so a benchmark run never disturbs the real corpus.
BENCHMARK_COLLECTION = "benchmark_documents"


@dataclass
class SampleResult:
    """Timings for one sample of pages."""

    name: str
    pages: int = 0
    raw_bytes: int = 0
    clean_bytes: int = 0
    words: int = 0
    clean_ms: list[float] = field(default_factory=list)
    normalize_ms: list[float] = field(default_factory=list)
    store_ms_total: float = 0.0
    batch_size: int | None = None
    runs: int = 1

    @property
    def avg_page_kb(self) -> float:
        return round(self.raw_bytes / self.pages / 1024, 1) if self.pages else 0.0

    @property
    def cpu_ms_total(self) -> float:
        """CPU time for **one** pass over the sample.

        ``--repeat`` times the CPU stages several times over the same pages to
        reduce measurement noise, so the accumulated timings are divided by
        the number of repetitions: the throughput figures below must describe
        one pass, not the whole measurement.
        """
        return (sum(self.clean_ms) + sum(self.normalize_ms)) / max(1, self.runs)

    @property
    def wall_ms(self) -> float:
        return self.cpu_ms_total + self.store_ms_total

    @property
    def pages_per_second(self) -> float:
        return round(self.pages / (self.wall_ms / 1000.0), 1) if self.wall_ms else 0.0

    @property
    def mb_per_second(self) -> float:
        if not self.wall_ms:
            return 0.0
        return round((self.raw_bytes / 1_048_576) / (self.wall_ms / 1000.0), 2)

    @property
    def reduction_ratio(self) -> float:
        if not self.raw_bytes:
            return 0.0
        return round(1.0 - (self.clean_bytes / self.raw_bytes), 4)

    @property
    def store_ms_per_page(self) -> float:
        return round(self.store_ms_total / self.pages, 3) if self.pages else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample": self.name,
            "pages": self.pages,
            "runs": self.runs,
            "batch_size": self.batch_size,
            "avg_page_kb": self.avg_page_kb,
            "raw_bytes": self.raw_bytes,
            "clean_bytes": self.clean_bytes,
            "reduction_ratio": self.reduction_ratio,
            "words": self.words,
            "clean_ms": {
                "mean": round(statistics.fmean(self.clean_ms), 3) if self.clean_ms else 0.0,
                "p50": _percentile(self.clean_ms, 0.50),
                "p95": _percentile(self.clean_ms, 0.95),
                "stdev": (
                    round(statistics.pstdev(self.clean_ms), 3)
                    if len(self.clean_ms) > 1 else 0.0
                ),
            },
            "normalize_ms": {
                "mean": (
                    round(statistics.fmean(self.normalize_ms), 3)
                    if self.normalize_ms else 0.0
                ),
                "p50": _percentile(self.normalize_ms, 0.50),
                "p95": _percentile(self.normalize_ms, 0.95),
            },
            "store_ms_total": round(self.store_ms_total, 1),
            "store_ms_per_page": self.store_ms_per_page,
            "pages_per_second": self.pages_per_second,
            "mb_per_second": self.mb_per_second,
            # CPU milliseconds spent per megabyte of raw HTML: the figure
            # that makes samples of different page sizes comparable.
            "ms_per_mb": (
                round(self.cpu_ms_total / (self.raw_bytes / 1_048_576), 1)
                if self.raw_bytes else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Measuring one sample
# ---------------------------------------------------------------------------


def measure_cpu_stages(
    name: str,
    rows: Sequence[dict[str, Any]],
    *,
    repeat: int = 1,
) -> tuple[SampleResult, list[dict[str, Any]]]:
    """Time the clean and normalise stages over a set of already-loaded rows.

    The documents from the last repetition are returned as well, so the store
    stage can be measured over exactly the same payload.
    """
    result = SampleResult(name=name, runs=repeat)
    documents: list[dict[str, Any]] = []

    for iteration in range(repeat):
        documents = []
        for row in rows:
            started = time.perf_counter()
            cleaned = clean_html(row.get("raw_html") or "")
            clean_ms = (time.perf_counter() - started) * 1000.0

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
                    "fetched_at": row.get("fetched_at"),
                    "fetched_by": row.get("fetched_by"),
                },
                processed_by="benchmark",
            )
            validate_document(document)
            normalize_ms = (time.perf_counter() - started) * 1000.0

            result.clean_ms.append(clean_ms)
            result.normalize_ms.append(normalize_ms)
            documents.append(document)

            # Byte counters only once, otherwise repeats would multiply them.
            if iteration == 0:
                result.pages += 1
                result.raw_bytes += len(row.get("raw_html") or "")
                result.clean_bytes += document["stats"]["char_count"]
                result.words += document["stats"]["word_count"]

    return result, documents


def measure_store(
    store: DocumentStore,
    documents: Sequence[dict[str, Any]],
    *,
    batch_size: int,
) -> float:
    """Write the documents in batches of ``batch_size``; return the total ms."""
    if not documents:
        return 0.0
    started = time.perf_counter()
    for offset in range(0, len(documents), batch_size):
        store.upsert_many(documents[offset : offset + batch_size])
    return (time.perf_counter() - started) * 1000.0


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


def load_sample(
    *,
    site: str | None = None,
    limit: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
) -> list[dict[str, Any]]:
    """Read a sample of raw pages into memory before any timing starts."""
    return list(
        ProcessingPipeline.iter_raw_pages(
            site=site,
            limit=limit,
            min_length=min_length,
            max_length=max_length,
            batch_size=50,
        )
    )


def available_sites() -> list[str]:
    from sqlalchemy import func, select

    from src.storage.database import session_scope
    from src.storage.models import RawPage

    with session_scope() as session:
        rows = session.execute(
            select(RawPage.site, func.count(RawPage.id))
            .group_by(RawPage.site)
            .order_by(func.count(RawPage.id).desc())
        ).all()
    return [row[0] for row in rows]


def run_suite(
    *,
    limit_per_sample: int = 200,
    repeat: int = 1,
    use_store: bool = True,
    batch_sizes: Sequence[int] = BATCH_SIZES,
) -> dict[str, Any]:
    """Run every sample and return the full result set."""
    store: DocumentStore | None = None
    if use_store:
        store = DocumentStore(collection=BENCHMARK_COLLECTION)
        try:
            store.require()
            store.drop()
            store.ensure_indexes()
        except StoreUnavailable as exc:
            logger.warning("MongoDB unavailable, measuring CPU stages only: %s", exc)
            store = None

    results: list[SampleResult] = []
    everything = load_sample(limit=limit_per_sample)
    if not everything:
        raise RuntimeError(
            "no raw pages found in PostgreSQL - run a crawl first "
            "(python scripts/demo_task2.py)"
        )

    # --- axis 1: per source website ---
    for site in available_sites():
        rows = [r for r in everything if r["site"] == site]
        if not rows:
            continue
        result, documents = measure_cpu_stages(f"site: {site}", rows, repeat=repeat)
        if store is not None:
            result.batch_size = settings.processing_batch_size
            result.store_ms_total = measure_store(
                store, documents, batch_size=settings.processing_batch_size
            )
        results.append(result)

    # --- axis 2: per document size ---
    for label, low, high in SIZE_BUCKETS:
        rows = [
            r
            for r in everything
            if len(r.get("raw_html") or "") >= low
            and (high is None or len(r.get("raw_html") or "") < high)
        ]
        if not rows:
            continue
        result, documents = measure_cpu_stages(f"size: {label}", rows, repeat=repeat)
        if store is not None:
            result.batch_size = settings.processing_batch_size
            result.store_ms_total = measure_store(
                store, documents, batch_size=settings.processing_batch_size
            )
        results.append(result)

    # --- axis 3: write batch size, over the whole corpus ---
    batch_results: list[SampleResult] = []
    if store is not None:
        _cpu, documents = measure_cpu_stages("all pages", everything, repeat=1)
        for batch_size in batch_sizes:
            store.drop()
            store.ensure_indexes()
            elapsed = measure_store(store, documents, batch_size=batch_size)
            entry = SampleResult(
                name=f"batch size {batch_size}",
                pages=len(documents),
                batch_size=batch_size,
                store_ms_total=elapsed,
            )
            entry.raw_bytes = sum(d["source"]["raw_length"] for d in documents)
            entry.clean_bytes = sum(d["stats"]["char_count"] for d in documents)
            entry.words = sum(d["stats"]["word_count"] for d in documents)
            batch_results.append(entry)

    # --- the whole corpus, end to end ---
    overall, documents = measure_cpu_stages("ALL pages", everything, repeat=repeat)
    if store is not None:
        store.drop()
        store.ensure_indexes()
        overall.batch_size = settings.processing_batch_size
        overall.store_ms_total = measure_store(
            store, documents, batch_size=settings.processing_batch_size
        )

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "mongodb": (store.server_version() if store is not None else None),
            "store_measured": store is not None,
            "repeat": repeat,
        },
        "samples": [r.as_dict() for r in results],
        "batch_sizes": [r.as_dict() for r in batch_results],
        "overall": overall.as_dict(),
    }

    if store is not None:
        # Leave nothing behind: the benchmark collection is scratch space.
        store.drop()

    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(report: dict[str, Any]) -> None:
    env = report["environment"]
    print()
    print("=" * 94)
    print(" DATA-PROCESSING EFFICIENCY BENCHMARK")
    print("=" * 94)
    print(f" python {env['python']} on {env['platform']}")
    print(f" MongoDB: {env['mongodb'] or 'not measured'}    "
          f"repetitions per sample: {env['repeat']}")

    header = (
        f"\n {'sample':<26} {'pages':>6} {'avg KB':>7} {'clean p50':>10} "
        f"{'norm p50':>9} {'store/pg':>9} {'pages/s':>8} {'MB/s':>6} {'noise':>7}"
    )
    print(header)
    print(" " + "-" * 92)
    for sample in report["samples"]:
        print(
            f" {sample['sample']:<26} {sample['pages']:>6} {sample['avg_page_kb']:>7} "
            f"{sample['clean_ms']['p50']:>10.2f} {sample['normalize_ms']['p50']:>9.2f} "
            f"{sample['store_ms_per_page']:>9.2f} {sample['pages_per_second']:>8} "
            f"{sample['mb_per_second']:>6} "
            f"{sample['reduction_ratio'] * 100:>6.1f}%"
        )

    overall = report["overall"]
    print(" " + "-" * 92)
    print(
        f" {overall['sample']:<26} {overall['pages']:>6} {overall['avg_page_kb']:>7} "
        f"{overall['clean_ms']['p50']:>10.2f} {overall['normalize_ms']['p50']:>9.2f} "
        f"{overall['store_ms_per_page']:>9.2f} {overall['pages_per_second']:>8} "
        f"{overall['mb_per_second']:>6} "
        f"{overall['reduction_ratio'] * 100:>6.1f}%"
    )

    if report["batch_sizes"]:
        print(f"\n WRITE BATCH SIZE (same {report['batch_sizes'][0]['pages']} documents, "
              f"MongoDB round trips)")
        print(f" {'batch size':<26} {'round trips':>12} {'total ms':>10} "
              f"{'ms/document':>12} {'docs/s':>9}")
        print(" " + "-" * 73)
        for entry in report["batch_sizes"]:
            size = entry["batch_size"] or 1
            trips = -(-entry["pages"] // size)          # ceiling division
            per_doc = entry["store_ms_per_page"]
            docs_per_s = round(entry["pages"] / (entry["store_ms_total"] / 1000), 1) \
                if entry["store_ms_total"] else 0.0
            print(f" {str(size):<26} {trips:>12} {entry['store_ms_total']:>10.1f} "
                  f"{per_doc:>12.3f} {docs_per_s:>9}")

    cpu = overall["clean_ms"]["mean"] + overall["normalize_ms"]["mean"]
    print(f"\n Per page: {overall['clean_ms']['mean']:.2f} ms cleaning + "
          f"{overall['normalize_ms']['mean']:.2f} ms normalising = {cpu:.2f} ms CPU"
          f" (+ {overall['store_ms_per_page']:.2f} ms to store)")
    print(f" CPU cost per MB of raw HTML: {overall['ms_per_mb']:.0f} ms")
    print(f" Noise removed from the corpus: "
          f"{overall['reduction_ratio'] * 100:.1f}% of {overall['raw_bytes']:,} bytes")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure data-processing efficiency over different samples."
    )
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Maximum pages loaded from PostgreSQL for the suite.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Repetitions of the CPU stages per sample (reduces noise).",
    )
    parser.add_argument(
        "--no-store", action="store_true",
        help="Skip the MongoDB stage and measure cleaning/normalising only.",
    )
    parser.add_argument(
        "--json", nargs="?", const="-", default=None, metavar="PATH",
        help="Write the report as JSON to PATH (or to stdout when omitted).",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Also save the JSON report under data/benchmarks/.",
    )
    args = parser.parse_args(argv)

    quiet_noisy_loggers()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    try:
        report = run_suite(
            limit_per_sample=args.limit,
            repeat=max(1, args.repeat),
            use_store=not args.no_store,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json == "-":
        print(json.dumps(report, indent=2))
    elif args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.json}")
    else:
        print_report(report)

    if args.save:
        directory = settings.data_dir / "benchmarks"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"processing-benchmark-{stamp}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
