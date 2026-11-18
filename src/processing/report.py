"""Command-line report over the processed documents in MongoDB.

The PostgreSQL counterpart of this script is ``src/storage/report.py``; this
one shows what the *processing* stage produced, which is the quickest proof
that the cleaning, the normalisation and the indexing all worked:

    python -m src.processing.report
    python -m src.processing.report --search "fiction"
    python -m src.processing.report --site quotes.toscrape.com --samples 2
    python -m src.processing.report --doc-url https://quotes.toscrape.com/
    python -m src.processing.report --json
"""

from __future__ import annotations

import argparse
import json
import sys

from src.common.logging_config import quiet_noisy_loggers
from src.processing.normalizer import to_json
from src.processing.store import DocumentStore, StoreUnavailable


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def print_report(
    store: DocumentStore,
    *,
    site: str | None = None,
    samples: int = 1,
    search: str | None = None,
) -> None:
    stats = store.stats()

    _rule("DOCUMENT STORE")
    print(f"  MongoDB version : {store.server_version()}")
    print(f"  database        : {stats['database']}")
    print(f"  collection      : {stats['collection']}")
    print(f"  documents       : {stats['totals']['documents']}")

    _rule("INDEXES")
    for index in store.indexes():
        keys = ", ".join(
            f"{name}:{direction}" for name, direction in index["key"].items()
        )
        unique = "  UNIQUE" if index.get("unique") else ""
        print(f"  {index['name']:<22} {keys}{unique}")
    print(
        "\n  The text index is what makes keyword retrieval a database "
        "operation\n  instead of a scan in Python."
    )

    _rule("DOCUMENTS PER SITE")
    if not stats["sites"]:
        print("(no documents stored yet - run python -m src.processing.pipeline)")
    else:
        print(f"  {'site':<26} {'docs':>5} {'words':>9} {'raw bytes':>11} "
              f"{'avg words':>10} {'noise removed':>14}")
        print("  " + "-" * 80)
        for row in stats["sites"]:
            if site and row["site"] != site:
                continue
            print(f"  {row['site']:<26} {row['documents']:>5} {row['words']:>9,} "
                  f"{row['raw_bytes']:>11,} {row['avg_words']:>10} "
                  f"{row['avg_reduction_ratio'] * 100:>13.1f}%")

    duplicates = store.duplicates(limit=5)
    _rule("NEAR-DUPLICATE DETECTION (identical clean text)")
    if duplicates:
        for group in duplicates:
            print(f"  {group['count']} documents share one content hash:")
            for url in group["urls"][:4]:
                print(f"      {url}")
    else:
        print("  (every stored document has distinct text)")

    if search:
        _rule(f"FULL-TEXT SEARCH: {search!r}")
        hits = store.search(
            search,
            site=site,
            limit=10,
            projection=("doc_id", "url", "site", "title", "stats"),
        )
        if not hits:
            print("  (no matches)")
        for hit in hits:
            print(f"  score {hit.get('score', 0):>6.2f}  {(hit.get('title') or '')[:58]}")
            print(f"                {hit['url']}")

    if samples > 0:
        _rule(f"SAMPLE DOCUMENTS (up to {samples} per site)")
        for row in stats["sites"]:
            if site and row["site"] != site:
                continue
            print(f"\n--- {row['site']} ---")
            cursor = store.collection.find(
                {"site": row["site"]}, {"_id": 0}
            ).limit(samples)
            for document in cursor:
                text = document.get("text") or ""
                print(f"  doc_id    : {document['doc_id']}")
                print(f"  url       : {document['url']}")
                print(f"  title     : {(document.get('title') or '')[:64]}")
                print(f"  language  : {document.get('language')}")
                print(f"  stats     : {json.dumps(document.get('stats', {}))}")
                print(f"  keywords  : {', '.join(document.get('keywords', [])[:10])}")
                print(f"  fields    : {', '.join(sorted(document.get('fields', {})))}")
                print(f"  noise     : "
                      f"{document.get('processing', {}).get('reduction_ratio')}")
                excerpt = text[:220].replace("\n", " / ")
                print(f"  text[:220]: {excerpt}")
                print()
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report on the processed documents stored in MongoDB."
    )
    parser.add_argument("--site", default=None, help="Limit the report to one site.")
    parser.add_argument(
        "--samples", type=int, default=1, help="Sample documents printed per site."
    )
    parser.add_argument(
        "--search", default=None, help="Run a full-text query against the index."
    )
    parser.add_argument(
        "--doc-url", default=None,
        help="Print the complete JSON document stored for one URL.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit the statistics as JSON.",
    )
    args = parser.parse_args(argv)

    quiet_noisy_loggers()
    store = DocumentStore()

    try:
        store.require()
    except StoreUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.doc_url:
        document = store.get_by_url(args.doc_url)
        if document is None:
            print(f"No document stored for {args.doc_url}", file=sys.stderr)
            return 1
        print(to_json(document))
        return 0

    if args.as_json:
        print(json.dumps(store.stats(), indent=2, default=str))
        return 0

    print_report(store, site=args.site, samples=args.samples, search=args.search)
    return 0


if __name__ == "__main__":
    sys.exit(main())
