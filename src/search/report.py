"""Command-line report over the Elasticsearch index (Task 6).

Where ``src/rag/report.py`` shows the vector index, this one shows the
full-text index that the web UI searches: the cluster, the mapping actually
in force, how much of the document store has been projected into it, and
what a query returns.

    python -m src.search.report
    python -m src.search.report --query "a light in the attic"
    python -m src.search.report --suggest lig
    python -m src.search.report --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.common.config import settings
from src.common.logging_config import quiet_noisy_loggers
from src.processing.store import DocumentStore, StoreUnavailable
from src.search.index import SearchIndex, SearchUnavailable


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def coverage(search: SearchIndex, store: DocumentStore) -> dict[str, Any]:
    """How much of MongoDB has made it into the index.

    The two counts should match. A gap means the indexer has not run since
    the last crawl, which is exactly the question this report exists to
    answer - the UI searching a stale index is otherwise invisible.
    """
    indexed = search.count()
    try:
        stored = store.count()
    except StoreUnavailable:
        stored = None
    return {
        "indexed": indexed,
        "stored": stored,
        "missing": (stored - indexed) if stored is not None else None,
        "complete": stored is not None and stored == indexed,
    }


def print_report(
    search: SearchIndex,
    store: DocumentStore,
    *,
    query: str | None = None,
    suggest: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Print the whole report; returns the same data for ``--json``."""
    data: dict[str, Any] = {}

    _rule("Cluster")
    health = search.health()
    data["cluster"] = {"version": search.server_version(), **health}
    print(f"  version              {data['cluster']['version']}")
    print(f"  cluster status       {health['status']}")
    print(f"  index                {health['index']}")
    print(f"  index exists         {'yes' if health['index_exists'] else 'no'}")
    print(f"  documents indexed    {health['documents']}")

    _rule("Coverage")
    data["coverage"] = coverage(search, store)
    print(f"  documents in MongoDB     {data['coverage']['stored']}")
    print(f"  documents in the index   {data['coverage']['indexed']}")
    if data["coverage"]["complete"]:
        print("  the index is up to date with the document store")
    elif data["coverage"]["stored"] is not None:
        print(f"  {data['coverage']['missing']} document(s) not indexed - "
              "run: python -m src.search.indexer")

    _rule("Indexed sites")
    data["sites"] = search.sites()
    for row in data["sites"]:
        print(f"  {str(row['value']):<40} {row['count']:>6} documents")
    if not data["sites"]:
        print("  (nothing indexed yet)")

    if query:
        _rule(f"Search: {query!r}")
        result = search.search(query, limit=limit)
        data["search"] = result
        print(f"  {result['total']} match(es), "
              f"cluster took {result['took_ms']} ms "
              f"(round trip {result['elapsed_ms']} ms)")
        for hit in result["hits"]:
            print()
            print(f"  [{hit['rank']}] {hit.get('title') or '(untitled)'}")
            print(f"       {hit.get('url')}")
            print(f"       score {hit['score']} (relevance {hit.get('relevance')})")
            snippet = (hit.get("snippet") or "").replace("\n", " ")
            if snippet:
                print(f"       {snippet[:220]}")
        if result["facets"].get("sites"):
            print()
            print("  facets by site:")
            for row in result["facets"]["sites"]:
                print(f"    {str(row['value']):<38} {row['count']:>6}")

    if suggest:
        _rule(f"Suggestions for {suggest!r}")
        data["suggestions"] = search.suggest(suggest)
        for row in data["suggestions"]:
            print(f"  {row['text']}  ({row['site']})")
        if not data["suggestions"]:
            print("  (no suggestion matched)")

    print()
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report on the Elasticsearch full-text index."
    )
    parser.add_argument("--query", default=None, help="Run this search and show the hits.")
    parser.add_argument("--suggest", default=None, help="Show as-you-type suggestions.")
    parser.add_argument("--limit", type=int, default=5, help="Hits to show.")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Print the data as JSON."
    )
    args = parser.parse_args(argv)

    quiet_noisy_loggers()
    search = SearchIndex()
    store = DocumentStore()

    try:
        data = print_report(
            search, store, query=args.query, suggest=args.suggest, limit=args.limit
        )
    except SearchUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Start the cluster with: docker compose -f docker/docker-compose.yml "
            "up -d elasticsearch",
            file=sys.stderr,
        )
        return 2

    if args.as_json:
        print(json.dumps(data, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
