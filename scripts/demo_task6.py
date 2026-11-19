"""Task 6 end-to-end demonstration.

One command that produces the evidence for the Task 6 section of the report:

    python scripts/demo_task6.py

What it does, in order:

1. reports the cluster and the index, and shows the mapping that is actually
   in force - the analyzer, the suggestion field and the disabled-indexing
   ``fields`` object
2. projects every stored MongoDB document into Elasticsearch, with the
   timings, then re-runs it to show the write path is idempotent
3. runs a set of searches and prints the ranked hits with their highlighted
   snippets, which is what the result cards in the UI display
4. demonstrates the three things only a search engine gives you: stemming
   (a different word form still matches), facet counts over the whole
   matching set, and as-you-type suggestions from the edge n-grams
5. compares a keyword search in MongoDB with the same search in
   Elasticsearch, so the reason for the fourth store is visible rather than
   asserted
6. prints the API calls the web UI makes, for the report's screenshots

``--rebuild`` drops the index first (needed after a mapping change), and
``--site`` limits the demonstration to one website.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Guard against being launched by the wrong interpreter. Set once a re-launch
# has happened, so a venv that genuinely lacks the dependencies reports the
# missing package instead of re-launching itself forever.
_REEXEC_FLAG = "DEMO_TASK6_REEXEC"


def project_python() -> Path | None:
    """Return the project's own virtual-environment interpreter, if present.

    Same reasoning as in the other demonstration scripts: ``sys.executable``
    cannot be trusted, because a ``python`` that resolves to the system
    interpreter - through a shell alias, or a PATH re-ordered after the
    virtual environment was activated, so the prompt still shows ``(venv)`` -
    runs this script without the project's dependencies.
    """
    candidates = [
        PROJECT_ROOT / "venv" / "Scripts" / "python.exe",   # Windows
        PROJECT_ROOT / "venv" / "bin" / "python",           # Linux / macOS
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def relaunch_in_project_python() -> None:
    """Re-run this script under the project interpreter when it differs."""
    target = project_python()
    if target is None or os.environ.get(_REEXEC_FLAG):
        return

    current = Path(sys.executable)
    try:
        if target.samefile(current):
            return
    except OSError:  # pragma: no cover - one of the paths vanished
        if str(target).lower() == str(current).lower():
            return

    print(f"This script was started by {current}")
    print(f"Re-launching with the project interpreter: {target}\n", flush=True)
    completed = subprocess.run(  # noqa: S603 - fixed, trusted argument list
        [str(target), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=PROJECT_ROOT,
        env={**os.environ, _REEXEC_FLAG: "1"},
        check=False,
    )
    raise SystemExit(completed.returncode)


relaunch_in_project_python()

try:
    from src.common.config import settings  # noqa: E402
    from src.common.logging_config import quiet_noisy_loggers  # noqa: E402
    from src.processing.store import DocumentStore, StoreUnavailable  # noqa: E402
    from src.search.index import (  # noqa: E402
        INDEX_MAPPINGS,
        SearchIndex,
        SearchUnavailable,
        to_index_document,
    )
    from src.search.indexer import SearchIndexer  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    print(f"ERROR: {exc}", file=sys.stderr)
    print(
        f"\nThe interpreter in use is {sys.executable}\n"
        "Install the project's dependencies into it with:\n"
        f'  "{project_python() or sys.executable}" -m pip install -r requirements.txt',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

QUERIES = [
    "light in the attic",
    "poetry for children",
    "thinking",
]


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(f" {text}")
    print("=" * 78)


def show_cluster(search: SearchIndex) -> None:
    banner("1. The cluster and the index")
    print(f"  Elasticsearch        {search.server_version()}")
    health = search.health()
    print(f"  cluster status       {health['status']}")
    print(f"  index                {health['index']}")
    print(f"  index exists         {'yes' if health['index_exists'] else 'no'}")
    print(f"  documents indexed    {health['documents']}")
    print()
    print("  The three parts of the mapping that do the work:")
    properties = INDEX_MAPPINGS["properties"]
    print(f"    text          analyzer={properties['text']['analyzer']!r}"
          "   (lowercase, stop words, English stemmer)")
    suggest = properties["title"]["fields"]["suggest"]
    print(f"    title.suggest analyzer={suggest['analyzer']!r} "
          f"search_analyzer={suggest['search_analyzer']!r}")
    print(f"    fields        {properties['fields']}"
          "   (stored, never indexed: one field, not thousands)")


def show_indexing(indexer: SearchIndexer, site: str | None, rebuild: bool) -> None:
    banner("2. Projecting MongoDB into Elasticsearch")
    stats = indexer.run(site=site, rebuild=rebuild)
    stats.print_report()

    print()
    print("  Running it again - the write path keys on doc_id, so the second")
    print("  run replaces rather than duplicates:")
    before = indexer.search.count()
    again = indexer.run(site=site)
    after = indexer.search.count()
    print(f"    documents before     {before}")
    print(f"    documents written    {again.documents_indexed}")
    print(f"    documents after      {after}")
    print(f"    duplicates created   {after - before}")


def show_one_projection(store: DocumentStore, site: str | None) -> None:
    banner("3. One document, as stored and as indexed")
    filters = {"site": site} if site else {}
    document = store.collection.find_one(filters, {"_id": 0})
    if document is None:
        print("  (no document in MongoDB to show)")
        return

    indexed = to_index_document(document)
    print(f"  MongoDB document : {len(json.dumps(document, default=str))} bytes, "
          f"{len(document)} top-level keys")
    print(f"  index document   : {len(json.dumps(indexed, default=str))} bytes, "
          f"{len(indexed)} flat fields")
    print()
    print("  flattened for search:")
    print(f"    fields       {json.dumps(document.get('fields') or {}, ensure_ascii=False)[:160]}")
    print(f"    fields_text  {indexed.get('fields_text', '')[:160]!r}")


def show_searches(search: SearchIndex, queries: list[str]) -> None:
    banner("4. Searches, with the snippets the UI shows")
    for query in queries:
        result = search.search(query, limit=3)
        print()
        print(f"  {query!r} -> {result['total']} match(es) in {result['took_ms']} ms")
        for hit in result["hits"]:
            print(f"    [{hit['rank']}] {hit.get('title') or '(untitled)'}  "
                  f"score {hit['score']} (relevance {hit.get('relevance')})")
            snippet = (hit.get("snippet") or "").replace("\n", " ")
            if snippet:
                print(f"         {snippet[:200]}")


def show_engine_features(search: SearchIndex) -> None:
    banner("5. What a search engine adds over a substring match")

    print("  Stemming - the query word form need not be the stored one:")
    for query in ("drawing", "draw", "draws"):
        result = search.search(query, limit=1, facets=False)
        print(f"    {query!r:<12} {result['total']} match(es)")

    print()
    print("  Facets, counted over every match rather than over the page shown:")
    result = search.search("")
    for name in ("sites", "languages", "enriched"):
        rows = result["facets"].get(name) or []
        formatted = ", ".join(f"{row['value']}={row['count']}" for row in rows[:6])
        print(f"    {name:<12} {formatted}")

    print()
    print("  As-you-type suggestions, from the edge n-grams built at index time:")
    for prefix in ("li", "lig", "ligh"):
        rows = search.suggest(prefix, limit=3)
        print(f"    {prefix!r:<8} -> {[row['text'] for row in rows]}")


def compare_with_mongo(search: SearchIndex, store: DocumentStore, query: str) -> None:
    banner("6. The same query in MongoDB and in Elasticsearch")
    print(f"  query: {query!r}")

    started = time.perf_counter()
    try:
        mongo_rows = store.search(query, limit=5)
    except Exception as exc:  # noqa: BLE001 - the comparison is the point
        mongo_rows = []
        print(f"  MongoDB text search failed: {exc}")
    mongo_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    result = search.search(query, limit=5)
    elastic_ms = (time.perf_counter() - started) * 1000.0

    print()
    print(f"  MongoDB       {len(mongo_rows)} document(s) in {mongo_ms:.1f} ms")
    print("                ranked by textScore, no snippet, no facet counts")
    for row in mongo_rows[:3]:
        print(f"                  - {row.get('title') or row.get('url')}")

    print()
    print(f"  Elasticsearch {result['total']} document(s) in {elastic_ms:.1f} ms")
    print("                ranked, with marked snippets and facet counts")
    for hit in result["hits"][:3]:
        print(f"                  - {hit.get('title') or hit.get('url')}")
    if result["hits"]:
        print(f"                  snippet: {(result['hits'][0].get('snippet') or '')[:120]}")


def show_ui_calls() -> None:
    banner("7. The calls the web UI makes")
    base = f"http://localhost:{settings.api_port}"
    for description, call in [
        ("search as you type", f"GET {base}/v1/fulltext?q=light&sort=relevance&limit=10"),
        ("narrow by facet", f"GET {base}/v1/fulltext?q=light&site=books.toscrape.com"),
        ("suggestions", f"GET {base}/v1/fulltext/suggest?q=lig"),
        ("site filter list", f"GET {base}/v1/fulltext/sites"),
        ("related pages", f"GET {base}/v1/fulltext/similar/<doc_id>"),
        ("open a result", f"GET {base}/v1/documents/<doc_id>"),
        ("the Ask tab", f"POST {base}/v1/ask"),
        ("the Raw data tab", f"GET {base}/v1/raw/pages"),
        ("the status bar", f"GET {base}/health/ready"),
    ]:
        print(f"  {description:<22} {call}")
    print()
    print("  Start them with:")
    print("    python -m src.api          (the API on :8080)")
    print("    cd ui && npm run dev       (the UI on :5173, proxying /v1)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Demonstrate the Task 6 full-text search layer."
    )
    parser.add_argument("--site", default=None, help="Limit the demo to one site.")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Drop the index first, so the mapping is reapplied.",
    )
    parser.add_argument(
        "--skip-index", action="store_true",
        help="Search the index as it is, without rewriting it.",
    )
    parser.add_argument(
        "--query", action="append", default=None,
        help="Run this query instead of the defaults (repeatable).",
    )
    args = parser.parse_args(argv)

    quiet_noisy_loggers()

    print("=" * 78)
    print(" TASK 6 DEMONSTRATION - full-text search and the web UI")
    print("=" * 78)
    print(f" interpreter : {sys.executable}")
    print(f" cluster     : {settings.elastic_scheme}://{settings.elastic_host}:"
          f"{settings.elastic_port}")
    print(f" index       : {settings.elastic_index}")

    search = SearchIndex()
    store = DocumentStore()

    try:
        show_cluster(search)

        if not args.skip_index:
            indexer = SearchIndexer(store=store, search=search)
            show_indexing(indexer, args.site, args.rebuild)

        show_one_projection(store, args.site)
        show_searches(search, args.query or QUERIES)
        show_engine_features(search)
        compare_with_mongo(search, store, (args.query or QUERIES)[0])
    except SearchUnavailable as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "Start the cluster with: docker compose -f docker/docker-compose.yml "
            "up -d elasticsearch",
            file=sys.stderr,
        )
        return 2
    except StoreUnavailable as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "Start MongoDB with: docker compose -f docker/docker-compose.yml "
            "up -d mongodb",
            file=sys.stderr,
        )
        return 2

    show_ui_calls()
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
