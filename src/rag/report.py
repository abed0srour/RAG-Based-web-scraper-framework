"""Command-line report over the RAG index and the enriched documents.

Where ``src/storage/report.py`` shows the crawl and ``src/processing/report.py``
shows the processed documents, this one shows what Task 4 added: the vector
index, the embedding model behind it, and the effect the augmentation stage
had on the stored documents.

    python -m src.rag.report
    python -m src.rag.report --query "books about history" --top-k 5
    python -m src.rag.report --compare "a quote about life"
    python -m src.rag.report --doc-url https://quotes.toscrape.com/
    python -m src.rag.report --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.common.config import settings
from src.common.logging_config import quiet_noisy_loggers
from src.processing.store import DocumentStore, StoreUnavailable
from src.rag.embedder import EmbeddingUnavailable, get_embedder
from src.rag.llm import get_llm
from src.rag.retriever import MODES, Retriever
from src.rag.vector_store import VectorStore, VectorStoreUnavailable


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def enrichment_stats(store: DocumentStore) -> dict[str, Any]:
    """Aggregate the enrichment state of the document collection.

    This is the data-quality number for the whole corpus rather than for one
    run: how many documents carry a summary, which provider produced it, and
    what the average checklist score is before and after.
    """
    pipeline = [
        {
            "$group": {
                "_id": "$enrichment.provider",
                "documents": {"$sum": 1},
                "with_summary": {
                    "$sum": {"$cond": [{"$ifNull": ["$enrichment.summary", False]}, 1, 0]}
                },
                "score_before": {"$avg": "$enrichment.quality.score_before"},
                "score_after": {"$avg": "$enrichment.quality.score_after"},
                "related": {"$avg": {"$size": {"$ifNull": ["$enrichment.related", []]}}},
            }
        },
        {"$sort": {"documents": -1}},
    ]
    rows = list(store.collection.aggregate(pipeline))

    total = store.count()
    enriched = sum(row["with_summary"] for row in rows)
    return {
        "documents": total,
        "enriched": enriched,
        "coverage": round(enriched / total, 4) if total else 0.0,
        "missing_description": store.collection.count_documents(
            {"$or": [{"description": None}, {"description": ""}]}
        ),
        "per_provider": [
            {
                "provider": row["_id"] or "(not enriched)",
                "documents": row["documents"],
                "with_summary": row["with_summary"],
                "score_before": round(row["score_before"], 4) if row.get("score_before") else None,
                "score_after": round(row["score_after"], 4) if row.get("score_after") else None,
                "avg_related": round(row["related"], 2) if row.get("related") else 0.0,
            }
            for row in rows
        ],
    }


def print_report(
    vectors: VectorStore,
    store: DocumentStore,
    *,
    samples: int = 1,
    query: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Print the full report and return the data behind it."""
    index = vectors.stats()
    embedder = get_embedder()
    llm = get_llm()

    _rule("VECTOR STORE")
    print(f"  Chroma version   : {vectors.server_version()}")
    print(f"  endpoint         : {index['endpoint']}")
    print(f"  collection       : {index['collection']}")
    print(f"  vectors (chunks) : {index['vectors']:,}")
    print(f"  documents        : {index['documents']:,}")
    print(f"  chunks/document  : {index['chunks_per_document']}")
    print(f"  avg words/chunk  : {index['avg_words_per_chunk']}")
    print(f"  distance metric  : {index['metadata'].get('hnsw:space', 'cosine')}")

    _rule("EMBEDDING MODEL")
    description = embedder.describe()
    print(f"  model            : {description['model']}")
    print(f"  dimensions       : {description['dimension']}")
    print(f"  runtime          : {description['runtime']}")
    print(f"  query prefix     : {'yes (asymmetric search)' if description['query_prefix'] else 'no'}")
    print(f"  chunking         : {settings.chunk_words} words, "
          f"{settings.chunk_overlap_words} overlap")
    print("\n  The model runs locally, so indexing and retrieval cost nothing"
          "\n  and work without network access once the model is cached.")

    _rule("LLM PROVIDER")
    for key, value in llm.describe().items():
        print(f"  {key:<17}: {value}")

    _rule("CHUNKS PER SITE")
    if not index["per_site"]:
        print("  (the index is empty - run 'python -m src.rag.indexer')")
    for row in index["per_site"]:
        print(f"  {row['site']:<32} {row['chunks']:>8,} chunks")

    quality: dict[str, Any] = {}
    try:
        quality = enrichment_stats(store)
        _rule("DATA QUALITY (AUGMENTATION)")
        print(f"  documents stored     : {quality['documents']:,}")
        print(f"  with a summary       : {quality['enriched']:,} "
              f"({quality['coverage'] * 100:.1f}%)")
        print(f"  still no description : {quality['missing_description']:,}")
        for row in quality["per_provider"]:
            print(f"  - {row['provider']:<28} {row['documents']:>6,} docs, "
                  f"quality {row['score_before']} -> {row['score_after']}, "
                  f"{row['avg_related']} related links")
    except StoreUnavailable as exc:
        print(f"  (MongoDB unavailable: {exc})")

    if samples:
        _rule("SAMPLE INDEXED CHUNKS")
        for row in vectors.peek(samples):
            text = " ".join(str(row.get("text") or "").split())
            print(f"  id      : {row.get('chunk_id')}")
            print(f"  page    : {row.get('title')}  <{row.get('url')}>")
            print(f"  words   : {row.get('word_count')}")
            print(f"  text    : \"{text[:240]}{'…' if len(text) > 240 else ''}\"")
            print()

    if query:
        retriever = Retriever(vectors=vectors, store=store, embedder=embedder)
        for mode in MODES:
            result = retriever.retrieve(query, mode=mode, top_k=top_k or settings.rag_top_k)
            result.print_report(snippet=140)

    return {"index": index, "model": description, "quality": quality}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report on the vector index and the enriched documents."
    )
    parser.add_argument(
        "--samples", type=int, default=1, help="Indexed chunks printed as samples."
    )
    parser.add_argument(
        "--query", default=None,
        help="Run this query in every retrieval mode and show the rankings.",
    )
    parser.add_argument(
        "--compare", default=None,
        help="Alias of --query: compares vector, keyword and hybrid retrieval.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Passages per mode.")
    parser.add_argument(
        "--doc-url", default=None,
        help="Print the enrichment stored for one URL.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit the statistics as JSON.",
    )
    args = parser.parse_args(argv)

    quiet_noisy_loggers()
    vectors = VectorStore()
    store = DocumentStore()

    try:
        vectors.require()
    except VectorStoreUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.doc_url:
        try:
            document = store.get_by_url(args.doc_url)
        except StoreUnavailable as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if document is None:
            print(f"No document stored for {args.doc_url}", file=sys.stderr)
            return 1
        print(json.dumps(document.get("enrichment") or {}, indent=2, ensure_ascii=False))
        return 0

    try:
        data = print_report(
            vectors,
            store,
            samples=args.samples,
            query=args.query or args.compare,
            top_k=args.top_k,
        )
    except (VectorStoreUnavailable, EmbeddingUnavailable) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(data, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
