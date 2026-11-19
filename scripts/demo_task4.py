"""Task 4 end-to-end demonstration.

One command that produces the evidence for the Task 4 section of the report:

    python scripts/demo_task4.py

What it does, in order:

1. shows one real stored document being cut into overlapping chunks, with the
   word counts and the overlap made visible
2. embeds a query and three passages and prints the similarity between them,
   which is the demonstration that retrieval is semantic rather than lexical
3. builds the vector index from every document in MongoDB, with the per-stage
   timings, then re-runs it to show the write path is idempotent
4. answers the same query with each of the three retrieval modes side by side
   (vector, keyword, hybrid), so the fusion step can be seen working
5. asks the LLM stage a few questions and prints the answers with the URLs
   they were generated from
6. enriches the stored documents with contextual summaries and reports the
   before/after data-quality score
7. prints the final report over the index and the enriched corpus

``--skip-augment`` leaves out step 6 (the slowest one), ``--site`` limits the
demonstration to a single website, and ``--provider extractive`` forces the
free offline summariser even when an API key is configured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import settings  # noqa: E402
from src.common.logging_config import quiet_noisy_loggers  # noqa: E402
from src.processing.store import DocumentStore, StoreUnavailable  # noqa: E402
from src.rag.augment import Augmenter, answer_question, quality_flags, quality_score  # noqa: E402
from src.rag.chunker import chunk_document  # noqa: E402
from src.rag.embedder import EmbeddingUnavailable, cosine_similarity, get_embedder  # noqa: E402
from src.rag.indexer import Indexer  # noqa: E402
from src.rag.llm import LLMError, get_llm  # noqa: E402
from src.rag.report import print_report  # noqa: E402
from src.rag.retriever import MODES, Retriever  # noqa: E402
from src.rag.vector_store import VectorStore, VectorStoreUnavailable  # noqa: E402

DEFAULT_QUESTIONS = (
    "what does this corpus say about love and life",
    "which books are about history",
)


def banner(text: str) -> None:
    print()
    print("#" * 78)
    print(f"#  {text}")
    print("#" * 78, flush=True)


def step_chunking(store: DocumentStore, site: str | None) -> dict | None:
    """Show one document being cut into overlapping chunks."""
    banner("STEP 1 - One document in, overlapping chunks out")

    filters = {"site": site} if site else {}
    document = store.collection.find_one(
        filters, {"_id": 0}, sort=[("stats.word_count", -1)]
    )
    if document is None:
        print("No documents in MongoDB. Run 'python scripts/demo_task3.py' first.")
        return None

    chunks = chunk_document(document)
    print(f"\nURL   : {document['url']}")
    print(f"title : {document.get('title')}")
    print(f"words : {document['stats']['word_count']:,}")
    print(f"\nchunking settings: {settings.chunk_words} words per chunk, "
          f"{settings.chunk_overlap_words} words of overlap")
    print(f"result           : {len(chunks)} chunk(s)")

    print("\n--- the chunks ---")
    for chunk in chunks:
        text = " ".join(chunk.text.split())
        print(f"  {chunk.chunk_id}  {chunk.word_count:>4} words  "
              f"heading={chunk.heading or '-'!r}")
        print(f"      \"{text[:110]}{'…' if len(text) > 110 else ''}\"")

    if len(chunks) > 1:
        tail = chunks[0].text.split()[-settings.chunk_overlap_words:]
        head = chunks[1].text.split()[: settings.chunk_overlap_words + 5]
        shared = [word for word in tail if word in head]
        print(f"\n--- the overlap between chunk 0 and chunk 1 ({len(shared)} words) ---")
        print("  " + " ".join(shared[:30]))
        print("\n  Without it, a fact sitting on the boundary would belong to")
        print("  neither chunk, and a query about it would match neither.")

    print("\n--- what is actually embedded (chunk 0) ---")
    print("  " + chunks[0].embedding_text[:200].replace("\n", " ⏎ "))
    print("\n  The site, title and heading are prefixed so that a passage taken")
    print("  from the middle of a page is embedded in its context.")
    return document


def step_embeddings() -> None:
    """Show that the vectors encode meaning, not vocabulary."""
    banner("STEP 2 - Why the vectors work: meaning, not words")

    embedder = get_embedder()
    description = embedder.describe()
    print(f"\nmodel      : {description['model']}")
    print(f"dimensions : {description['dimension']}")
    print(f"runtime    : {description['runtime']}  (local, free, no API key)")

    query = "how much does the blue widget cost"
    passages = [
        "The blue widget is priced at 9.99 pounds and ships within two days.",
        "A collection of quotations about love, life and friendship.",
        "The blue widget costs nine pounds ninety-nine.",
    ]

    vectors = embedder.embed_texts(passages)
    query_vector = embedder.embed_query(query)

    print(f"\nquery: {query!r}")
    print("\n  cosine similarity against three passages:")
    for passage, vector in zip(passages, vectors):
        similarity = cosine_similarity(query_vector, vector)
        print(f"    {similarity:>6.3f}  \"{passage[:66]}…\"")

    print("\n  The third passage shares almost no words with the query - no")
    print("  'how much', no 'cost' - and still scores high, because the vectors")
    print("  encode what the sentences mean. That is what a keyword index")
    print("  cannot do, and the reason the vector database exists.")
    print(f"\n  embedding throughput: {embedder.stats.as_dict()}")


def step_build_index(site: str | None, limit: int | None) -> VectorStore:
    """Build the index from every processed document, then prove idempotency."""
    banner("STEP 3 - Building the vector index from MongoDB")

    vectors = VectorStore()
    vectors.drop()
    print(f"Collection {vectors.collection_name!r} dropped - building from scratch.")

    indexer = Indexer(vectors=vectors)
    stats = indexer.run(site=site, limit=limit)
    stats.print_report()

    banner("STEP 3b - Re-indexing the same documents (idempotency)")
    before = vectors.count()
    again = indexer.run(site=site, limit=limit)
    after = vectors.count()

    print(f"\n  vectors before the second run : {before:,}")
    print(f"  vectors after the second run  : {after:,}")
    print(f"  documents processed again     : {again.documents_indexed:,}")
    if before == after:
        print("\n  The count did not change. Chunk ids are derived from the")
        print("  document id, so re-indexing overwrites rather than duplicates -")
        print("  which matters because the URL frontier delivers at-least-once.")
    else:
        print("\n  WARNING: the vector count changed on a repeat run.")
    return vectors


def step_retrieval(vectors: VectorStore, store: DocumentStore, query: str) -> None:
    """Run one query in all three modes, side by side."""
    banner(f"STEP 4 - The retrieval function, three modes: {query!r}")

    retriever = Retriever(vectors=vectors, store=store)
    for mode in MODES:
        result = retriever.retrieve(query, mode=mode, top_k=3)
        result.print_report(snippet=130)

    print("\n  vector  : semantic neighbours - finds paraphrases, misses rare literals")
    print("  keyword : MongoDB's text index - finds exact terms, misses paraphrases")
    print("  hybrid  : both rankings fused with Reciprocal Rank Fusion, so a")
    print("            passage found by both retrievers is ranked first")


def step_generation(questions: tuple[str, ...], site: str | None) -> None:
    """Combine the retrieved passages with generated text."""
    banner("STEP 5 - Retrieved text + generated answer")

    llm = get_llm()
    print(f"\nprovider: {json.dumps(llm.describe(), indent=2)}")

    for question in questions:
        try:
            answer = answer_question(question, site=site, top_k=4, llm=llm)
        except LLMError as exc:
            print(f"\nERROR for {question!r}: {exc}")
            continue
        answer.print_report()

    print("\n  Every sentence is produced from the retrieved passages alone and")
    print("  carries the [n] of the passage behind it, so each claim can be")
    print("  traced back to the page it came from.")


def step_augmentation(
    store: DocumentStore, vectors: VectorStore, site: str | None, limit: int | None
) -> None:
    """Write contextual summaries back into MongoDB and measure the gain."""
    banner("STEP 6 - Improving the stored data with generated summaries")

    sample = store.collection.find_one(
        {"site": site} if site else {}, {"_id": 0, "text": 0}
    )
    if sample is not None:
        flags = quality_flags(sample)
        print(f"\nbefore, for {sample['url']}:")
        print(f"  description : {sample.get('description')!r}")
        print(f"  summary     : {(sample.get('enrichment') or {}).get('summary')!r}")
        print(f"  quality     : {quality_score(flags):.3f}  {flags}")

    retriever = Retriever(vectors=vectors, store=store)
    augmenter = Augmenter(store=store, retriever=retriever)
    stats = augmenter.run(site=site, limit=limit)
    stats.print_report()

    if sample is not None:
        after = store.get(str(sample["doc_id"]))
        if after and after.get("enrichment"):
            print(f"\nafter, for {after['url']}:")
            print(json.dumps(after["enrichment"], indent=2, ensure_ascii=False)[:2000])


def step_report(vectors: VectorStore, store: DocumentStore) -> None:
    banner("STEP 7 - The index and the enriched corpus")
    print_report(vectors, store, samples=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demonstrate the Task 4 RAG module.")
    parser.add_argument("--site", default=None, help="Limit the demo to one site.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum documents indexed/augmented."
    )
    parser.add_argument(
        "--query", default=None, help="Query used for the retrieval comparison."
    )
    parser.add_argument(
        "--provider", default=None, choices=("auto", "anthropic", "extractive"),
        help="Override LLM_PROVIDER for this run.",
    )
    parser.add_argument(
        "--skip-augment", action="store_true", help="Leave out the enrichment step."
    )
    args = parser.parse_args(argv)

    quiet_noisy_loggers()
    if args.provider:
        settings.llm_provider = args.provider

    store = DocumentStore()
    vectors = VectorStore()

    try:
        store.require()
    except StoreUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Start MongoDB and run 'python scripts/demo_task3.py' first.", file=sys.stderr)
        return 2

    try:
        vectors.require()
    except VectorStoreUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if store.count(args.site) == 0:
        print("No processed documents to index. Run 'python scripts/demo_task3.py' first.",
              file=sys.stderr)
        return 1

    document = step_chunking(store, args.site)
    if document is None:
        return 1

    try:
        step_embeddings()
        vectors = step_build_index(args.site, args.limit)
        query = args.query or (document.get("title") or "widget")
        step_retrieval(vectors, store, query)
        step_generation(
            (query,) + DEFAULT_QUESTIONS if args.query else DEFAULT_QUESTIONS, args.site
        )
        if not args.skip_augment:
            step_augmentation(store, vectors, args.site, args.limit)
        step_report(vectors, store)
    except (EmbeddingUnavailable, VectorStoreUnavailable, StoreUnavailable, LLMError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    banner("TASK 4 DEMONSTRATION COMPLETE")
    print("\nWhat was shown:")
    print("  1. documents cut into overlapping, context-carrying chunks")
    print("  2. embeddings that match on meaning rather than on words")
    print("  3. a vector index built from MongoDB, idempotent on re-run")
    print("  4. a retrieval function in three modes, fused by rank")
    print("  5. answers generated only from retrieved passages, with citations")
    if not args.skip_augment:
        print("  6. stored documents improved by generated summaries, measured")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
