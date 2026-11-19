"""Tests for the RAG module (Task 4).

The tests are layered by what they need, so that ``pytest`` stays green on a
machine where no container is running and no API key is set:

* **Always run** - chunking, the free extractive summariser, rank fusion, the
  distance/relevance conversion, the quality checklist and the context
  builder. These are pure functions over Python objects.
* **Need the embedding model** - skipped when the model is not cached and
  cannot be downloaded.
* **Need Chroma and/or MongoDB** - skipped when the services are unreachable.

The Claude path is tested with a stub provider rather than a real API call:
the thing worth testing is that the prompt is grounded in the retrieved
context and that a provider failure degrades to the extractive summariser,
and neither of those needs to spend money to verify.
"""

from __future__ import annotations

import pytest

from src.processing.store import DocumentStore, StoreUnavailable
from src.rag.augment import (
    Augmenter,
    answer_question,
    quality_flags,
    quality_score,
)
from src.rag.chunker import (
    Chunk,
    ChunkStats,
    chunk_document,
    chunk_documents,
    split_paragraphs,
    split_text,
)
from src.rag.embedder import (
    Embedder,
    EmbeddingUnavailable,
    cosine_similarity,
)
from src.rag.llm import (
    BaseLLM,
    ExtractiveLLM,
    FallbackLLM,
    LLMError,
    LLMResponse,
    content_words,
    get_llm,
    split_sentences,
)
from src.rag.retriever import (
    RRF_K,
    RetrievalError,
    RetrievalResult,
    RetrievedChunk,
    Retriever,
    query_terms,
)
from src.rag.vector_store import (
    VectorStore,
    VectorStoreUnavailable,
    relevance_from_distance,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

BODY = (
    "Everything you need to know about widgets.\n"
    "Overview\n"
    "The blue widget costs 9.99 pounds. It is in stock and ships in two days.\n"
    "Pricing\n"
    + " ".join(f"Clause{index} explains a detail." for index in range(90))
    + "\nThe red widget costs 12.50 pounds and is out of stock.\n"
)

DOCUMENT = {
    "doc_id": "0123456789abcdef",
    "url": "https://widgets.test/catalogue",
    "site": "widgets.test",
    "title": "Widget Catalogue",
    "description": "Everything about widgets",
    "headings": ["Overview", "Pricing"],
    "keywords": ["widget", "blue", "price"],
    "text": BODY,
    "stats": {"word_count": len(BODY.split())},
}


def make_chunk(index: int, *, doc_id: str = "doc", text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}:{index:04d}",
        doc_id=doc_id,
        chunk_index=index,
        text=text,
        url=f"https://example.test/{doc_id}",
        site="example.test",
        title=f"Page {doc_id}",
    )


class StubLLM(BaseLLM):
    """A provider that records its prompt instead of calling an API."""

    name = "stub"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "stub-1"

    def available(self) -> bool:
        return True

    def summarise(self, query, context, **kwargs):  # type: ignore[override]
        if self.fail:
            raise LLMError("stub provider is down")
        self.calls.append((query, context))
        return LLMResponse(
            text="A generated summary [1].",
            provider=self.name,
            model=self.model,
            sentences=1,
        )


def _mongodb_available() -> bool:
    try:
        return DocumentStore().ping()
    except StoreUnavailable:
        return False


def _chroma_available() -> bool:
    try:
        return VectorStore().ping()
    except VectorStoreUnavailable:
        return False


def _embedder_or_skip() -> Embedder:
    embedder = Embedder()
    try:
        embedder.embed_texts(["warm up"])
    except EmbeddingUnavailable as exc:
        pytest.skip(f"embedding model unavailable: {exc}")
    return embedder


needs_mongodb = pytest.mark.skipif(
    not _mongodb_available(), reason="MongoDB is not reachable"
)
needs_chroma = pytest.mark.skipif(
    not _chroma_available(), reason="Chroma is not reachable"
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestChunker:
    def test_paragraphs_are_recovered_from_clean_text(self) -> None:
        blocks = split_paragraphs("one\n\ntwo\nthree\n\n\n")
        assert blocks == ["one", "two", "three"]

    def test_chunks_respect_the_word_budget(self) -> None:
        chunks = chunk_document(DOCUMENT)
        assert len(chunks) > 1
        assert all(chunk.word_count <= 180 for chunk in chunks)

    def test_consecutive_chunks_overlap(self) -> None:
        chunks = chunk_document(DOCUMENT)
        tail = set(chunks[0].text.split()[-40:])
        head = set(chunks[1].text.split()[:45])
        assert tail & head, "consecutive chunks must share their boundary text"

    def test_chunk_ids_are_deterministic(self) -> None:
        first = chunk_document(DOCUMENT)
        second = chunk_document(DOCUMENT)
        assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
        assert first[0].chunk_id == f"{DOCUMENT['doc_id']}:0000"

    def test_headings_label_chunks_and_are_not_chunks(self) -> None:
        chunks = chunk_document(DOCUMENT)
        assert any(chunk.heading in ("Overview", "Pricing") for chunk in chunks)
        assert not any(chunk.text.strip() in ("Overview", "Pricing") for chunk in chunks)

    def test_embedding_text_carries_the_page_context(self) -> None:
        chunk = chunk_document(DOCUMENT)[0]
        assert "widgets.test" in chunk.embedding_text
        assert "Widget Catalogue" in chunk.embedding_text
        # The stored text stays clean - the header is only for the vector.
        assert not chunk.text.startswith("widgets.test")

    def test_short_page_still_produces_one_chunk(self) -> None:
        chunks = chunk_document(
            {"doc_id": "short", "text": "A page with only one short sentence."}
        )
        assert len(chunks) == 1

    def test_empty_document_produces_no_chunks(self) -> None:
        assert chunk_document({"doc_id": "empty", "text": "   "}) == []

    def test_document_without_doc_id_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            chunk_document({"text": "some text"})

    def test_a_paragraph_longer_than_the_budget_is_split(self) -> None:
        wall = " ".join(f"Sentence{index} says something." for index in range(200))
        pieces = split_text(wall, words_per_chunk=60, overlap_words=10)
        assert len(pieces) > 3
        assert all(len(body.split()) <= 60 for body, _heading in pieces)

    def test_metadata_is_flat_and_json_safe(self) -> None:
        metadata = chunk_document(DOCUMENT)[0].metadata()
        assert all(isinstance(value, (str, int, float)) for value in metadata.values())

    def test_stats_are_accumulated(self) -> None:
        stats = ChunkStats()
        chunks = list(chunk_documents([DOCUMENT, {"doc_id": "x", "text": ""}], stats=stats))
        assert stats.documents == 1
        assert stats.documents_skipped == 1
        assert stats.chunks == len(chunks)
        assert stats.as_dict()["chunks_per_document"] == pytest.approx(
            len(chunks), abs=0.01
        )


# ---------------------------------------------------------------------------
# The free extractive provider
# ---------------------------------------------------------------------------


class TestExtractiveLLM:
    context = (
        "[1] Widget Catalogue\n"
        "URL: https://widgets.test/catalogue\n"
        "The blue widget costs 9.99 pounds. It is in stock and ships in two days.\n"
        "\n"
        "[2] Clearance\n"
        "URL: https://widgets.test/clearance\n"
        "The red widget costs 12.50 pounds and is out of stock."
    )

    def test_sentences_are_split_conservatively(self) -> None:
        sentences = split_sentences("It costs 9.99 pounds. It ships in two days.")
        assert len(sentences) == 2

    def test_stopwords_are_dropped_from_content_words(self) -> None:
        assert "the" not in content_words("The blue widget")
        assert "widget" in content_words("The blue widget")

    def test_summary_uses_only_source_sentences(self) -> None:
        response = ExtractiveLLM().summarise(
            "how much does the blue widget cost", self.context, max_sentences=2
        )
        assert response.provider == "extractive"
        assert response.grounded is True
        assert "9.99" in response.text
        # Every sentence is copied, so nothing can appear that is not in the
        # context (the citation markers aside).
        for sentence in split_sentences(response.text.replace(" [1]", "").replace(" [2]", "")):
            assert sentence.strip(" .") in self.context.replace("\n", " ")

    def test_summary_is_query_focused(self) -> None:
        red = ExtractiveLLM().summarise("red widget price", self.context, max_sentences=1)
        assert "12.50" in red.text

    def test_citation_markers_point_at_the_passage(self) -> None:
        response = ExtractiveLLM().summarise("blue widget", self.context, max_sentences=1)
        assert "[1]" in response.text

    def test_summary_respects_the_sentence_limit(self) -> None:
        response = ExtractiveLLM().summarise("widget", self.context, max_sentences=1)
        assert response.sentences == 1

    def test_empty_context_is_reported_not_invented(self) -> None:
        response = ExtractiveLLM().summarise("anything", "", max_sentences=3)
        assert response.text == ""
        assert response.notes

    def test_duplicate_sentences_are_not_repeated(self) -> None:
        duplicated = (
            "[1] A\nURL: u\nThe blue widget costs 9.99 pounds.\n\n"
            "[2] B\nURL: v\nThe blue widget costs 9.99 pounds."
        )
        response = ExtractiveLLM().summarise("blue widget", duplicated, max_sentences=3)
        assert response.sentences == 1


class TestProviderSelection:
    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(LLMError):
            get_llm("gpt")

    def test_extractive_is_always_available(self) -> None:
        assert get_llm("extractive").available() is True

    def test_auto_without_a_key_is_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.common.config.settings.anthropic_api_key", "")
        provider = get_llm("auto")
        assert provider.name == "extractive"

    def test_a_failing_provider_degrades_instead_of_raising(self) -> None:
        provider = FallbackLLM(StubLLM(fail=True), ExtractiveLLM())
        response = provider.summarise(
            "blue widget",
            "[1] A\nURL: u\nThe blue widget costs 9.99 pounds.",
            max_sentences=1,
        )
        assert response.provider == "extractive"
        assert any("unavailable" in note for note in response.notes)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class TestRelevance:
    def test_distance_zero_is_full_relevance(self) -> None:
        assert relevance_from_distance(0.0) == 1.0

    def test_opposite_vectors_score_zero(self) -> None:
        assert relevance_from_distance(2.0) == 0.0

    def test_missing_distance_is_not_an_error(self) -> None:
        assert relevance_from_distance(None) == 0.0

    def test_cosine_similarity_matches_itself(self) -> None:
        vector = [0.1, 0.2, 0.3]
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_cosine_similarity_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


class TestFusion:
    def test_query_terms_drop_stopwords(self) -> None:
        assert query_terms("what is the price of the blue widget") == [
            "price",
            "blue",
            "widget",
        ]

    def test_rrf_score_matches_the_formula(self) -> None:
        fused = Retriever.fuse([make_chunk(0)])
        assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))

    def test_a_chunk_found_twice_outranks_a_chunk_found_once(self) -> None:
        both = make_chunk(0, doc_id="both")
        both.vector_rank, both.vector_score, both.retrievers = 2, 0.7, ["vector"]
        same = make_chunk(0, doc_id="both")
        same.keyword_rank, same.keyword_score, same.retrievers = 2, 3.1, ["keyword"]
        only = make_chunk(1, doc_id="only")
        only.vector_rank, only.vector_score, only.retrievers = 1, 0.9, ["vector"]

        fused = Retriever.fuse([only, both], [same])
        assert fused[0].doc_id == "both"
        assert fused[0].retrievers == ["vector", "keyword"]
        # Both retrievers' evidence survives the merge.
        assert fused[0].vector_score == 0.7
        assert fused[0].keyword_score == 3.1

    def test_fusion_is_deterministic(self) -> None:
        chunks = [make_chunk(index) for index in range(5)]
        first = [chunk.chunk_id for chunk in Retriever.fuse(chunks)]
        second = [chunk.chunk_id for chunk in Retriever.fuse(chunks)]
        assert first == second


class TestRetrievalResult:
    def _result(self) -> RetrievalResult:
        chunks = [
            make_chunk(0, doc_id="a", text="The blue widget costs 9.99 pounds."),
            make_chunk(1, doc_id="b", text="The red widget costs 12.50 pounds."),
        ]
        for rank, chunk in enumerate(chunks, start=1):
            chunk.score = 1.0 / rank
        return RetrievalResult(query="widget price", mode="hybrid", chunks=chunks)

    def test_context_is_numbered_and_cited(self) -> None:
        context = self._result().context()
        assert context.startswith("[1] ")
        assert "[2] " in context
        assert "URL: https://example.test/a" in context

    def test_context_respects_the_character_budget(self) -> None:
        context = self._result().context(max_chars=80)
        assert len(context) <= 120  # the budget, plus one truncation marker

    def test_sources_are_deduplicated(self) -> None:
        result = self._result()
        result.chunks.append(make_chunk(2, doc_id="a", text="More about the blue widget."))
        sources = result.sources()
        assert len(sources) == 2
        assert sources[0]["chunks"] == 2

    def test_as_dict_can_omit_the_passage_text(self) -> None:
        data = self._result().as_dict(include_text=False)
        assert "text" not in data["chunks"][0]
        assert data["results"] == 2

    def test_empty_result_is_falsey(self) -> None:
        assert not RetrievalResult(query="q", mode="hybrid")


class TestRetrieverGuards:
    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(RetrievalError):
            Retriever().retrieve("widget", mode="magic")

    def test_empty_query_is_rejected(self) -> None:
        with pytest.raises(RetrievalError):
            Retriever().retrieve("   ")


# ---------------------------------------------------------------------------
# Augmentation and data quality
# ---------------------------------------------------------------------------


class TestQualityScore:
    def test_a_complete_document_scores_one(self) -> None:
        flags = quality_flags(DOCUMENT, summary="A summary.")
        assert quality_score(flags) == 1.0

    def test_a_missing_description_lowers_the_score(self) -> None:
        document = {**DOCUMENT, "description": None}
        assert quality_score(quality_flags(document)) < 1.0

    def test_the_summary_is_what_the_stage_adds(self) -> None:
        before = quality_flags(DOCUMENT)
        after = quality_flags(DOCUMENT, summary="A generated summary.")
        assert before["has_summary"] is False
        assert after["has_summary"] is True
        assert quality_score(after) > quality_score(before)


class TestAnswerQuestion:
    class _Retriever:
        """A retriever that returns a fixed result, so no service is needed."""

        def __init__(self, chunks: list[RetrievedChunk]) -> None:
            self.chunks = chunks

        def retrieve(self, query, **kwargs):  # type: ignore[no-untyped-def]
            return RetrievalResult(query=query, mode="hybrid", chunks=self.chunks)

    def test_the_prompt_contains_only_retrieved_text(self) -> None:
        chunk = make_chunk(0, doc_id="a", text="The blue widget costs 9.99 pounds.")
        chunk.score = 0.9
        provider = StubLLM()

        answer = answer_question(
            "what does the blue widget cost",
            retriever=self._Retriever([chunk]),  # type: ignore[arg-type]
            llm=provider,
        )

        _query, context = provider.calls[0]
        assert "9.99" in context
        assert answer.answer == "A generated summary [1]."
        assert answer.citations[0].url == chunk.url
        assert answer.citations[0].index == 1

    def test_no_retrieval_means_no_generated_answer(self) -> None:
        provider = StubLLM()
        answer = answer_question(
            "something absent",
            retriever=self._Retriever([]),  # type: ignore[arg-type]
            llm=provider,
        )
        assert answer.answer == ""
        assert provider.calls == []
        assert answer.notes


# ---------------------------------------------------------------------------
# Tests that need the embedding model
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_vectors_are_normalised_and_the_right_width(self) -> None:
        embedder = _embedder_or_skip()
        vectors = embedder.embed_texts(["the blue widget costs 9.99 pounds"])
        assert len(vectors[0]) == embedder.dimension
        assert sum(value * value for value in vectors[0]) == pytest.approx(1.0, abs=1e-5)

    def test_semantic_neighbours_beat_unrelated_text(self) -> None:
        """The property the whole retrieval stage depends on."""
        embedder = _embedder_or_skip()
        passages = embedder.embed_texts(
            [
                "The blue widget costs 9.99 pounds and is in stock.",
                "A collection of quotations about love and life.",
            ]
        )
        query = embedder.embed_query("how much is the blue widget")
        assert cosine_similarity(query, passages[0]) > cosine_similarity(query, passages[1])

    def test_embedding_is_deterministic(self) -> None:
        embedder = _embedder_or_skip()
        first = embedder.embed_texts(["the same sentence"])[0]
        second = embedder.embed_texts(["the same sentence"])[0]
        assert cosine_similarity(first, second) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests that need the services
# ---------------------------------------------------------------------------


@needs_chroma
class TestVectorStore:
    @pytest.fixture()
    def store(self) -> VectorStore:
        vectors = VectorStore(collection_name="pytest_rag_chunks")
        vectors.drop()
        yield vectors
        vectors.drop()

    def test_upsert_then_query_returns_the_chunk(self, store: VectorStore) -> None:
        embedder = _embedder_or_skip()
        chunks = chunk_document(DOCUMENT)[:2]
        store.ensure_collection({"embedding_model": embedder.model_name})
        store.upsert_chunks(
            chunks, embedder.embed_texts([chunk.embedding_text for chunk in chunks])
        )

        hits = store.query(embedder.embed_query("blue widget price"), top_k=2)
        assert hits
        assert hits[0]["doc_id"] == DOCUMENT["doc_id"]
        assert 0.0 <= hits[0]["score"] <= 1.0

    def test_reindexing_is_idempotent(self, store: VectorStore) -> None:
        embedder = _embedder_or_skip()
        chunks = chunk_document(DOCUMENT)
        vectors = embedder.embed_texts([chunk.embedding_text for chunk in chunks])

        store.upsert_chunks(chunks, vectors)
        first = store.count()
        store.upsert_chunks(chunks, vectors)
        assert store.count() == first

    def test_deleting_a_document_removes_its_chunks(self, store: VectorStore) -> None:
        embedder = _embedder_or_skip()
        chunks = chunk_document(DOCUMENT)
        store.upsert_chunks(
            chunks, embedder.embed_texts([chunk.embedding_text for chunk in chunks])
        )
        store.delete_document(str(DOCUMENT["doc_id"]))
        assert store.get_document_chunks(str(DOCUMENT["doc_id"])) == []

    def test_metadata_filter_scopes_the_search(self, store: VectorStore) -> None:
        embedder = _embedder_or_skip()
        chunks = chunk_document(DOCUMENT)
        store.upsert_chunks(
            chunks, embedder.embed_texts([chunk.embedding_text for chunk in chunks])
        )
        query = embedder.embed_query("widget")
        assert store.query(query, top_k=3, site="widgets.test")
        assert store.query(query, top_k=3, site="nowhere.test") == []


@needs_chroma
@needs_mongodb
class TestEndToEnd:
    """Index, retrieve and augment one document through the real services."""

    @pytest.fixture()
    def pipeline(self):  # type: ignore[no-untyped-def]
        from src.rag.indexer import Indexer

        embedder = _embedder_or_skip()
        documents = DocumentStore(collection="pytest_rag_documents")
        documents.ensure_indexes()
        documents.upsert(
            {
                **DOCUMENT,
                "content_hash": "0" * 64,
                "schema_version": "1.0",
                "processing": {"processed_at": "2026-01-01T00:00:00+00:00"},
            }
        )
        vectors = VectorStore(collection_name="pytest_rag_e2e")
        vectors.drop()

        indexer = Indexer(store=documents, vectors=vectors, embedder=embedder)
        retriever = Retriever(vectors=vectors, store=documents, embedder=embedder)
        yield indexer, retriever, documents, vectors

        vectors.drop()
        documents.drop()

    def test_index_then_retrieve_then_summarise(self, pipeline) -> None:  # type: ignore[no-untyped-def]
        indexer, retriever, documents, _vectors = pipeline

        stats = indexer.run()
        assert stats.documents_indexed == 1
        assert stats.chunks_written >= 1

        result = retriever.retrieve("what does the blue widget cost", mode="vector")
        assert result.chunks
        assert result.chunks[0].doc_id == DOCUMENT["doc_id"]

        answer = answer_question(
            "what does the blue widget cost",
            retriever=retriever,
            llm=ExtractiveLLM(),
            mode="vector",
        )
        assert answer.citations
        assert answer.provider == "extractive"

    def test_augmentation_raises_the_quality_score(self, pipeline) -> None:  # type: ignore[no-untyped-def]
        indexer, retriever, documents, _vectors = pipeline
        indexer.run()

        # Start from a document with no description, which is the gap the
        # augmentation stage is meant to close.
        documents.collection.update_one(
            {"doc_id": DOCUMENT["doc_id"]}, {"$set": {"description": None}}
        )

        augmenter = Augmenter(store=documents, retriever=retriever, llm=ExtractiveLLM())
        stats = augmenter.run()

        assert stats.documents_augmented == 1
        assert stats.quality_gain > 0

        stored = documents.get(str(DOCUMENT["doc_id"]))
        assert stored is not None
        enrichment = stored["enrichment"]
        assert enrichment["summary"]
        assert enrichment["provider"] == "extractive"
        assert stored["description"], "a missing description must be filled in"
        assert "description" in enrichment["quality"]["fields_filled"]

    def test_hybrid_retrieval_finds_an_exact_term(self, pipeline) -> None:  # type: ignore[no-untyped-def]
        indexer, retriever, _documents, _vectors = pipeline
        indexer.run()

        result = retriever.retrieve("9.99", mode="hybrid")
        assert result.chunks
        assert any("9.99" in chunk.text for chunk in result.chunks)
