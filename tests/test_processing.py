"""Tests for the data-processing module (Task 3).

The cleaning and normalisation tests are pure functions over HTML strings, so
they always run. The store and pipeline tests need MongoDB (and PostgreSQL)
and are skipped automatically when those are not reachable, which keeps
``pytest`` green on a machine where the containers are not started.
"""

from __future__ import annotations

import json

import pytest

from src.processing.cleaner import (
    clean_html,
    clean_value,
    filter_lines,
    normalise_characters,
    normalise_whitespace,
    strip_tags,
)
from src.processing.normalizer import (
    NormalizationError,
    canonical_url,
    content_hash,
    document_id,
    extract_keywords,
    normalize_page,
    text_statistics,
    to_json,
    validate_document,
)
from src.processing.store import DocumentStore, StoreUnavailable

PAGE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>  Widget   Catalogue  </title>
    <meta name="description" content="Everything about widgets">
    <style>body { color: red; }</style>
    <script>var tracker = {id: 42}; alert("tracking");</script>
  </head>
  <body onload="init()">
    <!-- a comment that must not appear in the text -->
    <nav><a href="/">Home</a><a href="/cart">Cart</a></nav>
    <header><h1>Widget&nbsp;Catalogue</h1></header>
    <main>
      <h2>Overview</h2>
      <p>The <b>blue</b> widget costs &pound;9.99.</p>
      <p>It is in stock.</p>
      <noscript>Enable JavaScript</noscript>
    </main>
    <footer><p>All rights reserved</p></footer>
    <iframe src="https://ads.example.com"></iframe>
  </body>
</html>
"""


# ---------------------------------------------------------------------------
# Stage 1: cleaning
# ---------------------------------------------------------------------------


class TestCleanHtml:
    @pytest.fixture(scope="class")
    def cleaned(self):  # type: ignore[no-untyped-def]
        return clean_html(PAGE)

    def test_script_and_style_contents_are_gone(self, cleaned) -> None:  # type: ignore[no-untyped-def]
        assert "tracker" not in cleaned.text
        assert "alert" not in cleaned.text
        assert "color: red" not in cleaned.text

    def test_comments_are_removed(self, cleaned) -> None:  # type: ignore[no-untyped-def]
        assert "must not appear" not in cleaned.text
        assert cleaned.removed_comments == 1

    def test_no_markup_survives(self, cleaned) -> None:  # type: ignore[no-untyped-def]
        assert "<" not in cleaned.text
        assert ">" not in cleaned.text

    def test_content_text_is_kept(self, cleaned) -> None:  # type: ignore[no-untyped-def]
        assert "blue" in cleaned.text
        assert "It is in stock." in cleaned.text

    def test_entities_are_decoded(self, cleaned) -> None:  # type: ignore[no-untyped-def]
        assert "£9.99" in cleaned.text
        assert "&pound;" not in cleaned.text
        assert "&nbsp;" not in cleaned.text

    def test_boilerplate_elements_are_dropped(self, cleaned) -> None:  # type: ignore[no-untyped-def]
        assert "Cart" not in cleaned.text          # inside <nav>
        assert "All rights reserved" not in cleaned.text   # inside <footer>
        assert cleaned.removed_elements.get("nav") == 1
        assert cleaned.removed_elements.get("iframe") == 1

    def test_boilerplate_can_be_kept(self) -> None:
        kept = clean_html(PAGE, remove_boilerplate=False, drop_navigation_lines=False)
        assert "Cart" in kept.text

    def test_paragraphs_are_separated(self, cleaned) -> None:  # type: ignore[no-untyped-def]
        # Without forced block breaks these two sentences would be glued.
        assert "£9.99It is in stock" not in cleaned.text

    def test_measurements_are_reported(self, cleaned) -> None:  # type: ignore[no-untyped-def]
        assert cleaned.raw_length == len(PAGE)
        assert 0 < cleaned.length < cleaned.raw_length
        assert 0.0 < cleaned.reduction_ratio < 1.0
        assert cleaned.word_count > 0
        assert cleaned.duration_ms > 0
        assert cleaned.summary()["clean_length"] == cleaned.length

    def test_empty_input_is_handled(self) -> None:
        for value in ("", "   ", "\n\n"):
            result = clean_html(value)
            assert result.text == ""
            assert result.word_count == 0
        # An empty body has nothing to compare against, so the ratio is 0;
        # a whitespace-only body was 100% noise.
        assert clean_html("").reduction_ratio == 0.0
        assert clean_html("   ").reduction_ratio == 1.0

    def test_malformed_html_does_not_raise(self) -> None:
        result = clean_html("<div><p>unclosed <b>tags <span>here")
        assert "unclosed" in result.text

    def test_double_encoded_markup_is_stripped(self) -> None:
        result = clean_html("<p>&lt;script&gt;evil()&lt;/script&gt;Real text</p>")
        assert "evil" not in result.text
        assert "Real text" in result.text


class TestTextHelpers:
    def test_strip_tags(self) -> None:
        assert strip_tags("<p>hello <b>world</b></p>").split() == ["hello", "world"]

    def test_strip_tags_removes_script_bodies(self) -> None:
        assert "evil" not in strip_tags("<script>evil()</script>keep")

    def test_strip_tags_on_empty(self) -> None:
        assert strip_tags("") == ""

    def test_normalise_characters_removes_invisibles(self) -> None:
        text = normalise_characters("a​b﻿c­d")
        assert text == "abcd"

    def test_normalise_characters_applies_nfkc(self) -> None:
        # Fullwidth and ligature forms collapse to their ASCII equivalents.
        assert normalise_characters("ｗｉｄｇｅｔ") == "widget"
        assert normalise_characters("ﬁle") == "file"

    def test_normalise_characters_strips_control_codes(self) -> None:
        assert normalise_characters("a\x00\x07b") == "ab"

    def test_normalise_whitespace_collapses_runs(self) -> None:
        assert normalise_whitespace("a   \t  b") == "a b"

    def test_normalise_whitespace_caps_blank_lines(self) -> None:
        assert normalise_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_filter_lines_drops_navigation_and_duplicates(self) -> None:
        lines, dropped = filter_lines(
            ["Home", "Real content here", "Real content here", "x", "More content"]
        )
        assert lines == ["Real content here", "More content"]
        assert dropped == 3

    def test_clean_value_walks_nested_structures(self) -> None:
        value = clean_value(
            {
                "price": "  &pound;10.00 ",
                "rows": ["<b>one</b>", {"k": "two&nbsp;words"}],
                "count": 3,
                "flag": True,
            }
        )
        assert value["price"] == "£10.00"
        assert value["rows"][0] == "one"
        assert value["rows"][1]["k"] == "two words"
        assert value["count"] == 3        # non-strings pass through untouched
        assert value["flag"] is True


# ---------------------------------------------------------------------------
# Stage 2: normalisation
# ---------------------------------------------------------------------------


class TestCanonicalisation:
    @pytest.mark.parametrize(
        "variant",
        [
            "https://Example.com/a/",
            "https://example.com/a",
            "https://EXAMPLE.COM/a/#section",
            "https://example.com/a/",
        ],
    )
    def test_variants_canonicalise_to_one_url(self, variant: str) -> None:
        assert canonical_url(variant) == "https://example.com/a"

    def test_query_string_is_significant(self) -> None:
        assert canonical_url("https://e.com/a?p=2") != canonical_url("https://e.com/a")

    def test_document_id_is_deterministic_and_url_derived(self) -> None:
        first = document_id("https://example.com/a/")
        assert first == document_id("https://EXAMPLE.com/a")
        assert first != document_id("https://example.com/b")
        assert len(first) == 40

    def test_content_hash_tracks_the_text(self) -> None:
        assert content_hash("abc") == content_hash("abc")
        assert content_hash("abc") != content_hash("abd")


class TestNormalizePage:
    @pytest.fixture(scope="class")
    def document(self):  # type: ignore[no-untyped-def]
        return normalize_page(
            url="https://Example.com/catalogue/widget/",
            raw_html=PAGE,
            title="  Widget   Catalogue  ",
            description="Everything about widgets",
            language="en",
            headings=["Widget Catalogue", "Overview", "Widget Catalogue"],
            structured_data={"price": "&pound;9.99", "tags": ["<b>blue</b>"]},
            source={
                "raw_page_id": 7,
                "job_id": 1,
                "depth": 2,
                "http_status": 200,
                "fetched_by": "host:123",
            },
            processed_by="pytest",
        )

    def test_identity_fields(self, document) -> None:  # type: ignore[no-untyped-def]
        assert document["url"] == "https://example.com/catalogue/widget"
        assert document["doc_id"] == document_id(document["url"])
        assert document["site"] == "example.com"
        assert document["path"] == "/catalogue/widget"
        assert document["original_url"] == "https://Example.com/catalogue/widget/"

    def test_scalar_fields_are_cleaned(self, document) -> None:  # type: ignore[no-untyped-def]
        assert document["title"] == "Widget Catalogue"
        assert document["language"] == "en"

    def test_headings_are_deduplicated(self, document) -> None:  # type: ignore[no-untyped-def]
        assert document["headings"] == ["Widget Catalogue", "Overview"]

    def test_site_specific_data_is_quarantined_and_cleaned(self, document) -> None:  # type: ignore[no-untyped-def]
        assert document["fields"]["price"] == "£9.99"
        assert document["fields"]["tags"] == ["blue"]

    def test_statistics_are_present(self, document) -> None:  # type: ignore[no-untyped-def]
        stats = document["stats"]
        assert stats["word_count"] > 0
        assert stats["char_count"] == len(document["text"])
        assert 0 < stats["lexical_diversity"] <= 1.0
        assert stats["reading_time_sec"] >= 0

    def test_provenance_is_carried_through(self, document) -> None:  # type: ignore[no-untyped-def]
        assert document["source"]["raw_page_id"] == 7
        assert document["source"]["http_status"] == 200
        assert document["source"]["fetched_by"] == "host:123"
        assert document["source"]["raw_length"] == len(PAGE)

    def test_processing_metadata_is_recorded(self, document) -> None:  # type: ignore[no-untyped-def]
        processing = document["processing"]
        assert processing["processed_by"] == "pytest"
        assert processing["cleaner_version"]
        assert processing["reduction_ratio"] > 0
        assert processing["clean_ms"] >= 0
        assert processing["processed_at"].startswith("20")

    def test_document_is_json_serialisable(self, document) -> None:  # type: ignore[no-untyped-def]
        restored = json.loads(to_json(document))
        assert restored["doc_id"] == document["doc_id"]

    def test_document_passes_validation(self, document) -> None:  # type: ignore[no-untyped-def]
        validate_document(document)      # must not raise

    def test_same_page_yields_the_same_id(self, document) -> None:  # type: ignore[no-untyped-def]
        again = normalize_page(url="https://example.com/catalogue/widget", raw_html=PAGE)
        assert again["doc_id"] == document["doc_id"]
        assert again["content_hash"] == document["content_hash"]

    def test_title_falls_back_to_the_first_heading(self) -> None:
        document = normalize_page(
            url="https://example.com/x",
            raw_html="<html><body><p>text</p></body></html>",
            headings=["A Heading"],
        )
        assert document["title"] == "A Heading"

    def test_url_is_required(self) -> None:
        with pytest.raises(NormalizationError):
            normalize_page(url="", raw_html=PAGE)

    def test_html_or_cleaned_text_is_required(self) -> None:
        with pytest.raises(NormalizationError):
            normalize_page(url="https://example.com/a")


class TestValidation:
    def test_missing_field_is_reported(self) -> None:
        document = normalize_page(url="https://example.com/a", raw_html=PAGE)
        del document["keywords"]
        with pytest.raises(NormalizationError, match="keywords"):
            validate_document(document)

    def test_wrong_type_is_reported(self) -> None:
        document = normalize_page(url="https://example.com/a", raw_html=PAGE)
        document["headings"] = "not a list"
        with pytest.raises(NormalizationError, match="headings"):
            validate_document(document)

    def test_unserialisable_value_is_reported(self) -> None:
        document = normalize_page(url="https://example.com/a", raw_html=PAGE)
        document["fields"] = {"bad": {1, 2, 3}}      # a set is not JSON
        with pytest.raises(NormalizationError, match="JSON"):
            validate_document(document)


class TestStatisticsAndKeywords:
    def test_keywords_exclude_stopwords(self) -> None:
        keywords = extract_keywords("the widget and the widget with a widget cost")
        assert keywords[0] == "widget"
        assert "the" not in keywords
        assert "and" not in keywords

    def test_keywords_on_empty_text(self) -> None:
        assert extract_keywords("") == []

    def test_statistics_of_empty_text(self) -> None:
        stats = text_statistics("", [])
        assert stats["word_count"] == 0
        assert stats["lexical_diversity"] == 0.0
        assert stats["reading_time_sec"] == 0


# ---------------------------------------------------------------------------
# Stage 3: the MongoDB document store
# ---------------------------------------------------------------------------


def _store_available() -> bool:
    try:
        return DocumentStore().ping()
    except StoreUnavailable:
        return False
    except Exception:  # noqa: BLE001
        return False


store_required = pytest.mark.skipif(
    not _store_available(), reason="MongoDB is not reachable"
)


@store_required
class TestDocumentStore:
    @pytest.fixture()
    def store(self):  # type: ignore[no-untyped-def]
        # A dedicated collection, so the tests never touch the real corpus.
        store = DocumentStore(collection="pytest_documents")
        store.drop()
        store.ensure_indexes()
        yield store
        store.drop()

    def _document(self, path: str, text: str = "widget catalogue text") -> dict:
        return normalize_page(
            url=f"https://pytest.local{path}",
            raw_html=f"<html><body><h1>Title {path}</h1><p>{text}</p></body></html>",
            site="pytest.local",
            structured_data={"price": "1.00"},
        )

    def test_indexes_are_created(self, store) -> None:  # type: ignore[no-untyped-def]
        names = {index["name"] for index in store.indexes()}
        assert {
            "uq_doc_id",
            "ix_site_processed_at",
            "ix_content_hash",
            "ix_word_count",
            "ix_fulltext",
        } <= names

    def test_ensure_indexes_is_idempotent(self, store) -> None:  # type: ignore[no-untyped-def]
        before = len(store.indexes())
        store.ensure_indexes()
        assert len(store.indexes()) == before

    def test_upsert_then_read_back(self, store) -> None:  # type: ignore[no-untyped-def]
        document = self._document("/a")
        assert store.upsert(document) is True
        stored = store.get(document["doc_id"])
        assert stored is not None
        assert stored["url"] == document["url"]
        assert stored["fields"]["price"] == "1.00"
        # processed_at is lifted to the top level for the compound index.
        assert stored["processed_at"] == document["processing"]["processed_at"]

    def test_reprocessing_updates_instead_of_duplicating(self, store) -> None:  # type: ignore[no-untyped-def]
        first = self._document("/b", text="first version")
        second = self._document("/b", text="second version")
        assert store.upsert(first) is True
        assert store.upsert(second) is False       # same doc_id: an update
        assert store.count() == 1
        assert "second version" in store.get(first["doc_id"])["text"]

    def test_upsert_many_reports_counts(self, store) -> None:  # type: ignore[no-untyped-def]
        documents = [self._document(f"/batch/{i}") for i in range(5)]
        result = store.upsert_many(documents)
        assert result == {"inserted": 5, "modified": 0, "failed": 0}
        assert store.count() == 5

        again = store.upsert_many(documents)
        assert again["inserted"] == 0
        assert again["failed"] == 0
        assert store.count() == 5

    def test_upsert_many_on_empty_list(self, store) -> None:  # type: ignore[no-untyped-def]
        assert store.upsert_many([]) == {"inserted": 0, "modified": 0, "failed": 0}

    def test_language_tag_does_not_break_the_text_index(self, store) -> None:  # type: ignore[no-untyped-def]
        """A page declaring lang="en-us" must still be indexable."""
        document = normalize_page(
            url="https://pytest.local/lang",
            raw_html="<html><body><p>regional language page</p></body></html>",
            language="en-us",
        )
        assert store.upsert(document) is True
        assert store.search("regional")

    def test_full_text_search_ranks_titles_higher(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert(
            normalize_page(
                url="https://pytest.local/title-match",
                raw_html="<html><body><h1>Umbrella</h1><p>nothing else</p></body></html>",
                title="Umbrella",
            )
        )
        store.upsert(
            normalize_page(
                url="https://pytest.local/body-match",
                raw_html="<html><body><h1>Other</h1><p>an umbrella appears here</p></body></html>",
                title="Other",
            )
        )
        hits = store.search("umbrella")
        assert len(hits) == 2
        assert hits[0]["url"].endswith("/title-match")
        assert hits[0]["score"] > hits[1]["score"]

    def test_search_can_be_restricted_to_a_site(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert(self._document("/site-filter", text="findable widget"))
        assert store.search("findable", site="pytest.local")
        assert store.search("findable", site="other.local") == []

    def test_get_by_url_accepts_any_url_variant(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert(self._document("/variant"))
        assert store.get_by_url("https://PYTEST.local/variant/") is not None

    def test_missing_document_returns_none(self, store) -> None:  # type: ignore[no-untyped-def]
        assert store.get("0" * 40) is None

    def test_duplicate_detection_groups_identical_text(self, store) -> None:  # type: ignore[no-untyped-def]
        # Two different URLs serving byte-identical content, which is exactly
        # what "/" and "/index.html" do on a real site.
        body = "<html><body><h1>Same</h1><p>identical body</p></body></html>"
        for path in ("/dup-1", "/dup-2"):
            store.upsert(
                normalize_page(url=f"https://pytest.local{path}", raw_html=body)
            )
        store.upsert(self._document("/unique", text="a different body"))
        groups = store.duplicates()
        assert len(groups) == 1
        assert groups[0]["count"] == 2

    def test_stats_aggregate_per_site(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert_many([self._document(f"/stats/{i}") for i in range(3)])
        stats = store.stats()
        assert stats["totals"]["documents"] == 3
        assert stats["sites"][0]["site"] == "pytest.local"
        assert stats["sites"][0]["documents"] == 3
        assert stats["sites"][0]["words"] > 0

    def test_delete_site(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert_many([self._document(f"/del/{i}") for i in range(4)])
        assert store.delete_site("pytest.local") == 4
        assert store.count() == 0

    def test_server_version_is_reported(self, store) -> None:  # type: ignore[no-untyped-def]
        assert store.server_version()[0].isdigit()


# ---------------------------------------------------------------------------
# The pipeline, end to end (needs both databases)
# ---------------------------------------------------------------------------


def _postgres_available() -> bool:
    try:
        from sqlalchemy import text

        from src.storage.database import get_engine

        with get_engine().connect() as connection:
            connection.execute(text("select 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    not (_store_available() and _postgres_available()),
    reason="MongoDB and PostgreSQL are both required",
)
class TestPipeline:
    @pytest.fixture()
    def pipeline(self):  # type: ignore[no-untyped-def]
        from src.processing.pipeline import ProcessingPipeline

        store = DocumentStore(collection="pytest_pipeline")
        store.drop()
        yield ProcessingPipeline(store=store, batch_size=10)
        store.drop()

    def test_process_page_builds_a_valid_document(self, pipeline) -> None:  # type: ignore[no-untyped-def]
        document, clean_ms, normalize_ms = pipeline.process_page(
            {"url": "https://pytest.local/one", "raw_html": PAGE, "site": "pytest.local"}
        )
        assert document is not None
        validate_document(document)
        assert clean_ms > 0 and normalize_ms > 0

    def test_textless_page_is_skipped_not_failed(self, pipeline) -> None:  # type: ignore[no-untyped-def]
        document, _clean_ms, _normalize_ms = pipeline.process_page(
            {"url": "https://pytest.local/empty", "raw_html": "<html><body></body></html>"}
        )
        assert document is None

    def test_run_over_the_stored_crawl(self, pipeline) -> None:  # type: ignore[no-untyped-def]
        stats = pipeline.run(limit=5)
        if stats.pages_read == 0:
            pytest.skip("no crawled pages in PostgreSQL to process")

        assert stats.documents_written > 0
        assert stats.failed == 0
        assert stats.raw_bytes > stats.clean_bytes      # noise was removed
        assert 0 < stats.reduction_ratio < 1
        assert stats.pages_per_second > 0
        assert stats.as_dict()["stage_ms"]["clean"]["p50"] >= 0

    def test_rerun_is_idempotent(self, pipeline) -> None:  # type: ignore[no-untyped-def]
        first = pipeline.run(limit=5)
        if first.pages_read == 0:
            pytest.skip("no crawled pages in PostgreSQL to process")
        before = pipeline.store.count()

        second = pipeline.run(limit=5)
        assert second.inserted == 0                  # nothing new
        assert pipeline.store.count() == before      # no duplicates

    def test_dry_run_writes_nothing(self) -> None:
        from src.processing.pipeline import ProcessingPipeline

        store = DocumentStore(collection="pytest_dry_run")
        store.drop()
        stats = ProcessingPipeline(store=store, dry_run=True).run(limit=3)
        if stats.pages_read == 0:
            pytest.skip("no crawled pages in PostgreSQL to process")
        assert stats.documents_written > 0
        assert store.count() == 0
