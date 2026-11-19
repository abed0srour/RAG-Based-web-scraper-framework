"""Tests for the Elasticsearch full-text layer (Task 6).

Two halves, for the same reason as the RAG tests:

* **The query builders and the document projection are tested directly.**
  They are pure functions from parameters to a dictionary, so the assertions
  are about the query that *would* be sent - the search equivalent of
  checking the generated SQL rather than the rows it returns. No cluster is
  needed, which is what keeps this half runnable anywhere.

* **The index itself is tested against a real cluster when one is running**,
  and skipped when it is not. Analysis, scoring, highlighting and facets are
  Elasticsearch's behaviour, not ours; asserting on them against a fake
  would only test the fake. These tests use their own index name and delete
  it afterwards, so they never touch the development corpus.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.common.config import settings
from src.search.index import (
    INDEX_MAPPINGS,
    INDEX_SETTINGS,
    RESULT_SOURCE,
    SEARCH_FIELDS,
    SORTS,
    SearchIndex,
    build_aggregations,
    build_filters,
    build_highlight,
    build_query,
    build_sort,
    flatten_fields,
    to_index_document,
)

# ---------------------------------------------------------------------------
# Sample documents, shaped like what Task 3 stores in MongoDB
# ---------------------------------------------------------------------------

BOOK = {
    "doc_id": "book-1",
    "url": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
    "site": "books.toscrape.com",
    "path": "/catalogue/a-light-in-the-attic_1000/index.html",
    "title": "A Light in the Attic | Books to Scrape",
    "description": "It's hard to imagine a world without A Light in the Attic.",
    "language": "en",
    "headings": ["A Light in the Attic", "Product Description"],
    "text": (
        "A Light in the Attic is a collection of poems and drawings for "
        "children by Shel Silverstein. It's hard to imagine a world without "
        "it. The poems are funny and the drawings are strange."
    ),
    "keywords": ["light", "attic", "poems", "children"],
    "fields": {"price": "£51.77", "rating": "Three", "availability": "In stock"},
    "stats": {"word_count": 32, "char_count": 210, "reading_time_sec": 9},
    "source": {
        "raw_page_id": 11,
        "job_id": 7,
        "depth": 1,
        "http_status": 200,
        "fetched_at": "2026-11-18T09:00:00+00:00",
    },
    "processing": {"processed_at": "2026-11-18T10:00:00+00:00"},
    "enrichment": {
        "summary": "A children's poetry collection by Shel Silverstein.",
        "quality": {"score_before": 0.6, "score_after": 0.9},
    },
}

QUOTE = {
    "doc_id": "quote-1",
    "url": "https://quotes.toscrape.com/page/1/",
    "site": "quotes.toscrape.com",
    "path": "/page/1/",
    "title": "Quotes to Scrape",
    "description": None,
    "language": "en",
    "headings": ["Quotes to Scrape"],
    "text": (
        "The world as we have created it is a process of our thinking. "
        "It cannot be changed without changing our thinking."
    ),
    "keywords": ["world", "thinking", "change"],
    "fields": {"author": "Albert Einstein", "tags": ["change", "thinking"]},
    "stats": {"word_count": 21, "char_count": 130, "reading_time_sec": 6},
    "source": {"raw_page_id": 2, "job_id": 7, "depth": 0, "http_status": 200},
    "processing": {"processed_at": "2026-11-18T11:00:00+00:00"},
}


# ---------------------------------------------------------------------------
# The index definition
# ---------------------------------------------------------------------------


class TestIndexDefinition:
    def test_the_mapping_is_strict(self) -> None:
        """A new document field must be added here deliberately, not mapped
        by accident - which is what keeps the field count bounded."""
        assert INDEX_MAPPINGS["dynamic"] == "strict"

    def test_the_site_specific_fields_are_stored_but_not_indexed(self) -> None:
        """The mapping-explosion guard: ``fields`` differs per website, so it
        is returned with every hit and never given index fields of its own."""
        assert INDEX_MAPPINGS["properties"]["fields"] == {
            "type": "object",
            "enabled": False,
        }
        # ...and its searchable flattening is one ordinary text field.
        assert INDEX_MAPPINGS["properties"]["fields_text"]["type"] == "text"

    def test_the_text_fields_use_the_english_analyzer(self) -> None:
        for field in ("title", "description", "headings", "text", "summary"):
            assert INDEX_MAPPINGS["properties"][field]["analyzer"] == "content"

        content = INDEX_SETTINGS["analysis"]["analyzer"]["content"]
        assert "english_stemmer" in content["filter"]
        assert "english_stop" in content["filter"]
        assert "lowercase" in content["filter"]

    def test_suggestions_are_n_grammed_at_index_time_only(self) -> None:
        """Edge n-grams on the index side make an as-you-type query a term
        lookup; applying them on the query side too would match far more
        than the user typed."""
        suggest = INDEX_MAPPINGS["properties"]["title"]["fields"]["suggest"]
        assert suggest["analyzer"] == "suggest_index"
        assert suggest["search_analyzer"] == "suggest_search"

        index_filters = INDEX_SETTINGS["analysis"]["analyzer"]["suggest_index"]["filter"]
        search_filters = INDEX_SETTINGS["analysis"]["analyzer"]["suggest_search"]["filter"]
        assert "edge_ngrams" in index_filters
        assert "edge_ngrams" not in search_filters

    def test_a_single_node_index_asks_for_no_replica(self) -> None:
        """An unassigned replica is what turns a one-node cluster yellow."""
        assert INDEX_SETTINGS["number_of_replicas"] == 0

    def test_keywords_are_both_a_facet_and_searchable(self) -> None:
        keywords = INDEX_MAPPINGS["properties"]["keywords"]
        assert keywords["type"] == "keyword"              # exact, for facets
        assert keywords["fields"]["text"]["type"] == "text"  # analysed, for search


# ---------------------------------------------------------------------------
# Projecting a stored document onto the index
# ---------------------------------------------------------------------------


class TestDocumentProjection:
    def test_the_flat_shape_is_produced_from_the_nested_document(self) -> None:
        indexed = to_index_document(BOOK)

        assert indexed["doc_id"] == "book-1"
        assert indexed["site"] == "books.toscrape.com"
        # Nested in MongoDB, flat in the index.
        assert indexed["word_count"] == 32
        assert indexed["reading_time_sec"] == 9
        assert indexed["http_status"] == 200
        assert indexed["depth"] == 1
        assert indexed["processed_at"] == "2026-11-18T10:00:00+00:00"
        assert indexed["fetched_at"] == "2026-11-18T09:00:00+00:00"

    def test_enrichment_becomes_a_searchable_summary_and_a_flag(self) -> None:
        indexed = to_index_document(BOOK)
        assert indexed["summary"].startswith("A children's poetry")
        assert indexed["enriched"] is True
        assert indexed["quality_score"] == 0.9

    def test_an_unenriched_document_is_flagged_as_such(self) -> None:
        indexed = to_index_document(QUOTE)
        assert indexed["enriched"] is False
        assert "summary" not in indexed
        assert "quality_score" not in indexed

    def test_every_projected_field_exists_in_the_mapping(self) -> None:
        """With ``dynamic: strict`` a field the mapping does not declare is
        rejected by the cluster, so this is the test that catches it before
        the cluster does."""
        allowed = set(INDEX_MAPPINGS["properties"])
        for document in (BOOK, QUOTE):
            assert set(to_index_document(document)) <= allowed

    def test_nulls_and_empty_strings_are_dropped(self) -> None:
        """``description`` is None on the quotes page: an explicit null costs
        a mapping check and means nothing, and "" is rejected for a date."""
        assert "description" not in to_index_document(QUOTE)

    def test_the_text_is_truncated_to_the_configured_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "elastic_max_text_chars", 20)
        indexed = to_index_document({**BOOK, "text": "x" * 500})
        assert len(indexed["text"]) == 20

    def test_the_full_text_is_never_returned_with_a_hit(self) -> None:
        """The highlighter returns the part that matched; a result page does
        not carry a megabyte of prose nobody displays."""
        assert "text" not in RESULT_SOURCE
        assert "doc_id" in RESULT_SOURCE


class TestFieldFlattening:
    def test_keys_and_values_both_become_searchable(self) -> None:
        flat = flatten_fields(BOOK["fields"])
        assert "price" in flat and "51.77" in flat
        assert "rating Three" in flat

    def test_nested_and_repeated_values_are_walked(self) -> None:
        flat = flatten_fields(QUOTE["fields"])
        assert "author Albert Einstein" in flat
        # A list contributes each of its values under the same key.
        assert flat.count("tags") == 2

    def test_missing_fields_flatten_to_nothing(self) -> None:
        assert flatten_fields(None) == ""
        assert flatten_fields({}) == ""

    def test_the_result_is_bounded(self) -> None:
        flat = flatten_fields({f"k{i}": "value" for i in range(500)}, limit=100)
        assert len(flat) <= 100


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


class TestQueryConstruction:
    def test_a_query_searches_every_weighted_field(self) -> None:
        query = build_query("light attic", [])
        multi = query["bool"]["must"][0]["multi_match"]
        assert multi["fields"] == list(SEARCH_FIELDS)
        assert multi["type"] == "best_fields"

    def test_the_title_outweighs_the_body(self) -> None:
        """A title hit is evidence about the document; a body hit is evidence
        about one sentence in it."""
        boosts = {
            field.split("^")[0]: float(field.split("^")[1]) if "^" in field else 1.0
            for field in SEARCH_FIELDS
        }
        assert boosts["title"] > boosts["text"]
        assert boosts["headings"] > boosts["text"]

    def test_an_exact_phrase_is_boosted_but_not_required(self) -> None:
        query = build_query("light in the attic", [])
        should = query["bool"]["should"]
        assert any("match_phrase" in clause for clause in should)
        # A should clause cannot exclude anything.
        assert len(query["bool"]["must"]) == 1

    def test_an_empty_query_browses_instead_of_failing(self) -> None:
        query = build_query("   ", [{"term": {"site": "x"}}])
        assert query["bool"]["must"] == [{"match_all": {}}]
        assert query["bool"]["filter"] == [{"term": {"site": "x"}}]

    def test_filters_do_not_contribute_to_the_score(self) -> None:
        """They are yes/no questions, so they go in ``filter``, where
        Elasticsearch can cache them."""
        filters = build_filters(site="example.com", min_words=100)
        query = build_query("widgets", filters)
        assert query["bool"]["filter"] == filters

    def test_site_filters_are_normalised_like_the_stored_value(self) -> None:
        assert build_filters(site="WWW.Example.COM") == [
            {"term": {"site": "example.com"}}
        ]

    def test_each_filter_has_its_clause(self) -> None:
        filters = build_filters(
            site="a.com", language="en", min_words=50, enriched=True
        )
        assert {"term": {"language": "en"}} in filters
        assert {"range": {"word_count": {"gte": 50}}} in filters
        assert {"term": {"enriched": True}} in filters

    def test_filtering_for_unenriched_documents_is_not_the_same_as_no_filter(
        self,
    ) -> None:
        assert build_filters(enriched=False) == [{"term": {"enriched": False}}]
        assert build_filters(enriched=None) == []

    @pytest.mark.parametrize("sort", SORTS)
    def test_every_sort_is_translatable(self, sort: str) -> None:
        assert build_sort(sort)

    def test_relevance_has_a_deterministic_tie_break(self) -> None:
        """Two equally scored hits must not swap places between identical
        requests."""
        assert build_sort("relevance") == ["_score", {"doc_id": {"order": "asc"}}]

    def test_a_date_sort_still_carries_the_score(self) -> None:
        assert build_sort("recent")[0] == {"processed_at": {"order": "desc"}}
        assert "_score" in build_sort("recent")

    def test_highlighting_marks_the_terms_and_tolerates_a_title_only_match(
        self,
    ) -> None:
        highlight = build_highlight()
        assert highlight["pre_tags"] == ["<mark>"]
        # Without these two a hit that matched only the title gets no snippet
        # and the result card looks empty.
        assert highlight["require_field_match"] is False
        assert highlight["no_match_size"] > 0

    def test_facets_cover_the_filters_the_ui_offers(self) -> None:
        aggregations = build_aggregations()
        assert set(aggregations) == {"sites", "languages", "keywords", "enriched"}
        assert aggregations["sites"]["terms"]["field"] == "site"


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------


class TestResponseFormatting:
    RESPONSE: dict[str, Any] = {
        "took": 7,
        "hits": {
            "total": {"value": 2, "relation": "eq"},
            "max_score": 8.0,
            "hits": [
                {
                    "_id": "book-1",
                    "_score": 8.0,
                    "_source": {"doc_id": "book-1", "url": "u1", "title": "t1"},
                    "highlight": {"text": ["a <mark>light</mark> in the attic"]},
                },
                {
                    "_id": "quote-1",
                    "_score": 2.0,
                    "_source": {
                        "doc_id": "quote-1",
                        "url": "u2",
                        "title": "t2",
                        "description": "a fallback description",
                    },
                },
            ],
        },
        "aggregations": {
            "sites": {"buckets": [{"key": "books.toscrape.com", "doc_count": 1}]}
        },
    }

    def _formatted(self) -> dict[str, Any]:
        return SearchIndex._format(
            self.RESPONSE, query="light", elapsed=12.5, sort="relevance"
        )

    def test_the_total_is_the_match_count_not_the_page_size(self) -> None:
        formatted = self._formatted()
        assert formatted["total"] == 2
        assert formatted["returned"] == 2

    def test_hits_are_ranked_from_one(self) -> None:
        assert [hit["rank"] for hit in self._formatted()["hits"]] == [1, 2]

    def test_relevance_is_relative_to_the_best_hit(self) -> None:
        """Elasticsearch scores are unbounded and only comparable within one
        result set, so the UI gets an explicit 0..1 fraction."""
        hits = self._formatted()["hits"]
        assert hits[0]["relevance"] == 1.0
        assert hits[1]["relevance"] == 0.25

    def test_a_highlighted_fragment_becomes_the_snippet(self) -> None:
        assert "<mark>light</mark>" in self._formatted()["hits"][0]["snippet"]

    def test_a_hit_with_no_highlight_still_gets_a_snippet(self) -> None:
        """An empty result card reads as a bug, so the description is used."""
        assert self._formatted()["hits"][1]["snippet"] == "a fallback description"

    def test_facets_are_flattened_to_value_and_count(self) -> None:
        assert self._formatted()["facets"]["sites"] == [
            {"value": "books.toscrape.com", "count": 1}
        ]

    def test_both_timings_are_reported(self) -> None:
        """A gap between the cluster's time and the round trip is a network
        problem, not a slow query."""
        formatted = self._formatted()
        assert formatted["took_ms"] == 7.0
        assert formatted["elapsed_ms"] == 12.5

    def test_an_empty_result_set_formats_without_a_max_score(self) -> None:
        empty = {"took": 1, "hits": {"total": {"value": 0}, "max_score": None, "hits": []}}
        formatted = SearchIndex._format(empty, query="x", elapsed=1.0, sort="relevance")
        assert formatted["hits"] == []
        assert formatted["total"] == 0


class TestSearchValidation:
    def test_an_unknown_sort_is_refused_before_the_cluster_is_touched(self) -> None:
        index = SearchIndex(index="unused", client=object())
        with pytest.raises(ValueError, match="unknown sort"):
            index.search("widgets", sort="sideways")


# ---------------------------------------------------------------------------
# Against a real cluster
# ---------------------------------------------------------------------------

TEST_INDEX = "test_scraped_documents"

#: Whether the cluster answered, probed once per session. Each failed probe
#: costs the connection timeout, and eighteen of them turned a two-second
#: skip into a two-minute one, so the answer is cached.
_CLUSTER_UP: bool | None = None


def _cluster_available() -> bool:
    global _CLUSTER_UP
    if _CLUSTER_UP is None:
        _CLUSTER_UP = SearchIndex(index=TEST_INDEX).ping()
    return _CLUSTER_UP


def _cluster_or_skip() -> SearchIndex:
    if not _cluster_available():
        pytest.skip("Elasticsearch is not reachable")
    return SearchIndex(index=TEST_INDEX)


@pytest.fixture()
def live_index() -> Any:
    """A throwaway index holding the two sample documents.

    Its own name, deleted afterwards: an integration test must not leave
    anything behind in the development corpus.
    """
    index = _cluster_or_skip()
    index.delete_index()
    index.ensure_index()
    index.index_documents([BOOK, QUOTE])
    yield index
    index.delete_index()


@pytest.mark.integration
class TestAgainstACluster:
    def test_the_index_is_created_with_its_mapping(self, live_index: SearchIndex) -> None:
        mapping = live_index.client.indices.get_mapping(index=TEST_INDEX)
        properties = mapping[TEST_INDEX]["mappings"]["properties"]
        assert properties["site"]["type"] == "keyword"
        assert properties["text"]["analyzer"] == "content"

    def test_both_documents_are_searchable(self, live_index: SearchIndex) -> None:
        assert live_index.count() == 2

    def test_stemming_matches_a_different_word_form(
        self, live_index: SearchIndex
    ) -> None:
        """"drawing" in the text is found by searching for "draw" - which is
        the thing a substring search over MongoDB cannot do."""
        result = live_index.search("draw")
        assert result["total"] >= 1
        assert result["hits"][0]["doc_id"] == "book-1"

    def test_the_matched_terms_come_back_marked(
        self, live_index: SearchIndex
    ) -> None:
        result = live_index.search("Silverstein")
        assert "<mark>" in result["hits"][0]["snippet"]

    def test_the_title_outranks_the_body_in_practice(
        self, live_index: SearchIndex
    ) -> None:
        result = live_index.search("quotes")
        assert result["hits"][0]["doc_id"] == "quote-1"

    def test_the_site_specific_fields_are_searchable_and_returned(
        self, live_index: SearchIndex
    ) -> None:
        """A price lives in a sub-document whose shape differs per site, is
        never an index field of its own, and is still findable."""
        result = live_index.search("Einstein")
        assert result["total"] == 1
        hit = result["hits"][0]
        assert hit["doc_id"] == "quote-1"
        assert hit["fields"]["author"] == "Albert Einstein"

    def test_facets_count_the_matching_set(self, live_index: SearchIndex) -> None:
        result = live_index.search("")
        sites = {row["value"]: row["count"] for row in result["facets"]["sites"]}
        assert sites == {"books.toscrape.com": 1, "quotes.toscrape.com": 1}

    def test_a_site_filter_narrows_the_results(self, live_index: SearchIndex) -> None:
        result = live_index.search("", site="quotes.toscrape.com")
        assert result["total"] == 1

    def test_a_length_filter_narrows_the_results(self, live_index: SearchIndex) -> None:
        assert live_index.search("", min_words=25)["total"] == 1

    def test_sorting_by_date_ignores_the_score(self, live_index: SearchIndex) -> None:
        recent = live_index.search("", sort="recent")["hits"]
        assert [hit["doc_id"] for hit in recent] == ["quote-1", "book-1"]

    def test_paging_walks_the_result_set(self, live_index: SearchIndex) -> None:
        first = live_index.search("", sort="title", limit=1, offset=0)
        second = live_index.search("", sort="title", limit=1, offset=1)
        assert first["total"] == second["total"] == 2
        assert first["hits"][0]["doc_id"] != second["hits"][0]["doc_id"]

    def test_suggestions_match_a_typed_prefix(self, live_index: SearchIndex) -> None:
        suggestions = live_index.suggest("lig")
        assert any("Light" in row["text"] for row in suggestions)

    def test_a_one_character_prefix_suggests_nothing(
        self, live_index: SearchIndex
    ) -> None:
        assert live_index.suggest("l") == []

    def test_reindexing_replaces_rather_than_duplicates(
        self, live_index: SearchIndex
    ) -> None:
        """The frontier delivers a URL at least once, so the write path has
        to be idempotent."""
        live_index.index_documents([BOOK, QUOTE])
        assert live_index.count() == 2

    def test_a_document_can_be_removed(self, live_index: SearchIndex) -> None:
        assert live_index.delete_document("book-1") is True
        live_index.refresh()
        assert live_index.count() == 1

    def test_similar_documents_exclude_the_document_itself(
        self, live_index: SearchIndex
    ) -> None:
        assert all(
            hit["doc_id"] != "book-1" for hit in live_index.similar("book-1", limit=5)
        )

    def test_health_reports_the_index(self, live_index: SearchIndex) -> None:
        health = live_index.health()
        assert health["index"] == TEST_INDEX
        assert health["index_exists"] is True
        assert health["documents"] == 2

    def test_searching_a_missing_index_is_empty_not_an_error(self) -> None:
        """The UI shows "nothing indexed yet" rather than an error page."""
        index = _cluster_or_skip()
        index.index_name = "test_index_that_does_not_exist_" + TEST_INDEX
        result = index.search("anything")
        assert result["total"] == 0
        assert result["notes"]
