"""Tests for the full-text search endpoints (Task 6).

Same approach as ``tests/test_api.py``: a fake search index is injected
through ``app.dependency_overrides``, and each test sends a real HTTP
request, so routing, authentication, scope checks, rate limiting, parameter
validation, error translation and serialisation are all exercised with no
cluster running.

What the fake records is the call it received. That is the point of most of
these tests: the HTTP contract is that ``?site=x&sort=recent&offset=20``
reaches the index as those arguments, and a test that only checked the
status code would not notice a parameter being dropped on the way.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_search_index_dep
from src.api.main import create_app
from src.api.rate_limit import reset_limiter
from src.api.security import _registry_for
from src.common.config import settings

READER_KEY = "test-reader-key"
ANALYST_KEY = "test-analyst-key"

KEY_SPEC = (
    f"reader:{READER_KEY}:raw,documents;"
    f"analyst:{ANALYST_KEY}:raw,documents,search,ask"
)

READER = {"X-API-Key": READER_KEY}
ANALYST = {"X-API-Key": ANALYST_KEY}


@pytest.fixture(autouse=True)
def api_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_auth_enabled", True)
    monkeypatch.setattr(settings, "api_keys", KEY_SPEC)
    monkeypatch.setattr(settings, "api_rate_limit", 1000)
    monkeypatch.setattr(settings, "api_search_rate_limit", 1000)
    monkeypatch.setattr(settings, "api_ask_rate_limit", 1000)
    monkeypatch.setattr(settings, "api_rate_window_seconds", 60)
    monkeypatch.setattr(settings, "api_rate_limit_backend", "memory")
    monkeypatch.setattr(settings, "api_page_size", 20)
    monkeypatch.setattr(settings, "api_max_page_size", 50)
    monkeypatch.setattr(settings, "api_docs_enabled", True)
    _registry_for.cache_clear()
    reset_limiter()


@pytest.fixture()
def client() -> Any:
    app = create_app()
    app.dependency_overrides[get_search_index_dep] = lambda: FakeSearchIndex()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def lenient_client() -> Any:
    """Returns the handled response instead of re-raising the exception.

    Starlette re-raises after the ``Exception`` handler has produced its
    response, so the default client never sees the 503 that an outage is
    translated into.
    """
    app = create_app()
    app.dependency_overrides[get_search_index_dep] = lambda: FakeSearchIndex()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------


class TestFullTextSearch:
    def test_hits_carry_their_highlights_and_facets(self, client: TestClient) -> None:
        body = client.get("/v1/fulltext?q=widgets", headers=ANALYST).json()

        assert body["total"] == 2
        assert body["returned"] == 2
        assert [hit["rank"] for hit in body["hits"]] == [1, 2]
        assert "<mark>widgets</mark>" in body["hits"][0]["snippet"]
        assert body["facets"]["sites"][0]["value"] == "example.com"

    def test_both_timings_are_reported(self, client: TestClient) -> None:
        body = client.get("/v1/fulltext?q=widgets", headers=ANALYST).json()
        assert body["took_ms"] >= 0
        assert body["elapsed_ms"] >= 0

    def test_the_site_specific_fields_survive_serialisation(
        self, client: TestClient
    ) -> None:
        """``fields`` has a different shape per website and the response
        model must not flatten that away."""
        body = client.get("/v1/fulltext?q=widgets", headers=ANALYST).json()
        assert body["hits"][0]["fields"] == {"price": "£51.77", "rating": "Three"}

    def test_an_empty_query_browses_the_index(self, client: TestClient) -> None:
        """Not a 422: the UI's initial view is a browse, and that is what an
        empty q with a sort means."""
        response = client.get("/v1/fulltext?sort=recent", headers=ANALYST)
        assert response.status_code == 200
        assert FakeSearchIndex.last_call["query"] == ""
        assert FakeSearchIndex.last_call["sort"] == "recent"

    def test_filters_reach_the_index(self, client: TestClient) -> None:
        client.get(
            "/v1/fulltext?q=widgets&site=example.com&language=en"
            "&min_words=100&enriched=true",
            headers=ANALYST,
        )
        call = FakeSearchIndex.last_call
        assert call["site"] == "example.com"
        assert call["language"] == "en"
        assert call["min_words"] == 100
        assert call["enriched"] is True

    def test_paging_reaches_the_index(self, client: TestClient) -> None:
        client.get("/v1/fulltext?q=widgets&limit=5&offset=10", headers=ANALYST)
        assert FakeSearchIndex.last_call["limit"] == 5
        assert FakeSearchIndex.last_call["offset"] == 10

    def test_the_page_size_is_capped(self, client: TestClient) -> None:
        """A listing must not be turnable into a full index dump."""
        client.get("/v1/fulltext?q=widgets&limit=5000", headers=ANALYST)
        assert FakeSearchIndex.last_call["limit"] == settings.api_max_page_size

    def test_highlights_and_facets_can_be_switched_off(
        self, client: TestClient
    ) -> None:
        client.get(
            "/v1/fulltext?q=widgets&highlight=false&facets=false", headers=ANALYST
        )
        assert FakeSearchIndex.last_call["highlight"] is False
        assert FakeSearchIndex.last_call["facets"] is False

    def test_an_unknown_sort_is_rejected_before_the_handler_runs(
        self, client: TestClient
    ) -> None:
        response = client.get("/v1/fulltext?q=widgets&sort=sideways", headers=ANALYST)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["type"] == "validation_error"
        assert error["problems"][0]["field"] == "sort"

    def test_an_over_long_query_is_rejected(self, client: TestClient) -> None:
        response = client.get("/v1/fulltext?q=" + "x" * 501, headers=ANALYST)
        assert response.status_code == 422

    def test_a_negative_offset_is_rejected(self, client: TestClient) -> None:
        assert (
            client.get("/v1/fulltext?q=a&offset=-1", headers=ANALYST).status_code == 422
        )

    @pytest.mark.parametrize("sort", ["relevance", "recent", "oldest", "words", "title"])
    def test_every_documented_sort_is_accepted(
        self, client: TestClient, sort: str
    ) -> None:
        response = client.get(f"/v1/fulltext?q=widgets&sort={sort}", headers=ANALYST)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Suggestions, sites, similar
# ---------------------------------------------------------------------------


class TestSuggest:
    def test_suggestions_are_returned_for_a_prefix(self, client: TestClient) -> None:
        body = client.get("/v1/fulltext/suggest?q=wid", headers=ANALYST).json()
        assert body[0]["text"] == "Widgets explained"
        assert body[0]["doc_id"] == "doc-1"

    def test_the_limit_reaches_the_index(self, client: TestClient) -> None:
        client.get("/v1/fulltext/suggest?q=wid&limit=3", headers=ANALYST)
        assert FakeSearchIndex.last_suggest == {"prefix": "wid", "limit": 3}

    def test_a_one_character_prefix_is_rejected(self, client: TestClient) -> None:
        """Enforced at the edge: a one-character prefix matches most of the
        index and would be requested on the first keystroke of every search."""
        assert (
            client.get("/v1/fulltext/suggest?q=w", headers=ANALYST).status_code == 422
        )

    def test_a_missing_prefix_is_rejected(self, client: TestClient) -> None:
        assert client.get("/v1/fulltext/suggest", headers=ANALYST).status_code == 422


class TestSites:
    def test_sites_come_back_with_their_counts(self, client: TestClient) -> None:
        body = client.get("/v1/fulltext/sites", headers=ANALYST).json()
        assert body == [{"value": "example.com", "count": 2}]


class TestSimilar:
    def test_similar_documents_are_returned(self, client: TestClient) -> None:
        body = client.get("/v1/fulltext/similar/doc-1", headers=ANALYST).json()
        assert body[0]["doc_id"] == "doc-2"

    def test_an_unindexed_document_is_a_404(self, client: TestClient) -> None:
        response = client.get("/v1/fulltext/similar/nope", headers=ANALYST)
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "not_found"


# ---------------------------------------------------------------------------
# Authentication, scopes, limits and outages
# ---------------------------------------------------------------------------


class TestAccessControl:
    def test_a_missing_key_is_rejected(self, client: TestClient) -> None:
        assert client.get("/v1/fulltext?q=widgets").status_code == 401

    def test_a_key_without_the_search_scope_is_forbidden(
        self, client: TestClient
    ) -> None:
        """Full-text search is the ``search`` scope, like the other search
        endpoints - a documents-only key cannot reach it."""
        response = client.get("/v1/fulltext?q=widgets", headers=READER)
        assert response.status_code == 403

    def test_every_route_in_the_group_is_protected(self, client: TestClient) -> None:
        for path in (
            "/v1/fulltext?q=widgets",
            "/v1/fulltext/suggest?q=wid",
            "/v1/fulltext/sites",
            "/v1/fulltext/similar/doc-1",
        ):
            assert client.get(path).status_code == 401, path

    def test_the_search_rate_limit_bucket_is_used(self, client: TestClient) -> None:
        """These endpoints are called on every keystroke, so they belong in
        the tighter search budget rather than the global one."""
        response = client.get("/v1/fulltext?q=widgets", headers=ANALYST)
        assert response.headers["X-RateLimit-Bucket"] == "search"

    def test_the_endpoints_are_documented(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        for path in (
            "/v1/fulltext",
            "/v1/fulltext/suggest",
            "/v1/fulltext/sites",
            "/v1/fulltext/similar/{doc_id}",
        ):
            assert path in paths, path


class TestOutage:
    def test_an_unreachable_cluster_becomes_a_503(
        self, lenient_client: TestClient
    ) -> None:
        """The request was fine, the dependency was not."""
        from src.search.index import SearchUnavailable

        def failing_index() -> Any:
            raise SearchUnavailable("Elasticsearch is not reachable at localhost:9200")

        lenient_client.app.dependency_overrides[get_search_index_dep] = failing_index
        response = lenient_client.get("/v1/fulltext?q=widgets", headers=ANALYST)

        assert response.status_code == 503
        assert response.headers["Retry-After"] == "10"
        error = response.json()["error"]
        assert error["type"] == "search_index_unavailable"
        assert "Elasticsearch" in error["message"]


# ---------------------------------------------------------------------------
# The fake
# ---------------------------------------------------------------------------


def _hit(rank: int, doc_id: str, score: float) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "url": f"https://example.com/{doc_id}",
        "site": "example.com",
        "path": f"/{doc_id}",
        "title": "Widgets explained",
        "description": "All about widgets",
        "language": "en",
        "keywords": ["widgets"],
        "headings": ["Widgets"],
        "word_count": 120,
        "reading_time_sec": 33,
        "processed_at": "2026-11-18T10:00:00+00:00",
        "fetched_at": "2026-11-18T09:00:00+00:00",
        "http_status": 200,
        "depth": 1,
        "summary": "A page about widgets.",
        "enriched": True,
        "quality_score": 0.9,
        "fields": {"price": "£51.77", "rating": "Three"},
        "rank": rank,
        "score": score,
        "relevance": round(score / 9.0, 4),
        "highlights": {"text": [f"a page about <mark>widgets</mark> ({doc_id})"]},
        "snippet": f"a page about <mark>widgets</mark> ({doc_id})",
    }


class FakeSearchIndex:
    """Stands in for ``SearchIndex`` - no cluster, predictable data."""

    last_call: dict[str, Any] = {}
    last_suggest: dict[str, Any] = {}

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        FakeSearchIndex.last_call = {"query": query, **kwargs}
        return {
            "query": query,
            "sort": kwargs.get("sort", "relevance"),
            "total": 2,
            "returned": 2,
            "hits": [_hit(1, "doc-1", 9.0), _hit(2, "doc-2", 3.0)],
            "facets": {
                "sites": [{"value": "example.com", "count": 2}],
                "languages": [{"value": "en", "count": 2}],
                "keywords": [{"value": "widgets", "count": 2}],
                "enriched": [{"value": "true", "count": 2}],
            },
            "took_ms": 7.0,
            "elapsed_ms": 9.5,
            "notes": [],
        }

    def suggest(self, prefix: str, limit: int = 8) -> list[dict[str, Any]]:
        FakeSearchIndex.last_suggest = {"prefix": prefix, "limit": limit}
        return [
            {
                "text": "Widgets explained",
                "doc_id": "doc-1",
                "url": "https://example.com/doc-1",
                "site": "example.com",
            }
        ]

    def sites(self) -> list[dict[str, Any]]:
        return [{"value": "example.com", "count": 2}]

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return {"doc_id": doc_id} if doc_id in ("doc-1", "doc-2") else None

    def similar(self, doc_id: str, limit: int = 5) -> list[dict[str, Any]]:
        return [_hit(1, "doc-2", 4.0)]
