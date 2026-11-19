"""Tests for the REST API (Task 5).

The whole suite runs with **no container started**. That is the point of the
design: every data access goes through a FastAPI dependency
(``src/api/deps.py``), so the tests inject fake stores with
``app.dependency_overrides`` and exercise the real HTTP path - routing,
authentication, scope checks, rate limiting, parameter validation, error
translation and response serialisation - against predictable data.

What is deliberately *not* faked: the application itself. These are not unit
tests of the handler functions; each one sends an HTTP request through
Starlette's test client and asserts on the status code, the headers and the
JSON body, which is what "test API responses" means for this deliverable.

The fakes are at the bottom of the file. They implement only the methods the
routers call, and they return the same shapes the real repository and store
return, so a change to those shapes breaks a test here rather than a client.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.deps import (
    get_document_store,
    get_repository,
    get_retriever_dep,
    get_vector_store,
)
from src.api.main import create_app
from src.api.rate_limit import reset_limiter
from src.api.security import _registry_for
from src.common.config import settings

READER_KEY = "test-reader-key"
ANALYST_KEY = "test-analyst-key"
NO_SCOPE_KEY = "test-empty-key"

KEY_SPEC = (
    f"reader:{READER_KEY}:raw,documents;"
    f"analyst:{ANALYST_KEY}:raw,documents,search,ask;"
    f"nobody:{NO_SCOPE_KEY}:admin"
)

READER = {"X-API-Key": READER_KEY}
ANALYST = {"X-API-Key": ANALYST_KEY}

NOW = datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def api_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the settings every test depends on.

    The real ``.env`` must not decide whether authentication is on or what
    the rate limits are, otherwise the suite passes or fails depending on the
    machine it runs on.
    """
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
    # A fresh in-memory limiter per test: counters must not leak between tests.
    reset_limiter()


@pytest.fixture()
def client() -> Any:
    """A test client whose data layer is entirely fake."""
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: FakeRepository
    app.dependency_overrides[get_document_store] = lambda: FakeStore()
    app.dependency_overrides[get_retriever_dep] = lambda: FakeRetriever()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def lenient_client() -> Any:
    """Same client, but it returns the handled response instead of re-raising.

    Starlette's ``ServerErrorMiddleware`` re-raises an exception after the
    ``Exception`` handler has produced its response, so the default test
    client (``raise_server_exceptions=True``) never gets to see the 503 that
    a dependency outage is translated into. The outage tests need the
    response, not the traceback.
    """
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: FakeRepository
    app.dependency_overrides[get_document_store] = lambda: FakeStore()
    app.dependency_overrides[get_retriever_dep] = lambda: FakeRetriever()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------


class TestServiceEndpoints:
    def test_health_needs_no_credentials(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == settings.api_version
        assert body["uptime_seconds"] >= 0

    def test_metrics_are_exposed_in_prometheus_format(self, client: TestClient) -> None:
        client.get("/health")
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "api_requests_total" in response.text

    def test_openapi_schema_documents_every_route(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        for path in (
            "/v1/raw/pages",
            "/v1/raw/pages/{page_id}",
            "/v1/documents",
            "/v1/documents/{doc_id}",
            "/v1/search",
            "/v1/ask",
            "/health",
        ):
            assert path in paths, f"{path} is missing from the OpenAPI schema"

    def test_root_points_at_the_documentation(self, client: TestClient) -> None:
        body = client.get("/").json()
        assert body["documentation"] == "/docs"
        assert body["endpoints"]["search"] == "/v1/search"

    def test_every_response_carries_a_request_id(self, client: TestClient) -> None:
        response = client.get("/health")
        assert len(response.headers["X-Request-ID"]) == 12
        assert float(response.headers["X-Response-Time-Ms"]) >= 0

    def test_a_supplied_request_id_is_echoed_back(self, client: TestClient) -> None:
        response = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
        assert response.headers["X-Request-ID"] == "trace-abc-123"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_missing_key_is_rejected_with_401(self, client: TestClient) -> None:
        response = client.get("/v1/raw/pages")
        assert response.status_code == 401
        error = response.json()["error"]
        assert error["type"] == "unauthenticated"
        assert error["path"] == "/v1/raw/pages"
        # The response must say how to authenticate, not only that it failed.
        assert "WWW-Authenticate" in response.headers

    def test_unknown_key_is_rejected_with_401(self, client: TestClient) -> None:
        response = client.get("/v1/raw/pages", headers={"X-API-Key": "not-a-key"})
        assert response.status_code == 401
        assert response.json()["error"]["message"] == "invalid API key"

    def test_bearer_token_is_accepted(self, client: TestClient) -> None:
        response = client.get(
            "/v1/raw/pages", headers={"Authorization": f"Bearer {READER_KEY}"}
        )
        assert response.status_code == 200

    def test_valid_key_without_the_scope_is_forbidden(self, client: TestClient) -> None:
        """A key is authenticated but still not authorised: 403, not 401."""
        response = client.get("/v1/raw/pages", headers={"X-API-Key": NO_SCOPE_KEY})
        assert response.status_code == 403
        assert response.json()["error"]["type"] == "forbidden"
        assert "'raw' scope" in response.json()["error"]["message"]

    def test_search_scope_is_separate_from_read_scopes(self, client: TestClient) -> None:
        """The reader key may read the corpus but may not spend retrieval time."""
        assert client.get("/v1/documents", headers=READER).status_code == 200
        assert client.get("/v1/search?q=life", headers=READER).status_code == 403
        assert client.get("/v1/search?q=life", headers=ANALYST).status_code == 200

    def test_ask_scope_is_separate_from_search_scope(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            settings, "api_keys", f"searcher:{READER_KEY}:search", raising=True
        )
        _registry_for.cache_clear()
        assert client.get("/v1/search?q=life", headers=READER).status_code == 200
        response = client.post("/v1/ask", headers=READER, json={"question": "why?"})
        assert response.status_code == 403

    def test_me_reports_the_scopes_of_the_calling_key(self, client: TestClient) -> None:
        body = client.get("/v1/me", headers=ANALYST).json()
        assert body["name"] == "analyst"
        assert body["authenticated"] is True
        assert body["scopes"] == ["ask", "documents", "raw", "search"]
        assert body["rate_limit"]["backend"] == "memory"
        # The secret itself must never appear in a response.
        assert ANALYST_KEY not in client.get("/v1/me", headers=ANALYST).text

    def test_auth_can_be_disabled_for_local_development(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "api_auth_enabled", False)
        response = client.get("/v1/raw/pages")
        assert response.status_code == 200
        assert client.get("/v1/me").json()["authenticated"] is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_headers_describe_the_remaining_allowance(self, client: TestClient) -> None:
        response = client.get("/v1/raw/pages", headers=READER)
        assert response.headers["X-RateLimit-Limit"] == "1000"
        assert response.headers["X-RateLimit-Remaining"] == "999"
        assert int(response.headers["X-RateLimit-Reset"]) > 0

    def test_the_allowance_decreases_per_request(self, client: TestClient) -> None:
        first = client.get("/v1/raw/pages", headers=READER)
        second = client.get("/v1/raw/pages", headers=READER)
        assert int(first.headers["X-RateLimit-Remaining"]) == 999
        assert int(second.headers["X-RateLimit-Remaining"]) == 998

    def test_exceeding_the_limit_returns_429_with_retry_after(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "api_rate_limit", 3)

        statuses = [
            client.get("/v1/raw/pages", headers=READER).status_code for _ in range(5)
        ]
        assert statuses == [200, 200, 200, 429, 429]

        refused = client.get("/v1/raw/pages", headers=READER)
        assert refused.status_code == 429
        assert int(refused.headers["Retry-After"]) >= 1
        error = refused.json()["error"]
        assert error["type"] == "rate_limited"
        assert "rate limit exceeded" in error["message"]

    def test_limits_are_counted_per_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One client exhausting its window must not affect another."""
        monkeypatch.setattr(settings, "api_rate_limit", 2)

        assert client.get("/v1/raw/pages", headers=READER).status_code == 200
        assert client.get("/v1/raw/pages", headers=READER).status_code == 200
        assert client.get("/v1/raw/pages", headers=READER).status_code == 429
        # A different key still has its full allowance.
        assert client.get("/v1/raw/pages", headers=ANALYST).status_code == 200

    def test_search_has_its_own_tighter_window(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The expensive bucket runs out while the global one still has room."""
        monkeypatch.setattr(settings, "api_rate_limit", 100)
        monkeypatch.setattr(settings, "api_search_rate_limit", 2)

        assert client.get("/v1/search?q=life", headers=ANALYST).status_code == 200
        second = client.get("/v1/search?q=life", headers=ANALYST)
        assert second.status_code == 200
        assert second.headers["X-RateLimit-Bucket"] == "search"

        refused = client.get("/v1/search?q=life", headers=ANALYST)
        assert refused.status_code == 429
        assert "search request(s)" in refused.json()["error"]["message"]
        # The cheap endpoints are unaffected by the search bucket.
        assert client.get("/v1/raw/pages", headers=ANALYST).status_code == 200

    def test_generation_has_the_tightest_window(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "api_ask_rate_limit", 1)
        payload = {"question": "what is this corpus about?"}

        assert client.post("/v1/ask", headers=ANALYST, json=payload).status_code == 200
        refused = client.post("/v1/ask", headers=ANALYST, json=payload)
        assert refused.status_code == 429
        assert refused.json()["error"]["type"] == "rate_limited"

    def test_a_zero_limit_disables_the_bucket(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "api_rate_limit", 0)
        statuses = [
            client.get("/v1/raw/pages", headers=READER).status_code for _ in range(4)
        ]
        assert statuses == [200, 200, 200, 200]

    def test_health_is_never_rate_limited(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A probe must keep working while a client is being throttled."""
        monkeypatch.setattr(settings, "api_rate_limit", 1)
        client.get("/v1/raw/pages", headers=READER)
        assert client.get("/v1/raw/pages", headers=READER).status_code == 429
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200


# ---------------------------------------------------------------------------
# Raw data endpoints
# ---------------------------------------------------------------------------


class TestRawDataEndpoints:
    def test_pages_listing_is_paged(self, client: TestClient) -> None:
        body = client.get("/v1/raw/pages?limit=1", headers=READER).json()
        assert body["meta"] == {
            "limit": 1, "offset": 0, "returned": 1, "total": 2, "next_offset": 1,
        }
        assert body["items"][0]["url"] == "https://example.com/a"
        # A listing never carries HTML bodies.
        assert "raw_html" not in body["items"][0]

    def test_last_page_reports_no_next_offset(self, client: TestClient) -> None:
        body = client.get("/v1/raw/pages?limit=1&offset=1", headers=READER).json()
        assert body["meta"]["next_offset"] is None

    def test_limit_is_capped_at_the_configured_maximum(self, client: TestClient) -> None:
        body = client.get("/v1/raw/pages?limit=5000", headers=READER).json()
        assert body["meta"]["limit"] == settings.api_max_page_size

    def test_filters_are_passed_through_to_the_repository(
        self, client: TestClient
    ) -> None:
        client.get(
            "/v1/raw/pages?site=example.com&http_status=200&url_contains=/a",
            headers=READER,
        )
        assert FakeRepository.last_call["site"] == "example.com"
        assert FakeRepository.last_call["http_status"] == 200
        assert FakeRepository.last_call["url_contains"] == "/a"

    def test_page_detail_includes_headers_and_html(self, client: TestClient) -> None:
        body = client.get("/v1/raw/pages/1", headers=READER).json()
        assert body["raw_html"].startswith("<html>")
        assert body["headers"]["Content-Type"] == "text/html"
        assert body["html_truncated"] is False

    def test_html_can_be_excluded_from_the_detail(self, client: TestClient) -> None:
        body = client.get("/v1/raw/pages/1?include_html=false", headers=READER).json()
        assert body["raw_html"] is None

    def test_oversized_html_is_truncated_and_flagged(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "api_max_html_chars", 10)
        body = client.get("/v1/raw/pages/1", headers=READER).json()
        assert body["html_truncated"] is True
        assert len(body["raw_html"]) == 10

    def test_html_endpoint_serves_the_body_as_html(self, client: TestClient) -> None:
        response = client.get("/v1/raw/pages/1/html", headers=READER)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.text.startswith("<html>")

    def test_unknown_page_is_a_404_with_the_common_error_shape(
        self, client: TestClient
    ) -> None:
        response = client.get("/v1/raw/pages/999", headers=READER)
        assert response.status_code == 404
        error = response.json()["error"]
        assert error == {
            "status": 404,
            "message": "no raw page with id 999",
            "type": "not_found",
            "path": "/v1/raw/pages/999",
            "request_id": error["request_id"],
        }

    def test_parsed_view_of_a_page(self, client: TestClient) -> None:
        body = client.get("/v1/raw/pages/1/parsed", headers=READER).json()
        assert body["title"] == "Page A"
        assert body["clean_text"] == "clean text of a"
        assert body["structured_data"] == {"price": "9.99"}

    def test_jobs_and_links_and_stats(self, client: TestClient) -> None:
        jobs = client.get("/v1/raw/jobs", headers=READER).json()
        assert jobs["items"][0]["job_name"] == "demo"

        links = client.get("/v1/raw/links?enqueued=true", headers=READER).json()
        assert links["items"][0]["target_url"] == "https://example.com/b"

        stats = client.get("/v1/raw/stats", headers=READER).json()
        assert stats["totals"]["raw_pages"] == 2


# ---------------------------------------------------------------------------
# Processed / enhanced content
# ---------------------------------------------------------------------------


class TestDocumentEndpoints:
    def test_listing_reports_enrichment_without_the_body_text(
        self, client: TestClient
    ) -> None:
        body = client.get("/v1/documents", headers=READER).json()
        assert body["meta"]["total"] == 2
        first = body["items"][0]
        assert first["enriched"] is True
        assert first["summary"] == "A page about widgets."
        assert "text" not in first

    def test_unenriched_documents_are_reported_as_such(self, client: TestClient) -> None:
        body = client.get("/v1/documents", headers=READER).json()
        second = body["items"][1]
        assert second["enriched"] is False
        assert second["summary"] is None

    def test_enriched_filter_is_passed_to_the_store(self, client: TestClient) -> None:
        client.get("/v1/documents?enriched=true&site=example.com", headers=READER)
        assert FakeStore.last_call["enriched"] is True
        assert FakeStore.last_call["site"] == "example.com"

    def test_detail_includes_text_fields_and_enrichment(self, client: TestClient) -> None:
        body = client.get("/v1/documents/doc-1", headers=READER).json()
        assert body["text"] == "the full cleaned text"
        assert body["fields"] == {"price": "9.99"}
        assert body["enrichment"]["topics"] == ["widgets"]

    def test_text_can_be_excluded(self, client: TestClient) -> None:
        body = client.get(
            "/v1/documents/doc-1?include_text=false", headers=READER
        ).json()
        assert body["text"] is None

    def test_lookup_by_url(self, client: TestClient) -> None:
        body = client.get(
            "/v1/documents/by-url?url=https://example.com/a", headers=READER
        ).json()
        assert body["doc_id"] == "doc-1"

    def test_unknown_document_is_a_404(self, client: TestClient) -> None:
        response = client.get("/v1/documents/missing-doc", headers=READER)
        assert response.status_code == 404
        assert "missing-doc" in response.json()["error"]["message"]

    def test_enrichment_endpoint_returns_only_the_generated_block(
        self, client: TestClient
    ) -> None:
        body = client.get("/v1/documents/doc-1/enrichment", headers=READER).json()
        assert set(body) == {"doc_id", "url", "enrichment"}
        assert body["enrichment"]["summary"] == "A page about widgets."

    def test_missing_enrichment_is_a_404_not_an_empty_object(
        self, client: TestClient
    ) -> None:
        """"Not generated" must be distinguishable from "generated and empty"."""
        response = client.get("/v1/documents/doc-2/enrichment", headers=READER)
        assert response.status_code == 404
        assert "has no enrichment yet" in response.json()["error"]["message"]

    def test_chunks_endpoint_shows_the_indexed_passages(self, client: TestClient) -> None:
        body = client.get("/v1/documents/doc-1/chunks", headers=READER).json()
        assert body[0]["chunk_id"] == "doc-1::0"

    def test_stats_and_sites(self, client: TestClient) -> None:
        assert client.get("/v1/documents/sites", headers=READER).json() == [
            "example.com"
        ]
        stats = client.get("/v1/documents/stats", headers=READER).json()
        assert stats["totals"]["documents"] == 2

    def test_static_paths_are_not_shadowed_by_the_doc_id_route(
        self, client: TestClient
    ) -> None:
        """``/documents/stats`` must not be read as a document called "stats"."""
        for path in ("/v1/documents/stats", "/v1/documents/sites",
                     "/v1/documents/duplicates"):
            assert client.get(path, headers=READER).status_code == 200


# ---------------------------------------------------------------------------
# Search and RAG
# ---------------------------------------------------------------------------


class TestSearchEndpoints:
    def test_search_returns_ranked_passages_with_their_evidence(
        self, client: TestClient
    ) -> None:
        body = client.get("/v1/search?q=widgets&top_k=2", headers=ANALYST).json()
        assert body["query"] == "widgets"
        assert body["mode"] == "hybrid"
        assert body["results"] == 2
        first = body["chunks"][0]
        assert first["score"] >= body["chunks"][1]["score"]
        assert first["retrievers"] == ["vector", "keyword"]
        assert first["vector_rank"] == 1

    def test_mode_is_passed_to_the_retriever(self, client: TestClient) -> None:
        body = client.get("/v1/search?q=widgets&mode=keyword", headers=ANALYST).json()
        assert body["mode"] == "keyword"
        assert FakeRetriever.last_call["mode"] == "keyword"

    def test_passage_text_can_be_excluded(self, client: TestClient) -> None:
        body = client.get(
            "/v1/search?q=widgets&include_text=false", headers=ANALYST
        ).json()
        assert body["chunks"][0].get("text") is None

    def test_an_unknown_mode_is_rejected_before_the_handler_runs(
        self, client: TestClient
    ) -> None:
        response = client.get("/v1/search?q=widgets&mode=sideways", headers=ANALYST)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["type"] == "validation_error"
        assert error["problems"][0]["field"] == "mode"

    def test_a_too_short_query_is_rejected(self, client: TestClient) -> None:
        response = client.get("/v1/search?q=a", headers=ANALYST)
        assert response.status_code == 422
        assert response.json()["error"]["problems"][0]["field"] == "q"

    def test_a_missing_query_is_rejected(self, client: TestClient) -> None:
        assert client.get("/v1/search", headers=ANALYST).status_code == 422

    def test_degradation_is_reported_in_notes_not_as_an_error(
        self, client: TestClient
    ) -> None:
        """One retriever being down is a note on a 200, not a 503."""
        response = client.get("/v1/search?q=degraded", headers=ANALYST)
        assert response.status_code == 200
        assert "vector search unavailable" in response.json()["notes"][0]

    def test_document_level_keyword_search(self, client: TestClient) -> None:
        body = client.get("/v1/search/documents?q=widgets", headers=ANALYST).json()
        assert body[0]["doc_id"] == "doc-1"

    def test_similar_documents(self, client: TestClient) -> None:
        body = client.get("/v1/search/similar/doc-1", headers=ANALYST).json()
        assert body[0]["doc_id"] == "doc-2"

    def test_similar_documents_404_when_nothing_is_indexed(
        self, client: TestClient
    ) -> None:
        response = client.get("/v1/search/similar/unindexed", headers=ANALYST)
        assert response.status_code == 404


class TestAskEndpoint:
    def test_answer_carries_citations_and_timings(self, client: TestClient) -> None:
        response = client.post(
            "/v1/ask",
            headers=ANALYST,
            json={"question": "what are widgets?", "top_k": 2, "max_sentences": 2},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["provider"]
        assert [citation["index"] for citation in body["citations"]] == [1, 2]
        assert body["timings_ms"]["total"] >= 0

    def test_context_is_excluded_unless_asked_for(self, client: TestClient) -> None:
        body = client.post(
            "/v1/ask", headers=ANALYST, json={"question": "what are widgets?"}
        ).json()
        chunks = (body.get("retrieval") or {}).get("chunks") or [{}]
        assert chunks[0].get("text") is None

        body = client.post(
            "/v1/ask",
            headers=ANALYST,
            json={"question": "what are widgets?", "include_context": True},
        ).json()
        assert body["retrieval"]["chunks"][0]["text"]

    def test_a_question_with_no_match_is_a_200_with_an_empty_answer(
        self, client: TestClient
    ) -> None:
        body = client.post(
            "/v1/ask", headers=ANALYST, json={"question": "nothing matches this"}
        ).json()
        assert body["answer"] == ""
        assert body["citations"] == []
        assert body["notes"]

    def test_an_empty_body_is_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/ask", headers=ANALYST, json={})
        assert response.status_code == 422
        assert response.json()["error"]["problems"][0]["field"] == "question"

    def test_out_of_range_parameters_are_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/v1/ask", headers=ANALYST, json={"question": "why?", "top_k": 999}
        )
        assert response.status_code == 422

    def test_get_is_not_allowed(self, client: TestClient) -> None:
        assert client.get("/v1/ask", headers=ANALYST).status_code == 405


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


class TestErrorTranslation:
    def test_an_unreachable_document_store_becomes_503(
        self, lenient_client: TestClient
    ) -> None:
        """A dependency outage is not the client's fault: 503 + Retry-After."""
        from src.processing.store import StoreUnavailable

        def failing_store() -> Any:
            raise StoreUnavailable("MongoDB is not reachable at localhost:27018")

        lenient_client.app.dependency_overrides[get_document_store] = failing_store
        response = lenient_client.get("/v1/documents", headers=READER)

        assert response.status_code == 503
        assert response.headers["Retry-After"] == "10"
        error = response.json()["error"]
        assert error["type"] == "document_store_unavailable"
        assert "MongoDB" in error["message"]

    def test_an_unreachable_vector_store_becomes_503(
        self, lenient_client: TestClient
    ) -> None:
        from src.rag.vector_store import VectorStoreUnavailable

        def failing_retriever() -> Any:
            raise VectorStoreUnavailable("Chroma is not reachable")

        lenient_client.app.dependency_overrides[get_retriever_dep] = failing_retriever
        response = lenient_client.get("/v1/search?q=widgets", headers=ANALYST)
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "vector_store_unavailable"

    def test_a_database_outage_becomes_503(self, lenient_client: TestClient) -> None:
        from sqlalchemy.exc import OperationalError

        class FailingRepository:
            @staticmethod
            def list_raw_pages(**_: Any) -> Any:
                raise OperationalError("SELECT 1", {}, Exception("connection refused"))

        lenient_client.app.dependency_overrides[get_repository] = (
            lambda: FailingRepository
        )
        response = lenient_client.get("/v1/raw/pages", headers=READER)
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "database_unavailable"

    def test_an_unexpected_error_becomes_a_500_that_leaks_nothing(self) -> None:
        """The message must not echo internals; the request id ties it to the log."""
        app = create_app()

        class ExplodingRepository:
            @staticmethod
            def list_raw_pages(**_: Any) -> Any:
                raise RuntimeError("postgresql://scraper:secret@db:5432 blew up")

        app.dependency_overrides[get_repository] = lambda: ExplodingRepository
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/v1/raw/pages", headers=READER)

        assert response.status_code == 500
        error = response.json()["error"]
        assert error["type"] == "internal_error"
        assert "secret" not in response.text
        assert error["request_id"]

    def test_an_unknown_route_is_a_404_in_the_common_shape(
        self, client: TestClient
    ) -> None:
        response = client.get("/v1/nothing-here", headers=READER)
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "not_found"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

RAW_PAGES = [
    {
        "id": 1,
        "job_id": 7,
        "url": "https://example.com/a",
        "site": "example.com",
        "depth": 0,
        "http_status": 200,
        "content_type": "text/html; charset=utf-8",
        "content_length": 24,
        "content_hash": "a" * 64,
        "fetched_at": NOW,
        "fetched_by": "node-1:100",
    },
    {
        "id": 2,
        "job_id": 7,
        "url": "https://example.com/b",
        "site": "example.com",
        "depth": 1,
        "http_status": 404,
        "content_type": "text/html",
        "content_length": 10,
        "content_hash": "b" * 64,
        "fetched_at": NOW,
        "fetched_by": "node-2:200",
    },
]

PARSED_PAGE = {
    "id": 11,
    "raw_page_id": 1,
    "url": "https://example.com/a",
    "site": "example.com",
    "title": "Page A",
    "description": "about widgets",
    "language": "en",
    "headings": ["Overview"],
    "clean_text": "clean text of a",
    "word_count": 4,
    "structured_data": {"price": "9.99"},
    "parser_used": "lxml",
    "parsed_at": NOW,
}

DOCUMENTS = [
    {
        "doc_id": "doc-1",
        "url": "https://example.com/a",
        "site": "example.com",
        "title": "Page A",
        "description": "about widgets",
        "language": "en",
        "keywords": ["widgets"],
        "headings": ["Overview"],
        "text": "the full cleaned text",
        "fields": {"price": "9.99"},
        "stats": {"word_count": 4, "char_count": 21},
        "source": {"raw_page_id": 1, "http_status": 200},
        "processing": {"schema_version": "1.0"},
        "enrichment": {
            "summary": "A page about widgets.",
            "topics": ["widgets"],
            "provider": "extractive",
        },
    },
    {
        "doc_id": "doc-2",
        "url": "https://example.com/b",
        "site": "example.com",
        "title": "Page B",
        "description": None,
        "language": "en",
        "keywords": [],
        "headings": [],
        "text": "another body",
        "fields": {},
        "stats": {"word_count": 2, "char_count": 12},
        "source": {"raw_page_id": 2, "http_status": 200},
        "processing": {"schema_version": "1.0"},
    },
]


class FakeRepository:
    """Stands in for ``ScrapeRepository`` - the PostgreSQL read side."""

    last_call: dict[str, Any] = {}

    @staticmethod
    def list_jobs(**kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        FakeRepository.last_call = kwargs
        return [
            {
                "id": 7,
                "job_name": "demo",
                "site": "example.com",
                "seed_urls": ["https://example.com/"],
                "status": "completed",
                "pages_scraped": 2,
                "pages_failed": 0,
                "started_at": NOW,
                "finished_at": NOW,
            }
        ], 1

    @staticmethod
    def list_raw_pages(
        *, limit: int = 20, offset: int = 0, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], int]:
        FakeRepository.last_call = {"limit": limit, "offset": offset, **kwargs}
        return RAW_PAGES[offset : offset + limit], len(RAW_PAGES)

    @staticmethod
    def get_raw_page(
        page_id: int, *, include_html: bool = True, max_html_chars: int | None = None
    ) -> dict[str, Any] | None:
        page = next((dict(p) for p in RAW_PAGES if p["id"] == page_id), None)
        if page is None:
            return None
        page["headers"] = {"Content-Type": "text/html"}
        page["raw_html"] = None
        page["html_truncated"] = False
        if include_html:
            html = "<html><body>page a</body></html>"
            if max_html_chars is not None and len(html) > max_html_chars:
                page["raw_html"] = html[:max_html_chars]
                page["html_truncated"] = True
            else:
                page["raw_html"] = html
        return page

    @staticmethod
    def get_raw_page_html(page_id: int) -> tuple[str, str] | None:
        if page_id != 1:
            return None
        return "<html><body>page a</body></html>", "text/html; charset=utf-8"

    @staticmethod
    def get_parsed_page(raw_page_id: int) -> dict[str, Any] | None:
        return dict(PARSED_PAGE) if raw_page_id == 1 else None

    @staticmethod
    def list_parsed_pages(**kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        FakeRepository.last_call = kwargs
        return [dict(PARSED_PAGE)], 1

    @staticmethod
    def list_links(**kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        FakeRepository.last_call = kwargs
        return [
            {
                "id": 3,
                "source_url": "https://example.com/a",
                "target_url": "https://example.com/b",
                "site": "example.com",
                "anchor_text": "next",
                "depth": 1,
                "enqueued": True,
                "discovered_at": NOW,
            }
        ], 1

    @staticmethod
    def stats() -> dict[str, Any]:
        return {
            "sites": [{"site": "example.com", "raw_pages": 2}],
            "totals": {"raw_pages": 2, "parsed_pages": 1, "discovered_links": 1,
                       "crawl_jobs": 1},
        }


class FakeStore:
    """Stands in for ``DocumentStore`` - the MongoDB read side."""

    last_call: dict[str, Any] = {}

    def list_documents(
        self, *, limit: int = 20, offset: int = 0, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], int]:
        FakeStore.last_call = {"limit": limit, "offset": offset, **kwargs}
        # A listing projects away the body text, like the real store.
        projected = [
            {key: value for key, value in document.items() if key != "text"}
            for document in DOCUMENTS
        ]
        return projected[offset : offset + limit], len(DOCUMENTS)

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return next((dict(d) for d in DOCUMENTS if d["doc_id"] == doc_id), None)

    def get_by_url(self, url: str) -> dict[str, Any] | None:
        return next((dict(d) for d in DOCUMENTS if d["url"] == url), None)

    def get_enrichment(self, doc_id: str) -> dict[str, Any] | None:
        document = self.get(doc_id)
        if document is None or not document.get("enrichment"):
            return None
        return {
            "doc_id": document["doc_id"],
            "url": document["url"],
            "enrichment": document["enrichment"],
        }

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"doc_id": "doc-1", "url": "https://example.com/a", "score": 1.5}]

    def duplicates(self, limit: int = 20) -> list[dict[str, Any]]:
        return [{"_id": "a" * 64, "count": 2, "urls": ["https://example.com/a"]}]

    def sites(self) -> list[str]:
        return ["example.com"]

    def stats(self) -> dict[str, Any]:
        return {
            "database": "processed_data",
            "collection": "documents",
            "sites": [{"site": "example.com", "documents": 2}],
            "totals": {"documents": 2, "words": 6, "raw_bytes": 34},
            "indexes": ["doc_id", "ix_fulltext"],
        }


class FakeVectorStore:
    def get_document_chunks(self, doc_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": f"{doc_id}::0",
                "doc_id": doc_id,
                "chunk_index": 0,
                "heading": "Overview",
                "text": "widgets are useful",
            }
        ]


def _fake_chunk(index: int, doc_id: str = "doc-1") -> Any:
    from src.rag.retriever import RetrievedChunk

    return RetrievedChunk(
        chunk_id=f"{doc_id}::{index}",
        doc_id=doc_id,
        # A real sentence, not a label: the extractive provider behind /v1/ask
        # ignores fragments with fewer than three content words, so a terser
        # placeholder would make the endpoint answer nothing.
        text=(
            f"Widgets are the modular parts described in passage {index} "
            "of this example page about widgets."
        ),
        url=f"https://example.com/{doc_id}",
        site="example.com",
        title=f"Page {doc_id}",
        heading="Overview",
        chunk_index=index,
        score=1.0 / (index + 1),
        vector_score=0.9 - index * 0.1,
        vector_rank=index + 1,
        keyword_score=2.0 - index,
        keyword_rank=index + 1,
        retrievers=["vector", "keyword"],
    )


class FakeRetriever:
    """Stands in for ``Retriever`` - no embedding model, no Chroma, no Mongo."""

    last_call: dict[str, Any] = {}

    def retrieve(self, query: str, **kwargs: Any) -> Any:
        from src.rag.retriever import RetrievalResult

        FakeRetriever.last_call = {"query": query, **kwargs}
        result = RetrievalResult(
            query=query,
            mode=kwargs.get("mode", "hybrid"),
            site=kwargs.get("site"),
            vector_candidates=2,
            keyword_candidates=2,
            embed_ms=1.0,
            vector_ms=2.0,
            keyword_ms=3.0,
            total_ms=6.0,
        )
        if query == "degraded":
            result.notes.append("vector search unavailable: Chroma is not reachable")
            return result
        if query == "nothing matches this":
            return result

        wanted = kwargs.get("top_k") or 2
        result.chunks = [_fake_chunk(index) for index in range(wanted)]
        return result

    def related_to_document(self, doc_id: str, *, top_k: int = 5) -> list[Any]:
        if doc_id != "doc-1":
            return []
        return [_fake_chunk(0, doc_id="doc-2")]
