"""The FastAPI application: the REST interface to the whole framework.

    uvicorn src.api.main:app --host 0.0.0.0 --port 8080
    python -m src.api                      # same thing, reading the settings

Run it and the interactive documentation is at ``/docs``, the OpenAPI schema
at ``/openapi.json``.

What this module is responsible for - the concerns that are the same for
every endpoint, and are therefore handled once here rather than in each
router:

**Composition.** Three data routers (``raw``, ``documents``, ``search``) are
mounted under ``/v1``, which carries the authentication and the global
rate-limit dependency; the service router (health, readiness, metrics) is
mounted at the root and stays open, because probes cannot authenticate.
Versioning the data surface means a later task can add ``/v2`` without
breaking a client.

**One error shape.** Every failure - a validation error, a missing document,
an exhausted rate limit, an unreachable database, an unhandled bug - comes
back as the same ``{"error": {...}}`` envelope with a status, a message, a
machine-readable type and the request id. A client therefore needs one error
path, not six.

**Degradation instead of 500s.** The data stores are remote services that can
be down. ``StoreUnavailable``, ``VectorStoreUnavailable``,
``EmbeddingUnavailable`` and SQLAlchemy's ``OperationalError`` are translated
to **503 with a Retry-After**, not 500: the request was fine, the dependency
was not, and the distinction is what tells a client to retry rather than to
fix its request.

**Request identity and timing.** One middleware stamps every request with an
id (echoed in ``X-Request-ID`` and in the error envelope, so a user-reported
failure can be found in the logs), measures it, writes one access-log line
and records the Prometheus series.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Depends, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api import metrics, rate_limit
from src.api.rate_limit import enforce_request_limit
from src.api.routers import documents, raw, search, system
from src.api.security import describe_keys
from src.common.config import settings
from src.common.logging_config import quiet_noisy_loggers

logger = logging.getLogger("api")

DESCRIPTION = """
REST interface to the distributed RAG web-scraping framework.

* **Raw data** (`/v1/raw`) - the crawl as it was fetched: HTTP status,
  headers, the unmodified HTML, the parsed view and the link graph.
* **Processed content** (`/v1/documents`) - the cleaned, normalised JSON
  documents and the summaries, topics and quality flags the RAG augmentation
  stage added to them.
* **Search** (`/v1/search`, `/v1/ask`) - vector, keyword and hybrid retrieval
  over the chunk index, and grounded question answering with citations.

**Authentication.** Every `/v1` endpoint needs an API key, sent either as
`X-API-Key: <key>` or as `Authorization: Bearer <key>`. Keys carry scopes
(`raw`, `documents`, `search`, `ask`); `GET /v1/me` reports the ones a key
holds. **Rate limiting** applies per key, with tighter windows for retrieval
and generation; every response carries `X-RateLimit-*` headers.
"""

#: Maps an exception class to the status code and error type it becomes.
#: Declaring it as data keeps the handler itself trivial and makes the
#: translation table reviewable in one place.
UNAVAILABLE_TYPES = {
    "StoreUnavailable": "document_store_unavailable",
    "VectorStoreUnavailable": "vector_store_unavailable",
    "EmbeddingUnavailable": "embedding_model_unavailable",
    "OperationalError": "database_unavailable",
    "DBAPIError": "database_unavailable",
    "InterfaceError": "database_unavailable",
}

ERROR_TYPES = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}


def error_body(
    *, status_code: int, message: str, request: Request, error_type: str | None = None
) -> dict[str, Any]:
    """Build the single error envelope used by every failure path."""
    return {
        "error": {
            "status": status_code,
            "message": message,
            "type": error_type or ERROR_TYPES.get(status_code, "error"),
            "path": request.url.path,
            "request_id": getattr(request.state, "request_id", None),
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log the effective configuration once, at start-up.

    The log line is the deployment's receipt: it says which stores this
    replica will talk to, which keys it loaded (by name and fingerprint,
    never the secret) and which rate-limit backend it ended up with, so a
    misconfigured replica is obvious from its first ten log lines.
    """
    quiet_noisy_loggers()
    keys = describe_keys()
    logger.info(
        "API %s starting | postgres=%s:%s | mongo=%s:%s | chroma=%s:%s",
        settings.api_version,
        settings.postgres_host, settings.postgres_port,
        settings.mongo_host, settings.mongo_port,
        settings.chroma_host, settings.chroma_port,
    )
    if settings.api_auth_enabled:
        logger.info(
            "Authentication enabled with %s key(s): %s",
            len(keys),
            ", ".join(f"{key['name']}[{key['key_id']}]" for key in keys) or "none",
        )
        if not keys:
            logger.error(
                "API_AUTH_ENABLED is true but API_KEYS is empty: every request "
                "will be refused with 401"
            )
    else:
        logger.warning(
            "Authentication DISABLED (API_AUTH_ENABLED=false): every caller is "
            "granted every scope"
        )
    logger.info(
        "Rate limit: %s requests / %ss per key (search %s, ask %s) via the %s backend",
        settings.api_rate_limit,
        settings.api_rate_window_seconds,
        settings.api_search_rate_limit,
        settings.api_ask_rate_limit,
        rate_limit.get_limiter().backend_name,
    )

    if settings.api_warm_embedder:
        # The embedding model is ~45 MB of ONNX weights and takes about a
        # second to load. Loading it here means the first client search pays
        # the retrieval cost only, not the model load.
        try:
            from src.rag.embedder import get_embedder

            embedder = get_embedder()
            logger.info(
                "Embedding model warmed: %s (%s dims)",
                settings.embedding_model, embedder.dimension,
            )
        except Exception as exc:  # noqa: BLE001 - a cold model is not fatal
            logger.warning("Could not warm the embedding model: %s", exc)

    yield
    logger.info("API shutting down")


def create_app() -> FastAPI:
    """Build the application. A factory, so tests can create a fresh one."""
    app = FastAPI(
        title="Distributed RAG Web Scraper API",
        description=DESCRIPTION,
        version=settings.api_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.api_docs_enabled else None,
        redoc_url="/redoc" if settings.api_docs_enabled else None,
        openapi_url="/openapi.json" if settings.api_docs_enabled else None,
    )

    # Let the rate limiter report refusals without importing the metrics
    # registry itself (which would make the two modules circular).
    rate_limit.on_rejection = metrics.observe_rate_limited

    origins = [
        origin.strip()
        for origin in (settings.api_cors_origins or "").split(",")
        if origin.strip()
    ]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials="*" not in origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=[
                "X-Request-ID",
                "X-Response-Time-Ms",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
                "X-RateLimit-Bucket",
                "Retry-After",
            ],
        )

    _install_middleware(app)
    _install_exception_handlers(app)

    # ---- the versioned data surface: authenticated and rate limited ----
    # The global rate limit is declared here, once, so no route can forget it;
    # authentication runs as its dependency, before any handler is reached.
    v1 = APIRouter(prefix="/v1", dependencies=[Depends(enforce_request_limit)])
    v1.include_router(raw.router)
    v1.include_router(documents.router)
    v1.include_router(search.router)
    v1.include_router(system.me_router)
    app.include_router(v1)

    # ---- probes and metrics: open, so an orchestrator can use them ----
    app.include_router(system.router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        """A pointer to the documentation, so ``/`` is not a 404."""
        return {
            "service": system.SERVICE_NAME,
            "version": settings.api_version,
            "documentation": "/docs" if settings.api_docs_enabled else None,
            "endpoints": {
                "raw_data": "/v1/raw/pages",
                "processed_content": "/v1/documents",
                "search": "/v1/search",
                "ask": "/v1/ask",
                "health": "/health",
                "readiness": "/health/ready",
                "metrics": "/metrics",
            },
        }

    return app


def _install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Stamp, time, log and count one request."""
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # The exception handlers below produce the response for anything
            # raised inside a route; reaching here means the failure happened
            # in the middleware stack itself, so it is logged and re-raised
            # after the metric has been recorded.
            duration = time.perf_counter() - started
            metrics.observe_request(request.method, _route_label(request), 500, duration)
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            raise

        duration = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration * 1000:.1f}"
        metrics.observe_request(
            request.method, _route_label(request), response.status_code, duration
        )

        if response.status_code == status.HTTP_401_UNAUTHORIZED:
            metrics.observe_auth_failure("invalid_or_missing_key")
        elif response.status_code == status.HTTP_403_FORBIDDEN:
            metrics.observe_auth_failure("missing_scope")

        # One line per request, at the level the status deserves.
        log = logger.info if response.status_code < 500 else logger.error
        log(
            "%s %s -> %s in %.1f ms [%s]",
            request.method, request.url.path, response.status_code,
            duration * 1000, request_id,
        )
        return response


def _route_label(request: Request) -> str:
    """The route template, so metrics do not fan out per path parameter."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Re-shape HTTPException into the common envelope, keeping headers."""
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                status_code=exc.status_code,
                message=str(exc.detail),
                request=request,
            ),
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """422 with the offending fields named, not a generic rejection."""
        problems = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())[1:]),
                "problem": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        body = error_body(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="the request parameters are invalid",
            request=request,
        )
        body["error"]["problems"] = problems
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=body)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        """Translate dependency outages to 503, and everything else to 500."""
        name = type(exc).__name__
        error_type = UNAVAILABLE_TYPES.get(name)

        if error_type is not None:
            logger.warning(
                "%s on %s %s: %s", name, request.method, request.url.path, exc
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=error_body(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    message=(
                        f"a backing service is unavailable ({name}): {exc}"[:500]
                    ),
                    request=request,
                    error_type=error_type,
                ),
                headers={"Retry-After": "10"},
            )

        if name in ("RetrievalError", "NormalizationError", "ValueError"):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=error_body(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    message=str(exc)[:500],
                    request=request,
                    error_type="bad_request",
                ),
            )

        logger.exception("Unhandled %s on %s %s", name, request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                # The exception text is deliberately not echoed: it can carry
                # connection strings and query fragments. The request id is
                # what ties the client's report to the full traceback in the
                # server log.
                message="an internal error occurred; quote the request id when reporting it",
                request=request,
            ),
        )


#: The ASGI application uvicorn serves.
app = create_app()


def main(argv: list[str] | None = None) -> int:
    """Run the service with uvicorn, using the central settings."""
    import argparse

    import uvicorn

    from src.common.logging_config import setup_logging

    parser = argparse.ArgumentParser(description="Run the REST API")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument(
        "--reload", action="store_true", help="auto-reload on code changes"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="number of worker processes (ignored with --reload)",
    )
    args = parser.parse_args(argv)

    setup_logging("api")
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=None if args.reload else args.workers,
        log_level="info",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
