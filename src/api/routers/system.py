"""Health, readiness, metrics and caller identity.

These are the endpoints the *operator* uses rather than the data consumer,
and they are deliberately the only ones that are not behind an API key:
a Kubernetes probe and a Prometheus scraper cannot carry credentials, and
refusing them would mean the orchestrator could never tell a healthy replica
from a sick one.

Liveness and readiness are separate on purpose:

``GET /health``
    Answers 200 as long as the process is serving requests. It touches no
    dependency, so a database outage never restarts an API replica that is
    perfectly able to serve the endpoints that do not need that database.

``GET /health/ready``
    Probes PostgreSQL, MongoDB, Chroma and the LLM provider and reports each
    one. It answers 503 only when *every* data store is unreachable - at that
    point the replica can serve nothing and should be taken out of the load
    balancer - and 200 with ``status: degraded`` when some stores are down,
    because the API can still answer from the ones that are up.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response

from src.api import metrics
from src.api.rate_limit import get_limiter
from src.api.schemas import (
    HealthResponse,
    IdentityResponse,
    ReadinessResponse,
)
from src.api.security import Principal, authenticate
from src.common.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["service"])

STARTED_AT = time.time()

SERVICE_NAME = "distributed-rag-scraper-api"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
)
def health() -> Any:
    """Is the process alive? No dependency is touched."""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": settings.api_version,
        "uptime_seconds": round(time.time() - STARTED_AT, 3),
    }


def _check_postgres() -> dict[str, str | None]:
    from src.storage.database import ScrapeRepository

    if ScrapeRepository.ping():
        return {"status": "up", "detail": f"{settings.postgres_host}:{settings.postgres_port}"}
    return {"status": "down", "detail": "raw data endpoints will fail"}


def _check_mongo() -> dict[str, str | None]:
    from src.processing.store import DocumentStore

    store = DocumentStore()
    if store.ping():
        return {
            "status": "up",
            "detail": f"{settings.mongo_db}.{settings.mongo_collection}",
        }
    return {"status": "down", "detail": "document and keyword-search endpoints will fail"}


def _check_chroma() -> dict[str, str | None]:
    from src.rag.vector_store import VectorStore

    store = VectorStore()
    if store.ping():
        return {"status": "up", "detail": f"collection {settings.chroma_collection}"}
    return {"status": "down", "detail": "vector search will fall back to keyword only"}


def _check_elasticsearch() -> dict[str, str | None]:
    from src.search.index import SearchIndex

    index = SearchIndex()
    if index.ping():
        try:
            documents = index.count()
        except Exception:  # noqa: BLE001 - a probe reports, never raises
            documents = 0
        return {
            "status": "up",
            "detail": f"index {index.index_name} ({documents} documents)",
        }
    return {"status": "down", "detail": "full-text search and the web UI will fail"}


def _check_llm() -> dict[str, str | None]:
    try:
        from src.rag.llm import get_llm

        llm = get_llm()
        return {
            "status": "up" if llm.available() else "disabled",
            "detail": f"{llm.name} ({llm.model})",
        }
    except Exception as exc:  # noqa: BLE001 - a broken provider is a report, not a crash
        return {"status": "down", "detail": str(exc)[:200]}


def _check_rate_limiter() -> dict[str, str | None]:
    limiter = get_limiter()
    return {
        "status": "degraded" if limiter.degraded else "up",
        "detail": f"{limiter.backend_name} backend",
    }


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={503: {"description": "No data store is reachable"}},
)
def readiness(response: Response) -> Any:
    """Which backing services are reachable right now.

    Each check is independent and failures are reported, never raised: the
    point of a readiness endpoint is to describe a partial outage, so it must
    not itself fail when one dependency is down.
    """
    checks = {
        "postgres": _check_postgres(),
        "mongodb": _check_mongo(),
        "chroma": _check_chroma(),
        "elasticsearch": _check_elasticsearch(),
        "llm": _check_llm(),
        "rate_limiter": _check_rate_limiter(),
    }

    stores = ("postgres", "mongodb", "chroma", "elasticsearch")
    stores_up = [name for name in stores if checks[name]["status"] == "up"]

    if not stores_up:
        response.status_code = 503
        status_text = "unavailable"
    elif len(stores_up) < len(stores):
        status_text = "degraded"
    else:
        status_text = "ready"

    return {"status": status_text, "checks": checks}


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    response_class=Response,
    responses={200: {"content": {metrics.PLAIN_TEXT: {}}}},
)
def prometheus_metrics() -> Response:
    """Request counts, latency histogram, refusals by auth and by rate limit."""
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


#: Mounted under the versioned router (``/v1/me``), so the identity endpoint
#: is subject to the same authentication and global rate limit as the data
#: endpoints - unlike the probes above, which must stay open.
me_router = APIRouter(tags=["service"])


@me_router.get(
    "/me",
    response_model=IdentityResponse,
    summary="Describe the calling API key",
)
def whoami(
    principal: Annotated[Principal, Depends(authenticate)],
) -> Any:
    """What the service thinks this key is allowed to do.

    This is the endpoint a client calls first: it confirms the key is valid,
    lists the scopes it carries and reports the rate limits that apply, which
    turns "403 on some endpoint" into something a client can diagnose without
    access to the server configuration.
    """
    return {
        **principal.describe(),
        "rate_limit": get_limiter().peek(principal.identity),
    }
