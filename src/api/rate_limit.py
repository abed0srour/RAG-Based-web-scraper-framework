"""Request rate limiting for the REST layer.

The service fronts a crawl database, a vector index and an LLM, and the cost
of a request is wildly uneven: listing raw pages is a single indexed SELECT,
while ``POST /v1/ask`` embeds a query, searches Chroma and MongoDB and then
calls a language model. Without a limit, one impatient client - or one loop
in a notebook - can saturate the workers that the crawl itself needs, or run
up a bill. So the limit is not one global number but **three windows per
client**:

================  ==========================  ================================
bucket            applies to                  default
================  ==========================  ================================
``requests``      every ``/v1`` endpoint      ``API_RATE_LIMIT`` per window
``search``        ``/v1/search*``             ``API_SEARCH_RATE_LIMIT``
``ask``           ``/v1/ask``                 ``API_ASK_RATE_LIMIT``
================  ==========================  ================================

A search therefore consumes one slot in the global window *and* one in the
search window, which is what lets a client make many cheap calls while the
expensive ones stay bounded.

**Algorithm.** A fixed window counter: the current time is divided into
windows of ``API_RATE_WINDOW_SECONDS``, and one counter per
``(bucket, client, window)`` is incremented. It is the cheapest correct
option - one ``INCR`` per request, no stored list of timestamps - and its
known weakness (a client can spend two windows' worth of requests across a
window boundary) is acceptable for an internal data API. Both backends
implement the same interface:

``RedisBackend``
    ``INCR`` plus ``EXPIRE`` in one pipeline. Redis is already part of the
    system (it is the Celery broker), and using it means the limit is
    **shared across API replicas**: three replicas behind a load balancer
    enforce one limit of 120 requests, not three limits of 120. This is the
    reason the counter does not live in the process.
``MemoryBackend``
    A dictionary of counters guarded by a lock. Correct for a single process
    and used automatically when Redis is unreachable, so an unavailable
    broker degrades the limit to per-replica instead of failing requests.

Every response carries ``X-RateLimit-*`` headers for the narrowest bucket the
request touched, and a refusal is a ``429`` with ``Retry-After``, so a client
can back off without guessing.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import Depends, HTTPException, Request, Response, status

from src.api.security import Principal, authenticate
from src.common.config import settings

logger = logging.getLogger(__name__)

#: Bucket names, also used as the Redis key prefix.
GLOBAL_BUCKET = "requests"
SEARCH_BUCKET = "search"
ASK_BUCKET = "ask"


@dataclass(frozen=True)
class RateLimitDecision:
    """The outcome of one limit check."""

    allowed: bool
    bucket: str
    limit: int
    remaining: int
    reset_after: int
    used: int
    backend: str

    def headers(self) -> dict[str, str]:
        """Standard rate-limit headers, returned on success and on refusal."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(self.remaining, 0)),
            "X-RateLimit-Reset": str(self.reset_after),
            "X-RateLimit-Bucket": self.bucket,
        }
        if not self.allowed:
            headers["Retry-After"] = str(max(self.reset_after, 1))
        return headers


class MemoryBackend:
    """Per-process fixed-window counters."""

    name = "memory"

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def incr(self, key: str, window: int) -> int:
        with self._lock:
            # The window index is part of the key, so an expired window is a
            # different key; old keys are swept when the window rolls over.
            count = self._counters.get(key, 0) + 1
            self._counters[key] = count
            if len(self._counters) > 10_000:  # pragma: no cover - safety valve
                self._sweep(key)
            return count

    def _sweep(self, keep_prefix: str) -> None:
        current_suffix = keep_prefix.rsplit(":", 1)[-1]
        self._counters = {
            key: value
            for key, value in self._counters.items()
            if key.endswith(current_suffix)
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()


class RedisBackend:
    """Fixed-window counters shared by every API replica."""

    name = "redis"

    def __init__(self, url: str | None = None) -> None:
        self.url = url or settings.redis_url
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import redis  # imported here so the API starts without redis

            self._client = redis.Redis.from_url(
                self.url,
                socket_timeout=0.5,          # a slow limiter must not slow the API
                socket_connect_timeout=0.5,
                decode_responses=True,
            )
        return self._client

    def incr(self, key: str, window: int) -> int:
        pipeline = self.client.pipeline()
        pipeline.incr(key, 1)
        # Refresh the TTL on every hit: harmless, and it means a key can never
        # be left without an expiry if a process died between INCR and EXPIRE.
        pipeline.expire(key, window)
        count, _ = pipeline.execute()
        return int(count)

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception as exc:  # noqa: BLE001 - any error means "not usable"
            logger.warning("Redis rate-limit backend unavailable: %s", exc)
            return False


class RateLimiter:
    """The limiter: one backend, three buckets, one decision per check."""

    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend
        self._memory = MemoryBackend()
        self._degraded = False

    # ---------------- backend selection ----------------

    @property
    def backend(self) -> Any:
        """Pick the backend once, honouring ``API_RATE_LIMIT_BACKEND``."""
        if self._backend is None:
            choice = (settings.api_rate_limit_backend or "auto").lower()
            if choice == "memory":
                self._backend = self._memory
            elif choice in ("redis", "auto"):
                candidate = RedisBackend()
                if candidate.ping():
                    self._backend = candidate
                elif choice == "redis":
                    # Explicitly configured for Redis: say loudly that the
                    # shared limit has become a per-replica one.
                    logger.error(
                        "API_RATE_LIMIT_BACKEND=redis but Redis is unreachable; "
                        "falling back to per-process counters"
                    )
                    self._backend = self._memory
                    self._degraded = True
                else:
                    logger.info(
                        "Rate limiting with per-process counters (Redis not reachable)"
                    )
                    self._backend = self._memory
            else:
                logger.warning(
                    "Unknown API_RATE_LIMIT_BACKEND %r; using per-process counters",
                    choice,
                )
                self._backend = self._memory
        return self._backend

    @property
    def backend_name(self) -> str:
        return str(getattr(self.backend, "name", "unknown"))

    @property
    def degraded(self) -> bool:
        """True when Redis was requested but the limiter fell back."""
        return self._degraded

    # ---------------- the check ----------------

    def check(
        self, identity: str, *, bucket: str, limit: int, window: int | None = None
    ) -> RateLimitDecision:
        """Count one request and decide whether it may proceed."""
        seconds = int(window or settings.api_rate_window_seconds)
        seconds = max(seconds, 1)

        if limit <= 0:  # 0 or negative disables this bucket
            return RateLimitDecision(
                allowed=True, bucket=bucket, limit=0, remaining=0,
                reset_after=seconds, used=0, backend=self.backend_name,
            )

        now = time.time()
        window_index = int(now // seconds)
        reset_after = int(seconds - (now % seconds)) or seconds
        key = f"ratelimit:{bucket}:{identity}:{window_index}"

        try:
            used = self.backend.incr(key, seconds)
        except Exception as exc:  # noqa: BLE001 - backend trouble, not client trouble
            # Failing open is deliberate: a rate limiter that cannot reach its
            # store must not take the API down with it. The fallback is still a
            # real limit, just a per-process one.
            logger.warning("Rate-limit backend failed (%s); using local counters", exc)
            self._backend = self._memory
            self._degraded = True
            used = self._memory.incr(key, seconds)

        remaining = limit - used
        return RateLimitDecision(
            allowed=used <= limit,
            bucket=bucket,
            limit=limit,
            remaining=max(remaining, 0),
            reset_after=reset_after,
            used=used,
            backend=self.backend_name,
        )

    def peek(self, identity: str) -> dict[str, Any]:
        """Describe the configured limits (used by ``GET /v1/me``)."""
        return {
            "backend": self.backend_name,
            "degraded": self.degraded,
            "window_seconds": settings.api_rate_window_seconds,
            "limits": {
                GLOBAL_BUCKET: settings.api_rate_limit,
                SEARCH_BUCKET: settings.api_search_rate_limit,
                ASK_BUCKET: settings.api_ask_rate_limit,
            },
        }


_LIMITER: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    """The process-wide limiter."""
    global _LIMITER
    if _LIMITER is None:
        _LIMITER = RateLimiter()
    return _LIMITER


def reset_limiter(backend: Any | None = None) -> RateLimiter:
    """Replace the limiter - used by the tests and by the demo script."""
    global _LIMITER
    _LIMITER = RateLimiter(backend=backend if backend is not None else MemoryBackend())
    return _LIMITER


#: Metrics hook, set by src/api/main.py so a refusal can be counted without
#: this module importing the metrics registry (and the circular import that
#: would come with it).
on_rejection: Callable[[str, str], None] | None = None


def _refuse(decision: RateLimitDecision, principal: Principal, path: str) -> None:
    logger.warning(
        "Rate limit hit by %r on %s: %s/%s in the %r window",
        principal.name, path, decision.used, decision.limit, decision.bucket,
    )
    if on_rejection is not None:
        on_rejection(decision.bucket, principal.name)

    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"rate limit exceeded: at most {decision.limit} "
            f"{decision.bucket} request(s) per {settings.api_rate_window_seconds}s "
            f"for this API key; retry in {decision.reset_after}s"
        ),
        headers=decision.headers(),
    )


def _limit(
    request: Request,
    response: Response,
    principal: Principal,
    *,
    bucket: str,
    limit: int,
) -> RateLimitDecision:
    decision = get_limiter().check(principal.identity, bucket=bucket, limit=limit)
    # The headers go out on every response, not only on a refusal, so a
    # well-behaved client can slow down before it is refused.
    for name, value in decision.headers().items():
        response.headers[name] = value
    if not decision.allowed:
        _refuse(decision, principal, request.url.path)
    return decision


def enforce_request_limit(
    request: Request,
    response: Response,
    principal: Principal = Depends(authenticate),
) -> RateLimitDecision:
    """Global per-key limit, declared once on the ``/v1`` router."""
    return _limit(
        request, response, principal,
        bucket=GLOBAL_BUCKET, limit=settings.api_rate_limit,
    )


def enforce_search_limit(
    request: Request,
    response: Response,
    principal: Principal = Depends(authenticate),
) -> RateLimitDecision:
    """Extra limit for retrieval endpoints (embedding + two index queries)."""
    return _limit(
        request, response, principal,
        bucket=SEARCH_BUCKET, limit=settings.api_search_rate_limit,
    )


def enforce_ask_limit(
    request: Request,
    response: Response,
    principal: Principal = Depends(authenticate),
) -> RateLimitDecision:
    """Extra limit for generation - the only endpoint that can cost money."""
    return _limit(
        request, response, principal,
        bucket=ASK_BUCKET, limit=settings.api_ask_rate_limit,
    )
