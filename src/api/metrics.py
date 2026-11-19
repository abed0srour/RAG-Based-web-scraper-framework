"""Prometheus metrics for the API layer.

The system already reports on itself at every other layer - the crawl report,
the processing benchmark, the RAG index report - and this is the same idea for
the HTTP surface: a request counter by route and status, a latency histogram,
and counters for the two ways a request can be turned away (authentication
and rate limiting). Those four series are enough to answer the questions that
matter when several API replicas sit behind a load balancer: is one replica
slower than the others, is a client being throttled, is someone probing with
a bad key.

Two implementation notes:

* The metrics live in a **private registry**, not in ``prometheus_client``'s
  default global one. Tests create several applications in one process, and a
  global registry would raise ``Duplicated timeseries`` on the second one.
* ``prometheus_client`` is treated as optional. If it is not installed the
  module keeps plain in-process counters and ``/metrics`` serves them in the
  same text format, so the endpoint never disappears just because a
  dependency is missing.

Labels use the **route template** (``/v1/raw/pages/{page_id}``) rather than
the request path, so one metric does not fan out into one series per page id.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

PLAIN_TEXT = "text/plain; version=0.0.4; charset=utf-8"

try:  # pragma: no cover - exercised by whichever branch the environment has
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False


class _FallbackMetrics:
    """Minimal counters used when ``prometheus_client`` is not installed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, ...], int] = defaultdict(int)
        self.rate_limited: dict[tuple[str, ...], int] = defaultdict(int)
        self.auth_failures: dict[tuple[str, ...], int] = defaultdict(int)
        self.duration_sum: dict[tuple[str, ...], float] = defaultdict(float)

    def render(self) -> str:
        lines = [
            "# HELP api_requests_total API requests by route and status.",
            "# TYPE api_requests_total counter",
        ]
        with self._lock:
            for (method, path, status), value in sorted(self.requests.items()):
                lines.append(
                    f'api_requests_total{{method="{method}",path="{path}",'
                    f'status="{status}"}} {value}'
                )
            lines += [
                "# HELP api_request_duration_seconds_sum Total request time.",
                "# TYPE api_request_duration_seconds_sum counter",
            ]
            for (method, path), value in sorted(self.duration_sum.items()):
                lines.append(
                    f'api_request_duration_seconds_sum{{method="{method}",'
                    f'path="{path}"}} {value:.6f}'
                )
            lines += [
                "# HELP api_rate_limited_total Requests refused by the rate limiter.",
                "# TYPE api_rate_limited_total counter",
            ]
            for (bucket, client), value in sorted(self.rate_limited.items()):
                lines.append(
                    f'api_rate_limited_total{{bucket="{bucket}",client="{client}"}} {value}'
                )
            lines += [
                "# HELP api_auth_failures_total Rejected authentication attempts.",
                "# TYPE api_auth_failures_total counter",
            ]
            for (reason,), value in sorted(self.auth_failures.items()):
                lines.append(f'api_auth_failures_total{{reason="{reason}"}} {value}')
        return "\n".join(lines) + "\n"


_fallback = _FallbackMetrics()
_registry: Any = None
_requests: Any = None
_latency: Any = None
_rate_limited: Any = None
_auth_failures: Any = None

if PROMETHEUS_AVAILABLE:
    _registry = CollectorRegistry()
    _requests = Counter(
        "api_requests_total",
        "API requests by route and status.",
        ("method", "path", "status"),
        registry=_registry,
    )
    _latency = Histogram(
        "api_request_duration_seconds",
        "Request duration in seconds, by route.",
        ("method", "path"),
        # Buckets chosen for this service: an indexed SELECT answers in a few
        # milliseconds, a hybrid search in tens to hundreds, and a generated
        # answer in seconds.
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        registry=_registry,
    )
    _rate_limited = Counter(
        "api_rate_limited_total",
        "Requests refused by the rate limiter.",
        ("bucket", "client"),
        registry=_registry,
    )
    _auth_failures = Counter(
        "api_auth_failures_total",
        "Rejected authentication attempts.",
        ("reason",),
        registry=_registry,
    )


def observe_request(method: str, path: str, status_code: int, duration: float) -> None:
    """Record one finished request."""
    if PROMETHEUS_AVAILABLE:
        _requests.labels(method=method, path=path, status=str(status_code)).inc()
        _latency.labels(method=method, path=path).observe(duration)
        return
    with _fallback._lock:  # noqa: SLF001 - same module
        _fallback.requests[(method, path, str(status_code))] += 1
        _fallback.duration_sum[(method, path)] += duration


def observe_rate_limited(bucket: str, client: str) -> None:
    """Record one refusal by the rate limiter."""
    if PROMETHEUS_AVAILABLE:
        _rate_limited.labels(bucket=bucket, client=client).inc()
        return
    with _fallback._lock:  # noqa: SLF001
        _fallback.rate_limited[(bucket, client)] += 1


def observe_auth_failure(reason: str) -> None:
    """Record one rejected authentication attempt."""
    if PROMETHEUS_AVAILABLE:
        _auth_failures.labels(reason=reason).inc()
        return
    with _fallback._lock:  # noqa: SLF001
        _fallback.auth_failures[(reason,)] += 1


def render() -> tuple[bytes, str]:
    """The metrics exposition, as ``(body, content_type)``."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(_registry), PLAIN_TEXT
    return _fallback.render().encode("utf-8"), PLAIN_TEXT
