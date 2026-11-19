"""Prometheus metrics for a scraper node (Task 7).

The API already reports on its HTTP surface (``src/api/metrics.py``). This is
the same idea for the crawler, and it exists because the two questions
monitoring has to answer about a distributed crawler cannot be answered from
outside it:

* **Is each node working, or merely running?** ``kubectl get pods`` reports
  processes. A node that holds its partitions and fetches nothing looks
  identical to a busy one from there. ``scraper_pages_total`` increasing is
  the difference, and ``scraper_heartbeat_age_seconds`` is the same signal as
  a number an alert rule can use.
* **Is the fleet keeping up with the frontier?** That is the queue depth
  against the fetch rate, and it is what decides whether another node is
  needed - the quantity a Grafana panel should show and an autoscaler should
  act on.

A node is not a web server, so this starts a metrics-only HTTP listener in a
background thread. Note the division of labour with
:mod:`src.worker.healthcheck`: this endpoint is for *observation*, and it
deliberately is not the liveness probe, because an HTTP server in a crawler
process keeps answering perfectly while the crawl itself is wedged. Liveness
reads the heartbeat file, which only advances when a page is stored.

``prometheus_client`` is treated as optional, as it is in the API: without it
the counters become no-ops and the crawler runs unchanged. Monitoring must
not be able to stop the thing it monitors.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

PLAIN_TEXT = "text/plain; version=0.0.4; charset=utf-8"

_server_started = False
_lock = threading.Lock()

try:  # pragma: no cover - whichever branch this environment has
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        start_http_server,
    )

    HAVE_PROMETHEUS = True
except ImportError:  # pragma: no cover
    HAVE_PROMETHEUS = False


class _NoopMetric:
    """Stands in for a metric when prometheus_client is absent."""

    def labels(self, *_args: Any, **_kwargs: Any) -> "_NoopMetric":
        return self

    def inc(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        return None


if HAVE_PROMETHEUS:
    # A private registry rather than the global default, for the same reason
    # as in the API: the tests build several of these in one process, and a
    # global registry raises "Duplicated timeseries" on the second one.
    REGISTRY = CollectorRegistry()

    # Counters only ever increase, which is what lets Prometheus compute a
    # rate across restarts without being confused by a reset.
    PAGES = Counter(
        "scraper_pages_total",
        "Pages fetched, by site and outcome",
        ["site", "outcome"],
        registry=REGISTRY,
    )
    PAGES_STORED = Counter(
        "scraper_pages_stored_total",
        "Pages committed to PostgreSQL and acknowledged on the frontier",
        ["site"],
        registry=REGISTRY,
    )
    URLS_PUBLISHED = Counter(
        "scraper_urls_published_total",
        "URLs discovered on a page and published back onto the frontier",
        ["site"],
        registry=REGISTRY,
    )
    URLS_PULLED = Counter(
        "scraper_urls_pulled_total",
        "URL messages taken from the frontier",
        registry=REGISTRY,
    )
    DEAD_LETTERED = Counter(
        "scraper_urls_dead_lettered_total",
        "URLs given up on after exhausting their retries",
        registry=REGISTRY,
    )
    FETCH_SECONDS = Histogram(
        "scraper_fetch_duration_seconds",
        "Time from request to stored page",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 60.0),
        registry=REGISTRY,
    )
    # Gauges describe a current state, so they are the right type for
    # "how far behind are we" and "how long since this node did anything".
    FRONTIER_DEPTH = Gauge(
        "scraper_frontier_depth",
        "URLs waiting on the frontier (RabbitMQ message count, or Kafka lag)",
        ["queue"],
        registry=REGISTRY,
    )
    HEARTBEAT_AGE = Gauge(
        "scraper_heartbeat_age_seconds",
        "Seconds since this node last stored a page",
        registry=REGISTRY,
    )
    IN_FLIGHT = Gauge(
        "scraper_requests_in_flight",
        "URLs pulled from the frontier but not yet acknowledged",
        registry=REGISTRY,
    )
    UP = Gauge(
        "scraper_up",
        "1 while the crawler process is running",
        ["node", "backend"],
        registry=REGISTRY,
    )
else:  # pragma: no cover - exercised only without the dependency
    REGISTRY = None  # type: ignore[assignment]
    PAGES = PAGES_STORED = URLS_PUBLISHED = _NoopMetric()  # type: ignore[assignment]
    URLS_PULLED = DEAD_LETTERED = FETCH_SECONDS = _NoopMetric()  # type: ignore[assignment]
    FRONTIER_DEPTH = HEARTBEAT_AGE = IN_FLIGHT = UP = _NoopMetric()  # type: ignore[assignment]


def start_metrics_server(port: int, node: str = "", backend: str = "") -> bool:
    """Expose ``/metrics`` on ``port`` in a background thread.

    Returns True when the listener is up. Failure is logged and swallowed: a
    port already taken - two nodes started on one host, which is the normal
    way to run this locally - must not stop the crawl.
    """
    global _server_started

    if not HAVE_PROMETHEUS:
        logger.info("prometheus_client is not installed; metrics are disabled")
        return False

    with _lock:
        if _server_started:
            return True
        try:
            start_http_server(port, registry=REGISTRY)
            _server_started = True
            UP.labels(node=node, backend=backend).set(1)
            logger.info("Scraper metrics on http://0.0.0.0:%d/metrics", port)
            return True
        except OSError as exc:
            logger.warning(
                "Could not start the metrics server on port %d: %s "
                "(the crawl continues without it)", port, exc,
            )
            return False


def render() -> bytes:
    """The current metrics in Prometheus text format (used by the tests)."""
    if not HAVE_PROMETHEUS:
        return b""
    return generate_latest(REGISTRY)


def observe_frontier_depth(depth: dict[str, Any]) -> None:
    """Publish the frontier depth reported by either broker.

    Both clients' ``depth()`` return a mapping of queue or topic name to a
    count, plus a couple of non-count keys, so the numeric entries are the
    ones that become samples.
    """
    for name, value in (depth or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if name in ("consumers", "partitions_assigned"):
            continue
        FRONTIER_DEPTH.labels(queue=str(name)).set(float(value))
