"""The distributed Scrapy spider.

A normal Scrapy spider has a hard-coded ``start_urls`` list and stops when
that list is exhausted. This spider has none: it takes every URL from the
RabbitMQ frontier, and publishes the links it discovers back onto the same
queue. Two consequences follow, and they are the whole point of the design:

1. **The crawl is dynamic.** URLs can be added to a running crawl at any time
   by publishing a message - from another process, another machine, or later
   from the REST API.
2. **The crawl is distributed.** Any number of these spider processes can run
   at once, on the same machine or on different nodes. RabbitMQ gives each one
   a different URL (``prefetch=1``), so the work is balanced across whoever is
   available, and no coordination between the spiders is needed.

Continuous consumption is implemented through Scrapy's ``spider_idle`` signal.
Scrapy fires it when the scheduler runs dry; the handler pulls the next batch
from RabbitMQ and raises ``DontCloseSpider`` so the engine keeps running. Once
the frontier has stayed empty for ``FRONTIER_IDLE_ROUNDS`` consecutive idle
events, the handler lets the spider close.

Message acknowledgement is deliberately *late*: the delivery tag of a URL is
only acked once its page has been committed to PostgreSQL (the pipeline calls
back into :meth:`ack_url`). If this process is killed mid-fetch, RabbitMQ
redelivers the URL to another spider instead of losing it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import scrapy
from scrapy import signals
from scrapy.exceptions import DontCloseSpider

from src.common.config import settings
from src.messaging.factory import describe_backend, get_frontier
from src.messaging.frontier import UrlMessage
from src.worker import metrics
from src.worker.heartbeat import Heartbeat, node_id
from src.scraper.items import RawPageItem
from src.scraper.parser import normalise_url, site_of


class FrontierSpider(scrapy.Spider):
    """Spider whose URL source is the frontier (RabbitMQ or Kafka)."""

    name = "frontier"

    custom_settings = {
        # Keep the spider alive while the frontier is briefly empty.
        "CLOSESPIDER_TIMEOUT": 0,
    }

    def __init__(
        self,
        job_id: int | None = None,
        max_depth: int = 2,
        allowed_sites: str | None = None,
        batch_size: int = 16,
        idle_rounds: int = 3,
        max_pages: int = 0,
        follow_links: str = "true",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)

        # Scrapy passes -a arguments as strings, so every one is coerced here.
        self.job_id = int(job_id) if job_id not in (None, "", "None") else None
        self.max_depth = int(max_depth)
        self.batch_size = int(batch_size)
        self.idle_rounds = int(idle_rounds)
        self.max_pages = int(max_pages)
        self.follow_links = str(follow_links).lower() in ("1", "true", "yes")
        self.allowed_sites: set[str] = {
            s.strip().lower().removeprefix("www.")
            for s in (allowed_sites or "").split(",")
            if s.strip()
        }

        # RabbitMQ or Kafka, decided by FRONTIER_BACKEND. The two clients
        # share an interface, so nothing below this line knows which broker
        # is carrying the work.
        self.frontier = get_frontier()
        # url -> delivery_tag, for the late acknowledgement described above.
        self._pending_tags: dict[str, int] = {}
        self._seen_urls: set[str] = set()
        self._empty_idle_rounds = 0
        self.pages_done = 0
        self.pages_failed = 0
        self.links_published = 0

        # --- node liveness and metrics (Task 7) ---
        # The heartbeat is written when a page is *stored*, so it reflects
        # progress rather than the process being alive. That is what lets a
        # Kubernetes livenessProbe restart a node that is running but wedged.
        self.node = node_id()
        self.heartbeat = Heartbeat(self.node) if settings.worker_heartbeat_enabled else None
        if settings.worker_metrics_enabled:
            metrics.start_metrics_server(
                settings.worker_metrics_port,
                node=self.node,
                backend=describe_backend()["backend"],
            )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):  # type: ignore[no-untyped-def]
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_idle, signal=signals.spider_idle)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def start_requests(self):  # type: ignore[no-untyped-def]
        """First batch of URLs, taken from the frontier."""
        self.frontier.connect()
        depth = self.frontier.depth()
        metrics.observe_frontier_depth(depth)
        self.logger.info(
            "Frontier connected (%s) - %s URL(s) waiting, %s consumer(s) attached",
            describe_backend()["backend"],
            depth.get(self.frontier.queue), depth.get("consumers"),
        )
        yield from self._requests_from_frontier()

    def spider_idle(self) -> None:
        """Scheduler is empty: pull more work, or allow the spider to finish."""
        requests = list(self._requests_from_frontier())

        if requests:
            self._empty_idle_rounds = 0
            for request in requests:
                self.crawler.engine.crawl(request)
            raise DontCloseSpider

        self._empty_idle_rounds += 1

        # An idle node is working correctly; it simply has nothing to do. The
        # heartbeat has to say so, or the liveness probe would treat a quiet
        # frontier as a wedged crawler and restart a perfectly healthy node
        # every few minutes. "Alive" therefore means "still polling the
        # frontier", and a stalled node is one that has stopped even doing
        # that.
        if self.heartbeat is not None:
            self.heartbeat.beat(
                pages_done=self.pages_done,
                pages_failed=self.pages_failed,
                links_published=self.links_published,
                in_flight=len(self._pending_tags),
                idle_rounds=self._empty_idle_rounds,
                state="idle",
            )
        metrics.HEARTBEAT_AGE.set(0)
        metrics.observe_frontier_depth(self._safe_depth())

        # idle_rounds=0 means "wait indefinitely", which is the right
        # behaviour for a long-lived node under a Deployment: exiting on a
        # briefly empty frontier gets the container restarted, and every
        # restart rejoins the Kafka consumer group and triggers a rebalance.
        # Two nodes doing that every few seconds keep the group in permanent
        # rebalance and no work is ever picked up - which is exactly what
        # happened the first time this ran with the default of 3.
        if self.idle_rounds <= 0:
            if self._empty_idle_rounds % 30 == 1:
                self.logger.info(
                    "Frontier empty (%s round(s)) - waiting for more URLs",
                    self._empty_idle_rounds,
                )
            raise DontCloseSpider

        if self._empty_idle_rounds < self.idle_rounds:
            self.logger.info(
                "Frontier empty (%s/%s) - waiting for more URLs",
                self._empty_idle_rounds, self.idle_rounds,
            )
            raise DontCloseSpider

        self.logger.info("Frontier empty for %s rounds - closing", self.idle_rounds)

    def _safe_depth(self) -> dict[str, Any]:
        """The frontier depth, or an empty mapping if it cannot be read.

        Reporting the queue length must never be able to interrupt a crawl,
        and on Kafka it is several network round trips.
        """
        try:
            return self.frontier.depth()
        except Exception as exc:  # noqa: BLE001 - a metric is not worth a crash
            self.logger.debug("Could not read the frontier depth: %s", exc)
            return {}

    def spider_closed(self, reason: str) -> None:
        """Requeue anything still unacknowledged, then disconnect."""
        if self._pending_tags:
            self.logger.warning(
                "Requeueing %s unfinished URL(s) on close", len(self._pending_tags)
            )
            for tag in list(self._pending_tags.values()):
                try:
                    self.frontier.nack(tag, requeue=True)
                except Exception:  # noqa: BLE001
                    pass
            self._pending_tags.clear()

        self.logger.info(
            "Crawl finished (%s): %s page(s) stored, %s failed, %s link(s) published",
            reason, self.pages_done, self.pages_failed, self.links_published,
        )
        # A deliberate exit must not leave a heartbeat behind to age into a
        # false alarm about a node that is gone on purpose.
        if self.heartbeat is not None:
            self.heartbeat.clear()
        self.frontier.close()

    # ------------------------------------------------------------------
    # frontier -> Scrapy requests
    # ------------------------------------------------------------------

    def _requests_from_frontier(self):  # type: ignore[no-untyped-def]
        """Pull a batch of URL messages and turn them into Scrapy requests."""
        if self.max_pages and self.pages_done >= self.max_pages:
            return

        pulled = self.frontier.pull(limit=self.batch_size)
        metrics.URLS_PULLED.inc(len(pulled))

        for tag, message in pulled:
            url = message.url

            # Already crawled in this process: ack and move on. Cheap
            # protection against the duplicate deliveries that an
            # at-least-once broker is allowed to produce.
            if url in self._seen_urls:
                self.frontier.ack(tag)
                continue
            if self.allowed_sites and message.site not in self.allowed_sites:
                self.logger.debug("Skipping out-of-scope URL %s", url)
                self.frontier.ack(tag)
                continue

            self._seen_urls.add(url)
            self._pending_tags[url] = tag
            metrics.IN_FLIGHT.set(len(self._pending_tags))

            yield scrapy.Request(
                url=url,
                callback=self.parse_page,
                errback=self.on_error,
                dont_filter=True,     # de-duplication is ours, not Scrapy's
                meta={
                    "frontier_message": message,
                    "depth_from_seed": message.depth,
                },
            )

    # ------------------------------------------------------------------
    # response handling
    # ------------------------------------------------------------------

    def parse_page(self, response):  # type: ignore[no-untyped-def]
        """Emit the raw page, and publish newly discovered links."""
        message: UrlMessage = response.meta["frontier_message"]
        depth = int(response.meta.get("depth_from_seed", 0))

        item = RawPageItem()
        item["url"] = response.url
        item["site"] = site_of(response.url)
        item["depth"] = depth
        item["job_id"] = self.job_id or message.job_id
        item["http_status"] = response.status
        item["content_type"] = (
            response.headers.get("Content-Type", b"").decode("latin-1") or None
        )
        item["headers"] = {
            k.decode("latin-1"): b", ".join(v).decode("latin-1")
            for k, v in response.headers.items()
        }
        item["raw_html"] = response.text
        item["discovered_from"] = message.discovered_from
        item["fetched_at"] = datetime.now(timezone.utc).isoformat()

        self.pages_done += 1
        metrics.PAGES.labels(
            site=item["site"] or "unknown",
            outcome="ok" if response.status < 400 else f"http_{response.status}",
        ).inc()
        yield item

        # --- dynamic URL management: feed the frontier from this page ---
        if self.follow_links and depth < self.max_depth:
            self._publish_discovered_links(response, depth, item["site"])

    def _publish_discovered_links(self, response, depth: int, site: str) -> None:  # type: ignore[no-untyped-def]
        """Publish in-scope links found on the page back onto the frontier."""
        published = 0
        for href in response.css("a::attr(href)").getall():
            target = normalise_url(response.url, href)
            if target is None or target in self._seen_urls:
                continue

            target_site = site_of(target)
            if target_site != site:
                continue    # stay on the site being crawled
            if self.allowed_sites and target_site not in self.allowed_sites:
                continue

            self.frontier.publish(
                UrlMessage(
                    url=target,
                    site=target_site,
                    depth=depth + 1,
                    job_id=self.job_id,
                    discovered_from=response.url,
                )
            )
            published += 1

        if published:
            self.links_published += published
            metrics.URLS_PUBLISHED.labels(site=site or "unknown").inc(published)
            self.logger.debug("Published %s new URL(s) from %s", published, response.url)

    def on_error(self, failure):  # type: ignore[no-untyped-def]
        """A request failed: requeue the URL so another worker can retry it."""
        request = failure.request
        url = request.url
        self.pages_failed += 1
        metrics.PAGES.labels(site=site_of(url) or "unknown", outcome="failed").inc()
        self.logger.error("Request failed for %s: %s", url, failure.value)

        tag = self._pending_tags.pop(url, None)
        message: UrlMessage | None = request.meta.get("frontier_message")

        if tag is None:
            return
        # Ack the original delivery and republish with an incremented attempt
        # counter, so a permanently broken URL eventually dead-letters instead
        # of cycling through the queue forever.
        try:
            self.frontier.ack(tag)
            if message is not None:
                message.attempts += 1
                from src.messaging.frontier import MAX_DELIVERY_ATTEMPTS

                if message.attempts >= MAX_DELIVERY_ATTEMPTS:
                    metrics.DEAD_LETTERED.inc()
                    self.frontier.dead_letter(message, f"fetch failed: {failure.value}")
                else:
                    self.frontier.publish(message)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Could not requeue %s: %s", url, exc)

    # ------------------------------------------------------------------
    # called by the storage pipeline
    # ------------------------------------------------------------------

    def ack_url(self, url: str, success: bool = True) -> None:
        """Acknowledge a URL once its page has been committed to PostgreSQL."""
        tag = self._pending_tags.pop(url, None)
        if tag is None:
            return
        try:
            if success:
                self.frontier.ack(tag)
            else:
                self.frontier.nack(tag, requeue=True)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Acknowledgement failed for %s: %s", url, exc)

        metrics.IN_FLIGHT.set(len(self._pending_tags))
        if not success:
            return

        # A page committed to PostgreSQL is the one unambiguous unit of
        # progress this node makes, so it is what the liveness signal and
        # the stored-page counter are driven by - not a request being sent,
        # which a wedged node can keep doing without finishing anything.
        metrics.PAGES_STORED.labels(site=site_of(url) or "unknown").inc()
        metrics.HEARTBEAT_AGE.set(0)
        if self.heartbeat is not None:
            self.heartbeat.beat(
                pages_done=self.pages_done,
                pages_failed=self.pages_failed,
                links_published=self.links_published,
                in_flight=len(self._pending_tags),
                last_url=url,
            )
