"""The RabbitMQ URL frontier.

This module is what makes the URL list *dynamic*: instead of a fixed list of
start URLs compiled into the spider, URLs live in a RabbitMQ queue that any
number of crawler processes consume from, and that the crawlers themselves
publish newly discovered links back into. Adding work to a running crawl is
just publishing another message.

Topology
--------

    publisher ──► exchange "scraper.frontier" (direct)
                        │  routing key "url"
                        ▼
                  queue "urls.frontier"   ◄── consumed by N crawler processes
                        │
                        │ message rejected / TTL expired
                        ▼
                  queue "urls.dead-letter"

Design choices that matter for a distributed system:

* **Durable queues + persistent messages** - the frontier survives a broker
  restart, so a crash does not lose the remaining work.
* **Manual acknowledgement** - a URL is only removed from the queue after it
  has been stored in PostgreSQL. If a worker dies mid-fetch, the unacked
  message is redelivered to another worker.
* **prefetch = 1** - the broker hands a worker one URL at a time, so a slow
  worker cannot hoard a backlog while others idle. This is the load-balancing
  mechanism.
* **Dead-letter queue** - a URL that keeps failing is routed aside instead of
  being retried forever.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPConnectionError

from src.common.config import settings

logger = logging.getLogger(__name__)

ROUTING_KEY = "url"
DLQ_ROUTING_KEY = "url.dead"

# A message that has been retried this many times is dead-lettered.
MAX_DELIVERY_ATTEMPTS = 3


@dataclass
class UrlMessage:
    """One unit of work on the frontier."""

    url: str
    site: str
    depth: int = 0
    job_id: int | None = None
    attempts: int = 0
    discovered_from: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def from_json(cls, body: bytes) -> "UrlMessage":
        data = json.loads(body.decode("utf-8"))
        known = {k: data.get(k) for k in cls.__dataclass_fields__ if k in data}
        known.setdefault("meta", {})
        return cls(**known)


class FrontierQueue:
    """Blocking client for the URL frontier (publisher and consumer)."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        vhost: str | None = None,
        queue: str | None = None,
        exchange: str | None = None,
        dlq: str | None = None,
        prefetch: int | None = None,
    ) -> None:
        self.host = host or settings.rabbitmq_host
        self.port = port or settings.rabbitmq_port
        self.user = user or settings.rabbitmq_user
        self.password = password or settings.rabbitmq_password
        self.vhost = vhost or settings.rabbitmq_vhost
        self.queue = queue or settings.frontier_queue
        self.exchange = exchange or settings.frontier_exchange
        self.dlq = dlq or settings.frontier_dlq
        self.prefetch = prefetch or settings.frontier_prefetch

        self._connection: pika.BlockingConnection | None = None
        self._channel: BlockingChannel | None = None

    # ------------------------------------------------------------------
    # connection handling
    # ------------------------------------------------------------------

    def _parameters(self) -> pika.ConnectionParameters:
        return pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=pika.PlainCredentials(self.user, self.password),
            heartbeat=600,
            blocked_connection_timeout=300,
            connection_attempts=3,
            retry_delay=2.0,
        )

    def connect(self, max_retries: int = 5) -> "FrontierQueue":
        """Open a connection, retrying with backoff if the broker is not up."""
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                self._connection = pika.BlockingConnection(self._parameters())
                self._channel = self._connection.channel()
                self._declare_topology()
                self._channel.basic_qos(prefetch_count=self.prefetch)
                # Publisher confirms: basic_publish blocks until the broker has
                # accepted the message, so a URL can never be silently lost
                # between the crawler and the queue.
                self._channel.confirm_delivery()
                logger.info(
                    "Connected to RabbitMQ at %s:%s (queue=%s, prefetch=%s)",
                    self.host, self.port, self.queue, self.prefetch,
                )
                return self
            except AMQPConnectionError as exc:
                last_error = exc
                logger.warning(
                    "RabbitMQ connection attempt %s/%s failed: %s - retrying in %.1fs",
                    attempt, max_retries, exc, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 15.0)
        raise RuntimeError(
            f"Could not connect to RabbitMQ at {self.host}:{self.port}"
        ) from last_error

    def _declare_topology(self) -> None:
        """Declare exchanges and queues (idempotent on every connect)."""
        ch = self._require_channel()

        # Dead-letter side first, so the main queue can reference it.
        ch.exchange_declare(
            exchange=f"{self.exchange}.dlx", exchange_type="direct", durable=True
        )
        ch.queue_declare(queue=self.dlq, durable=True)
        ch.queue_bind(
            queue=self.dlq,
            exchange=f"{self.exchange}.dlx",
            routing_key=DLQ_ROUTING_KEY,
        )

        # Main frontier queue.
        ch.exchange_declare(
            exchange=self.exchange, exchange_type="direct", durable=True
        )
        ch.queue_declare(
            queue=self.queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": f"{self.exchange}.dlx",
                "x-dead-letter-routing-key": DLQ_ROUTING_KEY,
            },
        )
        ch.queue_bind(
            queue=self.queue, exchange=self.exchange, routing_key=ROUTING_KEY
        )

    def _require_channel(self) -> BlockingChannel:
        if self._channel is None or self._channel.is_closed:
            raise RuntimeError("Frontier channel is not open - call connect() first")
        return self._channel

    def close(self) -> None:
        try:
            if self._connection is not None and self._connection.is_open:
                self._connection.close()
        except Exception:  # noqa: BLE001 - closing must never raise
            logger.debug("Ignoring error while closing RabbitMQ connection")
        finally:
            self._channel = None
            self._connection = None

    def __enter__(self) -> "FrontierQueue":
        return self.connect()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # producing
    # ------------------------------------------------------------------

    def publish(self, message: UrlMessage) -> None:
        """Publish one URL onto the frontier as a persistent message."""
        self._require_channel().basic_publish(
            exchange=self.exchange,
            routing_key=ROUTING_KEY,
            body=message.to_json(),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=pika.DeliveryMode.Persistent,  # survives restart
                headers={"site": message.site, "depth": message.depth},
            ),
        )

    def publish_many(self, messages: list[UrlMessage]) -> int:
        for message in messages:
            self.publish(message)
        return len(messages)

    def dead_letter(self, message: UrlMessage, reason: str) -> None:
        """Route a permanently failing URL to the dead-letter queue."""
        message.meta["dead_letter_reason"] = reason
        self._require_channel().basic_publish(
            exchange=f"{self.exchange}.dlx",
            routing_key=DLQ_ROUTING_KEY,
            body=message.to_json(),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=pika.DeliveryMode.Persistent,
            ),
        )
        logger.warning("Dead-lettered %s (%s)", message.url, reason)

    # ------------------------------------------------------------------
    # consuming
    # ------------------------------------------------------------------

    def pull(self, limit: int = 1) -> list[tuple[int, UrlMessage]]:
        """Non-blocking pull of up to ``limit`` messages.

        Returns ``(delivery_tag, message)`` pairs. The caller MUST later call
        :meth:`ack` or :meth:`nack` for each tag. This polling style is what
        lets a Scrapy spider - which drives its own event loop - take work from
        the queue whenever it goes idle.
        """
        ch = self._require_channel()
        out: list[tuple[int, UrlMessage]] = []
        for _ in range(limit):
            method, _props, body = ch.basic_get(queue=self.queue, auto_ack=False)
            if method is None:
                break
            try:
                out.append((method.delivery_tag, UrlMessage.from_json(body)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.error("Discarding malformed frontier message: %s", exc)
                ch.basic_nack(method.delivery_tag, requeue=False)
        return out

    def consume(
        self, handler: Callable[[UrlMessage], bool], max_messages: int | None = None
    ) -> int:
        """Blocking consume loop used by the standalone worker.

        ``handler`` returns True on success (message acked) or False on failure
        (message requeued, or dead-lettered once it has been retried
        ``MAX_DELIVERY_ATTEMPTS`` times).
        """
        ch = self._require_channel()
        processed = 0
        for method, _props, body in ch.consume(
            self.queue, auto_ack=False, inactivity_timeout=5.0
        ):
            if method is None:  # idle timeout - queue is empty
                break
            message = UrlMessage.from_json(body)
            try:
                ok = handler(message)
            except Exception as exc:  # noqa: BLE001 - a crash must not lose the URL
                logger.exception("Handler raised for %s: %s", message.url, exc)
                ok = False

            if ok:
                ch.basic_ack(method.delivery_tag)
            else:
                message.attempts += 1
                ch.basic_ack(method.delivery_tag)  # remove the old copy
                if message.attempts >= MAX_DELIVERY_ATTEMPTS:
                    self.dead_letter(message, "max attempts exceeded")
                else:
                    self.publish(message)  # retry with an incremented counter

            processed += 1
            if max_messages is not None and processed >= max_messages:
                break
        ch.cancel()
        return processed

    def ack(self, delivery_tag: int) -> None:
        self._require_channel().basic_ack(delivery_tag)

    def nack(self, delivery_tag: int, requeue: bool = True) -> None:
        self._require_channel().basic_nack(delivery_tag, requeue=requeue)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def depth(self) -> dict[str, int]:
        """Current message count of the frontier and the dead-letter queue."""
        ch = self._require_channel()
        main = ch.queue_declare(queue=self.queue, passive=True)
        dead = ch.queue_declare(queue=self.dlq, passive=True)
        return {
            self.queue: main.method.message_count,
            self.dlq: dead.method.message_count,
            "consumers": main.method.consumer_count,
        }

    def purge(self) -> None:
        """Empty both queues (used between test runs)."""
        ch = self._require_channel()
        ch.queue_purge(self.queue)
        ch.queue_purge(self.dlq)
        logger.info("Purged %s and %s", self.queue, self.dlq)

    def drain(self) -> Iterator[UrlMessage]:
        """Yield and acknowledge every message currently on the frontier."""
        while True:
            batch = self.pull(limit=50)
            if not batch:
                return
            for tag, message in batch:
                self.ack(tag)
                yield message
