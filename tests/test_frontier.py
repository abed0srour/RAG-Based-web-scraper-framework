"""Tests for the RabbitMQ URL frontier (Task 2).

The message-serialisation tests run everywhere. The integration tests need a
live broker and are skipped automatically when one is not reachable, so the
suite still passes on a machine where the containers are not running.
"""

from __future__ import annotations

import json
import uuid

import pytest

from src.messaging.frontier import (
    MAX_DELIVERY_ATTEMPTS,
    FrontierQueue,
    UrlMessage,
)


class TestUrlMessage:
    def test_round_trips_through_json(self) -> None:
        original = UrlMessage(
            url="https://example.com/a",
            site="example.com",
            depth=2,
            job_id=7,
            attempts=1,
            discovered_from="https://example.com/",
            meta={"note": "seed"},
        )
        restored = UrlMessage.from_json(original.to_json())
        assert restored == original

    def test_serialises_to_valid_json(self) -> None:
        payload = json.loads(
            UrlMessage(url="https://a.com", site="a.com").to_json().decode()
        )
        assert payload["url"] == "https://a.com"
        assert payload["depth"] == 0
        assert payload["attempts"] == 0

    def test_defaults_are_applied(self) -> None:
        message = UrlMessage(url="https://a.com", site="a.com")
        assert message.depth == 0
        assert message.job_id is None
        assert message.meta == {}

    def test_tolerates_unknown_fields(self) -> None:
        """An older/newer producer must not break the consumer."""
        body = json.dumps(
            {"url": "https://a.com", "site": "a.com", "future_field": 123}
        ).encode()
        message = UrlMessage.from_json(body)
        assert message.url == "https://a.com"

    def test_retry_limit_is_defined(self) -> None:
        assert MAX_DELIVERY_ATTEMPTS >= 1


def _broker_available() -> bool:
    try:
        queue = FrontierQueue(queue=f"test.probe.{uuid.uuid4().hex[:8]}")
        queue.connect(max_retries=1)
        queue.close()
        return True
    except Exception:  # noqa: BLE001
        return False


broker_required = pytest.mark.skipif(
    not _broker_available(), reason="RabbitMQ is not reachable"
)


@broker_required
class TestFrontierIntegration:
    """End-to-end queue behaviour against a live RabbitMQ broker."""

    @pytest.fixture()
    def frontier(self):  # type: ignore[no-untyped-def]
        # A unique queue name per test keeps these from touching real crawl data.
        suffix = uuid.uuid4().hex[:8]
        queue = FrontierQueue(
            queue=f"test.urls.{suffix}",
            dlq=f"test.dead.{suffix}",
            exchange=f"test.exchange.{suffix}",
        )
        queue.connect()
        yield queue
        try:
            queue.purge()
        finally:
            queue.close()

    def test_publish_then_pull(self, frontier: FrontierQueue) -> None:
        frontier.publish(UrlMessage(url="https://a.com/1", site="a.com"))
        pulled = frontier.pull(limit=5)
        assert len(pulled) == 1
        tag, message = pulled[0]
        assert message.url == "https://a.com/1"
        frontier.ack(tag)

    def test_queue_depth_reflects_published_messages(
        self, frontier: FrontierQueue
    ) -> None:
        frontier.publish_many(
            [UrlMessage(url=f"https://a.com/{i}", site="a.com") for i in range(5)]
        )
        assert frontier.depth()[frontier.queue] == 5

    def test_acknowledged_message_does_not_come_back(
        self, frontier: FrontierQueue
    ) -> None:
        frontier.publish(UrlMessage(url="https://a.com/x", site="a.com"))
        tag, _ = frontier.pull(limit=1)[0]
        frontier.ack(tag)
        assert frontier.pull(limit=1) == []

    def test_nacked_message_is_redelivered(self, frontier: FrontierQueue) -> None:
        """The guarantee that a crashed worker does not lose its URL."""
        frontier.publish(UrlMessage(url="https://a.com/retry", site="a.com"))
        tag, message = frontier.pull(limit=1)[0]
        frontier.nack(tag, requeue=True)

        redelivered = frontier.pull(limit=1)
        assert len(redelivered) == 1
        assert redelivered[0][1].url == message.url
        frontier.ack(redelivered[0][0])

    def test_dead_letter_queue_receives_failed_url(
        self, frontier: FrontierQueue
    ) -> None:
        frontier.dead_letter(
            UrlMessage(url="https://a.com/broken", site="a.com"), "unit test"
        )
        assert frontier.depth()[frontier.dlq] == 1

    def test_consume_acks_on_success(self, frontier: FrontierQueue) -> None:
        frontier.publish_many(
            [UrlMessage(url=f"https://a.com/c{i}", site="a.com") for i in range(3)]
        )
        seen: list[str] = []
        processed = frontier.consume(
            lambda m: (seen.append(m.url), True)[1], max_messages=3
        )
        assert processed == 3
        assert len(seen) == 3
        assert frontier.depth()[frontier.queue] == 0

    def test_failing_handler_eventually_dead_letters(
        self, frontier: FrontierQueue
    ) -> None:
        """A URL that always fails must not cycle through the queue forever."""
        frontier.publish(UrlMessage(url="https://a.com/always-fails", site="a.com"))
        frontier.consume(lambda _m: False, max_messages=MAX_DELIVERY_ATTEMPTS)

        depth = frontier.depth()
        assert depth[frontier.dlq] == 1
        assert depth[frontier.queue] == 0

    def test_purge_empties_the_queue(self, frontier: FrontierQueue) -> None:
        frontier.publish_many(
            [UrlMessage(url=f"https://a.com/p{i}", site="a.com") for i in range(4)]
        )
        frontier.purge()
        assert frontier.depth()[frontier.queue] == 0

    def test_drain_yields_every_message(self, frontier: FrontierQueue) -> None:
        frontier.publish_many(
            [UrlMessage(url=f"https://a.com/d{i}", site="a.com") for i in range(6)]
        )
        drained = list(frontier.drain())
        assert len(drained) == 6
        assert frontier.depth()[frontier.queue] == 0
