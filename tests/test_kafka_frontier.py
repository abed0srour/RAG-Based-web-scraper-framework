"""Tests for the Kafka frontier and the backend factory (Task 7).

No broker is started. What is tested here is the part that is ours: the
client configuration, the keying, the retry and dead-letter policy, and the
offset handling. Whether Kafka then distributes partitions correctly is
Kafka's own business, and a test against a fake broker would only be testing
the fake.

The configuration assertions look pedantic and are the most valuable tests in
the file. ``enable.auto.commit`` being left on, or the assignment strategy
being wrong, does not fail anything - it silently changes the delivery
guarantee or makes every rebalance stop the whole fleet.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.common.config import settings
from src.messaging.factory import (
    BACKENDS,
    UnknownBackend,
    describe_backend,
    get_frontier,
    resolve_backend,
)
from src.messaging.frontier import MAX_DELIVERY_ATTEMPTS, FrontierQueue, UrlMessage
from src.messaging.kafka_frontier import KafkaFrontier

TOPIC = "test.frontier"
DLQ = "test.dead-letter"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRecord:
    """One Kafka message, as the consumer hands it back."""

    def __init__(self, value: bytes, *, partition: int = 0, offset: int = 0,
                 error: Any = None) -> None:
        self._value = value
        self._partition = partition
        self._offset = offset
        self._error = error

    def value(self) -> bytes:
        return self._value

    def error(self) -> Any:
        return self._error

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset


class FakeProducer:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.produced: list[dict[str, Any]] = []
        self.flushed = 0

    def produce(self, topic: str, key: bytes | None = None,
                value: bytes | None = None, **_: Any) -> None:
        self.produced.append({"topic": topic, "key": key, "value": value})

    def poll(self, _timeout: float = 0) -> int:
        return 0

    def flush(self, _timeout: float = 0) -> int:
        self.flushed += 1
        return 0   # nothing left undelivered

    def list_topics(self, timeout: float = 0) -> Any:
        return FakeMetadata()


class FakeMetadata:
    def __init__(self) -> None:
        self.topics = {TOPIC: FakeTopicMetadata(), DLQ: FakeTopicMetadata(1)}


class FakeTopicMetadata:
    def __init__(self, partitions: int = 3) -> None:
        self.partitions = {index: object() for index in range(partitions)}


class FakeConsumer:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.subscribed: list[str] = []
        self.records: list[FakeRecord] = []
        self.committed_records: list[FakeRecord] = []
        self.committed_offsets: list[Any] = []
        self.closed = False

    def subscribe(self, topics: list[str]) -> None:
        self.subscribed = list(topics)

    def poll(self, timeout: float = 0) -> FakeRecord | None:
        return self.records.pop(0) if self.records else None

    def commit(self, message: Any = None, offsets: Any = None,
               asynchronous: bool = True) -> None:
        if message is not None:
            self.committed_records.append(message)
        if offsets is not None:
            self.committed_offsets.extend(offsets)

    def list_topics(self, timeout: float = 0) -> Any:
        return FakeMetadata()

    def committed(self, partitions: list[Any], timeout: float = 0) -> list[Any]:
        for partition in partitions:
            partition.offset = 5
        return partitions

    def get_watermark_offsets(self, _partition: Any, timeout: float = 0) -> tuple[int, int]:
        return (0, 10)

    def assignment(self) -> list[Any]:
        return [object(), object()]

    def close(self) -> None:
        self.closed = True


class FakeKafkaError:
    _PARTITION_EOF = -191


class FakeTopicPartition:
    def __init__(self, topic: str, partition: int, offset: int = -1001) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset


class FakeConsumerGroupTopicPartitions:
    def __init__(self, group: str, partitions: list[Any]) -> None:
        self.group_id = group
        self.topic_partitions = partitions


class FakeFuture:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self, timeout: float = 0) -> Any:
        return self._value


class FakeAdminClient:
    """Serves committed offsets without joining the consumer group."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def list_topics(self, timeout: float = 0) -> Any:
        return FakeMetadata()

    def list_consumer_group_offsets(self, requests: list[Any]) -> dict[str, Any]:
        request = requests[0]
        for partition in request.topic_partitions:
            partition.offset = 5
        return {request.group_id: FakeFuture(request)}


class FakeKafkaModule:
    """Stands in for the ``confluent_kafka`` module."""

    KafkaError = FakeKafkaError
    TopicPartition = FakeTopicPartition
    Producer = FakeProducer
    Consumer = FakeConsumer
    ConsumerGroupTopicPartitions = FakeConsumerGroupTopicPartitions


@pytest.fixture()
def kafka(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A KafkaFrontier with a fake producer and consumer already attached."""
    monkeypatch.setattr(
        "src.messaging.kafka_frontier._require_kafka", lambda: FakeKafkaModule
    )
    frontier = KafkaFrontier(topic=TOPIC, dlq_topic=DLQ, client_id="test-node")
    frontier._producer = FakeProducer()
    frontier._consumer = FakeConsumer()
    frontier._probe = FakeConsumer()
    monkeypatch.setattr(
        frontier, "_admin_client", lambda: FakeAdminClient(), raising=True
    )
    return frontier


def _message(url: str = "https://example.com/a", site: str = "example.com",
             attempts: int = 0) -> UrlMessage:
    return UrlMessage(url=url, site=site, depth=1, job_id=7, attempts=attempts)


# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------


class TestConsumerConfiguration:
    def test_offsets_are_not_committed_automatically(self, kafka: KafkaFrontier) -> None:
        """The delivery guarantee lives in this one setting.

        With auto-commit on, an offset advances on a timer rather than when
        the page is stored, so a node that dies has its URLs marked done and
        the pages are silently never crawled.
        """
        assert kafka._consumer_config()["enable.auto.commit"] is False

    def test_rebalancing_is_cooperative(self, kafka: KafkaFrontier) -> None:
        """Eager rebalancing revokes every assignment from every member, so
        adding one node briefly stops the entire fleet."""
        assert (
            kafka._consumer_config()["partition.assignment.strategy"]
            == "cooperative-sticky"
        )

    def test_a_silent_node_is_detected_and_replaced(self, kafka: KafkaFrontier) -> None:
        """session.timeout.ms is how long the coordinator waits before
        reassigning a dead member's partitions - the detection half of
        failover."""
        config = kafka._consumer_config()
        assert config["session.timeout.ms"] == settings.kafka_session_timeout_ms
        assert config["max.poll.interval.ms"] == settings.kafka_max_poll_interval_ms

    def test_every_node_joins_the_same_group(self, kafka: KafkaFrontier) -> None:
        """One group is what makes the partitions split rather than every
        node receiving every URL."""
        assert kafka._consumer_config()["group.id"] == settings.kafka_group_id


class TestProducerConfiguration:
    def test_a_publish_waits_for_every_replica(self, kafka: KafkaFrontier) -> None:
        """A lost URL is a page never crawled, and the frontier is small
        enough that the latency does not matter."""
        assert kafka._producer_config()["acks"] == "all"

    def test_publishing_is_idempotent(self, kafka: KafkaFrontier) -> None:
        """Without this, a producer-side retry can write the same URL twice."""
        assert kafka._producer_config()["enable.idempotence"] is True

    def test_the_client_identifies_itself(self, kafka: KafkaFrontier) -> None:
        """The client id appears in Kafka's group description, which is how a
        rebalance is traced back to a node."""
        assert kafka._producer_config()["client.id"] == "test-node"


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


class TestPublishing:
    def test_a_url_is_keyed_by_its_site(self, kafka: KafkaFrontier) -> None:
        """The partitioner hashes the key, so one site lands on one
        partition and therefore on one node - which is what makes per-host
        politeness hold. Round-robin would scatter a host across the fleet,
        each node applying its own delay to a server being hit by all of
        them.
        """
        kafka.publish(_message())
        produced = kafka._producer.produced[0]
        assert produced["topic"] == TOPIC
        assert produced["key"] == b"example.com"

    def test_a_url_without_a_site_falls_back_to_its_own_url(
        self, kafka: KafkaFrontier
    ) -> None:
        """A null key would be distributed round-robin, losing the ordering
        the site key exists for."""
        kafka.publish(UrlMessage(url="https://x.test/a", site=""))
        assert kafka._producer.produced[0]["key"] == b"https://x.test/a"

    def test_the_payload_round_trips(self, kafka: KafkaFrontier) -> None:
        kafka.publish(_message())
        restored = UrlMessage.from_json(kafka._producer.produced[0]["value"])
        assert restored.url == "https://example.com/a"
        assert restored.job_id == 7
        assert restored.depth == 1

    def test_a_batch_is_flushed_before_it_is_called_delivered(
        self, kafka: KafkaFrontier
    ) -> None:
        """produce() only buffers; without the flush the function would
        report success for messages still in memory."""
        sent = kafka.publish_many([_message(f"https://example.com/{i}") for i in range(5)])
        assert sent == 5
        assert kafka._producer.flushed == 1

    def test_a_dead_letter_records_why(self, kafka: KafkaFrontier) -> None:
        kafka.dead_letter(_message(), "max attempts exceeded")
        produced = kafka._producer.produced[0]
        assert produced["topic"] == DLQ
        assert json.loads(produced["value"])["dead_letter_reason"] == (
            "max attempts exceeded"
        )


# ---------------------------------------------------------------------------
# Consuming
# ---------------------------------------------------------------------------


class TestPulling:
    def test_pull_returns_tagged_messages(self, kafka: KafkaFrontier) -> None:
        kafka._consumer.records = [FakeRecord(_message().to_json())]
        pulled = kafka.pull(limit=5)
        assert len(pulled) == 1
        tag, message = pulled[0]
        assert tag == 0
        assert message.url == "https://example.com/a"

    def test_pull_stops_at_the_limit(self, kafka: KafkaFrontier) -> None:
        kafka._consumer.records = [FakeRecord(_message().to_json()) for _ in range(5)]
        assert len(kafka.pull(limit=2)) == 2

    def test_an_empty_topic_returns_nothing(self, kafka: KafkaFrontier) -> None:
        assert kafka.pull(limit=3) == []

    def test_acking_commits_that_record(self, kafka: KafkaFrontier) -> None:
        record = FakeRecord(_message().to_json())
        kafka._consumer.records = [record]
        tag, _message_ = kafka.pull()[0]
        kafka.ack(tag)
        assert kafka._consumer.committed_records == [record]

    def test_an_unknown_tag_is_ignored_rather_than_raising(
        self, kafka: KafkaFrontier
    ) -> None:
        """A spider that acks twice, or acks after a close, must not crash the
        crawl over bookkeeping."""
        kafka.ack(99)
        assert kafka._consumer.committed_records == []

    def test_a_commit_rejected_after_a_rebalance_does_not_kill_the_node(
        self, kafka: KafkaFrontier
    ) -> None:
        """Regression test for a crash seen against a real broker.

        Between the pull and the ack the group can rebalance, and the broker
        then rejects the commit with ILLEGAL_GENERATION because the member's
        generation is stale. The first version of this client let that
        exception escape, which took the crawler node down over an event the
        protocol treats as routine - and left the group churning.

        The right response is to log it: the record will be re-delivered to
        whoever now owns the partition, which is the at-least-once guarantee
        this frontier already provides.
        """
        kafka._consumer.records = [FakeRecord(_message().to_json())]
        tag, _ = kafka.pull()[0]

        def rejected(**_kwargs: Any) -> None:
            raise RuntimeError(
                'Commit failed: Broker: Specified group generation id is not valid'
            )

        kafka._consumer.commit = rejected   # type: ignore[method-assign]

        kafka.ack(tag)          # must not raise
        kafka.nack(tag)         # nor here
        assert kafka.consume(lambda _m: True) == 0

    def test_a_malformed_record_is_committed_past_and_skipped(
        self, kafka: KafkaFrontier
    ) -> None:
        """It cannot be left uncommitted: a partition is ordered, so the
        whole partition would stop at the bad record forever.
        """
        bad = FakeRecord(b"{not json")
        good = FakeRecord(_message().to_json())
        kafka._consumer.records = [bad, good]

        pulled = kafka.pull(limit=2)
        assert len(pulled) == 1
        assert bad in kafka._consumer.committed_records


class TestRetryPolicy:
    def test_a_nack_republishes_with_an_incremented_counter(
        self, kafka: KafkaFrontier
    ) -> None:
        """There is no in-place requeue on a log: a consumer cannot be handed
        one record again without rewinding past everything after it. So the
        URL is published afresh."""
        kafka._consumer.records = [FakeRecord(_message().to_json())]
        tag, _ = kafka.pull()[0]

        kafka.nack(tag, requeue=True)

        republished = UrlMessage.from_json(kafka._producer.produced[0]["value"])
        assert republished.attempts == 1
        assert kafka._producer.produced[0]["topic"] == TOPIC

    def test_the_offset_moves_on_even_when_the_url_is_requeued(
        self, kafka: KafkaFrontier
    ) -> None:
        """The failed copy has been replaced by a new one, so leaving the
        offset behind would deliver it twice."""
        kafka._consumer.records = [FakeRecord(_message().to_json())]
        tag, _ = kafka.pull()[0]
        kafka.nack(tag, requeue=True)
        assert len(kafka._consumer.committed_records) == 1

    def test_the_last_attempt_dead_letters_instead(self, kafka: KafkaFrontier) -> None:
        kafka._consumer.records = [
            FakeRecord(_message(attempts=MAX_DELIVERY_ATTEMPTS - 1).to_json())
        ]
        tag, _ = kafka.pull()[0]

        kafka.nack(tag, requeue=True)

        assert kafka._producer.produced[0]["topic"] == DLQ

    def test_a_nack_without_requeue_drops_the_url(self, kafka: KafkaFrontier) -> None:
        kafka._consumer.records = [FakeRecord(_message().to_json())]
        tag, _ = kafka.pull()[0]
        kafka.nack(tag, requeue=False)
        assert kafka._producer.produced == []
        assert len(kafka._consumer.committed_records) == 1


class TestConsumeLoop:
    def test_a_successful_handler_commits_each_record(
        self, kafka: KafkaFrontier
    ) -> None:
        kafka._consumer.records = [
            FakeRecord(_message(f"https://example.com/{i}").to_json()) for i in range(3)
        ]
        seen: list[str] = []

        processed = kafka.consume(lambda message: bool(seen.append(message.url)) or True)

        assert processed == 3
        assert len(seen) == 3
        assert len(kafka._consumer.committed_records) == 3
        assert kafka._producer.produced == []

    def test_a_failing_handler_republishes_the_url(self, kafka: KafkaFrontier) -> None:
        kafka._consumer.records = [FakeRecord(_message().to_json())]
        kafka.consume(lambda _message: False)
        assert UrlMessage.from_json(kafka._producer.produced[0]["value"]).attempts == 1

    def test_a_handler_that_raises_is_treated_as_a_failure(
        self, kafka: KafkaFrontier
    ) -> None:
        """A crash in the handler must not lose the URL."""
        kafka._consumer.records = [FakeRecord(_message().to_json())]

        def explode(_message: UrlMessage) -> bool:
            raise RuntimeError("handler bug")

        assert kafka.consume(explode) == 1
        assert kafka._producer.produced[0]["topic"] == TOPIC

    def test_max_messages_is_honoured(self, kafka: KafkaFrontier) -> None:
        kafka._consumer.records = [FakeRecord(_message().to_json()) for _ in range(10)]
        assert kafka.consume(lambda _m: True, max_messages=4) == 4

    def test_an_idle_topic_ends_the_loop(self, kafka: KafkaFrontier) -> None:
        assert kafka.consume(lambda _m: True) == 0


class TestIntrospection:
    def test_depth_reports_consumer_group_lag(self, kafka: KafkaFrontier) -> None:
        """The Kafka equivalent of a queue length: end offset minus committed
        offset, which is the number of URLs published but not finished - what
        a dashboard plots and an autoscaler scales on.
        """
        depth = kafka.depth()
        # Three partitions, each 10 - 5 outstanding.
        assert depth[TOPIC] == 15
        assert depth[DLQ] == 5
        assert depth["partitions_assigned"] == 2

    def test_reading_the_depth_does_not_join_the_consumer_group(
        self, kafka: KafkaFrontier
    ) -> None:
        """Observing the group must not change it.

        A client that subscribes in order to read offsets becomes a member,
        which rebalances the whole fleet - so a monitor polling the queue
        depth every fifteen seconds would keep the crawl permanently
        rebalancing. Measured against a live two-node fleet before this was
        changed: the lookup rebalanced the group and then timed out.
        """
        kafka._consumer = None          # no subscription exists yet
        kafka.depth()
        assert kafka._consumer is None, "depth() subscribed to the topic"

    def test_the_depth_falls_back_to_the_consumer_if_admin_fails(
        self, kafka: KafkaFrontier, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An old broker without the admin offsets API still reports a depth,
        at the cost of joining the group."""
        def unavailable() -> Any:
            raise RuntimeError("admin API not supported")

        monkeypatch.setattr(kafka, "_admin_client", unavailable)
        assert kafka.depth()[TOPIC] == 15

    def test_the_queue_name_is_aliased_for_the_shared_interface(
        self, kafka: KafkaFrontier
    ) -> None:
        """The spider logs and indexes by ``.queue``; the alias is what lets
        it work against either broker unchanged."""
        assert kafka.queue == kafka.topic == TOPIC

    def test_closing_leaves_the_group(self, kafka: KafkaFrontier) -> None:
        """A clean close triggers an immediate rebalance instead of making
        the group wait out the session timeout."""
        consumer = kafka._consumer
        kafka.close()
        assert consumer.closed is True
        assert kafka._producer is None


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


class TestBackendFactory:
    def test_the_default_is_the_task_2_broker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "frontier_backend", "rabbitmq")
        assert isinstance(get_frontier(), FrontierQueue)

    def test_kafka_can_be_selected_by_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "frontier_backend", "kafka")
        assert isinstance(get_frontier(), KafkaFrontier)

    @pytest.mark.parametrize(
        "name,expected",
        [("kafka", "kafka"), ("RabbitMQ", "rabbitmq"), ("rabbit", "rabbitmq"),
         ("amqp", "rabbitmq"), ("  KAFKA  ", "kafka")],
    )
    def test_names_are_normalised(self, name: str, expected: str) -> None:
        assert resolve_backend(name) == expected

    def test_an_unknown_backend_fails_loudly(self) -> None:
        """Better than silently falling back: a typo in FRONTIER_BACKEND
        would otherwise run the whole crawl on the wrong broker."""
        with pytest.raises(UnknownBackend):
            resolve_backend("pigeon")

    def test_both_backends_expose_the_methods_the_spider_calls(self) -> None:
        """The interface is what makes the backend a setting rather than a
        code change, so it is asserted rather than assumed."""
        required = (
            "connect", "close", "publish", "publish_many", "pull", "ack",
            "nack", "consume", "depth", "purge", "drain", "dead_letter",
            "queue",
        )
        for backend in BACKENDS:
            client = get_frontier(backend)
            for name in required:
                assert hasattr(client, name), f"{backend} is missing {name}"

    def test_the_description_says_how_work_is_distributed(self) -> None:
        assert "partition" in describe_backend("kafka")["distribution"]
        assert "consumers" in describe_backend("rabbitmq")["distribution"]
        assert describe_backend("kafka")["partitions"] == settings.kafka_partitions
