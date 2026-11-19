"""The URL frontier on **Kafka** - dynamic task distribution (Task 7).

Task 2 put the frontier on RabbitMQ, where work is distributed by competing
consumers: every node takes from one queue and the broker hands each message
to whoever asks next. That works, and for a pure work queue it is the simpler
model. This module implements the same frontier on Kafka, because the thing
Task 7 asks for - task distribution that adapts as nodes come and go - is
Kafka's consumer-group protocol rather than something the application has to
arrange.

**How the distribution works.** The topic has
``KAFKA_PARTITIONS`` partitions and every scraper node joins the same
consumer group. The group coordinator assigns each partition to exactly one
member, so the work is split without any node knowing about the others. When
a node starts, a rebalance hands it a share of the partitions; when a node
dies or stops polling, its partitions are reassigned to the survivors after
``KAFKA_SESSION_TIMEOUT_MS``. Nothing coordinates that: adding a replica *is*
the scale-out operation, which is the property the Kubernetes manifests in
``k8s/`` depend on.

The partition count is therefore the ceiling on useful parallelism - a group
can never have more working consumers than partitions - which is why the
default is 6 rather than 1.

**Why the key is the site.** Each message is produced with its site as the
key, so Kafka's default partitioner sends every URL of one site to the same
partition, and therefore to the same node. That matters for a crawler:
politeness is per host, so one node owning a host means its delay and
concurrency limits actually hold. A round-robin distribution would scatter
one site across every node, and each would apply its own limit to a host that
is being hit by all of them.

**Offsets are committed by hand.** ``enable.auto.commit`` is off, so a URL is
only marked done once the page has been processed. A node that dies
mid-request leaves its offsets uncommitted, and the URLs are re-delivered to
whoever takes the partition over. That is at-least-once delivery, which is
the same guarantee RabbitMQ gives here and the reason every write downstream
is keyed by a deterministic id: re-processing a page overwrites it instead of
duplicating it.

**Retries and the dead-letter topic.** Kafka has no per-message redelivery or
nack: a partition is an ordered log, and a consumer cannot ask for one record
again without rewinding the whole partition and re-reading everything after
it. So a failed URL is re-*published* with an incremented attempt counter,
exactly as the RabbitMQ implementation does on failure, and after
``MAX_DELIVERY_ATTEMPTS`` it goes to the dead-letter topic instead. This is
also why ``nack`` cannot requeue in place here, and says so.

The class mirrors :class:`src.messaging.frontier.FrontierQueue` method for
method, so :mod:`src.messaging.factory` can return either one and the spider
does not know which broker it is talking to.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import Any, Callable, Iterator

from src.common.config import settings
from src.messaging.frontier import MAX_DELIVERY_ATTEMPTS, UrlMessage

logger = logging.getLogger(__name__)


class KafkaUnavailable(RuntimeError):
    """Raised when the Kafka cluster cannot be reached or used."""


def _require_kafka() -> Any:
    try:
        import confluent_kafka
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise KafkaUnavailable(
            "confluent-kafka is not installed; run 'pip install -r requirements.txt'"
        ) from exc
    return confluent_kafka


class KafkaFrontier:
    """Kafka-backed URL frontier (producer and consumer-group member)."""

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic: str | None = None,
        dlq_topic: str | None = None,
        group_id: str | None = None,
        partitions: int | None = None,
        client_id: str | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic
        self.dlq = dlq_topic or settings.kafka_dlq_topic
        self.group_id = group_id or settings.kafka_group_id
        self.partitions = partitions or settings.kafka_partitions
        # ``queue`` is the name the RabbitMQ client uses for the same thing.
        # Keeping the alias means callers that log or index by it - the
        # spider does both - work against either backend unchanged.
        self.queue = self.topic
        # The client id shows up in Kafka's own metrics and in the group
        # description, which is how a rebalance is traced back to a node.
        self.client_id = client_id or f"scraper-{socket.gethostname()}"

        self._producer: Any = None
        self._consumer: Any = None
        # A group-less consumer used only for watermark lookups; see depth().
        self._probe: Any = None
        # Kafka acknowledges by offset, not per message, so the "delivery
        # tag" the shared interface hands back is an index into this list.
        self._in_flight: list[Any] = []

    # ------------------------------------------------------------------
    # connection handling
    # ------------------------------------------------------------------

    def _producer_config(self) -> dict[str, Any]:
        return {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": self.client_id,
            # Wait for every in-sync replica before calling a publish done.
            # A lost URL is a page never crawled, and the frontier is small
            # enough that the latency does not matter.
            "acks": "all",
            "enable.idempotence": True,
            "retries": 5,
            "linger.ms": 20,
            "compression.type": "snappy",
        }

    def _consumer_config(self) -> dict[str, Any]:
        return {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": self.client_id,
            "group.id": self.group_id,
            "auto.offset.reset": settings.kafka_auto_offset_reset,
            # The heart of the matter: commit only once the work is done.
            "enable.auto.commit": False,
            "session.timeout.ms": settings.kafka_session_timeout_ms,
            "max.poll.interval.ms": settings.kafka_max_poll_interval_ms,
            # Cooperative rebalancing moves only the partitions that have to
            # move, instead of revoking every assignment from every member
            # and handing them all out again. With eager rebalancing, one new
            # node briefly stops the whole fleet.
            "partition.assignment.strategy": "cooperative-sticky",
        }

    def connect(self, max_retries: int = 5) -> "KafkaFrontier":
        """Create the producer and ensure the topics exist.

        Retries, because in a compose or Kubernetes start-up the brokers are
        usually still electing a controller when the first node comes up.
        """
        confluent_kafka = _require_kafka()

        last: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                self._producer = confluent_kafka.Producer(self._producer_config())
                # list_topics doubles as the connectivity check: it is a real
                # metadata round trip, unlike constructing the producer.
                self._producer.list_topics(timeout=10)
                self.ensure_topics()
                logger.info(
                    "Kafka frontier connected to %s (topic %s, %d partitions)",
                    self.bootstrap_servers, self.topic, self.partitions,
                )
                return self
            except Exception as exc:  # noqa: BLE001 - retried below
                last = exc
                wait = min(2 ** attempt, 10)
                logger.warning(
                    "Kafka not ready (attempt %d/%d): %s - retrying in %ss",
                    attempt, max_retries, exc, wait,
                )
                time.sleep(wait)

        raise KafkaUnavailable(
            f"Kafka is not reachable at {self.bootstrap_servers}: {last}"
        )

    def ensure_topics(self) -> list[str]:
        """Create the frontier and dead-letter topics if they are missing.

        Auto-creation is deliberately not relied on: a topic created
        implicitly gets the broker's default partition count, and the
        partition count is what caps how many scraper nodes can work.
        """
        confluent_kafka = _require_kafka()
        from confluent_kafka.admin import AdminClient, NewTopic

        admin = AdminClient({"bootstrap.servers": self.bootstrap_servers})
        existing = set(admin.list_topics(timeout=10).topics)

        wanted = [
            NewTopic(
                name,
                num_partitions=self.partitions if name == self.topic else 1,
                replication_factor=settings.kafka_replication_factor,
            )
            for name in (self.topic, self.dlq)
            if name not in existing
        ]
        if not wanted:
            return []

        created: list[str] = []
        for name, future in admin.create_topics(wanted).items():
            try:
                future.result()
                created.append(name)
                logger.info("Created Kafka topic %s", name)
            except Exception as exc:  # noqa: BLE001
                # A concurrent node winning the race is not an error.
                if "TOPIC_ALREADY_EXISTS" in str(exc) or "already exists" in str(exc):
                    continue
                raise KafkaUnavailable(f"could not create topic {name}: {exc}") from exc
        return created

    def _subscribe(self) -> Any:
        """Join the consumer group, creating the consumer on first use."""
        if self._consumer is None:
            confluent_kafka = _require_kafka()
            self._consumer = confluent_kafka.Consumer(self._consumer_config())
            self._consumer.subscribe([self.topic])
            logger.info(
                "Joined consumer group %r on topic %s", self.group_id, self.topic
            )
        return self._consumer

    def _require_producer(self) -> Any:
        if self._producer is None:
            self.connect()
        return self._producer

    def close(self) -> None:
        """Flush pending publishes and leave the group cleanly.

        Closing the consumer triggers an immediate rebalance instead of
        making the group wait out the session timeout, so a planned shutdown
        costs seconds rather than most of a minute.
        """
        if self._producer is not None:
            self._producer.flush(10)
            self._producer = None
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception as exc:  # noqa: BLE001 - already gone
                logger.debug("Consumer close failed: %s", exc)
            self._consumer = None
        if self._probe is not None:
            try:
                self._probe.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Probe consumer close failed: %s", exc)
            self._probe = None
        self._in_flight.clear()

    def __enter__(self) -> "KafkaFrontier":
        return self.connect()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # publishing
    # ------------------------------------------------------------------

    def publish(self, message: UrlMessage) -> None:
        """Put one URL on the frontier, keyed by its site."""
        producer = self._require_producer()
        producer.produce(
            self.topic,
            key=(message.site or message.url).encode("utf-8"),
            value=message.to_json(),
        )
        # Serve delivery callbacks without blocking; flush() is what waits.
        producer.poll(0)

    def publish_many(self, messages: list[UrlMessage]) -> int:
        """Publish a batch and wait for the broker to confirm all of it."""
        for message in messages:
            self.publish(message)
        remaining = self._require_producer().flush(30)
        if remaining:
            logger.error("%d message(s) were not delivered to Kafka", remaining)
        return len(messages) - remaining

    def dead_letter(self, message: UrlMessage, reason: str) -> None:
        """Park a URL that has failed too often on the dead-letter topic."""
        producer = self._require_producer()
        payload = json.loads(message.to_json())
        payload["dead_letter_reason"] = reason
        producer.produce(
            self.dlq,
            key=(message.site or message.url).encode("utf-8"),
            value=json.dumps(payload).encode("utf-8"),
        )
        producer.poll(0)
        logger.warning("Dead-lettered %s: %s", message.url, reason)

    # ------------------------------------------------------------------
    # consuming
    # ------------------------------------------------------------------

    def pull(self, limit: int = 1) -> list[tuple[int, UrlMessage]]:
        """Non-blocking-ish pull of up to ``limit`` URLs.

        Returns ``(tag, message)`` pairs, where the tag is an index into this
        client's in-flight list rather than a broker-side delivery tag -
        Kafka has no such thing. The caller must still ack or nack each tag,
        which keeps the Scrapy spider's polling loop identical for both
        brokers.
        """
        consumer = self._subscribe()
        out: list[tuple[int, UrlMessage]] = []

        for _ in range(max(limit, 0)):
            record = consumer.poll(timeout=settings.kafka_poll_timeout)
            if record is None:
                break
            if record.error():
                confluent_kafka = _require_kafka()
                if record.error().code() == confluent_kafka.KafkaError._PARTITION_EOF:
                    break
                logger.error("Kafka consume error: %s", record.error())
                break
            try:
                message = UrlMessage.from_json(record.value())
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                # A malformed record cannot be left uncommitted: the
                # partition would stop at it forever. Commit past it and
                # keep the payload in the dead-letter topic for inspection.
                logger.error("Discarding malformed frontier record: %s", exc)
                self._commit(record)
                continue

            self._in_flight.append(record)
            out.append((len(self._in_flight) - 1, message))

        return out

    def _commit(self, record: Any) -> bool:
        """Commit one record's offset, surviving a rebalance.

        A commit is **not** allowed to kill the node, and the reason is
        specific: between the pull and the commit the group may have
        rebalanced, in which case the broker rejects the commit with
        ``ILLEGAL_GENERATION`` (the member's generation is stale) or
        ``UNKNOWN_MEMBER_ID``, and the partition may no longer even belong to
        this consumer. Letting that exception escape takes down a crawler
        node over an event the protocol considers routine - which is exactly
        what happened the first time this ran against a real broker, and it
        then left the group churning while the survivor was restarted.

        The consequence of a failed commit is that the record is re-delivered
        to whoever now owns the partition. That is at-least-once delivery,
        which this frontier already guarantees and every write downstream
        already tolerates, so the correct response is to log it and carry on.
        """
        try:
            self._subscribe().commit(message=record, asynchronous=False)
            return True
        except Exception as exc:  # noqa: BLE001 - deliberately broad: see above
            name = type(exc).__name__
            logger.warning(
                "Offset commit failed (%s: %s); the URL will be re-delivered "
                "to whichever node now owns the partition", name, exc,
            )
            return False

    def ack(self, delivery_tag: int) -> None:
        """Commit the offset of one pulled record."""
        record = self._record_for(delivery_tag)
        if record is None:
            return
        self._commit(record)

    def nack(self, delivery_tag: int, requeue: bool = True) -> None:
        """Give up on a record, optionally re-publishing it.

        There is no in-place requeue: a partition is an ordered log, so a
        consumer cannot be handed one record again without rewinding past
        everything after it. ``requeue=True`` therefore means "publish it
        again with its attempt counter incremented", which is what the
        RabbitMQ implementation does on failure as well.
        """
        record = self._record_for(delivery_tag)
        if record is None:
            return

        if requeue:
            try:
                message = UrlMessage.from_json(record.value())
                message.attempts += 1
                if message.attempts >= MAX_DELIVERY_ATTEMPTS:
                    self.dead_letter(message, "max attempts exceeded")
                else:
                    self.publish(message)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.error("Cannot requeue malformed record: %s", exc)

        # Either way the offset moves on: the copy that failed has been
        # replaced by a new one, or deliberately dropped.
        self._commit(record)

    def _record_for(self, delivery_tag: int) -> Any:
        if 0 <= delivery_tag < len(self._in_flight):
            return self._in_flight[delivery_tag]
        logger.warning("Unknown delivery tag %s", delivery_tag)
        return None

    def consume(
        self, handler: Callable[[UrlMessage], bool], max_messages: int | None = None
    ) -> int:
        """Blocking consume loop used by the standalone worker.

        ``handler`` returns True on success (offset committed) or False on
        failure (the URL is re-published with an incremented counter, or
        dead-lettered once it has been retried ``MAX_DELIVERY_ATTEMPTS``
        times). The loop ends when the topic has been quiet for one poll
        timeout, matching the RabbitMQ implementation's idle exit.
        """
        consumer = self._subscribe()
        confluent_kafka = _require_kafka()
        processed = 0

        while True:
            record = consumer.poll(timeout=settings.kafka_poll_timeout)
            if record is None:
                break
            if record.error():
                if record.error().code() == confluent_kafka.KafkaError._PARTITION_EOF:
                    break
                logger.error("Kafka consume error: %s", record.error())
                break

            message = UrlMessage.from_json(record.value())
            try:
                ok = handler(message)
            except Exception as exc:  # noqa: BLE001 - a crash must not lose the URL
                logger.exception("Handler raised for %s: %s", message.url, exc)
                ok = False

            if not ok:
                message.attempts += 1
                if message.attempts >= MAX_DELIVERY_ATTEMPTS:
                    self.dead_letter(message, "max attempts exceeded")
                else:
                    self.publish(message)

            # The offset is committed either way: on success because the work
            # is done, on failure because a fresh copy now carries it.
            self._commit(record)
            processed += 1
            if max_messages is not None and processed >= max_messages:
                break

        return processed

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def depth(self) -> dict[str, int]:
        """Frontier depth, as consumer-group lag per topic.

        The equivalent of RabbitMQ's message count, computed the Kafka way:
        the end offset of every partition minus the group's committed offset.
        That is the number of URLs published but not yet finished - the
        number a Grafana panel should plot and a Kubernetes autoscaler should
        scale on.

        It is read through the **admin client** rather than by subscribing,
        and that is not a detail. A consumer that joins the group to look at
        its offsets *is* a new member, so it triggers a rebalance of the
        whole fleet - and a monitoring tool asking "how deep is the queue?"
        every fifteen seconds would keep the group permanently rebalancing
        and the crawl would barely progress. Observing the group must not
        change it. (Measured here: a host-side ``depth()`` against two live
        nodes both rebalanced the fleet and then timed out reading offsets.)
        """
        try:
            return self._depth_via_admin()
        except Exception as exc:  # noqa: BLE001 - fall back below
            logger.debug("Admin lag lookup failed (%s); using the consumer", exc)

        return self._depth_via_consumer()

    def _admin_client(self) -> Any:
        """An AdminClient for this cluster.

        A method rather than an inline import so that the tests can replace
        it, as they do the rest of the client surface.
        """
        _require_kafka()
        from confluent_kafka.admin import AdminClient

        return AdminClient({"bootstrap.servers": self.bootstrap_servers})

    def _depth_via_admin(self) -> dict[str, int]:
        """Lag from the admin API, without joining the consumer group."""
        confluent_kafka = _require_kafka()
        ConsumerGroupTopicPartitions = confluent_kafka.ConsumerGroupTopicPartitions
        TopicPartition = confluent_kafka.TopicPartition

        admin = self._admin_client()
        metadata = admin.list_topics(timeout=10)

        out: dict[str, int] = {}
        for topic in (self.topic, self.dlq):
            topic_metadata = metadata.topics.get(topic)
            if topic_metadata is None:
                out[topic] = 0
                continue

            partitions = [
                TopicPartition(topic, partition)
                for partition in topic_metadata.partitions
            ]
            request = ConsumerGroupTopicPartitions(self.group_id, partitions)
            committed = admin.list_consumer_group_offsets([request])[
                self.group_id
            ].result(timeout=10).topic_partitions

            lag = 0
            for position in committed:
                low, high = self._watermarks(topic, position.partition)
                offset = position.offset if position.offset >= 0 else low
                lag += max(0, high - offset)
            out[topic] = lag

        out["partitions_assigned"] = len(
            self._consumer.assignment() if self._consumer is not None else []
        )
        out["consumers"] = 1
        return out

    def _watermarks(self, topic: str, partition: int) -> tuple[int, int]:
        """Low and high offsets of one partition.

        Uses a throwaway consumer with a unique group id, which never joins
        the real group: watermark lookups do not need group membership, and
        this keeps the fleet undisturbed.
        """
        confluent_kafka = _require_kafka()
        TopicPartition = confluent_kafka.TopicPartition

        if getattr(self, "_probe", None) is None:
            self._probe = confluent_kafka.Consumer({
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": f"{self.group_id}-watermark-probe",
                "enable.auto.commit": False,
            })
        return self._probe.get_watermark_offsets(
            TopicPartition(topic, partition), timeout=10
        )

    def _depth_via_consumer(self) -> dict[str, int]:
        """The original path, used only if the admin API is unavailable."""
        _require_kafka()
        from confluent_kafka import TopicPartition

        consumer = self._subscribe()
        metadata = consumer.list_topics(timeout=10)

        out: dict[str, int] = {}
        for topic in (self.topic, self.dlq):
            topic_metadata = metadata.topics.get(topic)
            if topic_metadata is None:
                out[topic] = 0
                continue

            partitions = [
                TopicPartition(topic, partition)
                for partition in topic_metadata.partitions
            ]
            committed = consumer.committed(partitions, timeout=10)

            lag = 0
            for position in committed:
                low, high = consumer.get_watermark_offsets(position, timeout=10)
                # An unset offset means nothing has been consumed yet, so the
                # whole partition is outstanding.
                offset = (
                    position.offset
                    if position.offset >= 0
                    else low
                )
                lag += max(0, high - offset)
            out[topic] = lag

        assignment = consumer.assignment()
        out["partitions_assigned"] = len(assignment)
        out["consumers"] = len({member for member in [self.client_id]})
        return out

    def purge(self) -> None:
        """Skip to the end of every partition (used between test runs).

        Kafka keeps records for its retention period whether or not they have
        been read, so "empty the queue" means "commit the end offsets" rather
        than deleting anything. The records remain in the log, which is the
        property that makes a replay possible at all.
        """
        from confluent_kafka import TopicPartition

        consumer = self._subscribe()
        metadata = consumer.list_topics(timeout=10)
        for topic in (self.topic, self.dlq):
            topic_metadata = metadata.topics.get(topic)
            if topic_metadata is None:
                continue
            positions = []
            for partition in topic_metadata.partitions:
                tp = TopicPartition(topic, partition)
                _low, high = consumer.get_watermark_offsets(tp, timeout=10)
                positions.append(TopicPartition(topic, partition, high))
            if positions:
                consumer.commit(offsets=positions, asynchronous=False)

    def drain(self) -> Iterator[UrlMessage]:
        """Yield every outstanding URL, committing as it goes."""
        while True:
            batch = self.pull(limit=50)
            if not batch:
                return
            for tag, message in batch:
                yield message
                self.ack(tag)
