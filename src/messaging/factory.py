"""Choosing the frontier backend (Task 7).

The framework has two frontier implementations with the same interface:
RabbitMQ from Task 2 and Kafka from Task 7. Which one is used is a setting,
not a code change, and this module is the one place that reads it.

Keeping both is a deliberate decision rather than indecision. They fail
differently, and that difference is the point:

* **RabbitMQ** is a work queue. The broker hands each message to whichever
  consumer asks next, so distribution is dynamic by construction and a slow
  node simply gets less work. It has per-message redelivery, so one failed
  URL can be retried without touching the others.
* **Kafka** is a partitioned log. Work is distributed by *assigning
  partitions* to members of a consumer group, so adding a node triggers a
  rebalance and nothing has to be told about it - which is what makes it fit
  a Kubernetes deployment whose replica count moves. Because a partition is
  ordered, every URL of one site goes to one node, and politeness per host
  actually holds. The cost is that there is no per-message nack, and records
  stay in the log after they are read, which is also what makes a replay
  possible.

``FRONTIER_BACKEND=rabbitmq`` (the default) keeps the Task 2 behaviour;
``FRONTIER_BACKEND=kafka`` switches the crawl onto the log. Everything
downstream - the spider, the seeder, the worker - goes through
:func:`get_frontier` and never knows the difference.
"""

from __future__ import annotations

import logging
from typing import Any

from src.common.config import settings

logger = logging.getLogger(__name__)

BACKENDS = ("rabbitmq", "kafka")


class UnknownBackend(ValueError):
    """Raised when FRONTIER_BACKEND names something that does not exist."""


def resolve_backend(name: str | None = None) -> str:
    """Normalise and validate a backend name."""
    choice = (name or settings.frontier_backend or "rabbitmq").strip().lower()
    # Friendly aliases, because "amqp" and "rabbit" are what people type.
    aliases = {"rabbit": "rabbitmq", "amqp": "rabbitmq", "kafka": "kafka"}
    choice = aliases.get(choice, choice)
    if choice not in BACKENDS:
        raise UnknownBackend(
            f"FRONTIER_BACKEND={choice!r} is not one of {BACKENDS}"
        )
    return choice


def get_frontier(backend: str | None = None, **kwargs: Any) -> Any:
    """Return a connected-on-demand frontier client for the chosen backend.

    The object is not connected yet: both implementations connect lazily or
    on an explicit ``connect()``, so building one is cheap and a caller that
    only wants to inspect configuration does not open a socket.
    """
    choice = resolve_backend(backend)

    if choice == "kafka":
        from src.messaging.kafka_frontier import KafkaFrontier

        logger.debug("Frontier backend: Kafka (%s)", settings.kafka_bootstrap_servers)
        return KafkaFrontier(**kwargs)

    from src.messaging.frontier import FrontierQueue

    logger.debug("Frontier backend: RabbitMQ (%s)", settings.rabbitmq_host)
    return FrontierQueue(**kwargs)


def describe_backend(backend: str | None = None) -> dict[str, Any]:
    """What the chosen backend is and where it points, for the reports."""
    choice = resolve_backend(backend)
    if choice == "kafka":
        return {
            "backend": "kafka",
            "servers": settings.kafka_bootstrap_servers,
            "topic": settings.kafka_topic,
            "dead_letter_topic": settings.kafka_dlq_topic,
            "group": settings.kafka_group_id,
            "partitions": settings.kafka_partitions,
            "distribution": "partition assignment within a consumer group",
        }
    return {
        "backend": "rabbitmq",
        "servers": f"{settings.rabbitmq_host}:{settings.rabbitmq_port}",
        "queue": settings.frontier_queue,
        "dead_letter_queue": settings.frontier_dlq,
        "prefetch": settings.frontier_prefetch,
        "distribution": "competing consumers on one queue",
    }
