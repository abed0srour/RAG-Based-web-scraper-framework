"""Node liveness: the signal Kubernetes restarts a stuck scraper on (Task 7).

A crawler node can fail in two quite different ways, and only one of them is
easy to detect:

* **It dies.** The process exits, the container exits, and the Kubernetes
  Deployment's restart policy replaces it. Nothing in this module is needed
  for that - a non-zero exit is already a signal.
* **It wedges.** The process is alive, holding its Kafka partitions or its
  RabbitMQ consumer slot, and doing nothing: a DNS lookup that never returns,
  a socket in a state the library does not time out, a deadlock between the
  reactor and a blocking call. This is the dangerous case, because every
  liveness signal that comes from the *process* rather than from its
  *progress* still looks healthy. The work queue simply stops moving and no
  alarm fires.

So the liveness signal here is progress, not existence. The node writes a
heartbeat file every time it finishes a page, carrying a timestamp and its
counters; :mod:`src.worker.healthcheck` reads that file and fails when it has
gone stale. Wired to a Kubernetes ``livenessProbe``, a wedged node is killed
and replaced without anyone watching; wired to a Docker ``healthcheck``, the
same is true under compose.

Two implementation details that matter more than they look:

* **The write is atomic.** The file is written to a temporary name and then
  renamed over the target, because a probe reading a half-written file would
  fail to parse it and declare a healthy node dead. ``os.replace`` is atomic
  on both POSIX and Windows.
* **A missing file is not failure, at first.** A node that has just started
  has not finished a page yet. The heartbeat records ``started_at``, and
  :func:`HeartbeatStatus.healthy` allows a grace period from it, which is
  what the probe's ``initialDelaySeconds`` cannot express on its own when
  the first page is slow.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.config import settings

logger = logging.getLogger(__name__)

#: Default location, under the data directory so it is git-ignored and, in
#: Kubernetes, lands on an ``emptyDir`` that dies with the pod - which is
#: correct: a heartbeat must never outlive the process it describes.
#: ``HEARTBEAT_DIR`` overrides it, which is how the Kubernetes manifests
#: point it at a mounted volume shared with the probe.
DEFAULT_DIR = Path(
    os.environ.get("HEARTBEAT_DIR") or (Path(settings.data_dir) / "heartbeats")
)


def node_id() -> str:
    """This node's name: the pod name under Kubernetes, else host:pid.

    ``HOSTNAME`` is the pod name in a Kubernetes container, which is what
    makes a heartbeat traceable back to a specific replica in ``kubectl``.
    """
    explicit = os.environ.get("WORKER_ID") or os.environ.get("NODE_ID")
    if explicit:
        return explicit
    host = os.environ.get("HOSTNAME") or socket.gethostname()
    return f"{host}:{os.getpid()}"


@dataclass
class HeartbeatStatus:
    """A heartbeat file as read back by the health check."""

    path: Path
    exists: bool
    age_seconds: float | None = None
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def started_seconds_ago(self) -> float | None:
        started = self.data.get("started_at")
        if not isinstance(started, (int, float)):
            return None
        return max(0.0, time.time() - float(started))

    def healthy(self, max_age: float, grace: float = 120.0) -> bool:
        """Whether this node should be considered alive.

        ``max_age`` is how long a node may go without finishing a page.
        ``grace`` covers start-up: a node that has only just begun has not
        written a page-completion beat yet, and killing it for that would
        produce a restart loop that never gets as far as the first page.
        """
        if self.error is not None:
            return False
        if not self.exists:
            return False
        if self.age_seconds is None:
            return False
        if self.age_seconds <= max_age:
            return True

        # Stale, but still inside the start-up grace window.
        started = self.started_seconds_ago
        return started is not None and started <= grace

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "age_seconds": (
                round(self.age_seconds, 2) if self.age_seconds is not None else None
            ),
            "error": self.error,
            **self.data,
        }


class Heartbeat:
    """Writes one node's liveness file."""

    def __init__(
        self,
        name: str | None = None,
        directory: Path | str | None = None,
    ) -> None:
        self.name = name or node_id()
        self.directory = Path(directory) if directory else DEFAULT_DIR
        # A pod name contains no path separators, but a host:pid does on
        # nothing - still, ':' is illegal in a Windows filename.
        safe = self.name.replace(":", "_").replace("/", "_")
        self.path = self.directory / f"{safe}.json"
        self.started_at = time.time()
        self.beats = 0

    def beat(self, **stats: Any) -> Path:
        """Record progress. Called once per completed page.

        Any keyword arguments are stored alongside the timestamps, which is
        how the metrics endpoint and ``kubectl exec`` end up reporting the
        same counters the crawler itself has.
        """
        self.beats += 1
        payload = {
            "node": self.name,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": time.time(),
            "beats": self.beats,
            "uptime_seconds": round(time.time() - self.started_at, 3),
            **stats,
        }

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a probe must never read a partial file and
            # conclude that a healthy node is dead.
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=".heartbeat-",
                suffix=".tmp",
                delete=False,
            )
            try:
                json.dump(payload, handle, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            os.replace(handle.name, self.path)
        except OSError as exc:
            # A node that cannot write its heartbeat must not crash because
            # of it: that would turn a monitoring problem into an outage.
            logger.warning("Could not write heartbeat to %s: %s", self.path, exc)

        return self.path

    def clear(self) -> None:
        """Remove the file on a clean shutdown.

        A node that exits deliberately should not leave a heartbeat behind
        that ages into a false alarm.
        """
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - best effort
            logger.debug("Could not remove heartbeat %s: %s", self.path, exc)


def read_heartbeat(
    name: str | None = None, directory: Path | str | None = None
) -> HeartbeatStatus:
    """Read one node's heartbeat file."""
    resolved_dir = Path(directory) if directory else DEFAULT_DIR
    resolved_name = (name or node_id()).replace(":", "_").replace("/", "_")
    path = resolved_dir / f"{resolved_name}.json"

    if not path.is_file():
        return HeartbeatStatus(path=path, exists=False)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return HeartbeatStatus(path=path, exists=True, error=str(exc))

    updated = data.get("updated_at")
    age = (
        max(0.0, time.time() - float(updated))
        if isinstance(updated, (int, float))
        else None
    )
    return HeartbeatStatus(path=path, exists=True, age_seconds=age, data=data)


def list_heartbeats(directory: Path | str | None = None) -> list[HeartbeatStatus]:
    """Every node's heartbeat, for the fleet report."""
    resolved = Path(directory) if directory else DEFAULT_DIR
    if not resolved.is_dir():
        return []

    out: list[HeartbeatStatus] = []
    for path in sorted(resolved.glob("*.json")):
        out.append(read_heartbeat(path.stem, resolved))
    return out
