"""Uniform logging across every node of the distributed system.

Each log line carries the component name and the hostname/PID, which is what
makes it possible to tell *which* worker produced a message once several
replicas are running.
"""

from __future__ import annotations

import logging
import os
import socket
import sys

_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(component)s@%(node)s | %(name)s | %(message)s"
)


class _NodeFilter(logging.Filter):
    """Inject the component name and node identity into every record."""

    def __init__(self, component: str) -> None:
        super().__init__()
        self.component = component
        self.node = f"{socket.gethostname()}:{os.getpid()}"

    def filter(self, record: logging.LogRecord) -> bool:
        record.component = self.component
        record.node = self.node
        return True


def setup_logging(component: str = "app", level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger and return a logger for ``component``."""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_NodeFilter(component))
    root.addHandler(handler)

    return logging.getLogger(component)
