"""Tests for node liveness and the scraper's metrics (Task 7).

The heartbeat is what the Kubernetes ``livenessProbe`` reads, so a bug here
has two shapes and both are bad: a false negative restarts a working node in
a loop, and a false positive leaves a wedged node holding its partitions
forever. Hence the attention to the awkward cases - a half-written file, a
node that has started but not yet finished a page, a clock that has not
moved.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from src.worker import metrics
from src.worker.healthcheck import main as healthcheck_main
from src.worker.heartbeat import (
    Heartbeat,
    HeartbeatStatus,
    list_heartbeats,
    node_id,
    read_heartbeat,
)


@pytest.fixture()
def beat_dir(tmp_path: Path) -> Path:
    return tmp_path / "heartbeats"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class TestWritingAHeartbeat:
    def test_a_beat_records_progress_and_timestamps(self, beat_dir: Path) -> None:
        heartbeat = Heartbeat("node-1", beat_dir)
        heartbeat.beat(pages_done=4, pages_failed=1)

        data = json.loads(heartbeat.path.read_text(encoding="utf-8"))
        assert data["node"] == "node-1"
        assert data["pages_done"] == 4
        assert data["pages_failed"] == 1
        assert data["beats"] == 1
        assert data["updated_at"] >= data["started_at"]

    def test_the_directory_is_created_on_demand(self, beat_dir: Path) -> None:
        assert not beat_dir.exists()
        Heartbeat("node-1", beat_dir).beat()
        assert beat_dir.is_dir()

    def test_each_beat_replaces_the_previous_one(self, beat_dir: Path) -> None:
        """One file per node, not a log: the probe needs the latest state,
        and a growing file in a container would be a slow leak."""
        heartbeat = Heartbeat("node-1", beat_dir)
        heartbeat.beat(pages_done=1)
        heartbeat.beat(pages_done=2)

        assert len(list(beat_dir.glob("*.json"))) == 1
        assert json.loads(heartbeat.path.read_text(encoding="utf-8"))["pages_done"] == 2
        assert heartbeat.beats == 2

    def test_no_partial_file_is_ever_left_behind(self, beat_dir: Path) -> None:
        """The write is to a temporary name and then renamed, because a probe
        that read a half-written file would fail to parse it and declare a
        healthy node dead."""
        heartbeat = Heartbeat("node-1", beat_dir)
        for _ in range(5):
            heartbeat.beat()
        assert [p.name for p in beat_dir.iterdir()] == ["node-1.json"]

    def test_a_node_name_with_a_colon_becomes_a_legal_filename(
        self, beat_dir: Path
    ) -> None:
        """``host:pid`` is the fallback node id, and ':' is illegal in a
        Windows filename."""
        heartbeat = Heartbeat("myhost:1234", beat_dir)
        heartbeat.beat()
        assert heartbeat.path.name == "myhost_1234.json"
        assert heartbeat.path.is_file()

    def test_an_unwritable_directory_does_not_crash_the_crawler(
        self, beat_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A node that cannot write its heartbeat must not die of it - that
        would turn a monitoring problem into an outage."""
        heartbeat = Heartbeat("node-1", beat_dir)

        def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "mkdir", refuse)
        heartbeat.beat(pages_done=1)   # must not raise

    def test_a_clean_shutdown_removes_the_file(self, beat_dir: Path) -> None:
        """A node that exits deliberately should not leave a heartbeat to age
        into a false alarm about a node that is gone on purpose."""
        heartbeat = Heartbeat("node-1", beat_dir)
        heartbeat.beat()
        heartbeat.clear()
        assert not heartbeat.path.exists()

    def test_clearing_twice_is_harmless(self, beat_dir: Path) -> None:
        heartbeat = Heartbeat("node-1", beat_dir)
        heartbeat.clear()
        heartbeat.clear()


class TestNodeIdentity:
    def test_an_explicit_worker_id_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORKER_ID", "scraper-7")
        assert node_id() == "scraper-7"

    def test_the_id_is_derived_from_the_pod_name_in_a_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HOSTNAME is the pod name in a container, which is what makes a
        heartbeat traceable back to a replica in kubectl.

        The pid is still appended, because HOSTNAME is also exported by many
        shells and two nodes started on one host must not share an id and
        overwrite each other's heartbeat. In Kubernetes that suffix never
        appears: k8s/20-scraper.yaml sets WORKER_ID from metadata.name, which
        takes precedence (see the test above).
        """
        monkeypatch.delenv("WORKER_ID", raising=False)
        monkeypatch.delenv("NODE_ID", raising=False)
        monkeypatch.setenv("HOSTNAME", "scraper-5d4f-abcde")
        assert node_id().startswith("scraper-5d4f-abcde:")

    def test_two_nodes_on_one_host_get_different_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WORKER_ID", raising=False)
        monkeypatch.delenv("NODE_ID", raising=False)
        monkeypatch.delenv("HOSTNAME", raising=False)
        assert ":" in node_id()


# ---------------------------------------------------------------------------
# Reading and judging
# ---------------------------------------------------------------------------


class TestReadingAHeartbeat:
    def test_a_fresh_beat_is_healthy(self, beat_dir: Path) -> None:
        Heartbeat("node-1", beat_dir).beat(pages_done=1)
        status = read_heartbeat("node-1", beat_dir)
        assert status.exists
        assert status.age_seconds is not None and status.age_seconds < 5
        assert status.healthy(max_age=180) is True

    def test_a_missing_heartbeat_is_not_healthy(self, beat_dir: Path) -> None:
        status = read_heartbeat("never-ran", beat_dir)
        assert status.exists is False
        assert status.healthy(max_age=180) is False

    def test_an_unreadable_heartbeat_is_not_healthy(self, beat_dir: Path) -> None:
        beat_dir.mkdir(parents=True)
        (beat_dir / "node-1.json").write_text("{truncated", encoding="utf-8")
        status = read_heartbeat("node-1", beat_dir)
        assert status.error is not None
        assert status.healthy(max_age=180) is False

    def test_a_stale_heartbeat_fails(self, beat_dir: Path) -> None:
        """The failure the whole mechanism exists for: the process is alive,
        the metrics endpoint answers, and no page has been stored."""
        beat_dir.mkdir(parents=True)
        long_ago = time.time() - 600
        (beat_dir / "node-1.json").write_text(
            json.dumps({"node": "node-1", "started_at": long_ago, "updated_at": long_ago}),
            encoding="utf-8",
        )
        status = read_heartbeat("node-1", beat_dir)
        assert status.age_seconds is not None and status.age_seconds > 500
        assert status.healthy(max_age=180) is False

    def test_a_just_started_node_is_given_a_grace_period(
        self, beat_dir: Path
    ) -> None:
        """A node that has not finished its first page yet is not wedged.
        Without this the probe produces a restart loop that never reaches the
        first page.
        """
        beat_dir.mkdir(parents=True)
        now = time.time()
        (beat_dir / "node-1.json").write_text(
            json.dumps({
                "node": "node-1",
                # Started 90s ago, last beat 60s ago: stale by the 30s limit
                # below, but still inside a 120s start-up window.
                "started_at": now - 90,
                "updated_at": now - 60,
            }),
            encoding="utf-8",
        )
        status = read_heartbeat("node-1", beat_dir)
        assert status.healthy(max_age=30, grace=120) is True
        # The same file fails once the node is older than the grace window:
        # by then a quiet node is wedged rather than starting up.
        assert status.healthy(max_age=30, grace=45) is False

    def test_an_idle_node_is_healthy_if_it_is_still_polling(
        self, beat_dir: Path
    ) -> None:
        """An empty frontier is not a stalled node.

        Observed on a live fleet: a node that waits for work stores no page,
        so a heartbeat written only on a stored page goes stale and the
        liveness probe restarts a perfectly healthy node every few minutes.
        The spider therefore beats on each idle round too, recording
        ``state="idle"`` with its page count unchanged - so "alive" means
        "still polling the frontier", and a wedged node is one that has
        stopped even doing that.
        """
        heartbeat = Heartbeat("idle-node", beat_dir)
        heartbeat.beat(pages_done=0, idle_rounds=1, state="idle")
        heartbeat.beat(pages_done=0, idle_rounds=2, state="idle")

        status = read_heartbeat("idle-node", beat_dir)
        assert status.healthy(max_age=180) is True
        assert status.data["state"] == "idle"
        assert status.data["pages_done"] == 0
        assert status.data["beats"] == 2

    def test_a_heartbeat_without_a_timestamp_is_not_trusted(
        self, beat_dir: Path
    ) -> None:
        beat_dir.mkdir(parents=True)
        (beat_dir / "node-1.json").write_text('{"node": "node-1"}', encoding="utf-8")
        assert read_heartbeat("node-1", beat_dir).healthy(max_age=180) is False

    def test_the_fleet_can_be_listed(self, beat_dir: Path) -> None:
        for name in ("node-1", "node-2", "node-3"):
            Heartbeat(name, beat_dir).beat(pages_done=2)
        statuses = list_heartbeats(beat_dir)
        assert len(statuses) == 3
        assert all(status.healthy(max_age=180) for status in statuses)

    def test_listing_a_missing_directory_is_empty_not_an_error(
        self, tmp_path: Path
    ) -> None:
        assert list_heartbeats(tmp_path / "nothing-here") == []

    def test_the_status_serialises_for_the_json_output(self, beat_dir: Path) -> None:
        Heartbeat("node-1", beat_dir).beat(pages_done=3)
        data = read_heartbeat("node-1", beat_dir).as_dict()
        assert data["exists"] is True
        assert data["pages_done"] == 3
        assert isinstance(data["age_seconds"], float)


# ---------------------------------------------------------------------------
# The probe's exit code - the interface Kubernetes consumes
# ---------------------------------------------------------------------------


class TestHealthcheckExitCode:
    def test_a_working_node_exits_zero(self, beat_dir: Path) -> None:
        Heartbeat("node-1", beat_dir).beat(pages_done=1)
        assert healthcheck_main(
            ["--node", "node-1", "--dir", str(beat_dir), "--quiet"]
        ) == 0

    def test_a_stalled_node_exits_one(self, beat_dir: Path) -> None:
        """This non-zero exit is what gets the container restarted."""
        beat_dir.mkdir(parents=True)
        long_ago = time.time() - 600
        (beat_dir / "node-1.json").write_text(
            json.dumps({"node": "node-1", "started_at": long_ago, "updated_at": long_ago}),
            encoding="utf-8",
        )
        assert healthcheck_main(
            ["--node", "node-1", "--dir", str(beat_dir), "--quiet"]
        ) == 1

    def test_a_node_that_never_ran_exits_one(self, beat_dir: Path) -> None:
        assert healthcheck_main(
            ["--node", "ghost", "--dir", str(beat_dir), "--quiet"]
        ) == 1

    def test_the_max_age_is_configurable(self, beat_dir: Path) -> None:
        beat_dir.mkdir(parents=True)
        recent = time.time() - 60
        (beat_dir / "node-1.json").write_text(
            json.dumps({"node": "node-1", "started_at": recent - 600, "updated_at": recent}),
            encoding="utf-8",
        )
        args = ["--node", "node-1", "--dir", str(beat_dir), "--quiet"]
        assert healthcheck_main([*args, "--max-age", "120"]) == 0
        assert healthcheck_main([*args, "--max-age", "30"]) == 1

    def test_the_fleet_view_prints_every_node(
        self, beat_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        Heartbeat("node-1", beat_dir).beat(pages_done=5)
        Heartbeat("node-2", beat_dir).beat(pages_done=9)

        assert healthcheck_main(["--all", "--dir", str(beat_dir)]) == 0
        printed = capsys.readouterr().out
        assert "node-1" in printed and "node-2" in printed
        assert "2/2 node(s) making progress" in printed

    def test_the_fleet_is_unhealthy_only_when_nothing_works(
        self, beat_dir: Path
    ) -> None:
        """One dead node out of two is a restart, not an outage, so the fleet
        check must not fail on it."""
        beat_dir.mkdir(parents=True)
        long_ago = time.time() - 600
        (beat_dir / "dead.json").write_text(
            json.dumps({"node": "dead", "started_at": long_ago, "updated_at": long_ago}),
            encoding="utf-8",
        )
        assert healthcheck_main(["--all", "--dir", str(beat_dir), "--quiet"]) == 1

        Heartbeat("alive", beat_dir).beat(pages_done=1)
        assert healthcheck_main(["--all", "--dir", str(beat_dir), "--quiet"]) == 0

    def test_json_output_is_machine_readable(
        self, beat_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        Heartbeat("node-1", beat_dir).beat(pages_done=2)
        healthcheck_main(["--node", "node-1", "--dir", str(beat_dir), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["healthy"] is True
        assert payload["pages_done"] == 2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestScraperMetrics:
    def test_the_expected_series_are_defined(self) -> None:
        if not metrics.HAVE_PROMETHEUS:
            pytest.skip("prometheus_client is not installed")

        metrics.PAGES.labels(site="example.com", outcome="ok").inc()
        metrics.PAGES_STORED.labels(site="example.com").inc()
        metrics.URLS_PUBLISHED.labels(site="example.com").inc(3)
        metrics.URLS_PULLED.inc(5)
        metrics.DEAD_LETTERED.inc()
        metrics.HEARTBEAT_AGE.set(12)
        metrics.IN_FLIGHT.set(2)

        rendered = metrics.render().decode("utf-8")
        for name in (
            "scraper_pages_total",
            "scraper_pages_stored_total",
            "scraper_urls_published_total",
            "scraper_urls_pulled_total",
            "scraper_urls_dead_lettered_total",
            "scraper_heartbeat_age_seconds",
            "scraper_requests_in_flight",
        ):
            assert name in rendered, name

    def test_the_frontier_depth_of_either_broker_becomes_samples(self) -> None:
        """Both clients' depth() return queue/topic names mapped to counts,
        plus a couple of non-count keys that must not become gauges."""
        if not metrics.HAVE_PROMETHEUS:
            pytest.skip("prometheus_client is not installed")

        metrics.observe_frontier_depth({
            "urls.frontier": 42,
            "urls.dead-letter": 0,
            "consumers": 3,
            "partitions_assigned": 2,
        })
        rendered = metrics.render().decode("utf-8")
        assert 'scraper_frontier_depth{queue="urls.frontier"} 42.0' in rendered
        assert "partitions_assigned" not in rendered
        assert 'queue="consumers"' not in rendered

    def test_a_missing_dependency_does_not_break_the_crawler(self) -> None:
        """Monitoring must never be able to stop the thing it monitors, so
        the metrics degrade to no-ops rather than raising."""
        noop = metrics._NoopMetric()
        noop.labels(site="x").inc()
        noop.set(1)
        noop.observe(0.5)

    def test_metrics_use_a_private_registry(self) -> None:
        """A global registry raises "Duplicated timeseries" the second time a
        test process imports this, which is how the API's metrics module got
        the same treatment."""
        if not metrics.HAVE_PROMETHEUS:
            pytest.skip("prometheus_client is not installed")
        from prometheus_client import REGISTRY as GLOBAL_REGISTRY

        assert metrics.REGISTRY is not GLOBAL_REGISTRY

    def test_a_taken_port_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two nodes started on one host is the normal way to run this
        locally, and the second one must still crawl."""
        if not metrics.HAVE_PROMETHEUS:
            pytest.skip("prometheus_client is not installed")

        monkeypatch.setattr(metrics, "_server_started", False)

        def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("address already in use")

        monkeypatch.setattr(metrics, "start_http_server", refuse)
        assert metrics.start_metrics_server(9110, "node-1", "kafka") is False
