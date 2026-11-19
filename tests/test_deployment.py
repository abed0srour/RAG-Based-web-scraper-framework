"""Tests over the deployment configuration (Task 7).

Infrastructure in a repository fails in a particular way: two files that have
to agree slowly stop agreeing. The Nginx upstream names a service the compose
file renamed. The liveness probe's threshold and the alert rule's threshold
drift apart, so a node is restarted before anyone is told. The scraper's
autoscaler is allowed more replicas than Kafka has partitions, so the extra
pods idle. None of that fails a build - it just quietly does the wrong thing
in production.

These tests read the actual files and assert the relationships between them.
They need no Docker daemon and no cluster, which is the point: they are the
part of this stack that *can* be checked on a laptop.

What they deliberately do not do is validate the files against the Kubernetes
or Nginx schemas. That needs an API server and nginx itself; `kubectl apply
--dry-run=server` and `nginx -t` are the right tools and are named in the
Task 7 report as the remaining manual step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.common.config import settings
from src.worker.healthcheck import DEFAULT_MAX_AGE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = PROJECT_ROOT / "docker" / "docker-compose.yml"
NGINX_CONF = PROJECT_ROOT / "docker" / "nginx" / "nginx.conf"
PROMETHEUS = PROJECT_ROOT / "docker" / "prometheus" / "prometheus.yml"
ALERTS = PROJECT_ROOT / "docker" / "prometheus" / "alert_rules.yml"
DASHBOARD = PROJECT_ROOT / "docker" / "grafana" / "dashboards" / "rag-scraper.json"
K8S = PROJECT_ROOT / "k8s"


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_manifests(path: Path) -> list[dict[str, Any]]:
    return [
        document
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if document
    ]


def all_manifests() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(K8S.glob("*.yaml")):
        out.extend(load_manifests(path))
    return out


def find(kind: str, name: str) -> dict[str, Any]:
    for manifest in all_manifests():
        if manifest["kind"] == kind and manifest["metadata"]["name"] == name:
            return manifest
    raise AssertionError(f"{kind}/{name} is not declared in k8s/")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestComposeStack:
    def test_every_task_7_service_is_declared(self) -> None:
        services = load_yaml(COMPOSE)["services"]
        for name in (
            "kafka", "redis", "api1", "api2", "api3", "nginx",
            "scraper1", "scraper2", "prometheus", "grafana", "nginx-exporter",
        ):
            assert name in services, name

    def test_the_new_services_are_behind_a_profile(self) -> None:
        """The default `docker compose up -d` must still start only the data
        stores the earlier tasks need, not eleven more containers."""
        services = load_yaml(COMPOSE)["services"]
        for name in ("kafka", "api1", "nginx", "prometheus", "grafana", "scraper1"):
            assert services[name].get("profiles"), f"{name} has no profile"

    def test_the_stores_from_earlier_tasks_have_no_profile(self) -> None:
        services = load_yaml(COMPOSE)["services"]
        for name in ("postgres", "mongodb", "rabbitmq", "chroma", "elasticsearch"):
            assert "profiles" not in services[name], name

    def test_every_named_volume_is_declared(self) -> None:
        compose = load_yaml(COMPOSE)
        declared = set(compose.get("volumes") or {})
        for name, service in compose["services"].items():
            for volume in service.get("volumes") or []:
                if isinstance(volume, str) and not volume.startswith((".", "/")):
                    source = volume.split(":")[0]
                    assert source in declared, f"{name} mounts undeclared {source}"

    def test_the_api_replicas_are_identical_apart_from_their_name(self) -> None:
        """They sit behind one balancer, so a difference in configuration
        between them would show up as requests behaving differently
        depending on which replica served them."""
        services = load_yaml(COMPOSE)["services"]
        replicas = [services[f"api{i}"] for i in (1, 2, 3)]
        for replica in replicas[1:]:
            assert replica["image"] == replicas[0]["image"]
            for key in ("API_RATE_LIMIT_BACKEND", "REDIS_URL", "API_WARM_EMBEDDER"):
                assert replica["environment"][key] == replicas[0]["environment"][key]

    def test_the_replicas_share_a_rate_limiter(self) -> None:
        """Three replicas with in-memory counters would each allow the full
        limit, so a client would get three times what it was promised."""
        services = load_yaml(COMPOSE)["services"]
        for index in (1, 2, 3):
            environment = services[f"api{index}"]["environment"]
            assert environment["API_RATE_LIMIT_BACKEND"] == "redis"
            assert "redis" in environment["REDIS_URL"]

    def test_only_the_balancer_publishes_the_api_port(self) -> None:
        """A replica reachable directly on the host would bypass the
        balancer, its rate limit and its retry behaviour."""
        services = load_yaml(COMPOSE)["services"]
        for index in (1, 2, 3):
            assert "ports" not in services[f"api{index}"]
        assert "8080:80" in services["nginx"]["ports"]

    def test_a_crashed_scraper_node_is_restarted(self) -> None:
        services = load_yaml(COMPOSE)["services"]
        assert services["scraper1"]["restart"] == "always"

    def test_a_node_waits_for_work_instead_of_exiting(self) -> None:
        """Observed failure: with the old default the node exited after ~6s
        of an empty frontier, `restart: always` restarted it, and every
        restart rejoined the Kafka consumer group. Two nodes doing that kept
        the group in permanent rebalance and no work was ever picked up -
        13 and 14 restarts in fifteen minutes, zero pages stored.
        """
        dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.scraper").read_text(
            encoding="utf-8"
        )
        assert '"--idle-rounds", "0"' in dockerfile.replace("\\\n", "")

        args = find("Deployment", "scraper")["spec"]["template"]["spec"][
            "containers"
        ][0]["args"]
        assert args[args.index("--idle-rounds") + 1] == "0"

    def test_the_heartbeat_directory_does_not_survive_the_container(self) -> None:
        """A heartbeat that outlived its process would let a restarted node
        look healthy before it had done anything."""
        services = load_yaml(COMPOSE)["services"]
        mounts = services["scraper1"]["tmpfs"]
        assert any(m.split(":")[0] == "/var/run/scraper" for m in mounts)

    def test_the_heartbeat_directory_is_writable_by_the_container_user(self) -> None:
        """Observed failure: a tmpfs mount is created fresh and root-owned,
        which masks the ownership the Dockerfile set, so the non-root user
        could not write its heartbeat. The crawl kept working and the node
        was reported unhealthy indefinitely.
        """
        services = load_yaml(COMPOSE)["services"]
        mount = next(
            m for m in services["scraper1"]["tmpfs"]
            if m.split(":")[0] == "/var/run/scraper"
        )
        assert "uid=10001" in mount, "the tmpfs would be root-owned"
        assert "gid=10001" in mount

    def test_kafka_runs_without_zookeeper(self) -> None:
        environment = load_yaml(COMPOSE)["services"]["kafka"]["environment"]
        assert "controller" in environment["KAFKA_PROCESS_ROLES"]
        assert environment["KAFKA_CONTROLLER_QUORUM_VOTERS"]

    def test_kafka_does_not_auto_create_topics(self) -> None:
        """An implicitly created topic gets the broker default partition
        count, and the partition count is what caps the fleet size."""
        environment = load_yaml(COMPOSE)["services"]["kafka"]["environment"]
        assert environment["KAFKA_AUTO_CREATE_TOPICS_ENABLE"] == "false"

    def test_a_single_broker_can_satisfy_its_own_internal_topics(self) -> None:
        """With the default replication factor of 3, a one-broker cluster
        never finishes starting."""
        environment = load_yaml(COMPOSE)["services"]["kafka"]["environment"]
        assert environment["KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR"] == "1"
        assert environment["KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR"] == "1"


# ---------------------------------------------------------------------------
# Nginx
# ---------------------------------------------------------------------------


class TestNginx:
    @pytest.fixture(scope="class")
    def conf(self) -> str:
        return NGINX_CONF.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def directives(self) -> str:
        """The configuration with its comments stripped.

        Needed because several of these assertions are about a directive
        *not* being present, and the comment explaining why it is absent
        contains the very word being searched for.
        """
        return "\n".join(
            line for line in NGINX_CONF.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        )

    def test_the_upstream_names_the_compose_services(self, conf: str) -> None:
        """The drift this catches: renaming a service in the compose file
        leaves the balancer pointing at a host that no longer resolves."""
        services = set(load_yaml(COMPOSE)["services"])
        upstream = re.findall(r"^\s*server\s+([a-z0-9_-]+):(\d+)", conf, re.MULTILINE)
        assert upstream, "no upstream servers found"
        for host, port in upstream:
            assert host in services, f"upstream {host} is not a compose service"
            assert port == "8080"

    def test_every_api_replica_is_in_the_pool(self, conf: str) -> None:
        upstream = re.findall(r"^\s*server\s+(api\d):8080", conf, re.MULTILINE)
        assert sorted(upstream) == ["api1", "api2", "api3"]

    def test_requests_are_balanced_by_open_connections(self, conf: str) -> None:
        """These requests are not uniform: /v1/ask holds a connection for
        seconds while a listing returns in milliseconds, so counting
        requests does not track load."""
        assert "least_conn" in conf

    def test_a_failed_replica_is_taken_out_of_rotation(self, conf: str) -> None:
        assert "max_fails=3" in conf
        assert "fail_timeout=15s" in conf

    def test_a_request_is_retried_on_another_replica(self, conf: str) -> None:
        assert "proxy_next_upstream" in conf
        assert "http_502" in conf and "http_503" in conf

    def test_a_post_is_never_replayed(self, directives: str) -> None:
        """Replaying POST /v1/ask that may already have been processed would
        double a generation call, so non_idempotent must stay off."""
        assert "non_idempotent" not in directives

    def test_the_read_timeout_allows_for_generation(self, conf: str) -> None:
        """/v1/ask runs retrieval and generation; the default 60s would cut
        a legitimate request off."""
        match = re.search(r"proxy_read_timeout\s+(\d+)s", conf)
        assert match and int(match.group(1)) >= 120

    def test_a_request_id_is_always_set(self, conf: str) -> None:
        """Generated when the client did not send one, so a line in the
        Nginx log and a line in the API log can be tied together."""
        assert "X-Request-ID      $request_id" in conf

    def test_the_gateway_error_keeps_the_apis_json_shape(self, conf: str) -> None:
        """A client that sees nginx's stock HTML cannot tell a gateway
        failure from an API failure, and the UI's error handling breaks."""
        assert "no_backend_available" in conf
        assert "Retry-After" in conf

    def test_the_balancer_listens_on_ipv6_too(self, directives: str) -> None:
        """Observed failure: with IPv4 only, the container's own health check
        resolved localhost to ::1, got a connection refused, and reported a
        balancer that was serving traffic perfectly as unhealthy - which an
        orchestrator would act on.
        """
        assert "listen [::]:80" in directives

    def test_the_health_check_does_not_depend_on_name_resolution(self) -> None:
        """...and the check itself uses 127.0.0.1 rather than localhost, so
        it cannot be caught by the same thing again."""
        healthcheck = load_yaml(COMPOSE)["services"]["nginx"]["healthcheck"]["test"]
        assert any("127.0.0.1" in part for part in healthcheck)

    def test_the_probe_endpoints_are_not_logged(self, conf: str) -> None:
        """An orchestrator polling /health would otherwise be most of the
        access log."""
        health_block = conf.split("location /health")[1].split("}")[0]
        assert "access_log off" in health_block


# ---------------------------------------------------------------------------
# Prometheus and Grafana
# ---------------------------------------------------------------------------


class TestPrometheus:
    def test_every_api_replica_is_scraped_directly(self) -> None:
        """Not through the balancer: that would return whichever replica it
        picked, mixing three processes into one series and hiding the slow
        one - the main question per-replica metrics exist to answer."""
        jobs = {job["job_name"]: job for job in load_yaml(PROMETHEUS)["scrape_configs"]}
        targets = jobs["api"]["static_configs"][0]["targets"]
        assert sorted(targets) == ["api1:8080", "api2:8080", "api3:8080"]

    def test_the_scraper_nodes_are_scraped_on_their_metrics_port(self) -> None:
        jobs = {job["job_name"]: job for job in load_yaml(PROMETHEUS)["scrape_configs"]}
        for target in jobs["scrapers"]["static_configs"][0]["targets"]:
            assert target.endswith(f":{settings.worker_metrics_port}")

    def test_prometheus_scrapes_itself(self) -> None:
        """The series that distinguishes an outage from a monitoring
        failure."""
        names = [job["job_name"] for job in load_yaml(PROMETHEUS)["scrape_configs"]]
        assert "prometheus" in names

    def test_the_rule_file_is_loaded(self) -> None:
        assert load_yaml(PROMETHEUS)["rule_files"]


class TestAlertRules:
    @pytest.fixture(scope="class")
    def rules(self) -> list[dict[str, Any]]:
        return [
            rule
            for group in load_yaml(ALERTS)["groups"]
            for rule in group["rules"]
        ]

    def test_the_failures_that_matter_all_have_a_rule(
        self, rules: list[dict[str, Any]]
    ) -> None:
        names = {rule["alert"] for rule in rules}
        for expected in (
            "NoApiReplicaAvailable",     # the service is down
            "ApiReplicaDown",            # one replica is down
            "ScraperNodeStalled",        # a node runs but does not work
            "FrontierBacklogUnattended",  # work queued, nothing consuming
            "TargetMissing",             # monitoring itself has stopped
        ):
            assert expected in names, expected

    def test_every_rule_waits_before_firing(self, rules: list[dict[str, Any]]) -> None:
        """Without `for`, a replica restarting during a deploy pages someone."""
        for rule in rules:
            assert rule.get("for"), f"{rule['alert']} has no `for` duration"

    def test_every_rule_explains_itself(self, rules: list[dict[str, Any]]) -> None:
        for rule in rules:
            assert rule.get("annotations", {}).get("summary"), rule["alert"]
            assert rule.get("labels", {}).get("severity"), rule["alert"]

    def test_the_stall_alert_matches_the_liveness_threshold(
        self, rules: list[dict[str, Any]]
    ) -> None:
        """The drift this exists to catch: if the alert's threshold were
        higher than the probe's, a node would be restarted before anyone was
        told it had stalled, and the restart would look spontaneous.
        """
        stall = next(rule for rule in rules if rule["alert"] == "ScraperNodeStalled")
        threshold = int(re.search(r">\s*(\d+)", stall["expr"]).group(1))
        assert threshold == int(DEFAULT_MAX_AGE) == int(
            settings.worker_heartbeat_max_age
        )

    def test_the_error_rate_rule_cannot_divide_by_zero(
        self, rules: list[dict[str, Any]]
    ) -> None:
        """With no traffic the denominator is 0, and 0/0 is NaN - which never
        fires, so the alert would silently stop working at exactly the moment
        a total outage stopped all traffic."""
        rule = next(
            r for r in rules if r["alert"] == "ApiHighServerErrorRate"
        )
        assert "clamp_min" in rule["expr"]


class TestGrafanaDashboard:
    @pytest.fixture(scope="class")
    def dashboard(self) -> dict[str, Any]:
        return json.loads(DASHBOARD.read_text(encoding="utf-8"))

    def test_the_datasource_uid_matches_the_provisioned_one(
        self, dashboard: dict[str, Any]
    ) -> None:
        """A generated uid makes every panel show "datasource not found",
        which is the classic provisioning mistake."""
        provisioned = load_yaml(
            PROJECT_ROOT / "docker" / "grafana" / "provisioning"
            / "datasources" / "prometheus.yml"
        )["datasources"][0]
        assert provisioned["uid"] == "prometheus"

        for panel in dashboard["panels"]:
            if panel["type"] == "row":
                continue
            assert panel["datasource"]["uid"] == provisioned["uid"], panel["title"]

    def test_every_panel_queries_a_metric_the_code_exports(
        self, dashboard: dict[str, Any]
    ) -> None:
        """The other half of the drift problem: a renamed metric leaves a
        panel that is permanently empty and looks like a quiet system."""
        exported = {
            # src/api/metrics.py
            "api_requests_total",
            "api_request_duration_seconds_bucket",
            "api_rate_limited_total",
            "api_auth_failures_total",
            # src/worker/metrics.py
            "scraper_pages_total",
            "scraper_pages_stored_total",
            "scraper_urls_published_total",
            "scraper_urls_pulled_total",
            "scraper_urls_dead_lettered_total",
            "scraper_frontier_depth",
            "scraper_heartbeat_age_seconds",
            "scraper_requests_in_flight",
            "scraper_up",
            # built in, or from the nginx exporter
            "up",
            "nginx_http_requests_total",
            "nginx_connections_active",
            "nginx_connections_waiting",
        }
        for panel in dashboard["panels"]:
            for target in panel.get("targets") or []:
                referenced = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*[\({]", target["expr"]))
                referenced |= set(re.findall(r"\b(scraper_[a-z_]+|api_[a-z_]+|nginx_[a-z_]+)\b", target["expr"]))
                unknown = {
                    name for name in referenced
                    if name.startswith(("api_", "scraper_", "nginx_"))
                    and name not in exported
                }
                assert not unknown, f"{panel['title']}: unknown metric(s) {unknown}"


# ---------------------------------------------------------------------------
# Kubernetes
# ---------------------------------------------------------------------------


class TestKubernetesManifests:
    def test_every_manifest_parses_and_is_named(self) -> None:
        manifests = all_manifests()
        assert len(manifests) > 30
        for manifest in manifests:
            assert manifest.get("apiVersion")
            assert manifest["metadata"].get("name")

    def test_everything_lives_in_one_namespace(self) -> None:
        """Except the cluster-scoped objects, which cannot have one."""
        cluster_scoped = {"Namespace", "ClusterRole", "ClusterRoleBinding"}
        for manifest in all_manifests():
            if manifest["kind"] in cluster_scoped:
                continue
            assert manifest["metadata"].get("namespace") == "rag-scraper", (
                f"{manifest['kind']}/{manifest['metadata']['name']}"
            )

    def test_the_stateful_services_are_statefulsets(self) -> None:
        """A Deployment's rolling update would briefly run two PostgreSQL
        pods against one volume, which is how a data directory is
        corrupted."""
        kinds = {
            (manifest["kind"], manifest["metadata"]["name"])
            for manifest in all_manifests()
        }
        for name in ("postgres", "mongodb", "kafka", "elasticsearch", "chroma"):
            assert ("StatefulSet", name) in kinds, name

    # ---- the auto-restart deliverable ----

    def test_the_scraper_liveness_probe_checks_progress_not_existence(self) -> None:
        """The whole point of Task 7's auto-restart: an HTTP or TCP probe
        answers happily while the crawl is wedged, because the process is
        fine. This one runs the health check, which reads a heartbeat that
        only advances when a page is stored.
        """
        container = find("Deployment", "scraper")["spec"]["template"]["spec"][
            "containers"
        ][0]
        probe = container["livenessProbe"]
        assert "exec" in probe, "a wedged node would pass an httpGet probe"
        assert "src.worker.healthcheck" in " ".join(probe["exec"]["command"])

    def test_the_probe_threshold_matches_the_configured_one(self) -> None:
        container = find("Deployment", "scraper")["spec"]["template"]["spec"][
            "containers"
        ][0]
        command = container["livenessProbe"]["exec"]["command"]
        max_age = command[command.index("--max-age") + 1]
        assert int(max_age) == int(settings.worker_heartbeat_max_age)

    def test_a_starting_node_is_not_killed_before_its_first_page(self) -> None:
        """Probing too early produces a restart loop that never reaches the
        first page."""
        container = find("Deployment", "scraper")["spec"]["template"]["spec"][
            "containers"
        ][0]
        probe = container["livenessProbe"]
        assert probe["initialDelaySeconds"] >= 60
        assert probe["failureThreshold"] >= 2

    def test_a_restarting_node_cannot_inherit_a_stale_heartbeat(self) -> None:
        deployment = find("Deployment", "scraper")["spec"]["template"]["spec"]
        volume = next(v for v in deployment["volumes"] if v["name"] == "heartbeat")
        assert "emptyDir" in volume, "a PVC would outlive the pod"

    def test_the_scraper_has_time_to_requeue_on_shutdown(self) -> None:
        spec = find("Deployment", "scraper")["spec"]["template"]["spec"]
        assert spec["terminationGracePeriodSeconds"] >= 30

    def test_the_fleet_cannot_outgrow_the_kafka_partitions(self) -> None:
        """A consumer group cannot use more members than there are
        partitions, so a seventh pod would sit idle holding resources."""
        hpa = find("HorizontalPodAutoscaler", "scraper")["spec"]
        partitions = int(
            find("ConfigMap", "rag-scraper-config")["data"]["KAFKA_PARTITIONS"]
        )
        assert hpa["maxReplicas"] <= partitions

    def test_the_fleet_scales_down_slower_than_it_scales_up(self) -> None:
        """Removing a consumer triggers a Kafka rebalance; a fleet that
        scales down eagerly spends its time rebalancing instead of
        crawling."""
        behavior = find("HorizontalPodAutoscaler", "scraper")["spec"]["behavior"]
        assert (
            behavior["scaleDown"]["stabilizationWindowSeconds"]
            > behavior["scaleUp"]["stabilizationWindowSeconds"]
        )

    def test_the_cluster_uses_the_log_based_frontier(self) -> None:
        """Kafka rather than RabbitMQ here: a consumer group rebalances when
        a pod is added or lost, which is what a Deployment whose replica
        count moves needs."""
        config = find("ConfigMap", "rag-scraper-config")["data"]
        assert config["FRONTIER_BACKEND"] == "kafka"
        assert "kafka" in config["KAFKA_BOOTSTRAP_SERVERS"]

    # ---- the API ----

    def test_the_api_separates_liveness_from_readiness(self) -> None:
        """Restarting a replica whose database is down fixes nothing and
        loses its warm embedding model. Taking it out of rotation is the
        correct response, and that is readiness, not liveness.
        """
        container = find("Deployment", "api")["spec"]["template"]["spec"][
            "containers"
        ][0]
        assert container["livenessProbe"]["httpGet"]["path"] == "/health"
        assert container["readinessProbe"]["httpGet"]["path"] == "/health/ready"

    def test_a_slow_start_is_not_mistaken_for_a_hang(self) -> None:
        """The embedding model loads at start-up, so there is a startupProbe
        rather than a long initialDelay on liveness."""
        container = find("Deployment", "api")["spec"]["template"]["spec"][
            "containers"
        ][0]
        assert "startupProbe" in container

    def test_an_update_never_reduces_the_pool(self) -> None:
        strategy = find("Deployment", "api")["spec"]["strategy"]["rollingUpdate"]
        assert strategy["maxUnavailable"] == 0

    def test_a_node_drain_cannot_take_every_replica(self) -> None:
        """Without a disruption budget there is a window with no API at
        all."""
        budget = find("PodDisruptionBudget", "api")["spec"]
        assert budget["minAvailable"] >= 2

    def test_the_replicas_share_their_rate_limiter_here_too(self) -> None:
        config = find("ConfigMap", "rag-scraper-config")["data"]
        assert config["API_RATE_LIMIT_BACKEND"] == "redis"

    def test_the_api_is_replicated(self) -> None:
        assert find("Deployment", "api")["spec"]["replicas"] >= 2

    def test_the_balancer_is_not_a_single_point_of_failure(self) -> None:
        assert find("Deployment", "nginx")["spec"]["replicas"] >= 2

    # ---- monitoring ----

    def test_pods_are_discovered_rather_than_listed(self) -> None:
        """The pod set changes every time the autoscaler acts, so a static
        target list would stop matching reality."""
        config = find("ConfigMap", "prometheus-config")["data"]["prometheus.yml"]
        parsed = yaml.safe_load(config)
        job = next(
            j for j in parsed["scrape_configs"] if j["job_name"] == "kubernetes-pods"
        )
        assert job["kubernetes_sd_configs"][0]["role"] == "pod"

    def test_prometheus_may_read_the_pods_it_discovers(self) -> None:
        """Discovery without RBAC fails at run time with a 403 that is easy
        to miss, because Prometheus keeps running."""
        rules = find("ClusterRole", "rag-scraper-prometheus")["rules"]
        resources = {resource for rule in rules for resource in rule.get("resources", [])}
        assert {"pods", "endpoints", "services"} <= resources

    def test_the_workloads_ask_to_be_scraped(self) -> None:
        for name in ("api", "scraper", "nginx"):
            annotations = find("Deployment", name)["spec"]["template"]["metadata"][
                "annotations"
            ]
            assert annotations["prometheus.io/scrape"] == "true", name
            assert annotations.get("prometheus.io/port"), name

    def test_the_scraper_metrics_port_is_consistent_everywhere(self) -> None:
        """The port appears in the settings, the Dockerfile, the compose
        file, the pod annotation and the container port. All five have to
        agree or the series simply never appears."""
        port = settings.worker_metrics_port
        deployment = find("Deployment", "scraper")["spec"]["template"]
        assert int(deployment["metadata"]["annotations"]["prometheus.io/port"]) == port
        container = deployment["spec"]["containers"][0]
        assert container["ports"][0]["containerPort"] == port
        assert (
            int(find("ConfigMap", "rag-scraper-config")["data"]["WORKER_METRICS_PORT"])
            == port
        )
        dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.scraper").read_text(
            encoding="utf-8"
        )
        assert f"WORKER_METRICS_PORT={port}" in dockerfile

    def test_the_images_are_the_ones_the_dockerfiles_build(self) -> None:
        services = load_yaml(COMPOSE)["services"]
        for kind, image in (("api", "rag-scraper-api"), ("scraper", "rag-scraper-node")):
            manifest_image = find("Deployment", kind)["spec"]["template"]["spec"][
                "containers"
            ][0]["image"]
            assert manifest_image.split(":")[0] == image
            assert services[f"{kind}1"]["image"].split(":")[0] == image

    def test_a_locally_built_image_is_not_pulled(self) -> None:
        """With imagePullPolicy Always the kubelet looks for "latest" in a
        registry that does not have it, and the pod sits in ErrImagePull."""
        for name in ("api", "scraper"):
            container = find("Deployment", name)["spec"]["template"]["spec"][
                "containers"
            ][0]
            assert container["imagePullPolicy"] == "Never", name

    def test_every_workload_declares_its_resources(self) -> None:
        """An unbounded container is how one pod starves the node it shares."""
        for manifest in all_manifests():
            if manifest["kind"] not in ("Deployment", "StatefulSet"):
                continue
            for container in manifest["spec"]["template"]["spec"]["containers"]:
                assert container.get("resources", {}).get("requests"), (
                    f"{manifest['metadata']['name']}/{container['name']}"
                )
                assert container["resources"].get("limits"), (
                    f"{manifest['metadata']['name']}/{container['name']}"
                )


class TestDockerfiles:
    @pytest.mark.parametrize("name", ["Dockerfile.api", "Dockerfile.scraper"])
    def test_the_build_is_staged_and_runs_unprivileged(self, name: str) -> None:
        text = (PROJECT_ROOT / "docker" / name).read_text(encoding="utf-8")
        # Two stages, so the build tools never ship in the runtime image.
        assert text.count("FROM ") >= 2
        assert "USER " in text

    @pytest.mark.parametrize("name", ["Dockerfile.api", "Dockerfile.scraper"])
    def test_each_image_can_report_its_own_health(self, name: str) -> None:
        text = (PROJECT_ROOT / "docker" / name).read_text(encoding="utf-8")
        assert "HEALTHCHECK" in text

    def test_the_scraper_image_checks_progress_not_liveness(self) -> None:
        text = (PROJECT_ROOT / "docker" / "Dockerfile.scraper").read_text(
            encoding="utf-8"
        )
        assert "src.worker.healthcheck" in text

    def test_the_build_context_excludes_the_virtualenv_and_secrets(self) -> None:
        """The venv alone is about a gigabyte, and a stale .env would bake
        this machine's credentials into an image."""
        ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").split()
        assert "venv/" in ignored
        assert ".env" in ignored
