# Report Notes — Task 7: Load Balancing and Fault Tolerance

Material to paste into the project report (.docx), plus the screenshot list.

The measured figures come from the run recorded in `docs/task7-run.txt` on
this machine: Windows 11, Docker 29.8.0, Python 3.12.10, and a local
Kubernetes 1.36.1 cluster. The whole stack was brought up and exercised —
replicas were stopped under traffic, a scraper node was killed without a
clean leave, a crawl was frozen mid-flight — and the run found nine defects
that review and 64 configuration tests had not. Those are listed near the
end.

---

## Section text for the report

### 1. Introduction to Phase 7

Every phase so far added a capability. This one adds no feature at all: it
takes the system that exists and makes it survive the loss of its parts. The
deliverables are four, and they correspond to four different failures:

| Deliverable | The failure it answers |
|-------------|------------------------|
| Nginx as a load balancer | One API instance is a single point of failure, and one process cannot absorb load |
| Kafka for task distribution | Scraper nodes need work shared out as the fleet grows and shrinks, without a coordinator |
| Kubernetes auto-restart | A node can stay alive and stop working, and nothing notices |
| Prometheus + Grafana | None of the above is visible unless something is recording it |

What makes this phase possible is that the earlier ones left the right
seams. The API is stateless, so it can be replicated. Its rate limiter can
share state through Redis, so a limit survives being spread over three
processes. It already had separate liveness and readiness endpoints. The
frontier was already an interface with one implementation, so a second one
could be added beside it. None of that was accidental, and the work here was
mostly about using it rather than building it.

### 2. Load balancing with Nginx

Three API replicas run behind Nginx, which takes port 8080 — the port the
single instance used. Nothing downstream changes, the web UI included, and
that is the first property worth stating: the clients cannot tell.

**Why `least_conn` rather than round robin.** These requests are not
uniform. A `POST /v1/ask` runs retrieval and generation and holds its
connection for seconds; a raw-page listing is an indexed `SELECT` that
returns in milliseconds. Round robin counts requests, which assumes they
cost the same. Distributing by the number of open connections tracks what a
replica is actually doing, so a replica stuck on three slow RAG calls stops
being handed quick ones.

**Health checking, and what open-source Nginx can actually do.** Active
probing of an upstream's `/health/ready` is an Nginx Plus feature. The
open-source equivalent is passive: `max_fails=3 fail_timeout=15s` takes a
replica out of rotation after three failures inside fifteen seconds and
retries it after the same window. On its own that only helps the *next*
client. What helps the current one is `proxy_next_upstream`, which replays a
failed request against another replica — so a replica dying mid-request is
invisible to the caller rather than being a 502.

That retry has a deliberate limit: `non_idempotent` is **not** set. A `GET`
is safe to replay. Replaying a `POST /v1/ask` that may already have been
processed would double a generation call, which is both a wrong answer and,
with a paid provider, a double charge. The configuration comments say so,
and a test asserts the token is absent.

**The rate limiter had to become shared.** This is the part that is easy to
miss when a service is replicated. With one process, an in-memory fixed
window is correct. With three, each replica keeps its own counters, and a
client that was promised 120 requests a minute gets 360 — the limit silently
stops meaning anything. So the replicas are configured with
`API_RATE_LIMIT_BACKEND=redis` and a Redis container joins the stack. The
API already supported this; the deployment is what makes it necessary.

There is also an edge rate limit (`limit_req_zone` per IP) in front of the
per-key limiting the API does. They exist for different reasons: the API's
limiter protects the backends from a *client*, and this one protects them
from a flood that never gets as far as being authenticated.

**Errors keep their shape.** A client that receives Nginx's stock HTML error
page cannot tell a gateway failure from an API failure, and the web UI's
error handling — which reads `error.type` — breaks on it. So the balancer
returns the same JSON envelope the API uses, with
`type: "no_backend_available"` and a `Retry-After`.

### 3. Kafka: dynamic task distribution

The frontier is now a setting. `FRONTIER_BACKEND=rabbitmq` keeps the Task 2
behaviour; `kafka` runs the same crawl on a partitioned log. Both clients
implement the same interface, and `src/messaging/factory.py` is the only
place that chooses, so the spider, the seeder and the worker do not know
which broker is carrying the work. A test asserts that both expose every
method the spider calls, because that interface is what makes the backend a
setting rather than a rewrite.

Keeping both is a decision, not indecision. They fail differently:

| | RabbitMQ | Kafka |
|---|----------|-------|
| How work is shared | Competing consumers on one queue; the broker hands each message to whoever asks next | Partitions are **assigned** to members of a consumer group |
| A node joins | It starts asking for messages | A rebalance gives it a share of the partitions |
| A node dies | Its unacknowledged messages are redelivered | Its partitions are reassigned after `session.timeout.ms` |
| Per-message retry | Yes — a real nack, redelivering one message | No. A partition is an ordered log; a consumer cannot be given one record again without rewinding past everything after it |
| Ordering | None | Per partition |
| After reading | The message is gone | The record stays for the retention period, so a replay is possible |

**Why this is the fit for a fleet whose size moves.** Adding a scraper
replica *is* the scale-out operation: the new member triggers a rebalance and
receives partitions, and nothing had to be told it exists. That is the
property the Kubernetes Deployment and its autoscaler depend on, and it is
why the cluster ConfigMap sets `FRONTIER_BACKEND=kafka`.

**Why the message key is the site.** Kafka's default partitioner hashes the
key, so keying each URL by its site sends every URL of one host to one
partition and therefore to one node. That matters for a crawler specifically:
politeness is per host, so one node owning a host means its delay and
concurrency limits actually hold. A round-robin distribution would scatter
one site across every node, each applying its own limit to a server being hit
by all of them — the crawler would be impolite by construction, and the
per-site politeness built in Phase 2 would quietly stop working.

**Why offsets are committed by hand.** `enable.auto.commit` is off. An
offset advances when the page has been stored, not on a timer. A node that
dies mid-request leaves its offsets uncommitted, so the URLs are re-delivered
to whichever node takes the partition over. The guarantee is at-least-once —
the same one RabbitMQ gives here — which is exactly why every write
downstream is keyed by a deterministic id: re-processing a page overwrites it
rather than duplicating it. Had auto-commit been left on, a crashed node's
URLs would be marked done and those pages would silently never be crawled.
That single setting is the delivery guarantee, which is why there is a test
asserting it.

**Why the assignor is cooperative.** With the default eager protocol, a
rebalance revokes every assignment from every member and hands them all out
again, so one node joining briefly stops the entire fleet.
`cooperative-sticky` moves only the partitions that have to move.

**Retries without a nack.** Because a record cannot be redelivered
individually, a failed URL is re-*published* with its attempt counter
incremented, and after `MAX_DELIVERY_ATTEMPTS` it goes to a dead-letter
topic. The offset is committed either way: on success because the work is
done, on failure because a fresh copy now carries it. Leaving it uncommitted
would stop the partition at that record forever — which is also why a
malformed record is committed past rather than retried.

The partition count is the ceiling on parallelism: a consumer group can never
have more working members than partitions, so with `KAFKA_PARTITIONS=6` a
seventh pod would sit idle holding resources. The scraper autoscaler's
`maxReplicas` is therefore 6, and a test asserts those two numbers agree.

### 4. Auto-restart: liveness by progress, not by existence

This is the deliverable with the most thought in it, because the obvious
implementation does not work.

A crawler node fails in two ways. If the **process dies**, the Deployment's
restart policy replaces it; nothing clever is needed and no probe is
involved. The dangerous case is a node that stays alive and stops working: a
socket in a state the library never times out, a DNS lookup that hangs, a
deadlock between Scrapy's reactor and a blocking call. In that state the
process is running, its port is open, `kubectl get pods` shows `Running`, and
an HTTP health endpoint — if the crawler had one — would answer perfectly.
The queue stops moving and nothing fires.

So the liveness signal is **progress**. The crawler writes a heartbeat file
every time a page is committed to PostgreSQL, carrying a timestamp and its
counters. `src/worker/healthcheck.py` reads that file and exits non-zero when
it has gone stale, and the Kubernetes `livenessProbe` runs exactly that
command. A wedged node fails it and is killed and replaced. The same command
is the Docker `HEALTHCHECK`, so the behaviour is identical under compose.

Five details decide whether this helps or hurts:

- **The write is atomic** — to a temporary name, then `os.replace`. A probe
  that read a half-written file would fail to parse it and declare a healthy
  node dead.
- **A start-up grace period.** A node that has only just begun has not
  finished a page yet. Without the grace window the probe produces a restart
  loop that never reaches the first page — the classic way this mechanism
  backfires.
- **`initialDelaySeconds: 120` and `failureThreshold: 2`**, so a node is
  killed about a minute after it genuinely goes quiet, not on one slow check.
- **The heartbeat cannot outlive its process.** It lives on an `emptyDir`
  (`tmpfs` under compose), so a restarted node cannot inherit a stale file
  and look healthy before doing anything. A clean shutdown deletes it, so a
  node that exited on purpose does not age into a false alarm.
- **A node that cannot write its heartbeat does not crash.** That would turn
  a monitoring problem into an outage.

The threshold itself is 180 seconds, which is deliberately generous: the
fetch timeout is 20 s with up to three retries and politeness adds a
per-host delay, so a healthy node can legitimately be quiet for over a
minute.

**The API's probes are different on purpose.** It has both, and they do
different jobs: `livenessProbe` on `/health` ("is this process alive?" —
restart it) and `readinessProbe` on `/health/ready` ("are its dependencies
reachable?" — take it out of the Service's endpoints, but do **not** restart
it). Restarting a replica whose database is down fixes nothing, loses its
warm embedding model and turns one outage into a crash loop. Readiness also
makes the rolling update safe, since a new pod receives no traffic until it
can actually serve. The scraper, conversely, has *no* readiness probe:
readiness controls traffic from a Service, and a scraper receives none — it
pulls work from the broker.

A `PodDisruptionBudget` of `minAvailable: 2` completes the picture. Without
it, draining a node can evict all three API replicas at once; the Deployment
would recreate them, but there would be a window with no API at all.

### 5. Monitoring

Both layers export Prometheus metrics: the API already did
(`src/api/metrics.py`), and `src/worker/metrics.py` adds the crawler's. The
scraper series exist because the two questions monitoring must answer about a
distributed crawler cannot be answered from outside it — *is each node
working, or merely running?* and *is the fleet keeping up with the frontier?*
`scraper_pages_stored_total`, `scraper_heartbeat_age_seconds` and
`scraper_frontier_depth` are those two questions as numbers.

**Every target is scraped directly, never through the balancer.** Scraping
through Nginx returns whichever replica it happened to pick, so the series
would be an unusable mixture of three processes — and "which replica is slow"
is the main question per-replica metrics exist to answer. Under compose the
targets are a static list; under Kubernetes the same job uses
`kubernetes_sd_configs` to discover pods by annotation, so a replica created
by the autoscaler is monitored without a configuration change. That discovery
is why Prometheus needs RBAC in the cluster, and a missing ClusterRole fails
at run time with a 403 that is easy to miss because Prometheus keeps running.

The Grafana dashboard and its data source are **provisioned from files** in
the repository rather than clicked into the UI, so they are reviewable in a
diff and survive the volume being deleted. The data source has a fixed `uid`,
because the dashboard references it by `uid` and a generated one makes every
panel show "datasource not found".

The alert rules are the real output, because a dashboard is only read by
someone already looking and this system's failures are the quiet ones. Three
of the thirteen are worth describing:

- **`ScraperNodeStalled`** fires on the same 180-second threshold the
  liveness probe uses. If the alert's threshold were higher, a node would be
  restarted before anyone was told it had stalled, and the restart would look
  spontaneous. A test asserts the two numbers are equal.
- **`FrontierBacklogUnattended`** — work queued and nothing consuming it —
  is the condition the autoscaler is supposed to resolve, so it firing means
  the autoscaler itself is not working.
- **`TargetMissing`** exists because when scraping stops, every other rule
  goes quiet rather than firing. A monitoring system that has stopped looks
  exactly like a healthy one.

Every rule has a `for` duration, because a replica restarting during a deploy
must not page anyone.

### 6. Testing configuration

Infrastructure in a repository fails in a specific way: two files that have
to agree slowly stop agreeing. The Nginx upstream names a service the compose
file renamed. The probe's threshold and the alert's threshold drift apart. The
autoscaler is allowed more replicas than Kafka has partitions, so the extra
pods idle. The metrics port is changed in four files out of five. None of that
fails a build — it quietly does the wrong thing.

So `tests/test_deployment.py` reads the real files and asserts the
relationships between them: that every Nginx upstream is a real compose
service on the right port, that only the balancer publishes the API port,
that the replicas are identical and share a Redis-backed limiter, that the
stall alert matches the liveness threshold, that the error-rate alert cannot
divide by zero, that every dashboard panel queries a metric the code actually
exports, that the HPA cannot outgrow the partition count, that the scraper's
probe is an `exec` and not an `httpGet`, and that the metrics port agrees
across the settings, the Dockerfile, the compose file, the pod annotation and
the container port. It needs no Docker daemon and no cluster, which is the
point: it is the part of this stack that can be checked on a laptop.

The Python added for this phase is tested the same way as everything else:
`tests/test_kafka_frontier.py` covers the client configuration, the keying,
the retry and dead-letter policy and the offset handling against a fake
broker, and `tests/test_worker.py` covers the heartbeat and the health check,
including the awkward cases that decide whether the restart mechanism helps
or hurts — a half-written file, a node that has started but not finished a
page, an unwritable directory, a node that exited cleanly.

---

## Screenshot list for the report

1. `docker compose ps` with the full stack up: the data stores, Kafka,
   Redis, three API replicas, Nginx, Prometheus and Grafana.
2. Repeated `curl` against `localhost:8080`, with the Nginx access log beside
   it showing `upstream` cycling between `api1`, `api2` and `api3`.
3. `docker stop rag-api2`, then the same `curl` loop still returning 200 —
   the failover, with the log showing only the two survivors.
4. The Nginx JSON 503 when all three replicas are stopped
   (`type: no_backend_available`).
5. `kafka-consumer-groups --describe --group scrapers`: the partitions, their
   owners and the lag, with two scraper nodes running.
6. The same command after stopping one node — the partitions reassigned to
   the survivor.
7. `python -m src.worker.healthcheck --all`: the fleet view, with one node
   stalled.
8. A stalled node being restarted: `docker ps` showing the restart count
   increase, or `kubectl describe pod` with "Liveness probe failed".
9. Prometheus `/targets` with every job `UP`.
10. The Grafana dashboard: requests per replica, latency percentiles,
    heartbeat age per node and the frontier depth.

---

## What the live run showed

Full transcript in `docs/task7-run.txt`. Every figure below was measured
against the running stack, not estimated.

### Load balancing

| Measurement | Result |
|-------------|--------|
| 60 concurrent requests over three replicas | 25 / 20 / 15 — all 200 |
| 40 requests with one replica **stopped mid-traffic** | **40 × 200, zero errors reached the client** |
| Of those, requests retried onto a second replica | 9 (visible as two upstream addresses on one log line) |
| After restarting the replica | back in rotation: 9 / 8 / 7 |
| All three replicas stopped | `503`, `Retry-After: 10`, `{"error":{"type":"no_backend_available",…}}` — the API's own envelope, not Nginx's HTML |
| 150 requests against a 120/min limit | **120 × 200, 30 × 429**, spread 46/49/55 across replicas. Per-replica in-memory counters would have allowed all 150; the Redis-backed limiter held the real limit. |

### Kafka

| Measurement | Result |
|-------------|--------|
| Topic | `urls.frontier`, 6 partitions, confirmed by the broker |
| One node alone | held all 6 partitions, consumed 18 of 20 URLs |
| A second node joining | cooperative rebalance: 3 partitions each, confirmed in the broker's group description |
| URLs delivered twice across the two nodes | **0** (18 + 2 = 20, no duplicates) |
| A node **killed without a clean leave** | its 3 partitions reassigned to the survivor after the session timeout; the survivor then held all 6 |
| Live fleet throughput | 2 nodes, 0 restarts, 9 545 links discovered and re-published, 14 960 URLs outstanding, 0 dead-lettered |

### Auto-restart

| Measurement | Result |
|-------------|--------|
| Container state while the crawl was frozen with `SIGSTOP` | `running` for the whole six minutes — what every existence-based check sees |
| Heartbeat age | climbed 90 s → 151 s → 213 s → 274 s → 336 s → 397 s |
| The probe | flipped to unhealthy once the age passed 180 s; `exit code 1`, message `stale - no page finished in 394s (limit 180s)` |
| Docker's health status (same command) | `healthy` → **`unhealthy`** at t+240 s |
| The two healthy nodes meanwhile | 481 pages stored, `ok`, **0 restarts** |

### Monitoring

| Measurement | Result |
|-------------|--------|
| Prometheus targets up | `api1`, `api2`, `api3`, `scraper1`, `scraper2`, `prometheus` |
| Alert rules parsed by the real Prometheus | **13 rules in 3 groups**, all loaded |
| Metrics cross-check | Prometheus reported exactly 30 rate-limit refusals, matching the 30 × 429 measured by hand |
| Grafana | datasource `uid=prometheus` and dashboard `rag-scraper-overview` provisioned from the repository files; a panel query through Grafana's own proxy returned live per-node page counts |

### Kubernetes

| Measurement | Result |
|-------------|--------|
| API server | v1.36.1 (local Docker Desktop cluster) |
| Manifests validated with `kubectl apply --dry-run=server` | **all 40 objects accepted** across the 7 files |

**Not verified, and why.** No pod was actually run from these images: the
local cluster is kind-based with its own image store and the connection on
this machine (~56 kB/s) was too slow to populate it. So the restart *action*
is evidenced by the probe failing on a wedged node plus server-side manifest
validation, not by a `RESTARTS` counter incrementing in `kubectl`. The
`nginx-prometheus-exporter` target is down for the same reason — its image
was never pulled. The HorizontalPodAutoscalers need `prometheus-adapter`
installed to serve the external metric.

## Nine bugs the live run found that the tests had not

This is the honest headline of the phase: the configuration was complete,
reviewed and covered by 64 consistency tests, and running it still found nine
defects. Six of them would have been invisible in review.

| # | Defect | Why no test caught it |
|---|--------|-----------------------|
| 1 | Kafka's EXTERNAL listener was never published, so no client outside the compose network could reach the broker | The config was internally consistent; only a real client proved otherwise |
| 2 | A commit rejected after a rebalance (`ILLEGAL_GENERATION`) escaped and **killed a crawler node** | It needs a real rebalance to happen between a pull and a commit |
| 3 | `depth()` joined the consumer group to read offsets, **rebalancing the live fleet on every call** | Correct in isolation; harmful only against a running group |
| 4 | Nodes exited after ~6 s of an empty frontier; `restart: always` turned that into a restart loop that rejoined the consumer group every few seconds, so the group never stabilised and **no work was ever picked up** (13–14 restarts per node, 0 pages) | Two correct settings interacting badly |
| 5 | The compose `tmpfs` for the heartbeat was root-owned, so the non-root user could not write it and the node was **reported unhealthy forever** while crawling fine | A runtime mount masking an image-build `chown` |
| 6 | Nginx listened on IPv4 only while its health check resolved `localhost` to `::1`, marking a **healthy balancer unhealthy** | Both halves were individually reasonable |
| 7 | `nginx-exporter` depended on a service in a different compose profile, making `--profile monitoring up` an invalid project | Only that profile combination fails |
| 8 | Redis published on 6379 clashed with another project on the machine | Environment-specific |
| 9 | `return` in the Nginx config was split over several quoted strings, which Nginx does not concatenate — the container would not start | Needs `nginx -t`, which needs Nginx |

Each fix carries a regression test: #2, #3 and #4 in
`tests/test_kafka_frontier.py` and `tests/test_worker.py`, and #5, #6 and the
idle-heartbeat behaviour in `tests/test_deployment.py`. The comments in the
configuration files say what went wrong, so the next person does not undo
them.

## Results table

| Metric | Value |
|--------|-------|
| New Python modules | `src/messaging/kafka_frontier.py`, `src/messaging/factory.py`, `src/worker/heartbeat.py`, `src/worker/healthcheck.py`, `src/worker/metrics.py` |
| New infrastructure files | 2 Dockerfiles, `.dockerignore`, `docker/nginx/nginx.conf`, Prometheus config + 13 alert rules, Grafana provisioning + a 14-panel dashboard, 7 Kubernetes manifests (40 objects) |
| API replicas behind the balancer | 3 |
| Load-balancing algorithm | `least_conn`, passive health checks, retry on another replica |
| Kafka topic | `urls.frontier`, 6 partitions, keyed by site |
| Delivery guarantee | At-least-once (manual offset commits) |
| Rebalance protocol | `cooperative-sticky` |
| Liveness signal | Heartbeat file written on each stored page; 180 s threshold |
| Prometheus series added | 9 scraper series, plus the 4 the API already exported |
| Alert rules | 13 under compose, 5 in the cluster ConfigMap |
| Kubernetes objects | 40 across 7 manifests |
| Tests added | 144 (43 Kafka, 33 worker, 68 deployment configuration) |
| Project test suite | **518 passing, 0 skipped** — with the whole stack running, every previously-skipped integration test (PostgreSQL, MongoDB, Chroma, Elasticsearch) executed for real. 309 were passing before this phase. |
| New Python dependencies | `confluent-kafka==2.6.1`, `PyYAML==6.0.3` |
| New containers | Kafka 3.9.0 (KRaft), Redis 7.4, Nginx 1.27, Prometheus 3.5.0, Grafana 12.1.0, nginx-prometheus-exporter 1.4.0 |
| Image sizes built | `rag-scraper-api` 1.01 GB, `rag-scraper-node` 221 MB (no compiler in either: every dependency has a manylinux wheel) |
| Live fleet at the end of the run | 15 containers, 2 scraper nodes with 0 restarts, 503+ pages crawled, 14 960 URLs queued |
