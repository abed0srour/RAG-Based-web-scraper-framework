# Kubernetes manifests (Task 7)

What each file is for, and the order to apply it in.

| File | What it declares |
|------|------------------|
| `00-namespace.yaml` | The `rag-scraper` namespace and a resource quota |
| `01-config.yaml` | `ConfigMap` with the non-secret settings, `Secret` with the credentials |
| `02-stores.yaml` | PostgreSQL, MongoDB, RabbitMQ, Kafka, Redis, Chroma, Elasticsearch — as single-replica `StatefulSet`s for development |
| `10-api.yaml` | The API `Deployment` (3 replicas), its `Service`, `HorizontalPodAutoscaler` and `PodDisruptionBudget` |
| `11-nginx.yaml` | Nginx as the in-cluster load balancer, plus an `Ingress` |
| `20-scraper.yaml` | The scraper `Deployment`, **its liveness probe**, and the autoscaler that scales on frontier depth |
| `30-monitoring.yaml` | Prometheus and Grafana, with RBAC for pod discovery |

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-config.yaml
kubectl apply -f k8s/02-stores.yaml
kubectl -n rag-scraper rollout status statefulset/postgres

# build the images into the cluster's registry first, e.g. for minikube:
#   eval $(minikube docker-env)
#   docker build -f docker/Dockerfile.api     -t rag-scraper-api:latest .
#   docker build -f docker/Dockerfile.scraper -t rag-scraper-node:latest .

kubectl apply -f k8s/10-api.yaml
kubectl apply -f k8s/11-nginx.yaml
kubectl apply -f k8s/20-scraper.yaml
kubectl apply -f k8s/30-monitoring.yaml
```

## The auto-restart deliverable

A failed scraper node is restarted by the `livenessProbe` in
`20-scraper.yaml`, and the thing worth understanding is *what it probes*.

A crawler can fail in two ways. If the process dies, the Deployment replaces
the pod and nothing clever is needed. The harder case is a node that stays
alive and stops working: a socket in a state the library never times out, a
DNS lookup that hangs, a deadlock between Scrapy's reactor and a blocking
call. Every probe that asks "is the process running" — a TCP check, an HTTP
endpoint, `kubectl get pods` — answers yes, while the frontier stops moving
and no alarm fires.

So the probe runs `python -m src.worker.healthcheck`, which reads the
heartbeat file the crawler writes **when a page is committed to PostgreSQL**.
It is a progress signal, not an existence signal, and a wedged node fails it.

```bash
# watch it work: kill the crawl inside a pod without killing the process
kubectl -n rag-scraper exec deploy/scraper -- pkill -STOP -f run_crawler
kubectl -n rag-scraper get pods -w          # RESTARTS goes to 1
kubectl -n rag-scraper describe pod <name>  # "Liveness probe failed"
```

Three settings in that probe matter:

- **`initialDelaySeconds: 120`** — a node that has only just started has not
  finished a page yet. Probing too early produces a restart loop that never
  reaches the first page.
- **`periodSeconds: 30`, `failureThreshold: 2`** — a node is killed about a
  minute after it goes quiet past the limit, not on a single bad check.
- **`terminationGracePeriodSeconds: 60`** on the pod — long enough for the
  spider's `spider_closed` handler to requeue the URLs it had taken, so a
  restart does not lose work. The frontier is at-least-once anyway, which is
  the backstop if the grace period is not enough.

## Why the scraper has no readiness probe

Readiness controls whether a pod receives *traffic from a Service*. A scraper
receives no traffic: it pulls work from the broker. Giving it a readiness
probe would be decoration. The API has both, and they do different jobs:
liveness on `/health` (is the process alive — restart it if not) and
readiness on `/health/ready` (are its dependencies reachable — take it out of
rotation, but do **not** restart it, because restarting a replica whose
database is down fixes nothing and loses its warm embedding model).
