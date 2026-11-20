# Screenshot guide — every task

One place that says what to capture for each phase of the report, in the
order to capture it. Each task's own `docs/REPORT_TASK*.md` has the section
text; this file is only about the figures.

**Save them as** `docs/screenshots/figNN-short-name.png`, numbered per task
(`t1-fig1-...`, `t2-fig1-...`), so the numbers in the .docx captions match
the files.

---

## Before you start

### 0. Bring everything up

```bash
cd "<project root>"
venv\Scripts\activate

# the data stores (Tasks 2, 3, 4, 6)
docker compose -f docker/docker-compose.yml up -d

# Task 7's layers
docker compose -f docker/docker-compose.yml --profile kafka up -d
docker compose -f docker/docker-compose.yml --profile lb up -d --build
docker compose -f docker/docker-compose.yml --profile monitoring up -d
```

### 1. Have data to show

The figures for Tasks 3–6 need a populated corpus. If the databases are
empty, do this once, in order:

```bash
python -m src.scraper.seed_frontier --job-name books-crawl  --url https://books.toscrape.com/
python -m src.scraper.seed_frontier --job-name quotes-crawl --url https://quotes.toscrape.com/
python -m src.scraper.run_crawler --max-depth 2 --worker-id w1   # crawl  -> PostgreSQL
python -m src.processing.pipeline                                # clean  -> MongoDB
python -m src.rag.indexer                                        # embed  -> Chroma
python -m src.search.indexer                                     # index  -> Elasticsearch
```

### 2. Terminal hygiene

- Maximise the window and keep the font large enough to read in a printed
  page. A screenshot nobody can read is a figure that proves nothing.
- Keep the **command visible** at the top of every capture. A table of
  numbers with no command above it could have come from anywhere.
- Capture the whole relevant block in one image; do not stitch two scrolls
  together.
- Use `| more` or resize rather than scrolling, so the start of the output
  is still on screen.

---

## Task 1 — Environment setup

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| 1 | Python and the virtual environment | `python --version` then `where python` (should show the project `venv`) |
| 2 | The installed dependency set | `pip list` |
| 3 | **The environment verification report, all sections** | `python scripts/verify_env.py` |
| 4 | Docker and Kubernetes available | `docker --version` ; `docker compose version` ; `kubectl version --client` |
| 5 | The repository on GitHub | the repo page in a browser, showing the commit history |
| 6 | The project layout | `git ls-files | find /v "venv"` , or the folder tree in VS Code's explorer |

> Figure 3 is the one that matters: it is a single page proving the
> interpreter, the packages, the directories and every backing service in
> one report.

---

## Task 2 — Web scraping module

Full table with captions in [REPORT_TASK2.md](REPORT_TASK2.md). The eight
figures, in short:

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| 1 | Broker and database healthy | `docker compose -f docker/docker-compose.yml ps` |
| 2 | The two queues | <http://localhost:15672> (`scraper`/`scraper_pass`) → **Queues** |
| 3 | Seeding, and the queue depth after it | `python -m src.scraper.seed_frontier --job-name books-crawl --url https://books.toscrape.com/` then `--status` |
| 4 | **Two crawler nodes side by side** | two terminals: `python -m src.scraper.run_crawler --site quotes.toscrape.com --worker-id w1` and `... --worker-id w2` |
| 5 | The frontier growing during the crawl | `python -m src.scraper.seed_frontier --status` in a third terminal |
| 6 | Jobs, per-site totals, work per node | `python -m src.storage.report` |
| 7 | The per-site structured extraction | `python -m src.storage.report --samples 2` |
| 8 | The Phase 2 tests | `python -m pytest tests/test_parser.py tests/test_frontier.py tests/test_storage.py -q` |

> Figure 4 is the single most important one in this task — it is the proof
> the crawler is distributed. Put the two terminals side by side and make
> sure the different `--worker-id` values are readable in the log lines.

---

## Task 3 — Data processing module

Full table with captions in [REPORT_TASK3.md](REPORT_TASK3.md).

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| 1 | MongoDB in the stack, healthy | `docker compose -f docker/docker-compose.yml ps` |
| 2 | **Raw HTML beside the clean text** | `python scripts/demo_task3.py --skip-benchmark` (step 1) |
| 3 | **The complete normalised JSON document** | same run, step 2 |
| 4 | The full run, and the idempotency proof | same run, steps 3–4 |
| 5 | The indexes, per-site corpus, duplicates | `python -m src.processing.report` |
| 6 | **A keyword query answered by the text index** | `python -m src.processing.report --search "fiction"` |
| 7 | **The efficiency benchmark with the batch table** | `python -m src.processing.benchmark --repeat 5` |
| 8 | The tests | `python -m pytest tests/test_processing.py -q` |
| 9 | *(optional)* the documents in a GUI | <http://localhost:8081> (mongo-express, `scraper`/`scraper_pass`) |

> Figures 2, 3, 6 and 7 map one-to-one onto the four requirements of the
> task (strip markup, normalise to JSON, index in a NoSQL store, test
> processing efficiency). Capture those first.

---

## Task 4 — RAG integration

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| 1 | Chroma in the stack, healthy | `docker compose -f docker/docker-compose.yml ps` |
| 2 | **One document cut into overlapping chunks** | `python scripts/demo_task4.py` (step 1) — the word counts and the overlap are the point |
| 3 | **Semantic similarity between a query and passages** | same run, step 2 — this is the evidence retrieval is by meaning, not by keyword |
| 4 | The vector index being built, with timings | same run, step 3 — and the second run below it, showing the write path is idempotent |
| 5 | **The three retrieval modes side by side** | same run, step 4 — `vector`, `keyword`, `hybrid` for one query, so the fusion is visible |
| 6 | **A generated answer with its source URLs** | same run, step 5 |
| 7 | The enrichment and the data-quality score | same run, step 6 — before/after per document |
| 8 | The index report | `python -m src.rag.report` |
| 9 | A query and its ranked passages | `python -m src.rag.report --query "books about history" --top-k 5` |
| 10 | The tests | `python -m pytest tests/test_rag.py -q` |

> Two things a marker will look for. **Figure 3** is the one that
> distinguishes this phase from Task 3's keyword search — show a query
> matching a passage that shares no words with it. **Figure 6** must show
> the numbered citations next to the answer, because "grounded in retrieved
> text" is the claim being made.
>
> The provider is the free offline extractive summariser by default, which
> is worth a sentence in the caption: no API key, and every sentence is
> copied from a retrieved passage, so the answer cannot hallucinate. If you
> set `ANTHROPIC_API_KEY`, capture Figure 6 again and note the provider
> changed to `anthropic` in the response.

---

## Task 5 — REST API

The API is on <http://localhost:8080>. With Task 7's `lb` profile running
that port is Nginx in front of three replicas; either is fine for these
figures.

```bash
python -m src.api        # if you are NOT running the lb profile
```

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| 1 | **The interactive documentation, endpoints collapsed** | <http://localhost:8080/docs> — one image showing every route group |
| 2 | Raw scraped data | `/docs` → `GET /v1/raw/pages` → Execute, with the response body visible |
| 3 | The stored HTML of one page | `GET /v1/raw/pages/{page_id}/html` |
| 4 | **Processed / enhanced content** | `GET /v1/documents/{doc_id}` — expand `enrichment` so the generated summary shows |
| 5 | **Searching the indexed data** | `GET /v1/search?q=...&mode=hybrid` — the per-retriever scores and ranks in the response |
| 6 | A cited RAG answer over HTTP | `POST /v1/ask` with `{"question": "..."}` |
| 7 | **Authentication refused** | `curl -i http://localhost:8080/v1/documents` → 401, and then with a wrong key → 401 |
| 8 | **Scope enforcement** | `curl -i -H "X-API-Key: dev-reader-key" "http://localhost:8080/v1/search?q=widgets"` → 403, because the reader key has no `search` scope |
| 9 | Who a key is | `curl -s -H "X-API-Key: dev-analyst-key" http://localhost:8080/v1/me` |
| 10 | **Rate limiting, with the headers** | `curl -i -H "X-API-Key: dev-reader-key" http://localhost:8080/v1/raw/stats` — capture `X-RateLimit-Limit`, `-Remaining`, `-Reset`, `-Bucket` |
| 11 | **A 429 after the limit** | the loop in Task 7 figure 6 below, or `for /l %i in (1,1,150) do @curl -s -o nul -w "%%{http_code} " -H "X-API-Key: dev-reader-key" http://localhost:8080/v1/raw/stats` |
| 12 | A dependency outage becoming a 503 | `docker stop rag-mongodb` then `curl -i -H "X-API-Key: dev-reader-key" http://localhost:8080/v1/documents` → 503 + `Retry-After`; `docker start rag-mongodb` afterwards |
| 13 | Readiness per dependency | `curl -s http://localhost:8080/health/ready` |
| 14 | The API tests | `python -m pytest tests/test_api.py -q` |

> Figures 7, 8, 10 and 11 are the "authentication and rate limiting"
> deliverable — four small captures, and together they are the whole claim.
> Figure 12 is worth including even though it is not required: it shows the
> API distinguishing "your request was wrong" from "a backing service is
> down", which is the difference between a 4xx and a 503.

---

## Task 6 — Web UI and full-text search

Start the UI (the API must already be up):

```bash
cd ui
npm install      # first time only
npm run dev      # http://localhost:5173
```

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| 1 | Elasticsearch in the stack, healthy | `docker compose -f docker/docker-compose.yml ps` |
| 2 | The cluster answering | `curl http://localhost:9200` |
| 3 | **The index being built from MongoDB** | `python -m src.search.indexer` — documents read, indexed, batches, throughput |
| 4 | Coverage and a ranked query | `python -m src.search.report --query "light in the attic"` |
| 5 | The API response behind the UI | `/docs` → `GET /v1/fulltext?q=light` — show `highlights`, `facets` and both timings |
| 6 | **The UI search results** | <http://localhost:5173> — marked terms in the snippets, the facet sidebar, the result count and timing line |
| 7 | **The suggestion dropdown mid-typing** | type `lig` in the search box and capture with the list open |
| 8 | Narrowing by facet | click a site in the sidebar; capture the reduced result count |
| 9 | **The document drawer** | click a result — the AI summary (MongoDB) beside the related pages (Elasticsearch) |
| 10 | The Ask tab | the **Ask** tab with an answer and its numbered sources |
| 11 | The Raw data tab | the **Raw data** tab — the PostgreSQL crawl record with status codes |
| 12 | **A graceful failure** | `docker stop rag-elasticsearch`, reload the UI: the 503 message with recovery instructions and the red dot in the status bar. `docker start rag-elasticsearch` afterwards |
| 13 | The tests | `python -m pytest tests/test_search.py tests/test_api_fulltext.py -q` |

> Figure 6 is the headline figure of the whole project — it is the one
> screenshot that shows a user actually using the system. Search for
> something that matches in the body text, not just the title, so the
> `<mark>` highlighting is visible inside a sentence.
>
> Figure 9 is the one that shows the architecture from the outside: two
> different stores answering in one panel.

---

## Task 7 — Load balancing and fault tolerance

Full transcript of a verified run in [task7-run.txt](task7-run.txt) — every
figure below was captured from it, so you can follow it command by command.

### Load balancing

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| 1 | **The whole stack, 15 containers** | `docker compose -f docker/docker-compose.yml ps` |
| 2 | The three replicas behind one port | `docker compose -f docker/docker-compose.yml ps | find "api"` and `curl -s http://localhost:8080/health` |
| 3 | **Requests spread over three replicas** | run the concurrent loop, then `docker logs rag-nginx | find "upstream"` — capture the counts per upstream address |
| 4 | **40 requests with a replica stopped: all 200** | `docker stop rag-api2`, send 40 requests, capture the codes **and** the log lines showing two upstream addresses (the retry) |
| 5 | The replica rejoining | `docker start rag-api2`, wait, repeat the loop, capture the even spread |
| 6 | **The shared rate limit** | 150 requests against the 120/min limit → exactly 120 × 200 and 30 × 429, spread across all three replicas |
| 7 | **Every replica down** | `docker stop rag-api1 rag-api2 rag-api3` then `curl -i http://localhost:8080/v1/raw/stats` → the JSON `no_backend_available` 503 |

The loop for figures 3–6, in `cmd`:

```bat
for /l %i in (1,1,60) do @curl -s -o nul -H "X-API-Key: dev-reader-key" http://localhost:8080/v1/raw/stats
docker logs rag-nginx 2>&1 | findstr /c:"upstream"
```

### Kafka

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| 8 | **The topic and its 6 partitions** | `docker exec rag-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --describe --topic urls.frontier` |
| 9 | **Two nodes sharing the partitions** | `docker exec rag-kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group scrapers --members` → 3 partitions each |
| 10 | The lag, i.e. the frontier depth | same command without `--members` |
| 11 | **Partitions reassigned after a node dies** | `docker stop rag-scraper2`, wait ~60 s, repeat figure 9 → the survivor holds all 6 |
| 12 | Both nodes working | `docker logs rag-scraper1 | findstr /c:"stored"` and the same for `scraper2` |

### Auto-restart

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| 13 | **The fleet view** | `docker exec rag-scraper1 python -m src.worker.healthcheck --all` — node, heartbeat age, pages, state |
| 14 | A node's heartbeat file | `docker exec rag-scraper1 cat /var/run/scraper/scraper1.json` |
| 15 | **A wedged node detected** | see the sequence below — the container stays `running` while the probe fails |
| 16 | The manifests against a real cluster | `kubectl apply --dry-run=server -f k8s/20-scraper.yaml` (and the other files) |
| 17 | The probe in the manifest | `k8s/20-scraper.yaml` open in the editor, with the `livenessProbe` block visible |

The wedge sequence for figure 15 — this is the most interesting figure in
the task, because the container never stops looking fine:

```bash
docker run -d --name wedge --network rag-scraper_default \
  -e FRONTIER_BACKEND=kafka -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  -e WORKER_ID=wedge-test -e POSTGRES_HOST=postgres -e POSTGRES_PORT=5432 \
  -e POSTGRES_USER=scraper -e POSTGRES_PASSWORD=scraper_pass -e POSTGRES_DB=scraped_data \
  rag-scraper-node:latest sh -c "python -m src.scraper.run_crawler --max-depth 1 --idle-rounds 0 & wait"

# let it store a few pages, then freeze the crawl (not the container)
docker exec wedge sh -c "kill -STOP 7"

# capture this every minute for ~5 minutes, in one image if you can:
docker inspect -f "{{.State.Status}} {{.State.Health.Status}}" wedge
docker exec wedge python -m src.worker.healthcheck --node wedge-test

docker rm -f wedge      # afterwards
```

What the figure has to show: `running` all the way through, the heartbeat
age climbing past 180 s, the probe saying `stale - no page finished in ...`,
and the health status turning `unhealthy`.

### Monitoring

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| 18 | **Prometheus targets all up** | <http://localhost:9090/targets> |
| 19 | The alert rules loaded | <http://localhost:9090/alerts> — 13 rules in 3 groups |
| 20 | A query in Prometheus | <http://localhost:9090/graph> → `sum by (instance) (api_requests_total)` |
| 21 | **The Grafana dashboard, top half** | <http://localhost:3000> (admin/admin) → *RAG Scraper* → *system overview* — the availability row and "requests per replica" |
| 22 | **The dashboard, fleet half** | the same dashboard scrolled to "pages stored per second" and "heartbeat age per node" |
| 23 | A panel during the failover | re-run figure 4 with the dashboard open, so one replica's line drops to zero while the others rise |
| 24 | The tests | `python -m pytest tests/test_deployment.py tests/test_kafka_frontier.py tests/test_worker.py -q` |

> Figure 23 is the best figure in the whole report if you can get it: it
> shows the load balancer, the metrics pipeline and the failure handling in
> one image. Open Grafana first, set the time range to **Last 15 minutes**
> and the refresh to **10s**, then stop a replica and wait for the lines to
> separate before capturing.

---

## Whole-project figures

Keep these for the conclusion:

| # | What to capture | Command / where |
|---|-----------------|-----------------|
| A | **The full test suite** | `python -m pytest -q` — with the stack running this is 518 passing, 0 skipped |
| B | The repository history | `git log --oneline` — one commit per task |
| C | The architecture, as the system reports it | `curl -s http://localhost:8080/health/ready` — all five stores in one response |
| D | The project layout | `git ls-files` or the VS Code explorer expanded to `src/`, `ui/`, `k8s/`, `docker/` |

> Figure A is the one to put in the conclusion. "518 tests, nothing skipped"
> is only true while every container is up, so capture it at the end of a
> session with the whole stack running — and say so in the caption, because
> the skip count is itself evidence that the integration tests really ran.
