# Distributed RAG-Based Web Scraper Framework

A scalable, fault-tolerant framework for **distributed web scraping** combined with
**Retrieval-Augmented Generation (RAG)**. Scraping tasks are spread across a pool of
workers, the extracted content is cleaned, chunked, embedded and stored in a vector
database, and a Large Language Model (Claude) then uses the retrieved context to
answer questions and enrich the scraped data. Everything is exposed through a REST API
and deployed with Docker and Kubernetes.

> Distributed Systems lab project.

---

## Objectives

| # | Objective | How it is addressed |
|---|-----------|---------------------|
| 1 | Implement a scalable and distributed web scraper | Celery worker pool consuming scrape tasks from a Redis broker |
| 2 | Integrate a RAG-based system to enhance scraped data | Chunking + ONNX embeddings + Chroma vector store + Claude generation |
| 3 | Ensure fault tolerance and load balancing | Task retries with exponential backoff, acknowledge-late, dead-letter handling, Redis fair dispatch, Kubernetes replicas + probes |
| 4 | Use a distributed computing model for efficient task execution | Producer / broker / worker (task-queue) model with horizontal scaling |
| 5 | Provide APIs for data retrieval and analysis | FastAPI service: submit jobs, poll status, search the corpus, ask RAG questions, read metrics |

---

## Architecture (high level)

```
          ┌────────────┐      submit job       ┌──────────────────┐
  client ─►  REST API  ├──────────────────────►│  Redis broker    │
          │ (FastAPI)  │◄──────────────────────┤  (task queue)    │
          └─────┬──────┘      results          └────────┬─────────┘
                │                                       │ load-balanced
                │                                       │ task dispatch
                │                          ┌────────────┼────────────┐
                │                          ▼            ▼            ▼
                │                     ┌─────────┐  ┌─────────┐  ┌─────────┐
                │                     │ worker 1│  │ worker 2│  │ worker N│
                │                     └────┬────┘  └────┬────┘  └────┬────┘
                │                          └────────────┼────────────┘
                │                                       ▼
                │                            fetch → extract → chunk → embed
                │                                       │
                │           retrieve context            ▼
                └──────────────────────────────►┌──────────────────┐
                                                │  Chroma vector   │
                             augment / answer   │     store        │
                        ┌──────────────┐        └──────────────────┘
                        │  Claude LLM  │◄───────────────┘
                        └──────────────┘
```

A detailed per-phase flowchart is included in the project report
(`docs/`).

---

## Technology stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Crawling framework | Scrapy 2.12 (Twisted async engine) |
| HTML parsing / extraction | BeautifulSoup4 (lxml backend) + lxml XPath |
| Message queuing (URL frontier) | RabbitMQ 3.13 via pika |
| Raw + parsed data store | PostgreSQL 16 via SQLAlchemy 2 + psycopg 3 |
| Distributed task queue | Celery 5.4 (from Task 4) |
| Worker monitoring | Flower, RabbitMQ management UI |
| Embeddings | FastEmbed (ONNX Runtime, CPU-only) |
| Vector store | Chroma (client/server mode) |
| LLM | Anthropic Claude (`claude-sonnet-5-5`) |
| REST API | FastAPI + Uvicorn |
| Metrics | prometheus-client |
| Containerisation | Docker + Docker Compose |
| Orchestration | Kubernetes |
| Testing | pytest |

---

## Repository layout

```
.
├── README.md                 Project documentation
├── requirements.txt          Pinned Python dependencies
├── requirements.lock.txt     Full resolved dependency snapshot
├── .env.example              Template for environment variables
├── .gitignore
├── scrapy.cfg                Scrapy project entry point
├── src/
│   ├── common/
│   │   ├── config.py         Central Pydantic settings (all components)
│   │   └── logging_config.py Node-aware logging format
│   ├── messaging/
│   │   └── frontier.py       RabbitMQ URL frontier (publish / consume / DLQ)
│   ├── scraper/
│   │   ├── settings.py       Scrapy settings (politeness, retries, pipelines)
│   │   ├── items.py          RawPageItem definition
│   │   ├── parser.py         BeautifulSoup + lxml extraction, per-site XPath
│   │   ├── pipelines.py      Parsing pipeline -> PostgreSQL pipeline
│   │   ├── spiders/
│   │   │   └── frontier_spider.py   Spider that consumes URLs from RabbitMQ
│   │   ├── run_crawler.py    Launch one crawler node
│   │   └── seed_frontier.py  Publish seed URLs / inspect / purge the queues
│   ├── storage/
│   │   ├── models.py         PostgreSQL schema (4 tables)
│   │   ├── database.py       Engine, sessions, repository
│   │   └── report.py         CLI report over the scraped data
│   ├── worker/               Celery application and distributed tasks (Task 4)
│   ├── rag/                  Chunking, embeddings, vector store, LLM (Task 3)
│   └── api/                  FastAPI application (Task 5)
├── scripts/
│   ├── verify_env.py         Environment verification report (Task 1)
│   └── demo_task2.py         End-to-end Task 2 demonstration on two sites
├── tests/                    pytest test-suite
├── docker/
│   └── docker-compose.yml    RabbitMQ + PostgreSQL infrastructure
├── k8s/                      Kubernetes manifests (deployments, services, HPA)
├── data/                     Local run artefacts and worker logs (git-ignored)
└── docs/
    └── screenshots/          Screenshots used in the report
```

---

## Task 1 - Environment setup

### 1. Prerequisites

| Tool | Version used |
|------|--------------|
| Python | 3.12.10 |
| Git | 2.49.0 |
| Docker Engine | 29.8.0 |
| Docker Compose | v5.5.1 |
| kubectl | v1.36.1 |
| Kubernetes | Docker Desktop built-in cluster |

### 2. Clone the repository

```bash
git clone https://github.com/abed0srour/RAG-Based-web-scraper-framework.git
cd RAG-Based-web-scraper-framework
```

### 3. Create and activate the virtual environment

```bash
# Windows (PowerShell)
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3.12 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
cp .env.example .env     # Windows: copy .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

### 6. Verify the environment

```bash
python scripts/verify_env.py
```

The script prints the version of the interpreter, of every Python dependency and of
every external tool, and then checks that the Docker daemon and the Kubernetes
cluster are reachable. A fully configured machine ends with:

```
[ OK ] Environment is fully configured. Ready for Task 2.
```

### 7. Enable Kubernetes

Kubernetes is provided by Docker Desktop:
**Settings → Kubernetes → Enable Kubernetes → Apply & Restart**. Verify with:

```bash
kubectl config get-contexts
kubectl get nodes
```

---

## Task 2 - Web scraping module

### Pipeline

```
 seed_frontier.py            RabbitMQ                 Scrapy spiders
 (producer)                  "urls.frontier"          (N independent nodes)
      │                            │                          │
      │ publish seed URL           │   prefetch=1, one URL     │
      └───────────────────────────►│──────────────────────────►│ fetch raw HTML
                                   │                          │
                                   │   newly discovered links │
                                   │◄─────────────────────────┤ (dynamic frontier)
                                   │                          ▼
                                   │                   ParsingPipeline
                                   │              BeautifulSoup + lxml XPath
                                   │                          │
                                   │                          ▼
                                   │                   PostgresPipeline
                                   │                          │
                                   │  ack only after COMMIT   ▼
                                   │◄────────────────────  PostgreSQL
                                   │                   raw_pages
            on repeated failure    ▼                   parsed_pages
                            "urls.dead-letter"         discovered_links
                                                       crawl_jobs
```

### 1. Start the infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml ps
```

RabbitMQ management UI: <http://localhost:15672> (`scraper` / `scraper_pass`).
PostgreSQL is published on host port **5433** (5432 is commonly taken by a
locally installed PostgreSQL).

### 2. Create the database schema

```bash
python -c "from src.storage.database import init_schema; init_schema()"
```

### 3. Seed the URL frontier

```bash
python -m src.scraper.seed_frontier --job-name books-crawl --url https://books.toscrape.com/
python -m src.scraper.seed_frontier --status      # queue depth
```

Seeding is a separate process from crawling, which is what makes the URL list
*dynamic*: this command can be run again at any time - including while
crawlers are already running - and the new URLs are picked up immediately.

### 4. Run one or more crawler nodes

```bash
# one node
python -m src.scraper.run_crawler --site books.toscrape.com --max-depth 2

# a second node, in another terminal, consuming from the same queue
python -m src.scraper.run_crawler --site books.toscrape.com --worker-id w2
```

Each process is an independent consumer. RabbitMQ hands each one a different
URL, so the work is balanced across whichever nodes are available, and no
coordination between them is required.

### 5. Inspect the stored data

```bash
python -m src.storage.report              # table report
python -m src.storage.report --json       # machine-readable
```

### 6. Run the whole thing end to end

```bash
python scripts/demo_task2.py
```

Resets the database and queues, crawls `books.toscrape.com` with one worker,
then `quotes.toscrape.com` with two workers in parallel, and prints the final
report.

### Database schema

| Table | Contents |
|-------|----------|
| `crawl_jobs` | One row per crawl run (seeds, status, counters) |
| `raw_pages` | Unmodified HTTP response: status, headers, raw HTML, SHA-256, fetching node |
| `parsed_pages` | Title, description, language, headings, clean text, word count, per-site JSONB fields |
| `discovered_links` | Every link found, with its source page (frontier audit trail) |

### Fault tolerance in this module

| Mechanism | Where |
|-----------|-------|
| Durable queues, persistent messages | `frontier.py` - the frontier survives a broker restart |
| Publisher confirms | `frontier.py` - a URL cannot be silently lost on publish |
| Manual, late acknowledgement | URL acked only after the `COMMIT`, so a crashed worker's URL is redelivered |
| Dead-letter queue | A URL that fails `MAX_DELIVERY_ATTEMPTS` times is set aside, not retried forever |
| Idempotent writes | `ON CONFLICT` upserts, so an at-least-once redelivery cannot duplicate a page |
| HTTP retry + AutoThrottle | `settings.py` - retries 5xx/429 and adapts the request rate |
| Connection pre-ping | `database.py` - a connection dropped by a restarted database is replaced |

---

## Roadmap

- [x] **Task 1** - Environment setup (Python, venv, Git/GitHub, Docker, Kubernetes)
- [x] **Task 2** - Web scraping module (Scrapy, BeautifulSoup/lxml, RabbitMQ, PostgreSQL)
- [ ] **Task 3** - RAG integration (chunking, embeddings, vector store, LLM)
- [ ] **Task 4** - Fault tolerance and load balancing
- [ ] **Task 5** - REST APIs for data retrieval and analysis
- [ ] **Task 6** - Docker / Kubernetes deployment
- [ ] **Task 7** - Testing on two different websites

---

## License

Academic project - provided for educational purposes.
