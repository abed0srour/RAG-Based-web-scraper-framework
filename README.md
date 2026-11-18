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
| 2 | Integrate a RAG-based system to enhance scraped data | Cleaning + JSON normalisation + MongoDB indexing, then chunking, ONNX embeddings, Chroma vector store and Claude generation |
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
| Processed document store (NoSQL) | MongoDB 7 via pymongo (weighted full-text index) |
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
│   ├── processing/
│   │   ├── cleaner.py        Stage 1: strip tags, scripts, boilerplate, formatting
│   │   ├── normalizer.py     Stage 2: canonical JSON document + validation
│   │   ├── store.py          Stage 3: MongoDB document store and its indexes
│   │   ├── pipeline.py       Postgres -> clean -> normalise -> MongoDB
│   │   ├── benchmark.py      Processing-efficiency test over different samples
│   │   └── report.py         CLI report over the processed documents
│   ├── worker/               Celery application and distributed tasks (Task 5)
│   ├── rag/                  Chunking, embeddings, vector store, LLM (Task 4)
│   └── api/                  FastAPI application (Task 6)
├── scripts/
│   ├── verify_env.py         Environment verification report (Task 1)
│   ├── demo_task2.py         End-to-end Task 2 demonstration on two sites
│   └── demo_task3.py         End-to-end Task 3 demonstration (cleaning -> index)
├── tests/                    pytest test-suite
├── docker/
│   └── docker-compose.yml    RabbitMQ + PostgreSQL + MongoDB infrastructure
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

## Task 3 - Data processing module

Raw HTML is unusable as it stands: measured over the crawled corpus, **94.7%
of the stored bytes are markup, scripts, styles and navigation chrome** rather
than content. This module turns the stored responses into clean, uniform JSON
documents and indexes them in MongoDB.

### Pipeline

```
 raw_pages (PostgreSQL)
        |  server-side cursor, batches of 50
        v
 +----------------------------------------------------------+
 | STAGE 1  cleaner.py        BeautifulSoup + lxml          |
 |  1. script / style / noscript / iframe / comments        |
 |  2. nav / header / footer / aside / form (boilerplate)   |
 |  3. entities, Unicode NFKC, invisible + control chars    |
 |  4. whitespace, blank lines, nav lines, duplicate lines  |
 +---------------------------+------------------------------+
                             v
 +----------------------------------------------------------+
 | STAGE 2  normalizer.py     canonical JSON document       |
 |  doc_id = SHA-1(canonical URL)      -> idempotent writes |
 |  title, headings, keywords, stats, provenance            |
 |  site-specific fields quarantined under "fields"         |
 |  validate_document(): 12 required keys + JSON-encodable  |
 +---------------------------+------------------------------+
                             v
 +----------------------------------------------------------+
 | STAGE 3  store.py          MongoDB (NoSQL)               |
 |  bulk ReplaceOne upserts, ordered=False                  |
 |  5 indexes, including a weighted full-text index         |
 +---------------------------+------------------------------+
                             v
              processed_data.documents  (indexed, searchable)
```

### 1. Start MongoDB

```bash
docker compose -f docker/docker-compose.yml up -d mongodb mongo-express
docker compose -f docker/docker-compose.yml ps
```

Mongo Express UI: <http://localhost:8081> (`scraper` / `scraper_pass`).
MongoDB is published on host port **27018** (27017 is commonly taken by a
locally installed MongoDB, for the same reason PostgreSQL uses 5433).

### 2. Process the crawled pages

```bash
python -m src.processing.pipeline --reset          # rebuild from scratch
python -m src.processing.pipeline                  # add / refresh documents
python -m src.processing.pipeline --site quotes.toscrape.com --limit 50
python -m src.processing.pipeline --dry-run        # clean + normalise, write nothing
python -m src.processing.pipeline --sample         # print one stored document
```

Re-running is safe: the document key is derived from the canonical URL, so a
second run over the same pages reports `0 new, 45 updated` and the collection
size does not change. That is what makes the stage retryable after a crash,
and what keeps at-least-once delivery from the URL frontier from creating
duplicates.

### 3. Inspect the processed documents

```bash
python -m src.processing.report                    # corpus + indexes + duplicates
python -m src.processing.report --search "fiction" # query through the text index
python -m src.processing.report --doc-url https://books.toscrape.com/
python -m src.processing.report --json
```

### 4. Test processing efficiency

```bash
python -m src.processing.benchmark --repeat 5      # the full suite
python -m src.processing.benchmark --no-store      # CPU stages only
python -m src.processing.benchmark --save          # keep the JSON report
```

The benchmark runs the module over samples that differ along the three axes
that change the cost of processing a page - the source website, the document
size and the write batch size - and times each stage separately, because
cleaning and normalising are CPU-bound while storing is dominated by the round
trip to MongoDB.

| Sample | Pages | Avg page | Clean p50 | Pages/s | Noise removed |
|--------|------:|---------:|----------:|--------:|--------------:|
| `quotes.toscrape.com` | 29 | 5.4 KB | 3.24 ms | 188.3 | 78.8% |
| `books.toscrape.com` | 16 | 45.5 KB | 21.82 ms | 43.2 | 98.1% |
| small (< 10 KB) | 23 | 3.7 KB | 3.97 ms | 198.9 | 77.6% |
| medium (10-40 KB) | 10 | 21.2 KB | 9.50 ms | 78.0 | 92.3% |
| large (>= 40 KB) | 12 | 48.9 KB | 22.90 ms | 37.1 | 98.1% |
| **whole corpus** | 45 | 19.6 KB | 6.52 ms | **71.6** | **94.7%** |

Writing in batches of 50 costs 0.373 ms per document against 1.401 ms one at a
time - a 3.8x improvement, which is why the pipeline batches. Cleaning is
sub-linear in page size: 13x larger pages cost 5.8x more to clean.

### 5. Run the whole thing end to end

```bash
python scripts/demo_task3.py                 # everything, including the benchmark
python scripts/demo_task3.py --skip-benchmark
```

Shows one page before and after cleaning, prints its complete JSON document,
rebuilds the collection from every crawled page, proves the write path is
idempotent, reports the indexed corpus with a full-text query, and runs the
efficiency benchmark.

### The document schema

Every document has the same top-level keys whatever site it came from; only
`fields` varies per site.

| Key | Contents |
|-----|----------|
| `doc_id` | SHA-1 of the canonical URL - the deterministic primary key |
| `url`, `original_url`, `site`, `path`, `query` | identity, canonicalised |
| `title`, `description`, `language`, `headings` | cleaned metadata |
| `text` | the clean body text |
| `keywords` | 15 most frequent non-stop-words, boosted in the text index |
| `fields` | site-specific data (price, rating, quotes, authors, ...) |
| `content_hash` | SHA-256 of the clean text - change and duplicate detection |
| `stats` | word / char / line / sentence counts, lexical diversity, reading time |
| `source` | provenance: raw page id, job, HTTP status, which node fetched it |
| `processing` | schema and cleaner versions, timings, how much noise was removed |

### MongoDB indexes

| Index | Purpose |
|-------|---------|
| `uq_doc_id` (unique) | one document per canonical URL - idempotent writes |
| `ix_fulltext` (text) | weighted keyword search: title 10, description 5, headings 4, keywords 3, body 1 |
| `ix_site_processed_at` | per-site listings in recency order |
| `ix_content_hash` | near-duplicate detection before embedding |
| `ix_word_count` | range queries over document size |

The text index is created with `language_override="_text_language"`: MongoDB
otherwise reads the stemming language from a field named `language`, which in
this schema holds the page's HTML `lang` attribute, and a page declaring
`lang="en-us"` would be rejected on write.

### Fault tolerance in this module

| Mechanism | Where |
|-----------|-------|
| Derived document key | `normalizer.py` - re-processing replaces a document instead of duplicating it |
| Unique index on `doc_id` | `store.py` - the same guarantee enforced by the database |
| Unordered bulk writes | `store.py` - one rejected document does not discard its batch |
| Per-page error isolation | `pipeline.py` - a page that cannot be processed is counted, the run continues |
| Validation before the write | `normalizer.py` - a malformed document is rejected at the boundary, not discovered by a reader later |
| Streaming reads | `pipeline.py` - memory is bounded by the batch size, not by the corpus |
| Short server timeouts | `store.py` - an unreachable database fails fast instead of hanging a worker |

---

## Roadmap

- [x] **Task 1** - Environment setup (Python, venv, Git/GitHub, Docker, Kubernetes)
- [x] **Task 2** - Web scraping module (Scrapy, BeautifulSoup/lxml, RabbitMQ, PostgreSQL)
- [x] **Task 3** - Data processing module (cleaning, JSON normalisation, MongoDB indexing)
- [ ] **Task 4** - RAG integration (chunking, embeddings, vector store, LLM)
- [ ] **Task 5** - Fault tolerance and load balancing
- [ ] **Task 6** - REST APIs for data retrieval and analysis
- [ ] **Task 7** - Docker / Kubernetes deployment
- [ ] **Task 8** - Testing on two different websites

---

## License

Academic project - provided for educational purposes.
