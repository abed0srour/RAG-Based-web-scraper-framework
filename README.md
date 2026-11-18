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
| Distributed task queue | Celery 5.4 |
| Broker / result backend | Redis 7 |
| Worker monitoring | Flower |
| HTTP fetching | Requests + Tenacity (retries) |
| HTML parsing / extraction | BeautifulSoup4, lxml, Trafilatura |
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
├── src/
│   ├── common/               Shared configuration, logging, data models
│   ├── scraper/              Fetching, parsing and content extraction
│   ├── worker/               Celery application and distributed tasks
│   ├── rag/                  Chunking, embeddings, vector store, LLM augmentation
│   └── api/                  FastAPI application (retrieval + analysis endpoints)
├── scripts/
│   └── verify_env.py         Environment verification report (Task 1)
├── tests/                    pytest test-suite
├── docker/                   Dockerfiles and docker-compose stack
├── k8s/                      Kubernetes manifests (deployments, services, HPA)
├── data/                     Local run artefacts (git-ignored)
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

## Roadmap

- [x] **Task 1** - Environment setup (Python, venv, Git/GitHub, Docker, Kubernetes)
- [ ] **Task 2** - Distributed web scraper implementation
- [ ] **Task 3** - RAG integration (chunking, embeddings, vector store, LLM)
- [ ] **Task 4** - Fault tolerance and load balancing
- [ ] **Task 5** - REST APIs for data retrieval and analysis
- [ ] **Task 6** - Docker / Kubernetes deployment
- [ ] **Task 7** - Testing on two different websites

---

## License

Academic project - provided for educational purposes.
