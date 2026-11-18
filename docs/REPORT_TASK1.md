# Report Notes — Task 1: Environment Setup

Material to paste into the project report (.docx), plus the screenshot list.

---

## Section text for the report

### 1. Introduction to Phase 1

The first phase of the project establishes the development and deployment
environment for the Distributed RAG-Based Web Scraper Framework. Because the
framework is a distributed system, the environment has to support three distinct
concerns at once: an isolated and reproducible Python runtime for the application
code, a version control system so that the work of both team members can be
merged and traced, and a container/orchestration layer so that the individual
components (REST API, scraping workers, message broker, vector store) can later
be deployed as independent, horizontally scalable services. Phase 1 therefore
produces no scraping functionality yet; its deliverable is a verified foundation
on which all later phases are built.

### 2. Python and the virtual environment

Python 3.12.10 was selected as the implementation language because the entire
intended stack — Celery for distributed task execution, FastAPI for the REST
layer, and the Anthropic SDK for the LLM — is Python-native. To keep the project
reproducible and isolated from the system-wide interpreter, a virtual environment
was created in the project root with `py -3.12 -m venv venv` and activated with
`.\venv\Scripts\Activate.ps1`. All dependencies are installed inside this
environment only, so the project cannot be broken by unrelated packages on the
host machine, and the identical environment can be recreated on the second team
member machine directly from the requirements files.

Dependencies are declared in two files, which serve different purposes.
`requirements.txt` is the human-readable, grouped and commented list of the
**direct** dependencies with exact pinned versions (for example `celery==5.4.0`,
`fastapi==0.115.6`), organised by the role each library plays in the
architecture. `requirements.lock.txt` is the complete resolved snapshot of all 98
installed packages, including transitive dependencies, generated with
`pip freeze`. Pinning both levels guarantees that a later `pip install`
reproduces an identical environment rather than silently pulling newer, possibly
incompatible releases.

One practical problem was encountered and solved during installation. The
originally chosen vector database package, `chromadb`, bundles a native C++
extension (`chroma-hnswlib`) which has no prebuilt Windows wheel for Python 3.12
and therefore attempted to compile from source, failing with
`Microsoft Visual C++ 14.0 or greater is required`. Rather than installing a
multi-gigabyte compiler toolchain, the design was adjusted to run Chroma in
**client/server mode**: the vector store becomes a separate service (a container
in the deployment), and the application only needs the thin pure-Python HTTP
client `chromadb-client`. This change removes the build dependency entirely and
is in fact better aligned with the distributed architecture, because the vector
store becomes a shared network service that every worker replica can reach,
instead of an embedded library duplicated inside each worker process. For the
same reason, embeddings are produced with `fastembed`, which uses ONNX Runtime on
the CPU and avoids the very large PyTorch dependency.

### 3. Project structure

The repository is organised by architectural component rather than by file type,
so that each phase of the project maps onto one package. `src/scraper/` will hold
the fetching, parsing and content-extraction logic; `src/worker/` the Celery
application and the distributed task definitions; `src/rag/` the chunking,
embedding, vector-store and LLM-augmentation pipeline; `src/api/` the FastAPI
service that exposes the data-retrieval and analysis endpoints; and
`src/common/` the code shared by every node of the system. Supporting
directories hold the deployment descriptors (`docker/`, `k8s/`), the test suite
(`tests/`), utility scripts (`scripts/`) and the report material with its
screenshots (`docs/`).

Two shared modules were implemented in this phase, because every later component
depends on them. `src/common/config.py` defines a typed Pydantic `Settings`
class that loads all configuration — API keys, the Redis broker URL, the Chroma
host and port, scraper timeouts and retry limits — from environment variables or
from a local `.env` file, and exposes it as a single cached `settings` object.
Centralising configuration this way is essential in a distributed system: the API
process, each worker replica and the RAG pipeline are separate processes that
must agree on the same broker and the same vector-store endpoint, and in
Kubernetes those values will be injected as environment variables from a
ConfigMap and a Secret without changing a single line of code.
`src/common/logging_config.py` configures a uniform log format that stamps every
log line with the component name and the `hostname:pid` of the node that produced
it, which is what makes it possible to tell which of several identical worker
replicas emitted a given message when debugging the running cluster.

A `.env.example` template is committed to the repository to document every
required variable, while the real `.env` file is listed in `.gitignore` so that
the Anthropic API key is never pushed to GitHub. `.gitignore` also excludes the
virtual environment, Python bytecode caches, the local vector-store data and log
files, keeping the repository limited to source code.

### 4. Environment verification

To make the setup verifiable in a single reproducible step rather than by manual
inspection, the script `scripts/verify_env.py` was written. It prints a four-part
report: the interpreter version together with a check that the virtual
environment is actually active; the version of each of the 17 direct Python
dependencies, imported one by one so that a broken installation is detected
rather than assumed; the version of each external command-line tool (Git, Docker,
Docker Compose, kubectl); and finally a reachability check that queries the Docker
daemon and attempts to list the Kubernetes nodes. Each line is prefixed with
`[ OK ]` or `[FAIL]` and the script exits with a non-zero status if anything is
missing, so it can also be reused later as a pre-flight check in CI.

In addition, a pytest suite (`tests/test_environment.py`) was created. It asserts
that the interpreter is at least Python 3.11, imports each of the fifteen
critical dependencies as a parametrised test case, verifies that the `Settings`
object loads with valid defaults, and confirms that the logging configuration
attaches its handlers correctly. All 18 tests pass. This suite is the starting
point of the test strategy that the later phases extend with scraper, RAG and
API tests.

### 5. Version control

A Git repository was initialised in the project root and connected to the GitHub
remote `https://github.com/abed0srour/RAG-Based-web-scraper-framework.git`, with
`main` as the default branch. Using a hosted remote is a requirement for the
project both because the work is done in a group of two — GitHub provides the
shared history, branching and merge mechanism — and because the deployment phase
will build container images from the repository. The complete Task 1 scaffold was
committed as a single descriptive commit and pushed to `origin/main`. The commit
message follows the convention of a short summary line followed by a bulleted
body explaining what the change contains, so that the history itself documents
the progression of the project phase by phase.

### 6. Docker and Kubernetes

Docker Engine 29.8.0 with Docker Compose v5.5.1 was installed to containerise
each component of the framework. Containerisation matters here for a specific
reason: the system is made of several heterogeneous services — a Redis broker, a
Chroma vector store, the FastAPI application and a variable number of identical
Celery workers — and containers let each of them be built, versioned, started and
scaled independently, while guaranteeing that a worker behaves identically on a
developer laptop and on the cluster. Docker Compose will be used for local
multi-service development in the deployment phase.

Kubernetes was enabled through the single-node cluster built into Docker Desktop
(Settings → Kubernetes → Enable Kubernetes), and `kubectl` v1.36.1 is used as the
client. Kubernetes supplies exactly the two non-functional requirements stated in
the project objectives. Load balancing is provided by its Service abstraction,
which distributes incoming requests across the API replicas, together with the
broker-level fair dispatch of tasks across the worker replicas. Fault tolerance
is provided by ReplicaSets, which restart or reschedule a container that crashes,
and by liveness and readiness probes, which keep traffic away from an unhealthy
pod. The Horizontal Pod Autoscaler will later scale the worker Deployment up and
down with the depth of the scraping queue. The cluster was verified with
`kubectl config get-contexts` and `kubectl get nodes`.

### 7. Phase 1 outcome

At the end of Phase 1 the machine has a verified Python 3.12 virtual environment
with all 98 packages installed, a structured and version-controlled repository
pushed to GitHub, a passing test suite, and a running container runtime with a
reachable single-node Kubernetes cluster. The project is ready for Phase 2, the
implementation of the distributed web scraper.

---

## Screenshot list

| # | What to capture | Command / where | Caption for the report |
|---|-----------------|-----------------|------------------------|
| 1 | Versions of Python, Git, Docker, Compose and kubectl in one terminal | `py --version` ; `git --version` ; `docker --version` ; `docker compose version` ; `kubectl version --client` | *Figure 1 — Verification of the installed prerequisites: Python 3.12.10, Git 2.49.0, Docker Engine 29.8.0, Docker Compose v5.5.1 and kubectl v1.36.1.* |
| 2 | Creating the virtual environment, and the prompt showing `(venv)` | `py -3.12 -m venv venv` then `.\venv\Scripts\Activate.ps1` | *Figure 2 — Creation and activation of the project virtual environment; the `(venv)` prefix confirms the isolated interpreter is in use.* |
| 3 | Installed packages inside the venv | `pip list` (or the tail of `pip install -r requirements.txt` showing "Successfully installed …") | *Figure 3 — The 98 project dependencies installed inside the virtual environment, including Celery, Redis, FastAPI, Chroma, FastEmbed and the Anthropic SDK.* |
| 4 | The `requirements.txt` file open in the editor | VS Code | *Figure 4 — `requirements.txt`, grouped by architectural role, pinning the exact version of every direct dependency.* |
| 5 | The project folder tree in the VS Code explorer, expanded | VS Code sidebar | *Figure 5 — Repository layout: one package per framework component (`scraper`, `worker`, `rag`, `api`, `common`) plus the deployment, test and documentation directories.* |
| 6 | `src/common/config.py` open in the editor | VS Code | *Figure 6 — The centralised Pydantic settings module; every node of the distributed system reads its broker, vector-store and LLM configuration from this single typed source.* |
| 7 | Full output of the verification script | `python scripts/verify_env.py` | *Figure 7 — Output of `verify_env.py`: the interpreter, the active virtual environment, all 17 direct dependencies and all external tools report `[ OK ]`.* |
| 8 | pytest results | `python -m pytest` | *Figure 8 — The Phase 1 test suite: 18 tests passed, confirming the interpreter version, every critical import, the configuration layer and the logging setup.* |
| 9 | Git initialisation and remote | `git init` ; `git remote -v` ; `git status --short` | *Figure 9 — Initialisation of the local Git repository and its connection to the GitHub remote; the staged file list shows that the virtual environment and the `.env` file are correctly excluded.* |
| 10 | Commit and push | `git log --oneline` and the `git push -u origin main` output | *Figure 10 — The Task 1 scaffold committed and pushed to the `main` branch of the GitHub repository.* |
| 11 | The repository as it appears on GitHub | github.com page of the repo | *Figure 11 — The project repository on GitHub after the first push, showing the committed structure and the rendered README.* |
| 12 | Docker Desktop window: engine running, Kubernetes enabled | Docker Desktop GUI (Settings → Kubernetes) | *Figure 12 — Docker Desktop with the engine running and the built-in single-node Kubernetes cluster enabled.* |
| 13 | Docker actually running a container | `docker run --rm hello-world` | *Figure 13 — Functional test of the container runtime: the Docker daemon pulls an image from Docker Hub and runs it, confirming the engine works end to end.* |
| 14 | Kubernetes cluster reachable | `kubectl config get-contexts` ; `kubectl get nodes` | *Figure 14 — The Kubernetes cluster is reachable through the `docker-desktop` context and reports its node in the `Ready` state.* |
| 15 | `verify_env.py` re-run with the cluster up | `python scripts/verify_env.py` | *Figure 15 — Final environment verification with Kubernetes running: every check passes and the environment is ready for Phase 2.* |

> Tip: take screenshots 1, 2, 3, 7, 8, 9, 10 and 14 in a terminal where the
> `(venv)` prefix and the project path are visible — it shows the commands were
> run inside the project's own environment.

---

## Flowchart — Phase 1

Node sequence for the Phase 1 branch of the overall framework flowchart:

```
Start
  -> Install Python 3.12
  -> Create virtual environment (venv)
  -> Activate virtual environment
  -> Declare dependencies (requirements.txt)
  -> Install dependencies (pip install -r)
  -> Freeze resolved versions (requirements.lock.txt)
  -> Create project package structure (scraper / worker / rag / api / common)
  -> Implement shared config + logging modules
  -> Create .gitignore and .env.example
  -> Run verification script  --[FAIL]-->  Fix missing component --+
            |                                                      |
          [OK]  <--------------------------------------------------+
            -> Run pytest suite
            -> git init -> git add -> git commit -> git remote add
            -> git push to GitHub (main)
            -> Install Docker Engine + Compose
            -> Enable Kubernetes (Docker Desktop) + install kubectl
            -> Verify cluster (kubectl get nodes)
  -> Environment ready -> Phase 2 (Distributed Scraper)
```

Decision points to show as diamonds in the flowchart: the verification-script
check (`all dependencies OK?`) and the cluster check (`cluster reachable?`), each
looping back to a remediation step when the answer is no.
