"""Environment verification for Task 1.

Prints the version of every tool and library the Distributed RAG Web
Scraper Framework depends on, so the setup can be verified (and
screenshotted) in a single command:

    python scripts/verify_env.py
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Running a file as a script puts the script's own directory on sys.path, not
# the current one, so the project root is added explicitly - otherwise the
# service checks below cannot import the framework's own `src` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# (import name, friendly name) pairs for the Python dependencies
PACKAGES: list[tuple[str, str]] = [
    ("scrapy", "Scrapy (distributed web crawler)"),
    ("twisted", "Twisted (Scrapy async engine)"),
    ("pika", "pika (RabbitMQ / AMQP client)"),
    ("sqlalchemy", "SQLAlchemy (PostgreSQL ORM)"),
    ("psycopg", "psycopg 3 (PostgreSQL driver)"),
    ("pymongo", "pymongo (MongoDB document store)"),
    ("celery", "Celery (distributed task queue)"),
    ("redis", "Redis client (broker / backend)"),
    ("flower", "Flower (worker monitoring)"),
    ("requests", "Requests (HTTP fetching)"),
    ("bs4", "BeautifulSoup4 (HTML parsing)"),
    ("lxml", "lxml (fast XML/HTML parser)"),
    ("trafilatura", "Trafilatura (main-content extraction)"),
    ("tenacity", "Tenacity (retry / fault tolerance)"),
    ("chromadb", "Chroma HTTP client (vector store)"),
    ("fastembed", "FastEmbed (ONNX embeddings)"),
    ("numpy", "NumPy"),
    ("anthropic", "Anthropic SDK (Claude LLM)"),
    ("fastapi", "FastAPI (REST API)"),
    ("uvicorn", "Uvicorn (ASGI server)"),
    ("pydantic", "Pydantic (validation)"),
    ("prometheus_client", "Prometheus client (metrics)"),
    ("pytest", "pytest (testing)"),
]

# External command-line tools
TOOLS: list[tuple[str, list[str], str]] = [
    ("git", ["git", "--version"], "Version control"),
    ("docker", ["docker", "--version"], "Containerisation"),
    ("docker-compose", ["docker", "compose", "version"], "Multi-container orchestration"),
    ("kubectl", ["kubectl", "version", "--client=true", "-o", "json"], "Kubernetes CLI"),
]

OK = "[ OK ]"
BAD = "[FAIL]"


def header(title: str) -> None:
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def check_python() -> bool:
    header("1. PYTHON INTERPRETER")
    print(f"{OK} Python          : {sys.version.split()[0]}")
    print(f"       Implementation : {platform.python_implementation()}")
    print(f"       Executable     : {sys.executable}")
    print(f"       Platform       : {platform.platform()}")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    status = OK if in_venv else BAD
    print(f"{status} Virtual env    : {'active -> ' + sys.prefix if in_venv else 'NOT ACTIVE'}")
    return in_venv


def check_packages() -> int:
    header("2. PYTHON DEPENDENCIES")
    failures = 0
    for module, label in PACKAGES:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "installed")
            print(f"{OK} {label:<42} {version}")
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            failures += 1
            print(f"{BAD} {label:<42} {exc}")
    return failures


def check_tools() -> int:
    header("3. EXTERNAL TOOLING")
    failures = 0
    for name, cmd, purpose in TOOLS:
        if shutil.which(cmd[0]) is None:
            failures += 1
            print(f"{BAD} {name:<16} not found on PATH   ({purpose})")
            continue
        try:
            out = subprocess.run(  # noqa: S603 - fixed, trusted command list
                cmd, capture_output=True, text=True, timeout=60, check=False
            )
            lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            if name == "kubectl":
                # `kubectl version -o json` prints a JSON object; pull out the
                # client version instead of showing the opening brace.
                import json

                try:
                    data = json.loads(out.stdout)
                    summary = (
                        "Client Version: "
                        f"{data['clientVersion']['gitVersion']}"
                    )
                except Exception:  # noqa: BLE001
                    summary = lines[0] if lines else "(no output)"
            else:
                summary = lines[0] if lines else "(no output)"
            print(f"{OK} {name:<16} {summary}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{BAD} {name:<16} {exc}")
    return failures


def check_cluster() -> None:
    """Docker daemon + Kubernetes cluster reachability (informational)."""
    header("4. RUNTIME REACHABILITY (daemon / cluster)")

    if shutil.which("docker"):
        out = subprocess.run(  # noqa: S603
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            print(f"{OK} Docker daemon    : running (engine {out.stdout.strip()})")
        else:
            print(f"{BAD} Docker daemon    : not reachable (start Docker Desktop)")

    if shutil.which("kubectl"):
        out = subprocess.run(  # noqa: S603
            ["kubectl", "get", "nodes", "--request-timeout=15s"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if out.returncode == 0:
            lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
            print(f"{OK} Kubernetes       : cluster reachable, {len(lines) - 1} node(s)")
            for ln in lines:
                print(f"       {ln}")
        else:
            print(f"{BAD} Kubernetes       : no cluster reachable "
                  "(enable Kubernetes in Docker Desktop)")


def check_services() -> None:
    """RabbitMQ, PostgreSQL, MongoDB and Chroma reachability (informational).

    All four run as containers from ``docker/docker-compose.yml``; start them
    with ``docker compose -f docker/docker-compose.yml up -d``.
    """
    header("5. BACKING SERVICES (broker / databases)")

    # --- RabbitMQ ---
    try:
        from src.messaging.frontier import FrontierQueue

        with FrontierQueue() as frontier:
            depth = frontier.depth()
        print(f"{OK} RabbitMQ         : reachable")
        for key, value in depth.items():
            print(f"       {key:<22} {value}")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} RabbitMQ         : not reachable ({type(exc).__name__})")
        print("       start it with: docker compose -f docker/docker-compose.yml up -d")

    # --- PostgreSQL ---
    try:
        from sqlalchemy import text

        from src.storage.database import get_engine

        with get_engine().connect() as conn:
            version = conn.execute(text("select version()")).scalar_one()
            tables = conn.execute(
                text(
                    "select table_name from information_schema.tables "
                    "where table_schema = 'public' order by table_name"
                )
            ).scalars().all()
        print(f"{OK} PostgreSQL       : {version.split(',')[0]}")
        print(f"       tables                 {', '.join(tables) or '(none yet)'}")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} PostgreSQL       : not reachable ({type(exc).__name__})")
        print("       start it with: docker compose -f docker/docker-compose.yml up -d")

    # --- MongoDB (the processed-document store of Task 3) ---
    try:
        from src.processing.store import DocumentStore

        store = DocumentStore()
        if not store.ping():
            raise RuntimeError("ping failed")
        print(f"{OK} MongoDB          : {store.server_version()}")
        print(f"       database               {store.database_name}")
        print(f"       documents indexed      {store.count()}")
        print(f"       indexes                "
              f"{', '.join(i['name'] for i in store.indexes()) or '(none yet)'}")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} MongoDB          : not reachable ({type(exc).__name__})")
        print("       start it with: docker compose -f docker/docker-compose.yml "
              "up -d mongodb")

    # --- Chroma (the vector store of Task 4) ---
    try:
        from src.rag.vector_store import VectorStore

        vectors = VectorStore()
        if not vectors.ping():
            raise RuntimeError("heartbeat failed")
        print(f"{OK} Chroma           : {vectors.server_version()}")
        print(f"       collection             {vectors.collection_name}")
        try:
            print(f"       vectors indexed        {vectors.count()}")
        except Exception:  # noqa: BLE001 - an empty server has no collection yet
            print("       vectors indexed        (collection not created yet)")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} Chroma           : not reachable ({type(exc).__name__})")
        print("       start it with: docker compose -f docker/docker-compose.yml "
              "up -d chroma")

    # --- The embedding model (local, no service) ---
    try:
        from src.rag.embedder import get_embedder

        embedder = get_embedder()
        print(f"{OK} Embedding model  : {embedder.model_name} "
              f"({embedder.dimension} dims, CPU)")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} Embedding model  : not loadable ({type(exc).__name__})")
        print("       the first run downloads about 45 MB into "
              f"{PROJECT_ROOT / 'data' / 'models'}")


def main() -> int:
    print("Distributed RAG-Based Web Scraper Framework")
    print("Environment verification report")

    venv_ok = check_python()
    pkg_failures = check_packages()
    tool_failures = check_tools()
    check_cluster()
    check_services()

    header("SUMMARY")
    total = pkg_failures + tool_failures + (0 if venv_ok else 1)
    if total == 0:
        print(f"{OK} Environment is fully configured.")
    else:
        print(f"{BAD} {total} issue(s) found - see the FAIL lines above.")
    print()
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
