"""Task 2 end-to-end demonstration.

Runs the complete scraping module against the two target websites, from a
clean database, and prints a report at every stage. This is the single command
that produces the evidence for the Task 2 section of the report:

    python scripts/demo_task2.py

What it does, in order:

1. resets PostgreSQL and the RabbitMQ queues so the run is reproducible
2. seeds the frontier with the first site and crawls it with ONE worker
3. seeds the frontier with the second site and crawls it with TWO workers
   running at the same time, to show the load being balanced between them
4. prints the per-site totals and the per-node page counts

Use ``--keep`` to add to the existing data instead of starting clean.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_python() -> str:
    """Return the interpreter that has the project's dependencies installed.

    The demo launches its steps as child processes, and those children must
    use the project's virtual environment. Relying on ``sys.executable`` alone
    is not enough: if the script is started by a ``python`` that resolves to
    the system interpreter - which happens when something re-orders PATH after
    the virtual environment was activated - every child inherits the wrong
    interpreter and fails with ``ModuleNotFoundError``. So the venv inside the
    project is preferred whenever it exists, whatever started this script.
    """
    candidates = [
        PROJECT_ROOT / "venv" / "Scripts" / "python.exe",   # Windows
        PROJECT_ROOT / "venv" / "bin" / "python",           # Linux / macOS
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


PYTHON = project_python()

SITES = [
    {
        "name": "books-crawl",
        "seed": "https://books.toscrape.com/",
        "host": "books.toscrape.com",
        "workers": 1,
        "max_pages": 12,
        "max_depth": 1,
    },
    {
        "name": "quotes-crawl",
        "seed": "https://quotes.toscrape.com/",
        "host": "quotes.toscrape.com",
        "workers": 2,
        "max_pages": 10,
        "max_depth": 2,
    },
]


def check_dependencies() -> list[str]:
    """Return the names of required packages the chosen interpreter cannot import."""
    required = ["scrapy", "pika", "sqlalchemy", "psycopg", "bs4", "lxml"]
    probe = (
        # importlib.util must be imported explicitly: importing the parent
        # package alone does not make the submodule available.
        "import importlib.util;"
        "missing=[m for m in " + repr(required) + " "
        "if importlib.util.find_spec(m) is None];"
        "print(','.join(missing))"
    )
    result = subprocess.run(  # noqa: S603 - fixed, trusted argument list
        [PYTHON, "-c", probe],
        cwd=PROJECT_ROOT, text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        return required
    return [name for name in result.stdout.strip().split(",") if name]


def banner(text: str) -> None:
    print()
    print("#" * 78)
    print(f"#  {text}")
    print("#" * 78, flush=True)


def run(args: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    """Run a module in the project's interpreter."""
    print(f"\n$ python {' '.join(args)}", flush=True)
    return subprocess.run(  # noqa: S603 - fixed, trusted argument list
        [PYTHON, *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=capture,
        check=False,
    )


def reset_state() -> None:
    banner("STEP 1 - Reset PostgreSQL tables and RabbitMQ queues")
    run(["-c", "from src.storage.database import init_schema; init_schema(drop_first=True)"])
    run(["-m", "src.scraper.seed_frontier", "--purge"])


def seed(site: dict) -> int | None:
    """Publish the seed URL and return the crawl job id."""
    result = run(
        [
            "-m", "src.scraper.seed_frontier",
            "--job-name", site["name"],
            "--url", site["seed"],
        ],
        capture=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    for line in (result.stdout or "").splitlines():
        if line.startswith("job_id="):
            return int(line.split("=", 1)[1])
    return None


def crawl(site: dict, job_id: int | None) -> None:
    """Start the configured number of crawler processes and wait for them."""
    log_dir = PROJECT_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[str, subprocess.Popen, Path]] = []
    for index in range(1, site["workers"] + 1):
        worker_id = f"{site['host'].split('.')[0]}-w{index}"
        log_path = log_dir / f"{worker_id}.log"
        args = [
            PYTHON, "-m", "src.scraper.run_crawler",
            "--site", site["host"],
            "--max-depth", str(site["max_depth"]),
            "--max-pages", str(site["max_pages"]),
            "--worker-id", worker_id,
        ]
        if job_id is not None:
            args += ["--job-id", str(job_id)]

        print(f"  starting worker {worker_id} (log: {log_path.name})", flush=True)
        handle = log_path.open("w", encoding="utf-8")
        processes.append(
            (
                worker_id,
                subprocess.Popen(  # noqa: S603
                    args, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT
                ),
                log_path,
            )
        )

    # While they work, show the queue draining - the clearest sign that the
    # URL list is being managed dynamically through the broker.
    print("\n  queue depth while the workers run:")
    for _ in range(6):
        time.sleep(5)
        probe = run(["-m", "src.scraper.seed_frontier", "--status"], capture=True)
        depth = " ".join((probe.stdout or "").split())
        print(f"    {depth}", flush=True)
        if all(p.poll() is not None for _n, p, _l in processes):
            break

    for worker_id, process, log_path in processes:
        process.wait()
        stored = sum(
            1 for line in log_path.read_text(encoding="utf-8").splitlines()
            if "Stored " in line
        )
        print(f"  worker {worker_id} finished: {stored} page(s) stored", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 2 demonstration run.")
    parser.add_argument(
        "--keep", action="store_true",
        help="Do not reset the database and queues before running.",
    )
    args = parser.parse_args(argv)

    print("Distributed RAG-Based Web Scraper Framework")
    print("Task 2 - Scraping module demonstration on two websites")
    print(f"Interpreter: {PYTHON}")

    # Fail fast and clearly rather than letting every step die on an import.
    missing = check_dependencies()
    if missing:
        print()
        print("ERROR: the interpreter above is missing required packages:")
        for name in missing:
            print(f"  - {name}")
        print()
        print("Install them into the project virtual environment with:")
        print(f'  "{PYTHON}" -m pip install -r requirements.txt')
        return 1

    if not args.keep:
        reset_state()

    for step, site in enumerate(SITES, start=2):
        banner(
            f"STEP {step} - Crawl {site['host']} "
            f"with {site['workers']} worker process(es)"
        )
        job_id = seed(site)
        crawl(site, job_id)
        if job_id is not None:
            run(["-m", "src.scraper.seed_frontier", "--finish-job", str(job_id)])

    banner(f"STEP {len(SITES) + 2} - Final report from PostgreSQL")
    run(["-m", "src.storage.report", "--samples", "1"])

    print("\nTask 2 demonstration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
