"""Task 3 end-to-end demonstration.

One command that produces the evidence for the Task 3 section of the report:

    python scripts/demo_task3.py

What it does, in order:

1. shows what a raw page looks like before and after cleaning, on one real
   stored page, with the byte counts side by side
2. prints the complete normalised JSON document for that page
3. rebuilds the MongoDB collection from every page stored by the crawl, with
   the per-stage timings
4. proves the write path is idempotent by processing the same pages again and
   showing that the document count does not change
5. reports the stored corpus, the indexes and a full-text query answered by
   the text index
6. runs the efficiency benchmark over different data samples

``--skip-benchmark`` leaves out step 6 (the slowest one), and ``--site`` limits
the demonstration to a single website.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Guard against being launched by the wrong interpreter. Set once a re-launch
# has happened, so a venv that genuinely lacks the dependencies reports the
# missing package instead of re-launching itself forever.
_REEXEC_FLAG = "DEMO_TASK3_REEXEC"


def project_python() -> Path | None:
    """Return the project's own virtual-environment interpreter, if present.

    Same reasoning as in ``scripts/demo_task2.py``: ``sys.executable`` cannot
    be trusted, because a ``python`` that resolves to the system interpreter -
    through a shell alias, or a PATH re-ordered after the virtual environment
    was activated, so the prompt still shows ``(venv)`` - runs this script
    without the project's dependencies.
    """
    candidates = [
        PROJECT_ROOT / "venv" / "Scripts" / "python.exe",   # Windows
        PROJECT_ROOT / "venv" / "bin" / "python",           # Linux / macOS
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def relaunch_in_project_python() -> None:
    """Re-run this script under the project interpreter when it differs.

    This demonstration imports the framework in-process rather than spawning
    child processes, so the switch has to happen before those imports are
    attempted - otherwise the wrong interpreter fails on ``import sqlalchemy``
    before any of this code runs.
    """
    target = project_python()
    if target is None or os.environ.get(_REEXEC_FLAG):
        return

    current = Path(sys.executable)
    try:
        if target.samefile(current):
            return
    except OSError:  # pragma: no cover - one of the paths vanished
        if str(target).lower() == str(current).lower():
            return

    print(f"This script was started by {current}")
    print(f"Re-launching with the project interpreter: {target}\n", flush=True)
    completed = subprocess.run(  # noqa: S603 - fixed, trusted argument list
        [str(target), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=PROJECT_ROOT,
        env={**os.environ, _REEXEC_FLAG: "1"},
        check=False,
    )
    raise SystemExit(completed.returncode)


relaunch_in_project_python()

try:
    from src.common.logging_config import quiet_noisy_loggers  # noqa: E402
    from src.processing.benchmark import print_report, run_suite  # noqa: E402
    from src.processing.cleaner import clean_html  # noqa: E402
    from src.processing.normalizer import to_json  # noqa: E402
    from src.processing.pipeline import ProcessingPipeline  # noqa: E402
    from src.processing.report import print_report as print_store_report  # noqa: E402
    from src.processing.store import DocumentStore, StoreUnavailable  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    # One clear instruction beats a traceback: at this point the interpreter is
    # already the project's own, so a missing package means the environment is
    # incomplete rather than wrongly selected.
    print(f"ERROR: {exc}", file=sys.stderr)
    print(
        f"\nThe interpreter in use is {sys.executable}\n"
        "Install the project's dependencies into it with:\n"
        f'  "{project_python() or sys.executable}" -m pip install -r requirements.txt',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


def banner(text: str) -> None:
    print()
    print("#" * 78)
    print(f"#  {text}")
    print("#" * 78, flush=True)


def step_cleaning_demonstration(site: str | None) -> dict | None:
    """Show one page before and after the cleaning stage."""
    banner("STEP 1 - Raw HTML in, clean text out (one real page)")

    rows = list(ProcessingPipeline.iter_raw_pages(site=site, limit=1))
    if not rows:
        print("No pages in PostgreSQL. Run 'python scripts/demo_task2.py' first.")
        return None

    row = rows[0]
    raw_html = row["raw_html"] or ""
    cleaned = clean_html(raw_html)

    print(f"\nURL: {row['url']}")
    print(f"\n--- raw HTML, first 600 characters of {len(raw_html):,} ---")
    print(raw_html[:600].replace("\r", ""))
    print(f"\n--- clean text, first 600 characters of {cleaned.length:,} ---")
    print(cleaned.text[:600])

    print("\n--- what the cleaning removed ---")
    print(f"  raw bytes            : {cleaned.raw_length:,}")
    print(f"  clean characters     : {cleaned.length:,}")
    print(f"  noise removed        : {cleaned.reduction_ratio * 100:.1f}%")
    print(f"  elements removed     : {cleaned.removed_elements}")
    print(f"  HTML comments removed: {cleaned.removed_comments}")
    print(f"  junk lines dropped   : {cleaned.dropped_lines}")
    print(f"  time                 : {cleaned.duration_ms:.2f} ms")
    return row


def step_normalisation_demonstration(row: dict) -> None:
    """Print the canonical JSON document built from that same page."""
    banner("STEP 2 - The normalised JSON document for that page")

    pipeline = ProcessingPipeline(dry_run=True)
    document, _clean_ms, _normalize_ms = pipeline.process_page(row)
    if document is None:
        print("That page yielded no usable text.")
        return

    # The body text is replaced by an excerpt so the structure stays readable.
    preview = dict(document)
    preview["text"] = document["text"][:300] + " ... [truncated]"
    print(to_json(preview))

    print("\nEvery document has exactly these top-level keys, whatever the site:")
    print("  " + ", ".join(sorted(document)))
    print("\nThe site-specific fields are quarantined under 'fields':")
    print("  " + ", ".join(sorted(document["fields"])) or "  (none)")


def step_process_everything(site: str | None) -> DocumentStore:
    banner("STEP 3 - Clean, normalise and index every stored page")

    store = DocumentStore()
    store.require()
    store.drop()

    pipeline = ProcessingPipeline(store=store)
    stats = pipeline.run(site=site)
    stats.print_report()
    return store


def step_idempotency(store: DocumentStore, site: str | None) -> None:
    banner("STEP 4 - Re-processing the same pages does not duplicate them")

    before = store.count()
    stats = ProcessingPipeline(store=store).run(site=site, ensure_indexes=False)
    after = store.count()

    print(f"  documents before the second run : {before}")
    print(f"  pages processed again           : {stats.pages_read}")
    print(f"  newly inserted documents        : {stats.inserted}")
    print(f"  updated documents               : {stats.modified}")
    print(f"  documents after the second run  : {after}")
    print(
        "\n  The document key is a hash of the canonical URL, so a re-run "
        "replaces\n  each document in place. That is what makes the stage "
        "safe to retry after\n  a crash, and what keeps at-least-once "
        "delivery from the URL frontier from\n  producing duplicates."
    )


def step_store_report(store: DocumentStore, site: str | None) -> None:
    banner("STEP 5 - The indexed corpus, and a query answered by the text index")
    print_store_report(store, site=site, samples=1, search="fiction")


def step_benchmark() -> None:
    banner("STEP 6 - Processing efficiency over different data samples")
    report = run_suite(limit_per_sample=500, repeat=3, use_store=True)
    print_report(report)
    print(json.dumps(report["environment"], indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 3 demonstration.")
    parser.add_argument("--site", default=None, help="Limit the demo to one site.")
    parser.add_argument(
        "--skip-benchmark", action="store_true", help="Leave out the benchmark step."
    )
    args = parser.parse_args(argv)

    print("Distributed RAG-Based Web Scraper Framework")
    print("Task 3 - data processing module demonstration")
    print(f"interpreter : {sys.executable}")

    quiet_noisy_loggers()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )
    logging.getLogger("src.processing.store").setLevel(logging.WARNING)

    try:
        row = step_cleaning_demonstration(args.site)
        if row is None:
            return 1
        step_normalisation_demonstration(row)
        store = step_process_everything(args.site)
        step_idempotency(store, args.site)
        step_store_report(store, args.site)
        if not args.skip_benchmark:
            step_benchmark()
    except StoreUnavailable as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    banner("TASK 3 DEMONSTRATION COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
