"""The scraper node's liveness probe (Task 7).

    python -m src.worker.healthcheck                 # this node
    python -m src.worker.healthcheck --max-age 180
    python -m src.worker.healthcheck --all --json    # the whole fleet

Exit codes are the interface, because that is what both Kubernetes and
Docker consume:

* ``0`` - the node has made progress recently, or is still inside its
  start-up grace window.
* ``1`` - the heartbeat is stale, unreadable or missing. Under a Kubernetes
  ``livenessProbe`` this is what gets the container killed and restarted;
  under a Docker ``healthcheck`` it marks the container unhealthy.

Why an ``exec`` probe rather than an HTTP one: a Scrapy node is not a server.
Giving it an HTTP endpoint purely to be probed would mean running a web
server inside the crawler process - and, worse, that server would keep
answering while the crawl itself was wedged, which is the exact failure this
is meant to catch. Reading a file that only advances when a page completes
cannot be fooled that way.

``--all`` is for humans: it prints every node's heartbeat in one table, which
is the fleet view that neither ``kubectl get pods`` (which shows processes,
not progress) nor the queue depth (which shows work, not workers) gives.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.worker.heartbeat import list_heartbeats, node_id, read_heartbeat

#: A node that has not finished a page in this long is considered wedged.
#: Chosen to be comfortably more than the slowest reasonable page: the
#: crawler's own request timeout is 20 s with up to 3 retries, and politeness
#: adds a delay per host, so a healthy node can legitimately be quiet for
#: well over a minute.
DEFAULT_MAX_AGE = 180.0

#: How long after start-up a node is excused for not having finished a page.
DEFAULT_GRACE = 120.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a scraper node is still making progress."
    )
    parser.add_argument(
        "--node", default=None,
        help="Node name to check (default: this process's own id).",
    )
    parser.add_argument(
        "--dir", default=None, dest="directory",
        help="Heartbeat directory (default: HEARTBEAT_DIR or data/heartbeats).",
    )
    parser.add_argument(
        "--max-age", type=float, default=DEFAULT_MAX_AGE,
        help=f"Seconds a node may go without finishing a page "
             f"(default: {DEFAULT_MAX_AGE:.0f}).",
    )
    parser.add_argument(
        "--grace", type=float, default=DEFAULT_GRACE,
        help=f"Start-up grace period in seconds (default: {DEFAULT_GRACE:.0f}).",
    )
    parser.add_argument(
        "--all", action="store_true", dest="show_all",
        help="Report every node found, not just this one.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Print the status as JSON.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Print nothing; use the exit code only (for a probe).",
    )
    args = parser.parse_args(argv)

    if args.show_all:
        statuses = list_heartbeats(args.directory)
        healthy = [s for s in statuses if s.healthy(args.max_age, args.grace)]

        if args.as_json:
            print(json.dumps(
                {
                    "nodes": len(statuses),
                    "healthy": len(healthy),
                    "max_age_seconds": args.max_age,
                    "statuses": [s.as_dict() for s in statuses],
                },
                indent=2,
            ))
        elif not args.quiet:
            if not statuses:
                print("No heartbeats found - no scraper node has run yet.")
            else:
                print(f"{'NODE':<34}{'AGE':>8}{'PAGES':>8}{'UP':>10}  STATE")
                for status in statuses:
                    age = (
                        f"{status.age_seconds:.0f}s"
                        if status.age_seconds is not None else "-"
                    )
                    pages = status.data.get("pages_done", "-")
                    uptime = status.data.get("uptime_seconds")
                    up = f"{float(uptime):.0f}s" if uptime is not None else "-"
                    state = (
                        "ok" if status.healthy(args.max_age, args.grace) else "STALE"
                    )
                    print(f"{str(status.data.get('node', status.path.stem)):<34}"
                          f"{age:>8}{str(pages):>8}{up:>10}  {state}")
                print()
                print(f"{len(healthy)}/{len(statuses)} node(s) making progress")

        # The fleet is unhealthy only when nothing at all is working; one
        # dead node out of five is a restart, not an outage.
        return 0 if healthy or not statuses else 1

    status = read_heartbeat(args.node, args.directory)
    ok = status.healthy(args.max_age, args.grace)

    if args.as_json:
        print(json.dumps({"healthy": ok, **status.as_dict()}, indent=2))
    elif not args.quiet:
        name = args.node or node_id()
        if ok:
            age = status.age_seconds
            print(f"{name}: ok"
                  + (f" (last page {age:.0f}s ago)" if age is not None else ""))
        elif not status.exists:
            print(f"{name}: no heartbeat at {status.path}", file=sys.stderr)
        elif status.error:
            print(f"{name}: unreadable heartbeat: {status.error}", file=sys.stderr)
        else:
            print(
                f"{name}: stale - no page finished in "
                f"{status.age_seconds:.0f}s (limit {args.max_age:.0f}s)",
                file=sys.stderr,
            )

    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
