"""Command-line report over the scraped data.

Prints what is actually in PostgreSQL after a crawl, which is the quickest way
to verify the whole pipeline end to end and the basis of the data-analysis API
added in a later task.

    python -m src.storage.report
    python -m src.storage.report --site quotes.toscrape.com --samples 3
    python -m src.storage.report --json
"""

from __future__ import annotations

import argparse
import json
import sys

from sqlalchemy import text

from src.common.logging_config import quiet_noisy_loggers
from src.messaging.frontier import FrontierQueue
from src.storage.database import get_engine


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def print_report(site: str | None = None, samples: int = 2) -> None:
    engine = get_engine()
    with engine.connect() as c:

        _rule("CRAWL JOBS")
        rows = c.execute(
            text(
                """select id, job_name, site, status, pages_scraped,
                          started_at, finished_at
                     from crawl_jobs order by id"""
            )
        ).all()
        if rows:
            print(f"{'id':>4}  {'name':<16} {'site':<24} {'status':<10} {'pages':>6}")
            print("-" * 68)
            for r in rows:
                print(f"{r[0]:>4}  {r[1]:<16} {r[2]:<24} {r[3]:<10} {r[4]:>6}")
        else:
            print("(no crawl jobs recorded)")

        _rule("STORED DATA PER SITE")
        where = "where site = :site" if site else ""
        params = {"site": site} if site else {}
        rows = c.execute(
            text(
                f"""select r.site,
                           count(distinct r.id)            as raw_pages,
                           count(distinct p.id)            as parsed_pages,
                           coalesce(sum(r.content_length),0) as raw_bytes,
                           coalesce(sum(p.word_count),0)     as words,
                           count(distinct r.fetched_by)      as worker_nodes
                      from raw_pages r
                      left join parsed_pages p on p.raw_page_id = r.id
                      {where}
                     group by r.site order by r.site"""
            ),
            params,
        ).all()

        if not rows:
            print("(no pages stored yet)")
        else:
            print(
                f"{'site':<24} {'raw':>5} {'parsed':>7} {'bytes':>9} "
                f"{'words':>8} {'nodes':>6}"
            )
            print("-" * 64)
            for r in rows:
                print(
                    f"{r[0]:<24} {r[1]:>5} {r[2]:>7} {r[3]:>9} {r[4]:>8} {r[5]:>6}"
                )

        _rule("WORK DISTRIBUTION ACROSS WORKER NODES")
        rows = c.execute(
            text(
                """select fetched_by, site, count(*) pages
                     from raw_pages
                    group by fetched_by, site
                    order by fetched_by, site"""
            )
        ).all()
        if rows:
            print(f"{'worker node (host:pid)':<32} {'site':<24} {'pages':>6}")
            print("-" * 66)
            for r in rows:
                print(f"{r[0] or '(unknown)':<32} {r[1]:<24} {r[2]:>6}")
            print(
                "\nSeveral distinct nodes for one site is the evidence that the "
                "crawl\nwas executed in parallel by independent worker processes."
            )
        else:
            print("(no pages stored yet)")

        _rule("DISCOVERED LINKS (frontier audit trail)")
        rows = c.execute(
            text(
                """select site, count(*) total,
                          count(*) filter (where enqueued) internal
                     from discovered_links group by site order by site"""
            )
        ).all()
        for r in rows or []:
            print(f"  {r[0]:<24} {r[1]:>6} links  ({r[2]} internal / in-scope)")
        if not rows:
            print("(none)")

        if samples > 0:
            _rule(f"SAMPLE PARSED PAGES (up to {samples} per site)")
            site_rows = [r[0] for r in c.execute(
                text("select distinct site from parsed_pages order by site")
            ).all()]
            for s in site_rows:
                if site and s != site:
                    continue
                print(f"\n--- {s} ---")
                for r in c.execute(
                    text(
                        """select url, title, word_count, parser_used,
                                  structured_data
                             from parsed_pages where site = :s
                            order by id limit :n"""
                    ),
                    {"s": s, "n": samples},
                ).all():
                    print(f"  url     : {r[0]}")
                    print(f"  title   : {(r[1] or '')[:70]}")
                    print(f"  words   : {r[2]}   parser: {r[3]}")
                    keys = sorted(k for k in (r[4] or {}))
                    print(f"  fields  : {', '.join(keys) or '(none)'}")
                    print()

    # Queue state is read from RabbitMQ, not the database.
    _rule("RABBITMQ FRONTIER STATE")
    try:
        with FrontierQueue() as frontier:
            for key, value in frontier.depth().items():
                print(f"  {key:<24} {value}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (broker unreachable: {exc})")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report on the scraped data.")
    parser.add_argument("--site", default=None, help="Limit the report to one site.")
    parser.add_argument(
        "--samples", type=int, default=2, help="Sample parsed pages per site."
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit machine-readable JSON instead of the table report.",
    )
    args = parser.parse_args(argv)

    quiet_noisy_loggers()

    if args.as_json:
        from src.storage.database import ScrapeRepository

        print(json.dumps(ScrapeRepository.stats(), indent=2, default=str))
        return 0

    print_report(site=args.site, samples=args.samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
