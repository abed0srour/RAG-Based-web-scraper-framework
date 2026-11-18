"""Seed the RabbitMQ frontier with start URLs, and inspect/purge the queues.

This is the *producer* side of the message-queuing layer. It is a separate
process from the crawlers on purpose: it shows that URLs are managed
dynamically, because this command can be run at any time - including while
crawlers are already running - and the new URLs are picked up immediately.

    # register a crawl job and seed it
    python -m src.scraper.seed_frontier --url https://books.toscrape.com/ --job-name books

    # add more URLs to a crawl that is already running
    python -m src.scraper.seed_frontier --url https://quotes.toscrape.com/

    # inspect the queues / empty them between runs
    python -m src.scraper.seed_frontier --status
    python -m src.scraper.seed_frontier --purge
"""

from __future__ import annotations

import argparse
import json
import sys

from src.common.logging_config import setup_logging
from src.messaging.frontier import FrontierQueue, UrlMessage
from src.scraper.parser import normalise_url, site_of
from src.storage.database import ScrapeRepository, init_schema


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish seed URLs onto the RabbitMQ frontier."
    )
    parser.add_argument(
        "--url", action="append", default=[], dest="urls",
        help="Seed URL (repeat the flag for several).",
    )
    parser.add_argument(
        "--job-name", default=None,
        help="Create a crawl job with this name and attach the seeds to it.",
    )
    parser.add_argument(
        "--job-id", type=int, default=None,
        help="Attach the seeds to an existing crawl job id.",
    )
    parser.add_argument("--depth", type=int, default=0, help="Starting depth.")
    parser.add_argument(
        "--status", action="store_true", help="Show queue depths and exit."
    )
    parser.add_argument(
        "--purge", action="store_true", help="Empty both queues and exit."
    )
    parser.add_argument(
        "--init-db", action="store_true",
        help="Create the PostgreSQL tables before seeding.",
    )
    parser.add_argument(
        "--finish-job", type=int, default=None, metavar="JOB_ID",
        help="Mark a crawl job as completed and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging("seeder")

    # Closing a job touches only the database, so it needs no broker connection.
    if args.finish_job is not None:
        ScrapeRepository.finish_job(args.finish_job, status="completed")
        logger.info("Marked crawl job %s as completed", args.finish_job)
        return 0

    with FrontierQueue() as frontier:

        if args.purge:
            frontier.purge()
            print(json.dumps(frontier.depth(), indent=2))
            return 0

        if args.status:
            print(json.dumps(frontier.depth(), indent=2))
            return 0

        if args.init_db:
            init_schema()
            logger.info("PostgreSQL schema created")

        if not args.urls:
            logger.error("No --url given. Use --status or --purge to inspect.")
            return 2

        # Normalise and validate the seeds before anything is published.
        seeds: list[str] = []
        for raw in args.urls:
            url = normalise_url(raw, raw)
            if url is None:
                logger.error("Skipping unusable seed URL: %s", raw)
                continue
            seeds.append(url)

        if not seeds:
            logger.error("No usable seed URLs")
            return 2

        job_id = args.job_id
        if job_id is None and args.job_name:
            init_schema()
            job_id = ScrapeRepository.create_job(
                job_name=args.job_name, site=site_of(seeds[0]), seed_urls=seeds
            )
            logger.info("Created crawl job %s (%s)", job_id, args.job_name)

        published = frontier.publish_many(
            [
                UrlMessage(
                    url=url, site=site_of(url), depth=args.depth, job_id=job_id
                )
                for url in seeds
            ]
        )

        logger.info("Published %s seed URL(s) to the frontier", published)
        for url in seeds:
            logger.info("  seed: %s", url)

        depth = frontier.depth()
        logger.info("Frontier now holds %s message(s)", depth[frontier.queue])
        if job_id is not None:
            print(f"job_id={job_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
