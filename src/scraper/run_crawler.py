"""Launch one crawler process (one node of the distributed crawler).

Each invocation starts an independent Scrapy process that consumes URLs from
the shared RabbitMQ frontier. Running the command several times - in several
terminals, or as several containers - is what makes the crawl distributed; the
processes need no knowledge of each other, because RabbitMQ decides which of
them gets which URL.

    python -m src.scraper.run_crawler --site books.toscrape.com --max-depth 2
    python -m src.scraper.run_crawler --worker-id 2      # second node
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from src.common.logging_config import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one node of the distributed Scrapy crawler."
    )
    parser.add_argument(
        "--job-id", type=int, default=None,
        help="Crawl job id to attach the stored pages to.",
    )
    parser.add_argument(
        "--site", default=None,
        help="Comma-separated hosts the crawler is allowed to visit "
             "(URLs for other hosts are skipped and acknowledged).",
    )
    parser.add_argument(
        "--max-depth", type=int, default=2,
        help="Maximum link depth from the seed URLs (default: 2).",
    )
    parser.add_argument(
        "--max-pages", type=int, default=0,
        help="Stop after this many pages on this node (0 = no limit).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="How many URLs to pull from the frontier per idle round.",
    )
    parser.add_argument(
        "--idle-rounds", type=int, default=3,
        help="Idle rounds with an empty frontier before the node exits. "
             "0 waits indefinitely, which is what a long-lived node in a "
             "container should do: exiting gets it restarted, and every "
             "restart rejoins the broker's consumer group.",
    )
    parser.add_argument(
        "--no-follow", action="store_true",
        help="Do not publish discovered links back to the frontier.",
    )
    parser.add_argument(
        "--worker-id", default=None,
        help="Label for this node, shown in the logs (default: hostname:pid).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Override CONCURRENT_REQUESTS for this node.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"

    logger = setup_logging(f"crawler-{worker_id}")
    logger.info("Starting crawler node %s", worker_id)

    settings = get_project_settings()
    if args.concurrency:
        settings.set("CONCURRENT_REQUESTS", args.concurrency)

    # install_root_handler=False: our own node-tagged handler is already on the
    # root logger, and letting Scrapy add a second one prints every line twice.
    process = CrawlerProcess(settings, install_root_handler=False)
    process.crawl(
        "frontier",
        job_id=args.job_id,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        allowed_sites=args.site,
        batch_size=args.batch_size,
        idle_rounds=args.idle_rounds,
        follow_links="false" if args.no_follow else "true",
    )
    process.start()   # blocks until the spider closes

    logger.info("Crawler node %s finished", worker_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
