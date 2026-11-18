"""Scrapy item pipelines: parse, then persist to PostgreSQL.

Items pass through the pipelines in the order given by ``ITEM_PIPELINES``:

1. :class:`ParsingPipeline`  - BeautifulSoup/lxml extraction
2. :class:`PostgresPipeline` - write the raw page, the parsed page and the
   discovered links, then acknowledge the RabbitMQ message

Splitting them this way keeps the parsing logic independent of the database:
the parser can be unit-tested on an HTML string, and re-parsing stored pages
later means running stage 1 again over rows that stage 2 already saved.
"""

from __future__ import annotations

import logging

from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem

from src.scraper.parser import parse_html
from src.storage.database import ScrapeRepository, init_schema

logger = logging.getLogger(__name__)


class ParsingPipeline:
    """Stage 1: turn raw HTML into structured content."""

    def process_item(self, item, spider):  # type: ignore[no-untyped-def]
        adapter = ItemAdapter(item)
        html = adapter.get("raw_html") or ""
        url = adapter["url"]

        if not html.strip():
            # Nothing to parse - ack the URL so it is not retried forever.
            spider.ack_url(url, success=True)
            raise DropItem(f"Empty response body for {url}")

        try:
            parsed = parse_html(html, url)
        except Exception as exc:  # noqa: BLE001 - one bad page must not stop the crawl
            logger.exception("Parsing failed for %s", url)
            spider.ack_url(url, success=False)
            raise DropItem(f"Parsing failed for {url}: {exc}") from exc

        adapter["parsed"] = parsed
        return item


class PostgresPipeline:
    """Stage 2: persist everything, then acknowledge the message."""

    def __init__(self) -> None:
        self.repository = ScrapeRepository()
        self.saved_pages = 0
        self.saved_links = 0

    def open_spider(self, spider):  # type: ignore[no-untyped-def]
        # Idempotent: CREATE TABLE IF NOT EXISTS for every table.
        init_schema()
        logger.info("PostgreSQL pipeline ready")

    def close_spider(self, spider):  # type: ignore[no-untyped-def]
        logger.info(
            "PostgreSQL pipeline closed - %s page(s) and %s link(s) written",
            self.saved_pages, self.saved_links,
        )

    def process_item(self, item, spider):  # type: ignore[no-untyped-def]
        adapter = ItemAdapter(item)
        url = adapter["url"]
        parsed = adapter["parsed"]

        try:
            raw_id, _is_new = self.repository.save_raw_page(
                job_id=adapter.get("job_id"),
                url=url,
                site=adapter["site"],
                depth=adapter.get("depth") or 0,
                http_status=adapter.get("http_status"),
                content_type=adapter.get("content_type"),
                headers=adapter.get("headers") or {},
                raw_html=adapter["raw_html"],
            )

            self.repository.save_parsed_page(
                raw_page_id=raw_id,
                url=url,
                site=parsed.site,
                title=parsed.title,
                description=parsed.description,
                language=parsed.language,
                headings=parsed.headings,
                clean_text=parsed.clean_text,
                structured_data=parsed.structured_data,
                parser_used=parsed.parser_used,
            )

            if parsed.links:
                written = self.repository.save_links(
                    [
                        {
                            "source_url": url,
                            "target_url": link["url"],
                            "site": parsed.site,
                            "anchor_text": (link["anchor_text"] or "")[:2000] or None,
                            "depth": (adapter.get("depth") or 0) + 1,
                            "enqueued": bool(link["internal"]),
                        }
                        for link in parsed.links
                    ]
                )
                self.saved_links += written

            if adapter.get("job_id"):
                self.repository.bump_job_counters(adapter["job_id"], scraped=1)

        except Exception as exc:  # noqa: BLE001
            # The write failed, so the URL must NOT be acknowledged: RabbitMQ
            # has to redeliver it. This is the core fault-tolerance guarantee.
            logger.exception("Database write failed for %s", url)
            spider.ack_url(url, success=False)
            raise DropItem(f"Database write failed for {url}: {exc}") from exc

        # Committed: now, and only now, the URL leaves the queue for good.
        spider.ack_url(url, success=True)
        self.saved_pages += 1
        logger.info(
            "Stored %s (status=%s, %s words, %s links)",
            url, adapter.get("http_status"), parsed.word_count, len(parsed.links),
        )
        return item
