"""Scrapy items carried from the spider to the pipelines."""

from __future__ import annotations

import scrapy


class RawPageItem(scrapy.Item):
    """One fetched page, plus everything the pipelines need to persist it."""

    # --- identity ---
    url = scrapy.Field()
    site = scrapy.Field()
    depth = scrapy.Field()
    job_id = scrapy.Field()

    # --- the raw response, stored verbatim ---
    http_status = scrapy.Field()
    content_type = scrapy.Field()
    headers = scrapy.Field()
    raw_html = scrapy.Field()

    # --- provenance ---
    discovered_from = scrapy.Field()
    fetched_at = scrapy.Field()

    # --- filled in by the parsing pipeline ---
    parsed = scrapy.Field()
