"""Scrapy settings for the distributed crawler."""

from __future__ import annotations

from src.common.logging_config import quiet_noisy_loggers

# Scrapy imports this module while it builds the crawler, which is the
# earliest point at which the chatty pika/urllib3 loggers can be raised to
# WARNING so that the crawl log stays readable.
quiet_noisy_loggers()

BOT_NAME = "rag_scraper"

SPIDER_MODULES = ["src.scraper.spiders"]
NEWSPIDER_MODULE = "src.scraper.spiders"

# --- politeness -----------------------------------------------------------
# A distributed crawler can easily overwhelm a small site, so the limits are
# deliberately conservative: they apply per spider process, and several
# processes may run at once.
ROBOTSTXT_OBEY = True
USER_AGENT = "DistributedRAGScraper/1.0 (+academic project; contact via repository)"

CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 4
DOWNLOAD_DELAY = 0.25

# AutoThrottle adapts the delay to the latency the server actually shows,
# which protects the target site better than a fixed delay.
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 10.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0

# --- fault tolerance ------------------------------------------------------
DOWNLOAD_TIMEOUT = 20
RETRY_ENABLED = True
RETRY_TIMES = 2
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# --- caching / compression ------------------------------------------------
HTTPCACHE_ENABLED = False
COMPRESSION_ENABLED = True

# --- pipelines ------------------------------------------------------------
ITEM_PIPELINES = {
    "src.scraper.pipelines.ParsingPipeline": 100,
    "src.scraper.pipelines.PostgresPipeline": 200,
}

# --- logging --------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | scrapy | %(name)s | %(message)s"

# --- misc -----------------------------------------------------------------
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
TELNETCONSOLE_ENABLED = False
