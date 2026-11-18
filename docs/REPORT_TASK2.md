# Report Notes — Task 2: Web Scraping Module

Material to paste into the project report (.docx), plus the screenshot list.

---

## Section text for the report

### 1. Introduction to Phase 2

Phase 2 implements the data-acquisition half of the framework. The requirement
is not simply to download pages: it is to build a crawler whose workload can be
spread over several machines, whose list of URLs can grow while it is running,
and whose output is durable enough to be re-processed later by the RAG stage.
Four technologies were therefore combined, each handling one concern. Scrapy
performs the crawling and gives an asynchronous download engine for free.
BeautifulSoup and lxml turn the retrieved HTML into structured records.
RabbitMQ holds the URLs that still need to be visited, which decouples the
producers of work from the consumers of work and is what allows several
crawler processes to cooperate. PostgreSQL stores both the raw responses and
the parsed results. The phase ends with the pipeline demonstrated on two
structurally different websites.

### 2. The message-queuing layer: a dynamic URL frontier

The central design decision of this phase is that the crawler has **no**
`start_urls` list. A conventional Scrapy spider is given a fixed set of start
URLs at launch and terminates when that set is exhausted, which makes it a
single-process, fixed-workload program. In this framework the equivalent of
`start_urls` is a RabbitMQ queue named `urls.frontier`, and the spider pulls
from it. Two properties follow directly from that inversion, and together they
satisfy the first project objective.

First, the URL set becomes dynamic. The producer is a separate program,
`src/scraper/seed_frontier.py`, which publishes URL messages and can be run at
any moment, including while crawlers are already working. Adding new work to a
live crawl is therefore just publishing another message, from any process on
any machine. The crawlers themselves are also producers: whenever a page is
parsed, the in-scope links discovered on it are published back onto the same
queue, so the frontier grows as the crawl progresses and the crawl expands
outward without any central scheduler deciding what to visit next.

Second, the crawl becomes distributed. Any number of crawler processes may
consume from the queue simultaneously, on one machine or on many. They share no
state and never communicate with each other; RabbitMQ is the only coordination
point, and because each message is delivered to exactly one consumer, two
workers can never duplicate each other's effort. Work is balanced by setting
the consumer prefetch count to one (`basic_qos(prefetch_count=1)`), which tells
the broker to hand a worker a single unacknowledged URL at a time. A slow
worker therefore cannot accumulate a private backlog while other workers sit
idle; the next URL always goes to whichever worker is actually free. This is
the load-balancing mechanism required by the project objectives, obtained from
the broker rather than from application code.

The topology is declared in `src/messaging/frontier.py` and is idempotent, so
every process re-declares it on connect and the system can be started in any
order. A direct exchange `scraper.frontier` routes messages to the durable
queue `urls.frontier`. That queue names a dead-letter exchange, whose own queue
`urls.dead-letter` receives URLs that could not be processed. Each URL travels
as a small JSON document - the `UrlMessage` dataclass - carrying the URL, its
site, its depth from the seed, the crawl job it belongs to, a delivery-attempt
counter and the page it was discovered on, which is what makes the provenance
of every stored page reconstructible afterwards.

Several queue settings exist specifically for fault tolerance. The queues are
declared durable and the messages are published with `delivery_mode=Persistent`,
so the remaining work survives a broker restart instead of vanishing from
memory. Publisher confirms are enabled, meaning `basic_publish` does not return
until the broker has accepted the message, which removes the possibility of a
URL being lost in the gap between crawler and queue. Most importantly,
acknowledgement is **manual and late**: the message is acked only after its
page has been committed to PostgreSQL. If a worker is killed mid-fetch, its
unacknowledged URL is simply redelivered to another worker. The retry path uses
the attempt counter rather than an unbounded requeue loop: a failing URL is
republished with the counter incremented, and once it reaches
`MAX_DELIVERY_ATTEMPTS` (three) it is routed to the dead-letter queue, so one
permanently broken page cannot occupy the crawl forever.

### 3. The crawler: Scrapy driven by the queue

The spider is `FrontierSpider` in
`src/scraper/spiders/frontier_spider.py`. Making a Scrapy spider consume
continuously from an external queue requires some care, because Scrapy's
`start_requests` is a generator that is consumed once at startup. The solution
is the engine's `spider_idle` signal, which Scrapy fires whenever its scheduler
runs dry. The handler pulls the next batch of URLs from RabbitMQ, converts them
into `scrapy.Request` objects, hands them to the engine and raises
`DontCloseSpider` so the spider stays alive. Only after the frontier has been
found empty for several consecutive idle events does the handler allow the
spider to close. This is the same technique the well-known `scrapy-redis`
extension uses, implemented here against RabbitMQ.

Each request carries its originating queue message in `Request.meta`, and the
spider keeps a map from URL to RabbitMQ delivery tag. The storage pipeline
calls back into the spider's `ack_url` method once the database transaction has
committed, and only then is the tag acknowledged; on a database failure the
same method nacks the message so the broker redelivers it. The `errback`
handler covers the other failure mode - a request that never produced a
response at all - by republishing the URL with an incremented attempt counter
or dead-lettering it. When the spider shuts down, any URL still unacknowledged
is explicitly requeued, so even a clean shutdown in the middle of a crawl loses
nothing.

Scrapy's own settings (`src/scraper/settings.py`) supply the remaining
robustness and the politeness the target sites deserve. `ROBOTSTXT_OBEY` is
enabled. Retries are enabled for the transient HTTP failures (500, 502, 503,
504, 408 and 429) with two attempts, which sits *underneath* the queue-level
retry: a transient blip is absorbed inside the worker, and only a genuine
failure escalates to a redelivery. AutoThrottle is enabled, so the request rate
adapts to the latency the server is actually showing rather than being fixed in
advance - this matters more than usual here, because several worker processes
may be hitting the same site at once and each one only limits itself.

### 4. The parsing layer: BeautifulSoup and lxml

Parsing is implemented in `src/scraper/parser.py`, deliberately as a pure
function of an HTML string so that it can be unit-tested without a network, a
broker or a database. Both parsing technologies named in the task are used,
each where it is genuinely stronger.

BeautifulSoup, configured with the `lxml` tree builder, handles the work that
benefits from a forgiving document model. Real web pages contain unclosed tags
and invalid nesting, and BeautifulSoup tolerates them while the `lxml` backend
keeps the parsing itself in C and therefore fast. It is used to read the
metadata (title, with fallbacks to the Open Graph title and then the first
`h1`; the meta description; the document language), to collect the `h1`-`h3`
headings in document order without duplicates, to extract and normalise every
link, and to strip the boilerplate. Boilerplate removal decomposes the
`script`, `style`, `nav`, `header`, `footer`, `aside`, `form` and similar
elements before the text is taken, which is what makes the stored text useful
to the later RAG stage: without it, every chunk of every page would be
contaminated by the same navigation menu.

lxml is used directly, with XPath, for the per-site structured extraction. A
registry maps a hostname to an extractor function, and a site with no entry
still receives the full generic extraction. XPath is the right tool here
because these are precise, targeted queries against a known layout - for
example `//table[contains(@class,'table-striped')]//tr` to read a product
information table - where an expression is both clearer and faster than walking
the soup tree. Extractor failures are caught and logged rather than propagated,
on the principle that a layout change on one page must not stop the crawl.

URL normalisation deserves a mention because it directly affects how much work
the crawl wastes. Every discovered `href` is resolved against the page URL,
stripped of its fragment, lower-cased in the host part, relieved of a trailing
slash on non-root paths, and rejected outright if it is a `mailto:`, `tel:` or
`javascript:` link or points at a non-document extension such as `.pdf`,
`.png`, `.css` or `.zip`. Without this step the frontier fills with many
spellings of the same page and the crawler re-fetches it repeatedly.

### 5. Storage in PostgreSQL

The schema is defined with SQLAlchemy 2 in `src/storage/models.py` and holds
four tables. `crawl_jobs` records one row per crawl run with its seeds, status
and counters, so every stored page is traceable to the run that produced it.
`raw_pages` stores the HTTP response verbatim: status code, headers as JSONB,
the unmodified HTML body, its length, a SHA-256 content hash, the timestamp,
and - importantly for this project - `fetched_by`, the `hostname:pid` of the
worker that retrieved it. `parsed_pages` holds the output of the parsing stage,
with the per-site fields in a JSONB column so that sites with completely
different shapes can share one table. `discovered_links` records every link
with the page it was found on, giving an audit trail of how the frontier grew.

Keeping the raw HTML alongside the parsed result is a deliberate choice rather
than redundancy. The parsing rules will change as the project develops, and
re-parsing stored HTML is far cheaper - and far kinder to the target website -
than re-crawling it. The SHA-256 hash exists so that an unchanged page can be
recognised on a later crawl without comparing megabytes of markup.

Two details make the storage layer safe under the delivery semantics of the
broker. RabbitMQ guarantees *at-least-once* delivery, which means the same URL
can legitimately arrive twice - after a redelivery, or when two workers race.
Both writes therefore use PostgreSQL's `INSERT ... ON CONFLICT ... DO UPDATE`
against a unique constraint, so a duplicate updates the existing row instead of
creating a second one; the writes are idempotent, and the database is the
authority on de-duplication rather than any in-memory set. This is directly
observable in the demonstration run described below, where the crawl job
reports 32 fetches for `quotes.toscrape.com` while the table holds 29 distinct
pages: three redeliveries were absorbed by the upsert. Second, the engine is
created with `pool_pre_ping=True`, so a connection dropped by a restarted
database container is detected and replaced rather than raising on first use.

### 6. Demonstration on two websites

The pipeline was tested on two sites chosen because they are explicitly
published as scraping sandboxes and because their structures are different
enough to exercise the extraction layer properly.
`books.toscrape.com` is an e-commerce layout: product pages with a price, a
star rating, an availability string, a breadcrumb and a product-information
table, plus listing pages holding twenty product teasers each.
`quotes.toscrape.com` is a content layout: listing pages holding quote blocks
with text, author and tags, plus author detail pages with a biography. A
dedicated XPath extractor was written for each.

The complete run is automated by `scripts/demo_task2.py`, which resets the
database and the queues, crawls the first site with one worker, then crawls the
second site with **two workers running concurrently**, and prints a final
report. The results confirm every property the phase set out to demonstrate.
Sixteen pages were stored for `books.toscrape.com` and twenty-nine for
`quotes.toscrape.com`, each with a parsed counterpart, and 1,120 and 602
discovered links were recorded respectively. The report's work-distribution
table lists the pages stored per worker node and shows three distinct
`hostname:pid` identities, with the two concurrent workers on the second site
having stored sixteen and thirteen pages - that is, the load was genuinely
shared between two independent processes rather than being handled by one. The
queue depth printed at intervals during the run shows the frontier growing from
a single seed URL to several hundred entries as links were discovered and
published back, which is the dynamic URL management working in practice. The
dead-letter queue remained empty throughout, and the crawl was stopped by the
configured page limit rather than by any failure.

### 7. Testing

The test suite grew from 18 to 95 tests, all passing. The parsing tests run on
HTML strings and cover URL normalisation, boilerplate removal, metadata
extraction, malformed and empty documents, both per-site extractors, and the
requirement that a failing extractor is contained rather than fatal. The
frontier tests cover message serialisation (including tolerance of unknown
fields, so that a newer producer cannot break an older consumer) and then,
against a live broker, publish/pull, acknowledgement, redelivery after a nack,
dead-lettering after repeated failure, draining and purging. The storage tests
cover the schema, every repository method, and specifically that a duplicate
URL updates one row rather than inserting two. The integration tests detect
whether RabbitMQ and PostgreSQL are reachable and skip cleanly when they are
not, so the suite still passes on a machine where the containers are stopped.

### 8. Problems encountered

Three practical issues were resolved during this phase and are worth recording.

Scrapy 2.12 could not be imported after installation, failing on
`cannot import name '_safe_chars' from 'w3lib.url'` and then on
`cannot import name '_setAcceptableProtocols' from twisted.internet._sslverify`.
Both are private names that the newest `w3lib` and `Twisted` releases have
removed but that Scrapy 2.12 still imports. The fix was to pin the last
compatible versions, `w3lib==2.2.1` and `Twisted==24.11.0`, in
`requirements.txt` - a good illustration of why the direct dependencies are
pinned rather than left open.

PostgreSQL authentication failed against the container even though the
container itself was healthy, because port 5432 on the host was already bound
by a locally installed PostgreSQL 17 service, so the client was reaching the
wrong server. Rather than stopping a service unrelated to the project, the
container was republished on host port 5433 and the default was changed in the
configuration; inside Docker and Kubernetes the standard 5432 is still used.

Finally, the count of newly inserted links was always reported as -1. With
`ON CONFLICT DO NOTHING` the driver cannot report a row count, so the inserted
rows had to be counted explicitly with a `RETURNING` clause - which is also
what makes the "skipped duplicates" figure in the report meaningful.

### 9. Phase 2 outcome

The scraping module is complete and verified on two different websites. URLs
are managed dynamically through RabbitMQ, pages are fetched by any number of
cooperating Scrapy processes, the HTML is parsed with BeautifulSoup and lxml,
and both the raw and the parsed data are durably stored in PostgreSQL with
idempotent, fault-tolerant writes. The stored `parsed_pages.clean_text` column
is the input that Phase 3 will chunk, embed and index for retrieval-augmented
generation.

---

## Screenshot list

Eight figures, one per claim the phase has to support: the infrastructure, the
message queue as the URL frontier, dynamic URL management, the distributed
crawl, the extraction, the storage and the tests. Figure 4 is the single most
important one.

| # | What to capture | Command / where | Caption for the report |
|---|-----------------|-----------------|------------------------|
| 1 | The infrastructure started and healthy | `docker compose -f docker/docker-compose.yml up -d` ; `docker compose -f docker/docker-compose.yml ps` | *Figure 1 — The Phase 2 infrastructure: the RabbitMQ broker and the PostgreSQL database are created as containers from `docker/docker-compose.yml` and both report the `healthy` status from their container health checks.* |
| 2 | The two queues in the RabbitMQ management UI | <http://localhost:15672> (`scraper` / `scraper_pass`) → Queues | *Figure 2 — The declared queue topology: `urls.frontier`, the dynamic URL frontier the crawlers consume from, and `urls.dead-letter`, which receives URLs that failed repeatedly instead of being retried forever.* |
| 3 | Seeding the frontier, and the queue depth afterwards | `python -m src.scraper.seed_frontier --job-name books-crawl --url https://books.toscrape.com/` ; `python -m src.scraper.seed_frontier --status` | *Figure 3 — The producer publishes a seed URL onto the frontier and registers a crawl job. Seeding is a separate process from crawling, so URLs can be added at any time, including to a crawl that is already running.* |
| 4 | **Two crawler nodes running side by side** | two terminals: `python -m src.scraper.run_crawler --site quotes.toscrape.com --worker-id w1` and `... --worker-id w2` | *Figure 4 — Two independent crawler processes consuming from the same frontier. RabbitMQ hands each one different URLs, so the work is balanced between them without any coordination between the processes — the crawl is genuinely distributed.* |
| 5 | The frontier growing while the crawl runs | `python -m src.scraper.seed_frontier --status` in a third terminal, or the RabbitMQ queue graph | *Figure 5 — The frontier grows from one seed URL to hundreds of entries while the crawl is in progress, because every parsed page publishes its discovered links back onto the queue: dynamic URL management in action.* |
| 6 | The report: jobs, per-site totals and the work distribution | `python -m src.storage.report` (top and middle) | *Figure 6 — Both crawl jobs completed: 16 pages stored for `books.toscrape.com` and 29 for `quotes.toscrape.com`. The per-node table shows several distinct `hostname:pid` identities, direct evidence that the pages were fetched in parallel by independent worker processes.* |
| 7 | The structured extraction stored for both sites | `python -m src.storage.report --samples 2` (bottom), or `jsonb_pretty(structured_data)` in `psql` | *Figure 7 — Per-site fields produced by the lxml XPath extractors and stored as JSONB: price, rating and availability for the e-commerce site, quote texts, authors and tags for the content site — two structurally different websites sharing one table.* |
| 8 | The Phase 2 test suite | `python -m pytest tests/test_parser.py tests/test_frontier.py tests/test_storage.py -q` | *Figure 8 — The Phase 2 test suite: 71 tests passed, covering URL normalisation, the parsers for both sites, queue semantics (redelivery and dead-lettering) and the idempotency of the database writes.* |

> Tips for good screenshots:
> - Figure 4 proves the crawler is distributed. Put two terminals side by side,
>   both running `run_crawler` against the same site, and make sure the
>   different `--worker-id` values are visible in the log lines.
> - For a `psql` figure, run `docker exec -it rag-postgres psql -U scraper -d
>   scraped_data` once and then type the query, so the prompt is visible.
> - If the `£` sign in a price looks like `?` in your terminal, run
>   `$env:PYTHONIOENCODING="utf-8"` first — the data in the database is
>   correct, it is only the console's code page.

---

## Flowchart — Phase 2

Node sequence for the Phase 2 branch of the overall framework flowchart. The
two shaded regions are separate processes, which is the point of the diagram.

```
                          ┌─────────────────────────┐
  Start Phase 2  ───────► │  Start infrastructure   │
                          │  (RabbitMQ + Postgres)  │
                          └───────────┬─────────────┘
                                      ▼
                          ┌─────────────────────────┐
                          │  Create PostgreSQL      │
                          │  schema (4 tables)      │
                          └───────────┬─────────────┘
                                      ▼
   ══ PRODUCER PROCESS ══  ┌─────────────────────────┐
                          │  Seed frontier:         │
                          │  publish seed URL(s)    │
                          └───────────┬─────────────┘
                                      ▼
                          ╔═════════════════════════╗
                          ║  RabbitMQ queue         ║
                          ║  "urls.frontier"        ║◄────────────┐
                          ╚═══════════╤═════════════╝             │
                                      │ prefetch = 1              │
                                      │ (one URL per free worker) │
                                      ▼                           │
   ══ WORKER PROCESS ══    ┌─────────────────────────┐            │
   (N of these run         │  Spider pulls URL       │            │
    in parallel)           │  (spider_idle signal)   │            │
                          └───────────┬─────────────┘             │
                                      ▼                           │
                             ◇ already crawled? ──yes──► ack, skip │
                                      │ no                        │
                                      ▼                           │
                          ┌─────────────────────────┐             │
                          │  Scrapy fetches page    │             │
                          └───────────┬─────────────┘             │
                                      ▼                           │
                             ◇ fetch succeeded? ──no──► ◇ attempts │
                                      │ yes                 ≥ 3 ? │
                                      │                   yes│ no │
                                      │          ┌───────────┘    │
                                      │          ▼      republish ┘
                                      │   ╔═════════════════╗
                                      │   ║ "urls.dead-     ║
                                      │   ║  letter" queue  ║
                                      │   ╚═════════════════╝
                                      ▼
                          ┌─────────────────────────┐
                          │  ParsingPipeline        │
                          │  BeautifulSoup + lxml:  │
                          │  - strip boilerplate    │
                          │  - metadata, headings   │
                          │  - clean text           │
                          │  - normalise links      │
                          │  - XPath site fields    │
                          └───────────┬─────────────┘
                                      ▼
                          ┌─────────────────────────┐
                          │  PostgresPipeline       │
                          │  upsert raw_pages,      │
                          │  parsed_pages, links    │
                          └───────────┬─────────────┘
                                      ▼
                             ◇ COMMIT succeeded?
                             │                  │
                          no │                  │ yes
                             ▼                  ▼
                      nack → redeliver    ack the URL
                                                │
                                                ▼
                                     ◇ depth < max_depth ?
                                     │                  │
                                 yes │                  │ no
                                     ▼                  ▼
                        publish discovered         (stop expanding)
                        links to frontier ─────────────┘
                              (loop back to the queue)
                                      │
                                      ▼
                             ◇ frontier empty for
                               N idle rounds ?
                                 │          │
                              no │          │ yes
                                 └──loop────┘
                                            ▼
                          ┌─────────────────────────┐
                          │  Worker exits; mark     │
                          │  crawl job completed    │
                          └───────────┬─────────────┘
                                      ▼
                          ┌─────────────────────────┐
                          │  Report from PostgreSQL │
                          │  (per-site + per-node)  │
                          └───────────┬─────────────┘
                                      ▼
                        Phase 3 (RAG: chunk → embed → index)
```

Decision diamonds to draw: *already crawled?*, *fetch succeeded?*,
*attempts ≥ 3?*, *COMMIT succeeded?*, *depth < max_depth?* and *frontier empty
for N rounds?*. The two loops that matter visually are the **link-discovery
loop** (parsed page → discovered links → back into the queue), which is what
makes the frontier dynamic, and the **redelivery loop** (failed commit → nack →
queue), which is what makes it fault tolerant.

---

## Key numbers to quote in the report

| Metric | Value |
|--------|-------|
| Websites tested | 2 (`books.toscrape.com`, `quotes.toscrape.com`) |
| Pages stored — site 1 | 16 raw / 16 parsed |
| Pages stored — site 2 | 29 raw / 29 parsed (from 32 fetches — 3 duplicate deliveries absorbed by the upsert) |
| Links discovered | 1,120 + 602 = 1,722 |
| Concurrent worker nodes | 2 on site 2 (16 and 13 pages each) |
| Distinct worker identities recorded | 3 |
| Dead-lettered URLs | 0 |
| Database tables | 4 |
| Tests passing | 95 |
