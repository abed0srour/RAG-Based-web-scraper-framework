# Report Notes — Task 3: Data Processing Module

Material to paste into the project report (.docx), plus the screenshot list.

All figures quoted below come from the run recorded in
`data/benchmarks/processing-benchmark-*.json` on a Windows 11 machine,
Python 3.12.10, MongoDB 7.0.48, over the 45 pages crawled in Phase 2.

---

## Section text for the report

### 1. Introduction to Phase 3

Phase 2 ended with the raw HTTP responses of two websites stored verbatim in
PostgreSQL. That data is complete but unusable: measured over the 45 crawled
pages, 94.7% of the stored bytes are markup, inline JavaScript, CSS,
navigation menus and tracking fragments rather than content. Phase 3 is the
stage that turns those responses into data a retrieval system can work with.
It has three responsibilities, and the module is organised as three
independently testable stages that match them exactly: cleaning the raw text,
normalising it into a structured JSON representation, and indexing the result
in a NoSQL database. A fourth concern, measuring how efficiently all of this
runs over different kinds of input, is treated as part of the deliverable
rather than as an afterthought, because the processing stage is the one that
will have to keep up with a growing pool of crawler nodes in Phase 4.

The module lives in `src/processing/` and consists of five files: `cleaner.py`
(stage 1), `normalizer.py` (stage 2), `store.py` (stage 3), `pipeline.py`
(the orchestrator) and `benchmark.py` (the efficiency test), plus `report.py`,
a command-line report over the processed corpus.

### 2. Stage 1 — Removing tags, scripts and formatting

The cleaning stage takes a raw HTML document and returns plain text. It is
implemented in `src/processing/cleaner.py` with BeautifulSoup over the lxml
tree builder, and it works in four passes, each removing a different class of
noise. The order of the passes matters and is the main design decision of the
stage.

The first pass removes elements whose textual content is code or decoration
rather than prose: `script`, `style`, `noscript`, `template`, `svg`, `iframe`,
`object`, `embed` and the media elements, together with every HTML comment.
These are decomposed *with their contents*, and they are handled first,
because a text extractor run before them would happily treat JavaScript source
and CSS rules as page text. On the home page of `books.toscrape.com` this pass
alone removes 7 `script` elements and 14 comments.

The second pass removes the structural boilerplate: `nav`, `header`, `footer`,
`aside`, `form`, `button` and the form controls. These elements do contain
real words, but the same words on every page of the site, which is precisely
the content a retrieval index must not be filled with — a query would
otherwise match the cookie banner of a thousand pages before it matched
anything a user asked about. The pass is optional
(`clean_html(..., remove_boilerplate=False)`) because a page whose entire
content sits inside a `<header>` would otherwise be emptied; the benchmark
uses that switch to show what the pass costs and what it saves.

The third pass is character-level normalisation, and it is the pass that is
easiest to overlook. HTML entities are unescaped (`&pound;` becomes `£`), the
text is converted to Unicode NFKC form, and invisible codepoints — zero-width
spaces, soft hyphens, directional marks, byte-order marks — and C0 control
characters are deleted. Without this step the same word exists in several byte
representations, and an index lookup that should match simply does not. After
unescaping, the tag-stripping regular expression is applied a second time,
because markup that was *double-encoded* inside a text node only becomes
visible once the entities have been decoded; the test
`test_double_encoded_markup_is_stripped` covers exactly that case.

The fourth pass normalises layout. A newline is forced into every block-level
element before the text is taken, so that two adjacent paragraphs are not
glued into one sentence; runs of spaces and tabs are collapsed; runs of blank
lines are capped at one; and finally the lines are filtered — lines shorter
than two characters, lines matching a list of known navigation labels
("Home", "Cart", "Accept all cookies", "All rights reserved", …) and lines
identical to the line immediately before them are dropped. That last rule
catches the very common case of a menu rendered twice, once in a mobile
container and once in a desktop one.

Everything in this stage is a pure function of a string: no network, no
database. That is what makes it cheap to unit-test (22 of the 68 tests of this
phase test it on HTML literals) and cheap to re-run over the stored HTML when
the rules change — which is the reason Phase 2 kept the raw responses in the
first place.

The stage also *measures itself*. `clean_html` returns a `CleanedText` object
carrying the input size, the output size, the number of elements removed per
tag, the number of comments removed, the number of lines dropped and the
elapsed time. Those measurements are written into every document, so the
effect of the cleaning is visible per page rather than only in aggregate.

### 3. Stage 2 — Normalising into structured JSON

Cleaning produces text; normalisation produces a *document*. The second
stage, `src/processing/normalizer.py`, builds one JSON object per page in
which every key and every type is the same whatever website the page came
from:

```json
{
  "doc_id": "3060d727707e12334c7134afe04df480af3f0fdf",
  "schema_version": "1.0",
  "url": "https://books.toscrape.com/",
  "site": "books.toscrape.com",
  "path": "/",
  "title": "All products | Books to Scrape - Sandbox",
  "description": null,
  "language": "en-us",
  "headings": ["All products", "A Light in the ...", "..."],
  "text": "All products | Books to Scrape - Sandbox\nAll products\n...",
  "keywords": ["stock", "products", "books", "scrape", "..."],
  "fields": { "price": "£51.77", "rating": "Three", "product_count": 20 },
  "content_hash": "1f0a...",
  "stats": { "word_count": 157, "char_count": 953, "lexical_diversity": 0.59 },
  "source": { "raw_page_id": 1, "http_status": 200, "fetched_by": "host:1234" },
  "processing": { "cleaner_version": "1.0", "reduction_ratio": 0.9814 }
}
```

Three decisions define the schema.

**The identifier is derived, not generated.** `doc_id` is a SHA-1 hash over
the *canonical* form of the URL, where the scheme and host are lower-cased,
the fragment is removed and a trailing slash is stripped from non-root paths.
Processing the same page twice therefore produces the same key, and the write
into MongoDB is an idempotent replacement rather than a second copy. This is
not a cosmetic property: the URL frontier of Phase 2 delivers messages
at-least-once, the pipeline may be killed half-way and restarted, and both
situations converge to the same corpus because of this one rule. The
demonstration proves it by processing all 45 pages twice — the second run
reports 0 inserted, 45 updated, and the collection still holds 45 documents.

**Site-specific data is quarantined under one key.** Everything that differs
per website — a price, a star rating, a quote list, an author's biography —
lives inside `fields`, which is the only part of the document whose shape
varies. The top level is identical for every site. MongoDB is schema-less and
would happily accept per-site top-level keys, which is exactly why the rule
has to be stated explicitly: without it, the per-site knowledge leaks into
every consumer of the data, and the RAG layer of the next phase would need a
branch per website.

**The content hash is computed over the clean text, not over the HTML.** A
page whose markup, advertisements or session identifiers changed but whose
content did not produces the same hash, so it can be recognised as unchanged
and skipped by a future incremental run. Over the crawled corpus the hash also
immediately exposed a genuine duplicate: `https://books.toscrape.com/` and
`https://books.toscrape.com/index.html` are different URLs serving byte-identical
content, and the report groups them.

Besides the text, the stage derives the measurements that make documents
comparable — word, character, line and sentence counts, the number of unique
words, lexical diversity, average word length and an estimated reading time —
and a keyword list, which is the fifteen most frequent words of the page after
a stop-word list has been applied. The keywords are a cheap summary used for
faceting and, more importantly, as a boosted field in the full-text index; a
full NLP pipeline is deliberately not introduced here.

Finally, every document is validated before it leaves the stage.
`validate_document` checks that each of the twelve required keys is present
and has the right type, that the schema version matches, and that the whole
object survives `json.dumps` — a `set` or a `datetime` that slipped into a
site-specific field is caught here, at the boundary, instead of failing a
reader days later. Failures raise `NormalizationError` listing every problem
at once.

### 4. Stage 3 — Storing the documents in a NoSQL database

The normalised documents are written to **MongoDB 7**, which runs as the
third container of the project's `docker-compose.yml`, alongside RabbitMQ and
PostgreSQL. Using a second database rather than another PostgreSQL table is a
deliberate split of responsibilities:

| | PostgreSQL (Phase 2) | MongoDB (Phase 3) |
|---|---|---|
| Holds | the crawl record: raw responses, job bookkeeping, the link graph | the processed documents |
| Shape | fixed columns, foreign keys, a relational link table | one self-contained JSON object per page, `fields` differing per site |
| Access pattern | joins and aggregates over rows | fetch or search whole documents |
| Written by | the crawler nodes | the processing pipeline |

What MongoDB actually contributes here is **indexing**, and five indexes are
created idempotently by `DocumentStore.ensure_indexes()`:

| Index | Purpose |
|-------|---------|
| `uq_doc_id` (unique) | enforces one document per canonical URL — the database-level half of the idempotency guarantee |
| `ix_fulltext` (text, weighted) | keyword retrieval over title, description, headings, keywords and body |
| `ix_site_processed_at` | per-site listings in recency order, the most common read |
| `ix_content_hash` | finds pages with identical text — near-duplicate detection before embedding |
| `ix_word_count` | range queries over document size, used to build benchmark samples |

The text index is the one that matters for the next phase. It is a single
compound index over five fields with weights — title 10, description 5,
headings 4, keywords 3, body 1 — so that a page *about* a subject outranks a
page that merely mentions it. MongoDB allows only one text index per
collection, which is why all searchable fields are combined into this one. The
effect is visible in the demonstration: a query for `fiction` returns the
*Fiction* category page with a relevance score of 14.71 and an author page
that mentions the word in passing with 0.50, and the query is answered by the
database rather than by scanning documents in Python.

Building that index exposed a problem worth recording. MongoDB's text index
reads the stemming language from a field literally named `language`, and in
this schema `language` holds the HTML `lang` attribute of the page. Every page
of `books.toscrape.com` declares `lang="en-us"`, which is not a language
MongoDB knows, so the first full run rejected all 16 of them with
*"language override unsupported: en-us"* while the 29 pages of the other site
were written. The fix is to create the index with
`language_override="_text_language"`, pointing it at a field the application
never sets: the documents keep their language metadata, and every document is
stemmed as English. The regression test
`test_language_tag_does_not_break_the_text_index` pins the behaviour.

Writes go through `upsert_many`, which issues one `bulk_write` of `ReplaceOne`
operations with `ordered=False`, so that a single rejected document does not
discard the rest of its batch; the partial-failure path reports how many
documents were written and how many were not, instead of raising. Reads are
exposed as `get`, `get_by_url` (which canonicalises the URL first, so any
variant of it finds the document), `search`, `duplicates` and `stats`, the
last two implemented as aggregation pipelines.

### 5. The pipeline

`src/processing/pipeline.py` is the orchestrator that connects the three
stages, and it is also the module that will be wrapped as a Celery task in
Phase 4:

```
raw_pages (PostgreSQL) → clean_html → normalize_page → validate → MongoDB
```

It streams. Rows are read with a server-side cursor (`yield_per`) and joined
to their `parsed_pages` row so that the per-site fields extracted during the
crawl are carried into the document rather than re-derived, and documents are
accumulated into batches of `PROCESSING_BATCH_SIZE` (50 by default) before
being written. The memory it needs is therefore proportional to the batch
size, not to the size of the corpus — which matters because a single raw page
in this corpus can be 52 KB and a real crawl has millions of them.

It isolates failures. One page that cannot be processed is counted, its URL is
recorded, and the run continues; a page that yields fewer than
`PROCESSING_MIN_WORDS` clean words is reported as *skipped* rather than as a
failure, because an empty page is a fact about the crawl and not a defect of
the processing.

It is restartable, for the reasons given in section 3, and `--since-id` lets
it process only pages crawled after a given row when a full rebuild is not
wanted. `--dry-run` runs both CPU stages and writes nothing, which is how the
cost of the processing can be separated from the cost of the database.

And it measures itself: every stage is timed per page, and `ProcessingStats`
reports the throughput together with the mean, p50, p95 and maximum latency of
each stage. Percentiles rather than averages, because the distribution is
strongly skewed — in this corpus the mean cleaning time is 11 ms, the median
is 4.5 ms and the slowest page takes 50 ms.

### 6. Testing processing efficiency on different data samples

`src/processing/benchmark.py` measures the module over samples that differ
along the three axes that actually change the cost of processing a page. The
pages are loaded into memory before timing starts, so PostgreSQL read time is
excluded; `--repeat` runs the CPU stages several times over the same pages and
divides the accumulated timings by the number of repetitions, which keeps the
numbers stable on a machine that is also doing other things.

**Axis 1 — the source website.** The two sites are structurally different and
the measurements separate cleanly:

| Sample | Pages | Avg page | Clean p50 | Normalise p50 | Pages/s | Noise removed |
|--------|------:|---------:|----------:|--------------:|--------:|--------------:|
| `quotes.toscrape.com` | 29 | 5.4 KB | 3.24 ms | 0.34 ms | 188.3 | 78.8% |
| `books.toscrape.com` | 16 | 45.5 KB | 21.82 ms | 0.60 ms | 43.2 | 98.1% |

The e-commerce site serves pages eight times larger that yield *less* text:
98.1% of a `books.toscrape.com` page is markup, because its content is a
dense product grid wrapped in table and form elements. The content site is
mostly prose, so only 78.8% is noise. Cost tracks the markup, value tracks the
text — which is the quantitative argument for keeping the cleaning stage as
the first thing that touches a page.

**Axis 2 — the document size.** Running the same code over three size buckets
shows how the per-page cost scales:

| Sample | Pages | Avg page | Clean p50 | Pages/s | MB/s | Noise removed |
|--------|------:|---------:|----------:|--------:|-----:|--------------:|
| small (< 10 KB) | 23 | 3.7 KB | 3.97 ms | 198.9 | 0.72 | 77.6% |
| medium (10–40 KB) | 10 | 21.2 KB | 9.50 ms | 78.0 | 1.62 | 92.3% |
| large (≥ 40 KB) | 12 | 48.9 KB | 22.90 ms | 37.1 | 1.77 | 98.1% |

A 13× increase in page size produces a 5.8× increase in cleaning time, so the
stage is *sub-linear* in the payload: the per-page constants (building the
tree, the statistics, the hashing) dominate for small pages, while throughput
in MB/s improves with size. There is no size at which the parser degrades,
which is the property the test was looking for.

**Axis 3 — the write batch size.** The same 45 documents were written four
times, varying only how many go into one round trip:

| Batch size | Round trips | Total | Per document | Documents/s |
|-----------:|------------:|------:|-------------:|------------:|
| 1 | 45 | 63.0 ms | 1.401 ms | 714 |
| 10 | 5 | 22.3 ms | 0.497 ms | 2,018 |
| 50 | 1 | 16.8 ms | 0.373 ms | 2,679 |
| 200 | 1 | 17.8 ms | 0.395 ms | 2,528 |

Batching 50 documents per write is 3.8× faster per document than writing them
one at a time, and beyond one round trip there is nothing left to gain — which
is what justifies the default `PROCESSING_BATCH_SIZE=50` and shows that the
store stage is dominated by the round trip, not by the server's work.

**Overall.** Across the whole corpus the module processes 71.6 pages/s
(1.37 MB/s of raw HTML) on one core: 12.89 ms of cleaning plus 0.68 ms of
normalisation per page, and 0.40 ms to store it. Cleaning is 95% of the cost,
which is the stage to optimise if the need arises, and the work is CPU-bound
and per-page independent — in other words exactly the shape of workload that
Phase 4 can distribute over several Celery workers.

### 7. Testing

`tests/test_processing.py` adds 68 tests, bringing the project suite to 163,
all passing. They are organised by stage:

* **Cleaning (22 tests)** — script, style and comment contents are gone; no
  `<` or `>` survives; entities are decoded; `&nbsp;` and invisible codepoints
  are removed; NFKC folding works; boilerplate is dropped and can be kept;
  paragraphs are not glued together; malformed and double-encoded HTML are
  handled; empty and whitespace-only input are handled; the measurements are
  reported.
* **Normalisation (26 tests)** — four URL variants canonicalise to one
  identifier while a query string stays significant; the document identifier is
  deterministic; headings are de-duplicated; site-specific fields are cleaned
  recursively and kept under `fields`; statistics and provenance are correct;
  the document is JSON-serialisable; the validator rejects a missing key, a
  wrong type and an unserialisable value.
* **The store (15 tests)** — the five indexes are created and `ensure_indexes`
  is idempotent; an upsert of the same URL updates rather than duplicates; the
  text index ranks a title match above a body match; search can be restricted
  to one site; `en-us` pages are indexable; duplicate detection groups
  identical text; the aggregates are correct.
* **The pipeline (5 tests)** — a document built from a stored page passes
  validation; a text-free page is skipped rather than failed; a run removes
  noise and reports throughput; a second run inserts nothing and adds no
  documents; a dry run writes nothing.

The store and pipeline tests use their own collections (`pytest_documents`,
`pytest_pipeline`, `pytest_dry_run`) and drop them afterwards, so they never
touch the real corpus, and they skip themselves automatically when MongoDB or
PostgreSQL is not running.

### 8. Problems encountered

**MongoDB's language override rejected two thirds of the corpus.** Described
in section 4. Diagnosed from the `writeErrors` of the bulk write — the
partial-failure path reported "16 document(s) failed" while 29 succeeded,
which pointed straight at the site whose pages declare `lang="en-us"`.

**Port 27017 was already in use.** The development machine has a locally
installed MongoDB 8 serving unrelated databases, so the container's attempt to
authenticate as `scraper` failed against *the wrong server*: the ping went to
the local instance, which has no such user. The container is now published on
host port **27018**, exactly as PostgreSQL was published on 5433 in Phase 2
for the same reason, and `MONGO_PORT` defaults to it. Inside Docker and
Kubernetes the standard port is used.

**Text was being glued across block elements.** The first version of the
cleaner took `soup.get_text(separator="\n")` directly and produced
`…£9.99It is in stock…`: BeautifulSoup inserts the separator between *nodes*,
not between *blocks*, so two adjacent paragraphs with no whitespace in the
markup ran together. A newline is now appended inside every block-level
element before the text is taken.

**Double-encoded markup survived the first strip.** A page that contains
`&lt;script&gt;…&lt;/script&gt;` as *text* yields real tags once entities are
unescaped. Stripping tags before unescaping therefore misses them, and
unescaping before stripping promotes them into the document. The stage now
strips, unescapes and strips again, and removes script and style blocks with
their contents in the process.

**The benchmark's `--repeat` inflated its own throughput figures.** With
`--repeat 5` the accumulated CPU time covered five passes while the page count
covered one, so the reported pages/s was five times too low and contradicted
the pipeline's own measurement. The CPU total is now divided by the number of
repetitions, and the two measurements agree (71.6 vs 82 pages/s, the
difference being the pipeline's overlapped database reads).

### 9. Phase 3 outcome

The framework now has a complete path from a URL to an indexed document: the
frontier hands a URL to a crawler node, the page is stored verbatim in
PostgreSQL, the processing module strips 94.7% of it away as noise, normalises
what remains into one JSON schema shared by both websites, and indexes it in
MongoDB with a weighted full-text index. The corpus is 45 documents, 8,169
words, reproducible with one command (`python scripts/demo_task3.py`) and safe
to rebuild at any time because every write is idempotent. The next phase can
take the documents as its input without knowing anything about HTML or about
which site a page came from.

---

## Screenshot list

| # | What to capture | Command / where | Caption for the report |
|---|-----------------|-----------------|------------------------|
| 1 | MongoDB added to the infrastructure | `docker compose -f docker/docker-compose.yml up -d mongodb mongo-express` | *Figure 1 — The Phase 3 infrastructure: a MongoDB container is added to the broker and the relational database, published on host port 27018 to avoid the locally installed MongoDB.* |
| 2 | All containers healthy | `docker compose -f docker/docker-compose.yml ps` | *Figure 2 — RabbitMQ, PostgreSQL and MongoDB all reporting the `healthy` status from their container health checks.* |
| 3 | The MongoDB service definition | `docker/docker-compose.yml` in VS Code | *Figure 3 — The MongoDB service, its durable volume, its health check and the comment explaining the port choice.* |
| 4 | `src/processing/cleaner.py` open at the four passes | VS Code | *Figure 4 — Stage 1: the cleaning passes, with the tag lists for executable elements and for layout boilerplate.* |
| 5 | **Raw HTML next to the clean text for one page** | `python scripts/demo_task3.py --skip-benchmark` (step 1) | *Figure 5 — The effect of the cleaning stage on one real page: 51,274 bytes of raw HTML reduced to 953 characters of text, with 7 `script` elements, 14 comments and the header, footer, aside and 21 form elements removed — 98.1% of the payload was noise.* |
| 6 | `src/processing/normalizer.py` open at the schema | VS Code | *Figure 6 — Stage 2: the canonical document, the derived identifier and the required-field validator.* |
| 7 | **The complete normalised JSON document** | `python scripts/demo_task3.py --skip-benchmark` (step 2) | *Figure 7 — The structured JSON produced for that page: identity, metadata, clean text, keywords, statistics, provenance, and the site-specific fields quarantined under `fields`.* |
| 8 | The same document in MongoDB | `python -m src.processing.report --doc-url https://books.toscrape.com/` | *Figure 8 — The document as it is stored and read back from the NoSQL database.* |
| 9 | `src/processing/store.py` open at `ensure_indexes` | VS Code | *Figure 9 — Stage 3: the five indexes, including the weighted compound text index and the `language_override` that keeps `lang="en-us"` pages indexable.* |
| 10 | A full processing run | `python -m src.processing.pipeline --reset` | *Figure 10 — The pipeline cleaning, normalising and indexing all 45 crawled pages in 0.6 s: 94.7% of 905 KB removed as noise, 8,169 words extracted, with the per-stage p50/p95 latencies.* |
| 11 | The indexes listed from the database | `python -m src.processing.report` (top) | *Figure 11 — The five indexes created in MongoDB, with the unique constraint on `doc_id`.* |
| 12 | Documents per site | `python -m src.processing.report` (middle) | *Figure 12 — The processed corpus: 16 documents for `books.toscrape.com` with 98.2% of their bytes removed, 29 for `quotes.toscrape.com` with 79.1% — the cost of markup differs sharply between the two site structures.* |
| 13 | Duplicate detection | `python -m src.processing.report` (middle) | *Figure 13 — The content hash identifies `/` and `/index.html` as serving byte-identical content, a near-duplicate found without comparing documents.* |
| 14 | **A full-text query answered by the index** | `python -m src.processing.report --search "fiction"` | *Figure 14 — Keyword retrieval through the weighted text index: the Fiction category page scores 14.71 while a page that merely mentions the word scores 0.50.* |
| 15 | A sample document summary | `python -m src.processing.report --samples 2` (bottom) | *Figure 15 — Sample documents from both sites, showing the identical top-level schema and the different `fields` keys each site contributes.* |
| 16 | **Idempotency of the write path** | `python scripts/demo_task3.py --skip-benchmark` (step 4) | *Figure 16 — Processing the same 45 pages a second time inserts 0 new documents and updates 45; the collection still holds 45. The document key is derived from the canonical URL, so the stage is safe to retry.* |
| 17 | **The efficiency benchmark** | `python -m src.processing.benchmark --repeat 5` | *Figure 17 — Processing efficiency over different data samples: per site, per document size and per write batch size, with the per-stage latencies and the share of each sample that was noise.* |
| 18 | The batch-size table on its own | same command, lower half | *Figure 18 — The effect of batching MongoDB writes: 1.401 ms per document one at a time against 0.373 ms in batches of 50, a 3.8× improvement that justifies the default batch size.* |
| 19 | The saved benchmark report | `python -m src.processing.benchmark --save` then open `data/benchmarks/processing-benchmark-*.json` | *Figure 19 — The machine-readable benchmark report, kept with the run's environment so the numbers in this section are reproducible.* |
| 20 | The documents in Mongo Express | <http://localhost:8081> → `processed_data` → `documents` | *Figure 20 — The indexed collection browsed in Mongo Express, showing the stored JSON documents.* |
| 21 | The collection from the Mongo shell | `docker exec -it rag-mongodb mongosh -u scraper -p scraper_pass --authenticationDatabase admin --eval "db.getSiblingDB('processed_data').documents.countDocuments()"` | *Figure 21 — The document count read directly from the database server.* |
| 22 | An index-backed query plan | `docker exec -it rag-mongodb mongosh -u scraper -p scraper_pass --authenticationDatabase admin --eval "db.getSiblingDB('processed_data').documents.find({\$text:{\$search:'fiction'}}).explain('queryPlanner').queryPlanner.winningPlan"` | *Figure 22 — The query planner confirming that the keyword search is served by the `ix_fulltext` index rather than by a collection scan.* |
| 23 | The processing test module | `python -m pytest tests/test_processing.py -v` | *Figure 23 — The 68 tests of the data-processing module: cleaning, normalisation, the document store and the pipeline.* |
| 24 | The whole suite | `python -m pytest -q` | *Figure 24 — The full project test suite after Phase 3: 163 tests passed.* |
| 25 | The complete demonstration | `python scripts/demo_task3.py` | *Figure 25 — The single command that reproduces the whole phase: cleaning, normalisation, indexing, the idempotency proof, the corpus report and the efficiency benchmark.* |
| 26 | The commit and push | `git log --oneline` and `git push` output | *Figure 26 — The Phase 3 implementation committed and pushed to the GitHub repository.* |

> Tips for good screenshots:
> - Figures 5, 7, 14, 16 and 17 are the ones that demonstrate the four
>   requirements of the task (cleaning, JSON normalisation, NoSQL indexing,
>   efficiency testing). Capture those first.
> - If a `£` sign appears as `?` in the terminal, run
>   `$env:PYTHONIOENCODING="utf-8"` before the command — the data in MongoDB
>   is correct Unicode, it is only the console's code page.
> - Run `python scripts/demo_task3.py --skip-benchmark > docs/task3-run.txt`
>   once to keep the full transcript, then screenshot the sections from it if
>   the terminal scrolls too fast.

---

## Flowchart — Phase 3

Node sequence for the Phase 3 branch of the overall framework flowchart. The
dashed box is the part that repeats per page.

```
                          ┌──────────────────────────┐
  Start Phase 3  ───────► │  Start MongoDB container │
                          └────────────┬─────────────┘
                                       ▼
                          ┌──────────────────────────┐
                          │  ensure_indexes()        │
                          │  5 indexes (unique id,   │
                          │  text, site, hash, size) │
                          └────────────┬─────────────┘
                                       ▼
                          ┌──────────────────────────┐
                          │  Read batch of raw_pages │
                          │  from PostgreSQL         │
                          │  (server-side cursor)    │
                          └────────────┬─────────────┘
                                       ▼
   ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
                                       ▼                            per page
   │                      ┌──────────────────────────┐             │
                          │  STAGE 1  clean_html     │
   │                      │  1 scripts/styles/       │             │
                          │    comments removed      │
   │                      │  2 nav/header/footer     │             │
                          │  3 entities, NFKC,       │
   │                      │    invisible chars       │             │
                          │  4 whitespace + lines    │
   │                      └────────────┬─────────────┘             │
                                       ▼
   │                            ╱ ╲                                │
                         ╱ words < min? ╲ ── yes ──► count as
   │                            ╲ ╱                   skipped      │
                                 no
   │                             ▼                                 │
                          ┌──────────────────────────┐
   │                      │  STAGE 2 normalize_page  │             │
                          │  canonical URL → doc_id  │
   │                      │  title/headings/keywords │             │
                          │  stats + provenance      │
   │                      │  fields{} per site       │             │
                          └────────────┬─────────────┘             │
   │                                   ▼                           │
                          ┌──────────────────────────┐
   │                      │  validate_document()     │             │
                          └────────────┬─────────────┘
   │                            ╱ ╲                                │
                         ╱    valid?   ╲ ── no ──► count as failed,
   │                            ╲ ╱                 record URL     │
                                 yes               (run continues)
   │                             ▼                                 │
                          ┌──────────────────────────┐
   │                      │  append to batch         │             │
                          └────────────┬─────────────┘
   └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                                       ▼
                                ╱ ╲
                       ╱ batch full?  ╲ ── no ──► next page
                                ╲ ╱
                                 yes
                                  ▼
                          ┌──────────────────────────┐
                          │  STAGE 3 upsert_many     │
                          │  bulk ReplaceOne,        │
                          │  ordered=False,          │
                          │  key = doc_id (upsert)   │
                          └────────────┬─────────────┘
                                       ▼
                          ┌──────────────────────────┐
                          │  MongoDB: processed_data │
                          │  .documents (indexed)    │
                          └────────────┬─────────────┘
                                       ▼
                          ┌──────────────────────────┐
                          │  ProcessingStats:        │
                          │  throughput, p50/p95,    │
                          │  noise removed           │
                          └────────────┬─────────────┘
                                       ▼
                          ┌──────────────────────────┐
                          │  benchmark over samples: │
                          │  site / size / batch     │
                          └────────────┬─────────────┘
                                       ▼
                                  End Phase 3
```

---

## Key numbers to quote in the report

| Quantity | Value |
|----------|-------|
| Pages processed | 45 (16 `books.toscrape.com`, 29 `quotes.toscrape.com`) |
| Raw bytes in | 905,162 B |
| Clean characters out | 47,685 |
| Noise removed, whole corpus | **94.7%** |
| Noise removed, per site | 98.2% (books) / 79.1% (quotes) |
| Words extracted | 8,169 |
| Documents indexed | 45, one per canonical URL |
| Indexes created | 5 (`uq_doc_id` unique, `ix_fulltext` weighted text, site, hash, size) |
| Schema | 1 canonical JSON schema, 12 required keys, site-specific data under `fields` |
| Cleaning cost per page | 12.89 ms mean, 6.52 ms p50, 50 ms worst case |
| Normalisation cost per page | 0.68 ms mean |
| Store cost per document | 0.40 ms in batches of 50 |
| End-to-end throughput | **71.6 pages/s, 1.37 MB/s** on one core |
| CPU cost per MB of raw HTML | 707 ms |
| Batching gain | 1.401 ms → 0.373 ms per document (3.8×) |
| Size scaling | 13× larger pages cost 5.8× more to clean (sub-linear) |
| Re-run of all 45 pages | 0 inserted, 45 updated, 45 documents (idempotent) |
| Near-duplicates found | 1 pair (`/` and `/index.html`) |
| Tests added | 68 (project total 163, all passing) |
| New dependency | `pymongo==4.10.1`; MongoDB 7.0 container on host port 27018 |
