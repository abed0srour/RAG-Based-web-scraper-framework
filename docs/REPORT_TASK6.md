# Report Notes — Task 6: Web UI and Full-Text Search

Material to paste into the project report (.docx), plus the screenshot list.

The code figures below are measured. The figures marked **(from the demo
run)** come from `scripts/demo_task6.py` against the live cluster and have to
be filled in from that run — this machine's Docker daemon was not running
while the module was written, so every test that needs Elasticsearch is
skipped rather than guessed at.

---

## Section text for the report

### 1. Introduction to Phase 6

Phase 5 ended with a REST API over three stores: the crawl record in
PostgreSQL, the processed documents in MongoDB, and the chunk index in
Chroma. Everything a client could want was reachable, but only
programmatically — the deliverable of this phase is the part a person uses,
and that changes what the system has to be able to answer.

A program asks for a resource. A person types half a word and expects the
interface to react; narrows by site when the result list is too long; wants
to see *why* a page was returned rather than being told that it was; and
expects the whole thing to stay responsive while they do it. None of those
are retrieval problems, and the first part of this phase was recognising
that the existing stores cannot answer them. MongoDB's text index can report
which documents contain a set of words, and that is genuinely all it does:
it does not rank them in a way that survives being shown to a user, it
cannot return the fragment of the page that matched with the terms marked,
and it cannot count the matches per site in the same round trip. So this
phase adds a fourth store, Elasticsearch, and a React interface in front of
it.

The work is in three places: `src/search/` (the index, its mapping, the
queries and the projection from MongoDB), `src/api/routers/fulltext.py`
(four endpoints), and `ui/` (the interface itself).

### 2. Why Elasticsearch, and what each store is for

The framework now has four data stores, and the division between them is the
same argument each time — each answers a question the others cannot:

| Store | What it holds | The question it answers |
|-------|---------------|-------------------------|
| PostgreSQL | Raw responses, jobs, the link graph | "Which rows have this value?" |
| MongoDB | One JSON document per page | "Which documents contain these words?" |
| Chroma | One 384-dimensional vector per chunk | "Which passages are closest in *meaning* to this one?" |
| Elasticsearch | A flat projection of every document | "Give me the ranked pages for this phrase, the snippet from each showing why it matched, the counts per site, and the titles beginning with what I have typed." |

The fourth question is the one a search *interface* asks, and it is one
request, not four. `scripts/demo_task6.py` runs the same query against
MongoDB and against Elasticsearch and prints both results, so this argument
appears in the report as an observation rather than a claim.

It is worth saying what Elasticsearch is *not* doing here: it is not
replacing the vector store. The two are complementary, and the demonstration
makes that visible too — `/v1/fulltext/similar/{doc_id}` finds pages that use
the same words (`more_like_this` over high-TF-IDF terms), while
`/v1/search/similar/{doc_id}` from Phase 4 finds pages that mean the same
thing. A page about "poems for children" is lexically similar to another page
about poems; it is semantically similar to a page about verse for infants
that shares no vocabulary with it at all.

### 3. The index and its mapping

MongoDB remains the system of record. The Elasticsearch index is a
*projection* of it: `to_index_document()` flattens each stored document onto
about twenty fields, truncating the page text to `ELASTIC_MAX_TEXT_CHARS`
(60 000 by default) because a highlighted snippet never needs the whole of a
long page and the untruncated text is one MongoDB read away via the `doc_id`
that comes back with every hit. The index can therefore be deleted and
rebuilt at any time without touching the source data, which is what
`python -m src.search.indexer --rebuild` does.

Four decisions in the mapping carry the weight of the phase.

**The site-specific fields are stored but never indexed.** Every processed
document carries a `fields` sub-document whose shape differs per website — a
price, a rating and an availability string on the bookshop; an author and a
list of tags on the quotes site. Letting Elasticsearch map those dynamically
is the standard way to destroy a cluster: the field count grows with the
number of distinct keys ever seen, the mapping becomes part of the cluster
state, and the cluster state has to be replicated to every node on every
change. The literature calls it mapping explosion. The mapping here is
declared `dynamic: strict`, so an undeclared field is *rejected* rather than
mapped, and `fields` is declared as an object with `enabled: false`: stored
in `_source`, returned with every hit, invisible to the index. Its values
stay searchable through `fields_text`, a single analysed string built by
`flatten_fields()`, which renders `{"author": "Albert Einstein"}` as
`"author Albert Einstein"`. One field instead of unboundedly many, and
nothing is lost from the response. A test asserts that every field the
projection produces exists in the mapping, so the strictness is caught here
rather than by the cluster.

**Analysis is English-aware.** The `content` analyzer lowercases, strips
possessives, removes stop words and applies the English stemmer, so a search
for *drawing* matches a page containing *draw* or *draws*. This is the
concrete thing the previous stores could not do: a substring match over
MongoDB matches the characters it was given and nothing else.

**Suggestions are n-grammed at index time, not at query time.** The
`title.suggest` field is indexed through an edge n-gram filter, so "Light"
is stored as `li`, `lig`, `ligh`, `light`. The *search* analyzer deliberately
omits that filter, so the prefix the user has typed is matched whole against
those n-grams. The result is that an as-you-type query is an ordinary term
lookup rather than a wildcard scan, which is what makes it cheap enough to
issue on every keystroke. Had both sides been n-grammed, typing "li" would
have been matched as `l` + `li` and returned most of the corpus.

**The document id is the index id.** `_id` is the `doc_id` that MongoDB and
Chroma already key on, so re-indexing a page replaces it rather than adding a
second copy. This is the same idempotency property the other two stores have,
and it is needed for the same reason: the RabbitMQ frontier delivers a URL
at least once, not exactly once. The demonstration runs the indexer twice and
prints the document count before and after to show that the second run
creates no duplicates.

### 4. The query

`build_query()` returns a `bool` query with three parts, and each is a
decision worth stating:

- A `multi_match` of type `best_fields` over seven weighted fields
  (`title^4`, `headings^2.5`, `description^2`, `summary^2`,
  `keywords.text^1.5`, `fields_text^1.5`, `text`). `best_fields` scores a
  document by its single strongest field rather than by the sum across all of
  them, so a page whose *title* matches beats a page that happens to mention
  the words once in each of six fields. The title is weighted four times the
  body because a title is the one field an author wrote to describe the whole
  page: a hit there is evidence about the document, while a body hit is
  evidence about one sentence in it.
- `minimum_should_match: 70%`, which is the compromise between `and` —
  which discards a good hit over one missing word — and `or`, which returns
  every page containing *the*.
- Two `match_phrase` clauses in `should`, on the title and the body. A
  `should` clause cannot exclude anything; it only adds score, so an exact
  phrase outranks the same words scattered across a page.

Filters (site, language, minimum length, enrichment state) go in `filter`
rather than in the query, because they are yes/no questions: they contribute
nothing to the score and Elasticsearch can cache their results. The sort for
relevance carries `doc_id` as a tie-break so that two equally scored hits do
not swap places between two identical requests — paging over an unstable sort
shows the same document twice.

All of this is built by plain functions that return plain dictionaries, which
is what lets `tests/test_search.py` assert on the query a set of parameters
*produces* without a cluster running. It is the search equivalent of testing
the generated SQL rather than the rows.

### 5. The endpoints

Four endpoints, all requiring the `search` scope and sharing the search
rate-limit window with the Phase 5 retrieval endpoints — they are cheap
individually, but the UI calls them on every keystroke, so they belong in the
same budget:

| Endpoint | Returns |
|----------|---------|
| `GET /v1/fulltext` | Ranked hits, each with its highlighted fragments, plus facet counts over **every** match, five sort orders and offset paging over an exact total |
| `GET /v1/fulltext/suggest` | As-you-type suggestions (minimum two characters, enforced at the edge) |
| `GET /v1/fulltext/sites` | The indexed sites and their counts, which populates the filter before anything is typed |
| `GET /v1/fulltext/similar/{doc_id}` | Lexically related pages |

Two details. An empty `q` is *allowed* on `/v1/fulltext` and is not an error:
it means browse, and with a sort and a filter that is exactly what the UI's
initial view needs. And the facet counts are computed over the matching set
rather than over the page being displayed, which is what makes them worth
showing — "311 of your matches are from books.toscrape.com" is a decision a
user can act on, whereas "3 of the 10 results on screen" is not.

`SearchUnavailable` was added to the API's exception translation table, so an
unreachable cluster becomes a **503 with `Retry-After`** and the error type
`search_index_unavailable`, not a 500: the request was fine, the dependency
was not. `/health/ready` now reports Elasticsearch as a fifth check, and
because the stores degrade independently, a cluster outage shows as
`degraded` while the RAG endpoints keep working.

### 6. The interface

`ui/` is a React 18 + TypeScript application built with Vite, with no UI
framework — the interface is a search page, a form and a table, and a utility
framework would have outweighed the whole application. Three tabs sit over
three stores: **Search** (Elasticsearch), **Ask** (the Phase 4 RAG read path,
with its citations), and **Raw data** (the PostgreSQL crawl record). A status
bar polls `/health/ready` every fifteen seconds, so a partial outage is
visible in the interface rather than only as a query that failed.

Four implementation details are worth describing in the report.

**The search is debounced and every request is abortable.** Typing *widgets*
is seven state changes. Without debouncing that is seven requests, six of
them obsolete before they return, and on the suggest endpoint it is six
requests against the rate limit for nothing. Debouncing the *value* rather
than the request keeps the input itself fully responsive — the box shows
every keystroke immediately, only the query lags. Cancellation then solves
the harder problem: without it, a slow response for *widg* can arrive after
the response for *widgets* and overwrite it, so the results on screen would
not correspond to the text in the box. Each API call takes an `AbortSignal`,
the search effect aborts the previous request when its inputs change, and an
aborted request is reported as such rather than as a failure, so a replaced
keystroke never flashes an error.

**Highlight fragments are rendered as React nodes, never as HTML.** This is
the security decision of the phase. Elasticsearch wraps matched terms in
`<mark>` … `</mark>` and inserts those tags into the *stored text* — which
came from a scraped web page — and escapes nothing else in it. A crawled page
whose clean text happens to contain the characters `<img onerror=...>` comes
back in a fragment with those characters intact. Passing that fragment to
`dangerouslySetInnerHTML`, which is the obvious way to render a highlight,
would be a stored cross-site-scripting vulnerability whose payload is
supplied by whatever site the crawler visited. `ui/src/lib/highlight.tsx`
splits the fragment on the two markers and returns React elements, so
everything that is not a marker stays a text node and React escapes it on
render. No HTML from a crawled page can reach the DOM as markup.

**Relevance is shown as a bar, not as a number.** An Elasticsearch score is
unbounded and only comparable within one result set, so printing "8.42" next
to a result would imply a precision that does not exist. The API returns
`relevance`, the score as a fraction of the best hit in that result set, and
the card draws it as a bar with the raw score in the tooltip.

**Errors are read from the API's error envelope.** Every failure from Phase 5
carries `{type, message, request_id}`, so the UI can distinguish "this key
lacks the `search` scope" (403) from "Elasticsearch is down" (503) and say
something useful about each — the second one prints the `docker compose`
command that starts the cluster and the `python -m src.search.indexer`
command that fills it. The request id is displayed, which is what ties a
user's complaint to a line in the API log.

### 7. Testing

The split is the same one used for the RAG module, for the same reason.

`tests/test_search.py` (43 tests, no cluster needed) covers the mapping, the
projection and the query builders directly, because they are pure functions
from parameters to a dictionary. A further 18 integration tests run against a
real cluster when one is up and are skipped when it is not: analysis,
scoring, highlighting and faceting are Elasticsearch's behaviour rather than
ours, and asserting on them against a fake would only test the fake. Those
tests create their own index and delete it afterwards, so they never touch
the development corpus. The availability probe is cached for the session —
eighteen separate connection timeouts turned a two-second skip into a
two-minute one.

`tests/test_api_fulltext.py` (29 tests) covers the HTTP contract with a fake
index injected through `app.dependency_overrides`: authentication, the scope
check, parameter validation, the paging cap, the rate-limit bucket, the
OpenAPI documentation, and the 503 translation. Most of them assert on the
call the fake *received*, because the contract is that `?site=x&sort=recent`
reaches the index as those arguments — a test that only checked the status
code would not notice a parameter being dropped on the way.

---

## Screenshot list for the report

1. `docker compose ps` with all five services healthy, Elasticsearch among them.
2. `curl http://localhost:9200` — the cluster's version banner.
3. `python -m src.search.indexer` — the indexing run report (documents read,
   indexed, batches, throughput).
4. `python -m src.search.report --query "light in the attic"` — coverage,
   indexed sites, and the ranked hits with their snippets.
5. `GET /v1/fulltext?q=...` in `/docs` — the response body, showing
   `highlights`, `facets` and both timings.
6. The UI's search tab: results with marked terms, the facet sidebar, and the
   timing line.
7. The UI mid-typing, with the suggestion list open.
8. The document drawer: the AI summary from MongoDB beside the related pages
   from Elasticsearch.
9. The UI with Elasticsearch stopped — the 503 message, the recovery
   instructions, and the red dot in the status bar.
10. `scripts/demo_task6.py` section 6 — the same query in MongoDB and in
    Elasticsearch, side by side.

---

## Results table

| Metric | Value |
|--------|-------|
| New Python modules | 3 (`src/search/index.py`, `indexer.py`, `report.py`) — 1 449 lines |
| New API router | `fulltext.py`, 160 lines, 4 endpoints |
| API surface | 29 routes, 25 of them under `/v1` (25 and 21 before this phase) |
| Index mapping | 21 declared fields, `dynamic: strict`, 1 object with indexing disabled |
| Analysers | 3 (`content`, `suggest_index`, `suggest_search`) |
| Queried fields per search | 7, weighted `title^4` down to `text^1` |
| Facets per response | 4 (site, language, keyword, enrichment state) |
| UI | React 18 + TypeScript, 1 686 lines across 13 TypeScript files plus one stylesheet, no UI framework |
| UI bundle | 165 kB JS (53 kB gzipped) + 11.7 kB CSS |
| UI dependencies | 2 runtime (`react`, `react-dom`), 67 packages installed in total |
| Tests added | 90 (43 unit + 18 cluster-integration + 29 API) |
| Project test suite | **309 passing, 65 skipped** (237 passing before this phase) |
| New Python dependency | `elasticsearch==8.17.0` (client major must match the server) |
| New container | `docker.elastic.co/elasticsearch/elasticsearch:8.17.0`, 1 GB heap, 2 GB limit, host port 9200 |
| Documents indexed | *(from the demo run)* |
| Indexing throughput | *(from the demo run)* |
| Query latency, cluster | *(from the demo run)* |
| Query latency, round trip | *(from the demo run)* |
| MongoDB vs Elasticsearch on the same query | *(from the demo run)* |
