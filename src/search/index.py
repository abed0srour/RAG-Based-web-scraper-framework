"""The Elasticsearch index behind the web UI's search box (Task 6).

This is the fourth store in the framework, and as with the other three the
reason it exists is that the others cannot answer its question:

* *PostgreSQL* holds the crawl record - raw responses, jobs, the link graph.
* *MongoDB* holds the processed documents, read back whole, with a text
  index that answers "which documents contain these words".
* *Chroma* holds one vector per chunk and answers "which passages are
  closest in meaning to this one".
* *Elasticsearch* - here - answers what a **search interface** asks: give me
  the ranked documents for this phrase, the snippet from each one with the
  matched words marked, the counts per site so the user can narrow down, and
  a list of titles that start with what has been typed so far. One query,
  one round trip, tens of milliseconds.

Four design points, each of which shows up in the code below.

**The mapping is explicit, and the site-specific fields are not indexed.**
Every document carries a ``fields`` sub-document whose shape differs per
website (a price and a rating on a bookshop, an author and a tag on a quotes
site). Letting Elasticsearch map those dynamically is how an index ends up
with thousands of fields and a cluster state too big to replicate - the
"mapping explosion" problem. So ``fields`` is stored with indexing disabled,
and a flattened ``fields_text`` string carries the same key/value pairs in a
form the full-text query can search. One field instead of unboundedly many,
and nothing is lost from the response.

**Analysis is English-aware.** The ``content`` analyzer lowercases, strips
possessives, removes stop words and stems, so a search for "running shoes"
matches "run shoe" - which is the behaviour a user expects and the thing a
substring or regular-expression search over MongoDB cannot do.

**Suggestions use edge n-grams at index time, not at query time.** Indexing
``"Light"`` as ``li``, ``lig``, ``ligh``, ``light`` means an as-you-type
query is an ordinary term lookup rather than a wildcard scan, which is what
keeps the suggestion endpoint fast enough to call on every keystroke. The
search analyzer deliberately does *not* apply the n-gram filter, so the
typed prefix is matched whole against those n-grams.

**The id is the document id.** ``_id`` is the ``doc_id`` that MongoDB and
Chroma already use, so re-indexing a page replaces it rather than adding a
second copy - the same idempotency the other two stores have, and needed for
the same reason: the frontier delivers a URL at least once, not exactly once.

Every call fails fast with :class:`SearchUnavailable` when the cluster is
unreachable, so a stopped container produces one clear message instead of a
hang, and the API turns that into a 503.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.common.config import settings

logger = logging.getLogger(__name__)

# Fields queried for a full-text search, with their boosts. A title match is
# worth four body matches: the title is the one field an author wrote to
# describe the whole page, so a hit there is evidence about the document
# rather than about one sentence in it.
SEARCH_FIELDS = (
    "title^4",
    "headings^2.5",
    "description^2",
    "summary^2",
    "keywords.text^1.5",
    "fields_text^1.5",
    "text",
)

# Returned for every hit. The full text is deliberately excluded: the
# highlighter already returns the part of it that matched, and a result list
# of twenty pages would otherwise carry a megabyte of prose nobody displays.
RESULT_SOURCE = (
    "doc_id",
    "url",
    "site",
    "path",
    "title",
    "description",
    "language",
    "keywords",
    "headings",
    "word_count",
    "reading_time_sec",
    "processed_at",
    "fetched_at",
    "http_status",
    "depth",
    "summary",
    "enriched",
    "quality_score",
    "fields",
)

SORTS = ("relevance", "recent", "oldest", "words", "title")

_CLIENTS: dict[tuple[str, int, str], Any] = {}
_INDEX: SearchIndex | None = None


class SearchUnavailable(RuntimeError):
    """Raised when Elasticsearch cannot be reached or used."""


# ---------------------------------------------------------------------------
# Index definition
# ---------------------------------------------------------------------------

INDEX_SETTINGS: dict[str, Any] = {
    # One shard, no replica: a single-node development cluster cannot
    # allocate a replica anyway, and an unassigned replica is what turns the
    # cluster health yellow and makes a readiness probe look wrong.
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {
        "filter": {
            "english_stop": {"type": "stop", "stopwords": "_english_"},
            "english_stemmer": {"type": "stemmer", "language": "english"},
            "english_possessive": {
                "type": "stemmer",
                "language": "possessive_english",
            },
            # 2 is the shortest prefix worth suggesting on; 20 covers any
            # word a title realistically starts with.
            "edge_ngrams": {"type": "edge_ngram", "min_gram": 2, "max_gram": 20},
        },
        "analyzer": {
            "content": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": [
                    "lowercase",
                    "english_possessive",
                    "english_stop",
                    "english_stemmer",
                ],
            },
            "suggest_index": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "edge_ngrams"],
            },
            # The query side must not be n-grammed: the prefix the user has
            # typed is matched as one term against the indexed n-grams.
            "suggest_search": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase"],
            },
        },
        "normalizer": {
            "lowercase_keyword": {"type": "custom", "filter": ["lowercase"]},
        },
    },
}

INDEX_MAPPINGS: dict[str, Any] = {
    # Unknown fields are rejected rather than silently mapped, which is what
    # keeps the "mapping explosion" described in the module docstring from
    # happening by accident when the document schema grows.
    "dynamic": "strict",
    "properties": {
        "doc_id": {"type": "keyword"},
        # The URL is a filter and a link, not prose - but its words are worth
        # searching, since a slug often repeats the title.
        "url": {
            "type": "keyword",
            "fields": {"text": {"type": "text", "analyzer": "content"}},
        },
        "site": {"type": "keyword", "normalizer": "lowercase_keyword"},
        "path": {"type": "keyword"},
        "title": {
            "type": "text",
            "analyzer": "content",
            "fields": {
                # For sorting and exact-title faceting.
                "raw": {"type": "keyword", "ignore_above": 512},
                # For the as-you-type endpoint.
                "suggest": {
                    "type": "text",
                    "analyzer": "suggest_index",
                    "search_analyzer": "suggest_search",
                },
            },
        },
        "description": {"type": "text", "analyzer": "content"},
        "headings": {"type": "text", "analyzer": "content"},
        "text": {"type": "text", "analyzer": "content"},
        # Keywords are a facet (exact terms) and searchable prose at once.
        "keywords": {
            "type": "keyword",
            "fields": {
                "text": {"type": "text", "analyzer": "content"},
                "suggest": {
                    "type": "text",
                    "analyzer": "suggest_index",
                    "search_analyzer": "suggest_search",
                },
            },
        },
        "language": {"type": "keyword"},
        "word_count": {"type": "integer"},
        "reading_time_sec": {"type": "integer"},
        "processed_at": {"type": "date"},
        "fetched_at": {"type": "date"},
        "http_status": {"type": "integer"},
        "depth": {"type": "integer"},
        # The contextual summary generated in Task 4, when the document has
        # been enriched. Searchable, so a user can find a page by what the
        # summary says about it.
        "summary": {"type": "text", "analyzer": "content"},
        "enriched": {"type": "boolean"},
        "quality_score": {"type": "float"},
        # The site-specific sub-document: returned with every hit, never
        # indexed. See the module docstring.
        "fields": {"type": "object", "enabled": False},
        # ...and its searchable flattening, built by ``to_index_document``.
        "fields_text": {"type": "text", "analyzer": "content"},
    },
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _require_elasticsearch() -> Any:
    try:
        import elasticsearch
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise SearchUnavailable(
            "elasticsearch is not installed; run 'pip install -r requirements.txt'"
        ) from exc
    return elasticsearch


def get_client(
    host: str | None = None, port: int | None = None, scheme: str | None = None
) -> Any:
    """Return the process-wide Elasticsearch client for ``host:port``.

    One client per process, as with the other stores: it owns a connection
    pool, and building a second one per request would throw that away.
    """
    elasticsearch = _require_elasticsearch()
    key = (
        host or settings.elastic_host,
        port or settings.elastic_port,
        scheme or settings.elastic_scheme,
    )

    if key not in _CLIENTS:
        kwargs: dict[str, Any] = {
            "hosts": [{"host": key[0], "port": key[1], "scheme": key[2]}],
            "request_timeout": settings.elastic_timeout,
            # One retry on a connection error covers a cluster that is still
            # starting; more than that just delays a clear failure.
            "max_retries": 1,
            "retry_on_timeout": True,
        }
        if settings.elastic_user:
            kwargs["basic_auth"] = (settings.elastic_user, settings.elastic_password)
        _CLIENTS[key] = elasticsearch.Elasticsearch(**kwargs)
        logger.debug("Opened Elasticsearch client for %s://%s:%s", key[2], key[0], key[1])
    return _CLIENTS[key]


def reset_client() -> None:
    """Drop the cached clients and the shared index handle (used by tests)."""
    global _INDEX
    for client in _CLIENTS.values():
        try:
            client.close()
        except Exception:  # noqa: BLE001 - closing a dead client is not news
            pass
    _CLIENTS.clear()
    _INDEX = None


def get_search_index() -> SearchIndex:
    """The shared :class:`SearchIndex`, created on first use."""
    global _INDEX
    if _INDEX is None:
        _INDEX = SearchIndex()
    return _INDEX


# ---------------------------------------------------------------------------
# Document conversion
# ---------------------------------------------------------------------------


def flatten_fields(fields: Mapping[str, Any] | None, limit: int = 2000) -> str:
    """Render the site-specific ``fields`` sub-document as searchable text.

    ``{"price": "£51.77", "rating": "Three"}`` becomes
    ``"price £51.77 rating Three"``, so a query for *rating three* matches
    the document without ``price`` and ``rating`` ever becoming index
    fields. Nested values are walked; the keys are kept because they are
    meaningful words themselves on these sites.
    """
    if not fields:
        return ""

    parts: list[str] = []

    def walk(key: str, value: Any) -> None:
        if len(" ".join(parts)) > limit:
            return
        if isinstance(value, Mapping):
            for inner_key, inner in value.items():
                walk(str(inner_key), inner)
        elif isinstance(value, (list, tuple, set)):
            for inner in value:
                walk(key, inner)
        elif value is None or isinstance(value, bool):
            return
        else:
            text = str(value).strip()
            if text:
                parts.append(f"{key} {text}" if key else text)

    for name, value in fields.items():
        walk(str(name), value)

    return " ".join(parts)[:limit]


def to_index_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Project a stored MongoDB document onto the flat index mapping.

    The index is not a copy of the document store: it is the subset of each
    document that a search needs, flattened, with the page text truncated to
    ``ELASTIC_MAX_TEXT_CHARS``. A highlighted snippet never needs the whole
    of a 200 kB page, and the untruncated text is one MongoDB read away via
    the ``doc_id`` that comes back with the hit.
    """
    stats = document.get("stats") or {}
    source = document.get("source") or {}
    processing = document.get("processing") or {}
    enrichment = document.get("enrichment") or {}
    quality = enrichment.get("quality") or {}

    summary = str(enrichment.get("summary") or "")
    text = str(document.get("text") or "")

    indexed: dict[str, Any] = {
        "doc_id": document.get("doc_id"),
        "url": document.get("url"),
        "site": document.get("site"),
        "path": document.get("path"),
        "title": document.get("title"),
        "description": document.get("description"),
        "headings": list(document.get("headings") or []),
        "text": text[: settings.elastic_max_text_chars],
        "keywords": list(document.get("keywords") or []),
        "language": document.get("language"),
        "word_count": stats.get("word_count"),
        "reading_time_sec": stats.get("reading_time_sec"),
        "processed_at": processing.get("processed_at"),
        "fetched_at": source.get("fetched_at"),
        "http_status": source.get("http_status"),
        "depth": source.get("depth"),
        "summary": summary,
        "enriched": bool(summary),
        # The augmentation stage records the checklist score before and after
        # it ran; the post-enrichment one is the document's current quality.
        "quality_score": quality.get("score_after", quality.get("score_before")),
        "fields": document.get("fields") or {},
        "fields_text": flatten_fields(document.get("fields")),
    }
    # A null is not the same as an absent field for a date or an integer:
    # Elasticsearch rejects "" for both, and an explicit null costs a mapping
    # check for nothing. Drop them.
    return {key: value for key, value in indexed.items() if value not in (None, "")}


# ---------------------------------------------------------------------------
# Query construction
#
# These are plain functions returning plain dictionaries, which is what lets
# the tests assert on the query a set of parameters produces without a
# cluster running - the search equivalent of checking the SQL rather than
# the rows.
# ---------------------------------------------------------------------------


def build_filters(
    *,
    site: str | None = None,
    language: str | None = None,
    min_words: int | None = None,
    enriched: bool | None = None,
) -> list[dict[str, Any]]:
    """The ``filter`` clauses for a search.

    Filters, not query clauses: a filter is a yes/no question, so it does not
    contribute to the score and Elasticsearch can cache its result.
    """
    filters: list[dict[str, Any]] = []
    if site:
        filters.append({"term": {"site": site.lower().removeprefix("www.")}})
    if language:
        filters.append({"term": {"language": language}})
    if min_words is not None:
        filters.append({"range": {"word_count": {"gte": min_words}}})
    if enriched is not None:
        filters.append({"term": {"enriched": enriched}})
    return filters


def build_query(query: str, filters: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The scoring query for ``query``.

    ``best_fields`` scores a document by its single strongest field rather
    than by the sum over all of them, so one page whose title matches beats a
    page that mentions the words once in each of six fields.
    ``minimum_should_match: 70%`` is the compromise between ``and`` (which
    drops a good hit over one missing word) and ``or`` (which returns every
    page containing *the*). The two phrase clauses add a bonus - they cannot
    exclude anything - so an exact phrase outranks the same words scattered
    over a page.
    """
    text = (query or "").strip()
    if not text:
        # An empty query with filters is a browse, not a search: everything
        # matches, and the sort does the ordering.
        return {"bool": {"must": [{"match_all": {}}], "filter": list(filters)}}

    return {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": text,
                        "fields": list(SEARCH_FIELDS),
                        "type": "best_fields",
                        "minimum_should_match": "70%",
                        # One edit of tolerance, and only on terms long
                        # enough for it to be a typo rather than a different
                        # word. Off for phrases, where it is noise.
                        "fuzziness": "AUTO",
                        "prefix_length": 1,
                    }
                }
            ],
            "should": [
                {"match_phrase": {"title": {"query": text, "boost": 3}}},
                {"match_phrase": {"text": {"query": text, "boost": 2}}},
            ],
            "filter": list(filters),
        }
    }


def build_sort(sort: str) -> list[Any]:
    """Translate a sort name into Elasticsearch sort clauses."""
    if sort == "recent":
        return [{"processed_at": {"order": "desc"}}, "_score"]
    if sort == "oldest":
        return [{"processed_at": {"order": "asc"}}, "_score"]
    if sort == "words":
        return [{"word_count": {"order": "desc"}}, "_score"]
    if sort == "title":
        return [{"title.raw": {"order": "asc"}}, "_score"]
    # Relevance, with the document id as the tie-break so that two equally
    # scored hits do not swap places between identical requests.
    return ["_score", {"doc_id": {"order": "asc"}}]


def build_highlight() -> dict[str, Any]:
    """Highlighter configuration: the snippets that show *why* a hit matched."""
    return {
        "pre_tags": ["<mark>"],
        "post_tags": ["</mark>"],
        "fields": {
            "title": {"number_of_fragments": 0},
            "summary": {"number_of_fragments": 1},
            "headings": {"number_of_fragments": 1},
            "text": {
                "fragment_size": settings.elastic_highlight_size,
                "number_of_fragments": settings.elastic_highlight_fragments,
            },
            "fields_text": {"number_of_fragments": 1},
        },
        # Without this, a hit that matched only on the title gets no snippet
        # at all and the result card looks empty.
        "require_field_match": False,
        "no_match_size": settings.elastic_highlight_size,
    }


def build_aggregations() -> dict[str, Any]:
    """The facet counts returned alongside the hits.

    They are computed over the *matching* set, not the whole index, which is
    what makes them useful: "of the 412 pages matching your search, 311 are
    from books.toscrape.com".
    """
    return {
        "sites": {"terms": {"field": "site", "size": 20}},
        "languages": {"terms": {"field": "language", "size": 10}},
        "keywords": {"terms": {"field": "keywords", "size": 15}},
        "enriched": {"terms": {"field": "enriched", "size": 2}},
    }


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


class SearchIndex:
    """Typed wrapper around one Elasticsearch index."""

    def __init__(self, index: str | None = None, client: Any | None = None) -> None:
        self.index_name = index or settings.elastic_index
        self._client = client

    # ---------------- plumbing ----------------

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = get_client()
        return self._client

    def _guard(self, action: str, exc: Exception) -> SearchUnavailable:
        """Wrap a client error in one exception type with a usable message."""
        elasticsearch = _require_elasticsearch()
        if isinstance(exc, elasticsearch.ConnectionError):
            return SearchUnavailable(
                f"Elasticsearch is not reachable at "
                f"{settings.elastic_scheme}://{settings.elastic_host}:"
                f"{settings.elastic_port} ({action})"
            )
        return SearchUnavailable(f"Elasticsearch {action} failed: {exc}")

    def ping(self) -> bool:
        """True when the cluster answers. Never raises: this is a probe."""
        try:
            return bool(self.client.ping())
        except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
            logger.debug("Elasticsearch ping failed: %s", exc)
            return False

    def server_version(self) -> str:
        """The cluster's version string, for the environment report."""
        try:
            info = self.client.info()
        except Exception as exc:  # noqa: BLE001
            raise self._guard("info", exc) from exc
        return str((info.get("version") or {}).get("number") or "unknown")

    def health(self) -> dict[str, Any]:
        """Cluster status, document count and store size for the API probe."""
        try:
            cluster = self.client.cluster.health()
            exists = self.client.indices.exists(index=self.index_name)
            documents = self.count() if exists else 0
        except Exception as exc:  # noqa: BLE001
            raise self._guard("health", exc) from exc
        return {
            "status": str(cluster.get("status") or "unknown"),
            "index": self.index_name,
            "index_exists": bool(exists),
            "documents": documents,
        }

    # ---------------- write path ----------------

    def exists(self) -> bool:
        try:
            return bool(self.client.indices.exists(index=self.index_name))
        except Exception as exc:  # noqa: BLE001
            raise self._guard("indices.exists", exc) from exc

    def ensure_index(self) -> bool:
        """Create the index with its mapping if it is not there yet.

        Returns True when it was created by this call. The mapping is applied
        at creation because an analyzer cannot be added to a live index:
        changing analysis means a new index and a reindex, so getting it right
        up front is not a nicety.
        """
        if self.exists():
            return False
        elasticsearch = _require_elasticsearch()
        try:
            self.client.indices.create(
                index=self.index_name,
                settings=INDEX_SETTINGS,
                mappings=INDEX_MAPPINGS,
            )
        except elasticsearch.BadRequestError as exc:
            # Another worker won the race between exists() and create().
            if "resource_already_exists_exception" in str(exc):
                return False
            raise self._guard("indices.create", exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._guard("indices.create", exc) from exc
        logger.info("Created Elasticsearch index %s", self.index_name)
        return True

    def delete_index(self) -> bool:
        """Drop the index. Used by ``--rebuild`` and by the integration tests."""
        elasticsearch = _require_elasticsearch()
        try:
            self.client.indices.delete(index=self.index_name)
            return True
        except elasticsearch.NotFoundError:
            return False
        except Exception as exc:  # noqa: BLE001
            raise self._guard("indices.delete", exc) from exc

    def refresh(self) -> None:
        """Make the most recent writes visible to search.

        Elasticsearch refreshes once a second by default, which is invisible
        to a user and fatal to a test that indexes and then searches.
        """
        elasticsearch = _require_elasticsearch()
        try:
            self.client.indices.refresh(index=self.index_name)
        except elasticsearch.NotFoundError:
            return
        except Exception as exc:  # noqa: BLE001
            raise self._guard("indices.refresh", exc) from exc

    def index_document(self, document: Mapping[str, Any], refresh: bool = False) -> str:
        """Index one document, replacing any earlier copy of the same page."""
        body = to_index_document(document)
        doc_id = str(body.get("doc_id") or "")
        if not doc_id:
            raise SearchUnavailable("document has no doc_id; cannot be indexed")
        try:
            self.client.index(
                index=self.index_name,
                id=doc_id,
                document=body,
                refresh=bool(refresh),
            )
        except Exception as exc:  # noqa: BLE001
            raise self._guard("index", exc) from exc
        return doc_id

    def index_documents(
        self, documents: Iterable[Mapping[str, Any]], batch_size: int | None = None
    ) -> dict[str, Any]:
        """Bulk-index an iterable of stored documents.

        The bulk helper is used rather than a loop of single writes because
        the cost here is per request, not per document: one HTTP round trip
        carrying two hundred documents is roughly two hundred times cheaper
        than two hundred round trips.
        """
        elasticsearch = _require_elasticsearch()
        from elasticsearch import helpers

        self.ensure_index()
        started = time.perf_counter()
        indexed = 0
        failed: list[str] = []

        def actions() -> Iterator[dict[str, Any]]:
            nonlocal failed
            for document in documents:
                body = to_index_document(document)
                doc_id = body.get("doc_id")
                if not doc_id:
                    failed.append("document without doc_id")
                    continue
                yield {"_index": self.index_name, "_id": str(doc_id), "_source": body}

        try:
            success, errors = helpers.bulk(
                self.client,
                actions(),
                chunk_size=batch_size or settings.elastic_index_batch_size,
                raise_on_error=False,
                stats_only=False,
            )
            indexed = int(success)
            for error in errors or []:
                failed.append(str(error)[:300])
        except elasticsearch.ConnectionError as exc:
            raise self._guard("bulk", exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._guard("bulk", exc) from exc

        self.refresh()
        elapsed = (time.perf_counter() - started) * 1000.0
        if failed:
            logger.warning("%d document(s) failed to index", len(failed))
        return {
            "indexed": indexed,
            "failed": len(failed),
            "errors": failed[:10],
            "duration_ms": round(elapsed, 2),
            "documents_per_second": (
                round(indexed / (elapsed / 1000.0), 1) if elapsed > 0 else 0.0
            ),
        }

    def delete_document(self, doc_id: str) -> bool:
        """Remove one document from the index."""
        elasticsearch = _require_elasticsearch()
        try:
            self.client.delete(index=self.index_name, id=doc_id)
            return True
        except elasticsearch.NotFoundError:
            return False
        except Exception as exc:  # noqa: BLE001
            raise self._guard("delete", exc) from exc

    def count(self, query: Mapping[str, Any] | None = None) -> int:
        """How many documents match, or how many there are in total."""
        elasticsearch = _require_elasticsearch()
        try:
            if query:
                response = self.client.count(index=self.index_name, query=dict(query))
            else:
                response = self.client.count(index=self.index_name)
        except elasticsearch.NotFoundError:
            return 0
        except Exception as exc:  # noqa: BLE001
            raise self._guard("count", exc) from exc
        return int(response.get("count") or 0)

    # ---------------- read path ----------------

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
        site: str | None = None,
        language: str | None = None,
        min_words: int | None = None,
        enriched: bool | None = None,
        sort: str = "relevance",
        highlight: bool = True,
        facets: bool = True,
    ) -> dict[str, Any]:
        """Run a full-text search and return hits, facets and timings.

        The return value is already in the shape the API serialises, so the
        router does no reshaping of its own: hits with their highlights and a
        relevance proportional to the best hit, the facet counts over the
        matching set, and the total so a client can page through it.
        """
        if sort not in SORTS:
            raise ValueError(f"unknown sort {sort!r}; expected one of {SORTS}")

        filters = build_filters(
            site=site, language=language, min_words=min_words, enriched=enriched
        )
        body: dict[str, Any] = {
            "query": build_query(query, filters),
            "sort": build_sort(sort),
            "from_": max(offset, 0),
            "size": max(limit, 0),
            "source_includes": list(RESULT_SOURCE),
            # The exact total matters to a paging UI, and these corpora are
            # small enough that counting them is cheap.
            "track_total_hits": True,
        }
        if highlight and (query or "").strip():
            body["highlight"] = build_highlight()
        if facets:
            body["aggs"] = build_aggregations()

        started = time.perf_counter()
        elasticsearch = _require_elasticsearch()
        try:
            response = self.client.search(index=self.index_name, **body)
        except elasticsearch.NotFoundError:
            # Searching before anything has been indexed is an empty result,
            # not an error - the UI shows "nothing indexed yet".
            return {
                "query": query,
                "total": 0,
                "hits": [],
                "facets": {},
                "took_ms": 0.0,
                "notes": [f"index {self.index_name} does not exist yet"],
            }
        except Exception as exc:  # noqa: BLE001
            raise self._guard("search", exc) from exc

        elapsed = (time.perf_counter() - started) * 1000.0
        return self._format(response, query=query, elapsed=elapsed, sort=sort)

    def suggest(self, prefix: str, limit: int = 8) -> list[dict[str, Any]]:
        """Titles and keywords starting with ``prefix``, for the search box.

        Matched against the edge n-grams built at index time, so this is a
        term lookup rather than a wildcard scan and is safe to call on every
        keystroke.
        """
        text = (prefix or "").strip()
        if len(text) < 2:
            return []

        elasticsearch = _require_elasticsearch()
        try:
            response = self.client.search(
                index=self.index_name,
                query={
                    "bool": {
                        "should": [
                            {"match": {"title.suggest": {"query": text, "boost": 3}}},
                            {"match": {"keywords.suggest": {"query": text}}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                size=max(limit, 0),
                source_includes=["doc_id", "url", "site", "title"],
            )
        except elasticsearch.NotFoundError:
            return []
        except Exception as exc:  # noqa: BLE001
            raise self._guard("suggest", exc) from exc

        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for hit in (response.get("hits") or {}).get("hits") or []:
            source = hit.get("_source") or {}
            title = str(source.get("title") or "").strip()
            if not title or title.lower() in seen:
                continue
            seen.add(title.lower())
            out.append(
                {
                    "text": title,
                    "doc_id": source.get("doc_id"),
                    "url": source.get("url"),
                    "site": source.get("site"),
                }
            )
        return out

    def similar(self, doc_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Documents like this one, by shared significant terms.

        ``more_like_this`` is the lexical counterpart of the vector-store
        neighbour search in Task 4: it compares the document's high-TF-IDF
        terms rather than its embedding, so it finds pages that use the same
        words, where the vector search finds pages that mean the same thing.
        """
        elasticsearch = _require_elasticsearch()
        try:
            response = self.client.search(
                index=self.index_name,
                query={
                    "bool": {
                        "must": [
                            {
                                "more_like_this": {
                                    "fields": ["title", "text", "keywords.text"],
                                    "like": [
                                        {"_index": self.index_name, "_id": doc_id}
                                    ],
                                    "min_term_freq": 1,
                                    "min_doc_freq": 1,
                                    "max_query_terms": 25,
                                }
                            }
                        ],
                        "must_not": [{"term": {"doc_id": doc_id}}],
                    }
                },
                size=max(limit, 0),
                source_includes=list(RESULT_SOURCE),
            )
        except elasticsearch.NotFoundError:
            return []
        except Exception as exc:  # noqa: BLE001
            raise self._guard("similar", exc) from exc

        formatted = self._format(response, query="", elapsed=0.0, sort="relevance")
        return formatted["hits"]

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """The indexed form of one document, or None when it is not indexed."""
        elasticsearch = _require_elasticsearch()
        try:
            response = self.client.get(index=self.index_name, id=doc_id)
        except elasticsearch.NotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001
            raise self._guard("get", exc) from exc
        return dict(response.get("_source") or {})

    def sites(self) -> list[dict[str, Any]]:
        """Indexed sites with their document counts, for the UI's filter."""
        elasticsearch = _require_elasticsearch()
        try:
            response = self.client.search(
                index=self.index_name,
                size=0,
                aggs={"sites": {"terms": {"field": "site", "size": 50}}},
            )
        except elasticsearch.NotFoundError:
            return []
        except Exception as exc:  # noqa: BLE001
            raise self._guard("sites", exc) from exc
        buckets = ((response.get("aggregations") or {}).get("sites") or {}).get(
            "buckets"
        ) or []
        return [
            {"value": bucket.get("key"), "count": int(bucket.get("doc_count") or 0)}
            for bucket in buckets
        ]

    # ---------------- formatting ----------------

    @staticmethod
    def _format(
        response: Mapping[str, Any], *, query: str, elapsed: float, sort: str
    ) -> dict[str, Any]:
        """Turn a raw Elasticsearch response into the API's shape."""
        hits_section = response.get("hits") or {}
        total = hits_section.get("total")
        total_value = int(total.get("value") if isinstance(total, Mapping) else total or 0)
        raw_hits = list(hits_section.get("hits") or [])
        max_score = hits_section.get("max_score") or 0.0

        hits: list[dict[str, Any]] = []
        for rank, hit in enumerate(raw_hits, start=1):
            source = dict(hit.get("_source") or {})
            highlights = {
                field: list(snippets)
                for field, snippets in (hit.get("highlight") or {}).items()
                if snippets
            }
            score = float(hit.get("_score") or 0.0)
            hits.append(
                {
                    **source,
                    "rank": rank,
                    "score": round(score, 4),
                    # A score the UI can draw as a bar. Elasticsearch scores
                    # are unbounded and only comparable within one result
                    # set, so this is explicitly relative to the best hit.
                    "relevance": (
                        round(score / float(max_score), 4) if max_score else None
                    ),
                    "highlights": highlights,
                    "snippet": _first_snippet(highlights, source),
                }
            )

        facets: dict[str, list[dict[str, Any]]] = {}
        for name, aggregation in (response.get("aggregations") or {}).items():
            facets[name] = [
                {
                    "value": bucket.get("key_as_string", bucket.get("key")),
                    "count": int(bucket.get("doc_count") or 0),
                }
                for bucket in (aggregation.get("buckets") or [])
            ]

        return {
            "query": query,
            "sort": sort,
            "total": total_value,
            "returned": len(hits),
            "hits": hits,
            "facets": facets,
            # Two timings: what the cluster spent, and what the call took
            # including the network. A gap between them is a network problem,
            # not a slow query.
            "took_ms": round(float(response.get("took") or 0.0), 2),
            "elapsed_ms": round(elapsed, 2),
            "notes": [],
        }


def _first_snippet(
    highlights: Mapping[str, Sequence[str]], source: Mapping[str, Any]
) -> str:
    """One line of context for a result card, whatever matched.

    Preference order is the body text, then the generated summary, then the
    page description - the first of those that exists. A card with no snippet
    reads as a bug, so the description is used even when nothing highlighted.
    """
    for field in ("text", "summary", "headings", "fields_text"):
        snippets = highlights.get(field)
        if snippets:
            return " … ".join(str(snippet) for snippet in snippets[:3])
    for field in ("summary", "description"):
        value = str(source.get(field) or "").strip()
        if value:
            return value[:300]
    return ""
