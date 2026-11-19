"""Stage 3 of the data-processing module: the NoSQL document store.

The normalised documents are written to **MongoDB**, not to PostgreSQL, and
that split is intentional:

* PostgreSQL holds the *crawl record* - raw responses, job bookkeeping, the
  link graph. It is relational data with fixed columns and foreign keys.
* MongoDB holds the *processed documents*. Each one is a single JSON object,
  retrieved as a whole, whose ``fields`` sub-document has a different shape on
  every website. Storing that in a relational table would mean either one
  column per site or a JSON blob column with no index on its contents.

What MongoDB is actually used for here is **indexing**. Five indexes are
created by :meth:`DocumentStore.ensure_indexes`:

``doc_id`` (unique)
    The deterministic identifier from the normaliser. The uniqueness is what
    makes the write path idempotent: re-processing a page updates its document
    instead of inserting a second copy, which matters because the upstream
    queue delivers URLs at-least-once.
``text`` (full-text, weighted)
    A compound text index over title, description, headings, keywords and
    body, with the title weighted highest. This is what allows a keyword query
    to be answered by the database rather than by scanning documents in Python,
    and it is the retrieval path the API exposes in a later task.
``site + processed_at``
    Per-site listings in recency order, the most common access pattern.
``content_hash``
    Finds pages whose text is identical - near-duplicate detection before the
    documents are embedded.
``stats.word_count``
    Range queries, used by the benchmark to build size-based samples.

Every operation goes through one client per process, with short server
selection timeouts so that an unreachable database fails fast and visibly
instead of hanging the worker.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

from src.common.config import settings

logger = logging.getLogger(__name__)

# pymongo is imported lazily inside _require_pymongo() so that the cleaning and
# normalisation stages - and their tests - work on a machine where the driver
# is not installed.
_client: Any = None
_client_uri: str | None = None


class StoreUnavailable(RuntimeError):
    """Raised when MongoDB cannot be reached or the driver is missing."""


def _require_pymongo() -> Any:
    try:
        import pymongo  # noqa: PLC0415 - deliberately deferred
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise StoreUnavailable(
            "pymongo is not installed; run 'pip install -r requirements.txt'"
        ) from exc
    return pymongo


def get_client(uri: str | None = None) -> Any:
    """Return the process-wide MongoDB client, creating it on first use.

    One client per process is the documented pymongo pattern: it owns an
    internal connection pool and is thread-safe, so creating one per operation
    would mean a new handshake per document.
    """
    global _client, _client_uri
    pymongo = _require_pymongo()
    target = uri or settings.mongo_uri

    if _client is None or _client_uri != target:
        _client = pymongo.MongoClient(
            target,
            serverSelectionTimeoutMS=settings.mongo_timeout_ms,
            connectTimeoutMS=settings.mongo_timeout_ms,
            socketTimeoutMS=settings.mongo_timeout_ms * 4,
            retryWrites=True,
            appname="rag-scraper-processing",
            tz_aware=True,
        )
        _client_uri = target
        logger.info(
            "MongoDB client created for %s:%s", settings.mongo_host, settings.mongo_port
        )
    return _client


def reset_client() -> None:
    """Close the cached client. Used by the tests and by forked workers."""
    global _client, _client_uri
    if _client is not None:
        try:
            _client.close()
        except Exception:  # noqa: BLE001 - closing must never raise upwards
            pass
    _client = None
    _client_uri = None


class DocumentStore:
    """Read/write access to the processed-document collection."""

    def __init__(
        self,
        *,
        database: str | None = None,
        collection: str | None = None,
        uri: str | None = None,
    ) -> None:
        self.database_name = database or settings.mongo_db
        self.collection_name = collection or settings.mongo_collection
        self._uri = uri
        self._collection: Any = None

    # ---------------- connection ----------------

    @property
    def collection(self) -> Any:
        if self._collection is None:
            client = get_client(self._uri)
            self._collection = client[self.database_name][self.collection_name]
        return self._collection

    def ping(self) -> bool:
        """True when the server answers, False when it does not."""
        try:
            get_client(self._uri).admin.command("ping")
            return True
        except Exception as exc:  # noqa: BLE001 - any driver error means "down"
            logger.warning("MongoDB ping failed: %s", exc)
            return False

    def require(self) -> None:
        """Raise :class:`StoreUnavailable` unless the server answers."""
        if not self.ping():
            raise StoreUnavailable(
                f"MongoDB is not reachable at {settings.mongo_host}:"
                f"{settings.mongo_port} - start it with "
                "'docker compose -f docker/docker-compose.yml up -d mongodb'"
            )

    def server_version(self) -> str:
        return str(get_client(self._uri).server_info().get("version", "unknown"))

    # ---------------- indexes ----------------

    def ensure_indexes(self) -> list[str]:
        """Create every index the access patterns need. Idempotent."""
        pymongo = _require_pymongo()
        created = [
            self.collection.create_index(
                [("doc_id", pymongo.ASCENDING)], name="uq_doc_id", unique=True
            ),
            self.collection.create_index(
                [("site", pymongo.ASCENDING), ("processed_at", pymongo.DESCENDING)],
                name="ix_site_processed_at",
            ),
            self.collection.create_index(
                [("content_hash", pymongo.ASCENDING)], name="ix_content_hash"
            ),
            self.collection.create_index(
                [("stats.word_count", pymongo.ASCENDING)], name="ix_word_count"
            ),
            # One text index per collection is a MongoDB limit, so all the
            # searchable fields are combined into this single compound one.
            self.collection.create_index(
                [
                    ("title", pymongo.TEXT),
                    ("description", pymongo.TEXT),
                    ("headings", pymongo.TEXT),
                    ("keywords", pymongo.TEXT),
                    ("text", pymongo.TEXT),
                ],
                name="ix_fulltext",
                weights={
                    "title": 10,
                    "description": 5,
                    "headings": 4,
                    "keywords": 3,
                    "text": 1,
                },
                default_language="english",
                # By default MongoDB reads the stemming language from a field
                # literally named "language" - which in these documents holds
                # the HTML `lang` attribute, so a page declaring "en-us" (not
                # a language MongoDB knows) would be rejected on write. The
                # override is pointed at a field the application never sets,
                # so every document is simply stemmed as English.
                language_override="_text_language",
            ),
        ]
        logger.info("MongoDB indexes ready: %s", ", ".join(created))
        return created

    def indexes(self) -> list[dict[str, Any]]:
        return list(self.collection.list_indexes())

    # ---------------- writes ----------------

    @staticmethod
    def _as_record(document: Mapping[str, Any]) -> dict[str, Any]:
        """Copy the document and lift the processing timestamp to the top level.

        ``processed_at`` is duplicated out of ``processing`` because a
        compound index cannot be built efficiently over a field that is only
        reachable through a sub-document the application may reshape later.
        """
        record = dict(document)
        record["processed_at"] = record.get("processing", {}).get("processed_at")
        return record

    def upsert(self, document: Mapping[str, Any]) -> bool:
        """Insert or replace one document. Returns True when it was new."""
        record = self._as_record(document)
        result = self.collection.replace_one(
            {"doc_id": record["doc_id"]}, record, upsert=True
        )
        return result.upserted_id is not None

    def upsert_many(self, documents: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        """Write a batch in one round trip.

        ``ordered=False`` lets the remaining documents be written even when one
        of them is rejected, which keeps a single malformed page from
        discarding the whole batch.
        """
        if not documents:
            return {"inserted": 0, "modified": 0, "failed": 0}

        pymongo = _require_pymongo()
        operations = [
            pymongo.ReplaceOne(
                {"doc_id": record["doc_id"]}, record, upsert=True
            )
            for record in (self._as_record(d) for d in documents)
        ]

        try:
            result = self.collection.bulk_write(operations, ordered=False)
            return {
                "inserted": result.upserted_count,
                "modified": result.modified_count,
                "failed": 0,
            }
        except Exception as exc:  # noqa: BLE001
            details = getattr(exc, "details", None)
            if not isinstance(details, dict):
                raise
            # A partial failure: report what was written and how much was not.
            failed = len(details.get("writeErrors", []))
            logger.warning("Bulk write partially failed: %s document(s)", failed)
            return {
                "inserted": details.get("nUpserted", 0),
                "modified": details.get("nModified", 0),
                "failed": failed,
            }

    def delete_site(self, site: str) -> int:
        return self.collection.delete_many({"site": site}).deleted_count

    def drop(self) -> None:
        """Remove the whole collection, indexes included (used by the demo)."""
        self.collection.drop()
        self._collection = None
        logger.info("Collection %s dropped", self.collection_name)

    # ---------------- reads ----------------

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"doc_id": doc_id}, {"_id": 0})

    def get_by_url(self, url: str) -> dict[str, Any] | None:
        from src.processing.normalizer import document_id  # local: avoids a cycle

        return self.get(document_id(url))

    def count(self, site: str | None = None) -> int:
        return self.collection.count_documents({"site": site} if site else {})

    def search(
        self,
        query: str,
        *,
        site: str | None = None,
        limit: int = 10,
        projection: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search, answered by the ``ix_fulltext`` index.

        Results come back ordered by MongoDB's relevance score, which the
        index weights push towards title and keyword matches.
        """
        filters: dict[str, Any] = {"$text": {"$search": query}}
        if site:
            filters["site"] = site

        fields = {
            "_id": 0,
            "score": {"$meta": "textScore"},
            **{name: 1 for name in (projection or ("doc_id", "url", "site", "title"))},
        }
        cursor = (
            self.collection.find(filters, fields)
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )
        return list(cursor)

    def duplicates(self, limit: int = 20) -> list[dict[str, Any]]:
        """Groups of documents whose clean text is byte-identical."""
        pipeline = [
            {"$group": {"_id": "$content_hash", "count": {"$sum": 1},
                        "urls": {"$push": "$url"}}},
            {"$match": {"count": {"$gt": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return list(self.collection.aggregate(pipeline))

    def stats(self) -> dict[str, Any]:
        """Per-site aggregates over the stored documents."""
        pipeline = [
            {
                "$group": {
                    "_id": "$site",
                    "documents": {"$sum": 1},
                    "words": {"$sum": "$stats.word_count"},
                    "chars": {"$sum": "$stats.char_count"},
                    "raw_bytes": {"$sum": "$source.raw_length"},
                    "avg_reduction": {"$avg": "$processing.reduction_ratio"},
                    "avg_words": {"$avg": "$stats.word_count"},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        per_site = [
            {
                "site": row["_id"],
                "documents": row["documents"],
                "words": int(row.get("words") or 0),
                "chars": int(row.get("chars") or 0),
                "raw_bytes": int(row.get("raw_bytes") or 0),
                "avg_reduction_ratio": round(row.get("avg_reduction") or 0.0, 4),
                "avg_words": int(round(row.get("avg_words") or 0)),
            }
            for row in self.collection.aggregate(pipeline)
        ]
        return {
            "database": self.database_name,
            "collection": self.collection_name,
            "sites": per_site,
            "totals": {
                "documents": sum(s["documents"] for s in per_site),
                "words": sum(s["words"] for s in per_site),
                "raw_bytes": sum(s["raw_bytes"] for s in per_site),
            },
            "indexes": [index["name"] for index in self.indexes()],
        }

    # ---------------- read side (used by the Task 5 API) ----------------

    def list_documents(
        self,
        *,
        site: str | None = None,
        enriched: bool | None = None,
        min_words: int | None = None,
        limit: int = 20,
        offset: int = 0,
        projection: Iterable[str] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Page through the processed documents, newest first.

        The sort is ``site + processed_at`` when a site filter is given, which
        is exactly the compound index created by :meth:`ensure_indexes`, so a
        listing is served from the index rather than by scanning and sorting
        the collection in memory.

        Returns ``(documents, total)``; the total is what lets the API tell a
        client how many pages of results exist.
        """
        filters: dict[str, Any] = {}
        if site:
            filters["site"] = site
        if enriched is True:
            filters["enrichment.summary"] = {"$exists": True, "$ne": ""}
        elif enriched is False:
            filters["$or"] = [
                {"enrichment": {"$exists": False}},
                {"enrichment.summary": ""},
            ]
        if min_words is not None:
            filters["stats.word_count"] = {"$gte": min_words}

        fields: dict[str, Any] = {"_id": 0}
        if projection:
            fields.update({name: 1 for name in projection})

        total = self.collection.count_documents(filters)
        cursor = (
            self.collection.find(filters, fields)
            .sort([("processing.processed_at", -1), ("doc_id", 1)])
            .skip(max(offset, 0))
            .limit(max(limit, 0))
        )
        return list(cursor), total

    def get_enrichment(self, doc_id: str) -> dict[str, Any] | None:
        """The enrichment sub-document only, or None when not yet generated."""
        document = self.collection.find_one(
            {"doc_id": doc_id}, {"_id": 0, "doc_id": 1, "url": 1, "enrichment": 1}
        )
        if document is None or not document.get("enrichment"):
            return None
        return document

    def sites(self) -> list[str]:
        """Every site present in the collection, for the filter documentation."""
        return sorted(str(site) for site in self.collection.distinct("site"))
