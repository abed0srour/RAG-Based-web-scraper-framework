"""Stage 3 of the RAG module: the vector database.

The embedded chunks are stored in **Chroma**, which is the third database in
this framework, and the split between the three is deliberate:

* *PostgreSQL* holds the crawl record - raw responses, jobs, the link graph.
  Relational data with fixed columns and foreign keys.
* *MongoDB* holds the processed documents - one JSON object per page, read
  back whole, with a full-text index for keyword search.
* *Chroma* holds the **indexed text representations** - one 384-dimensional
  vector per chunk, with the chunk text and its metadata alongside it.

Neither of the first two can answer the question Chroma exists for. A
relational index answers "which rows have this value"; MongoDB's text index
answers "which documents contain these words"; a vector index answers "which
passages are closest in meaning to this one", which is a nearest-neighbour
search in a 384-dimensional space and needs a purpose-built index (Chroma uses
HNSW) to run in milliseconds instead of scanning every vector.

Three design points worth stating:

**Cosine distance.** The collection is created with ``hnsw:space=cosine``,
matching the normalised vectors from
:mod:`src.rag.embedder`. Chroma returns a *distance*
(``0`` = identical, ``2`` = opposite); :func:`relevance_from_distance`
converts it into the ``0 .. 1`` relevance score the rest of the module uses.

**Upsert by deterministic id.** Chunk ids are ``<doc_id>:<nnnn>``, so
re-indexing a page overwrites its chunks instead of adding a second copy -
the same idempotency property the MongoDB store has, and necessary for the
same reason: the upstream queue delivers URLs at-least-once. Chunks left over
from a page that got *shorter* are removed explicitly by
:meth:`VectorStore.delete_document`, since an upsert alone cannot delete.

**Client/server, not embedded.** Chroma runs as its own container. Several
worker nodes have to write into the same index and the API has to read it, so
an in-process index in each worker would give each node a different, partial
view of the corpus.

Every call goes through one client per process and fails fast with
:class:`VectorStoreUnavailable` when the service is unreachable, so a missing
container produces one clear message rather than a hang.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from src.common.config import settings
from src.rag.chunker import Chunk

logger = logging.getLogger(__name__)

# Chroma's cosine distance runs 0 (identical) .. 2 (opposite).
_MAX_COSINE_DISTANCE = 2.0

_CLIENTS: dict[tuple[str, int], Any] = {}


class VectorStoreUnavailable(RuntimeError):
    """Raised when the vector database cannot be reached or used."""


def relevance_from_distance(distance: float | None) -> float:
    """Map a cosine distance onto a ``0 .. 1`` relevance score.

    Scores are what get shown, logged and compared against a threshold, and a
    score that rises with relevance is far easier to reason about than a
    distance that falls.
    """
    if distance is None:
        return 0.0
    score = 1.0 - (float(distance) / _MAX_COSINE_DISTANCE)
    return max(0.0, min(1.0, score))


def _require_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise VectorStoreUnavailable(
            "chromadb-client is not installed; run 'pip install -r requirements.txt'"
        ) from exc
    return chromadb


def get_client(host: str | None = None, port: int | None = None) -> Any:
    """Return the process-wide Chroma HTTP client for ``host:port``."""
    chromadb = _require_chromadb()
    key = (host or settings.chroma_host, port or settings.chroma_port)

    if key not in _CLIENTS:
        from chromadb.config import Settings as ChromaSettings

        # chromadb-client 0.5.23 calls its telemetry hook with an argument
        # list the bundled posthog build does not accept, and logs the
        # resulting TypeError at ERROR level on every client start. Telemetry
        # is switched off below, so the message is pure noise - and an ERROR
        # line in an otherwise clean run is actively misleading.
        logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

        _CLIENTS[key] = chromadb.HttpClient(
            host=key[0],
            port=key[1],
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.debug("Opened Chroma client for %s:%s", *key)
    return _CLIENTS[key]


def reset_client() -> None:
    """Drop the cached clients (used by the tests)."""
    _CLIENTS.clear()


class VectorStore:
    """Thin, typed wrapper around one Chroma collection."""

    def __init__(
        self,
        collection_name: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_collection
        self.host = host or settings.chroma_host
        self.port = port or settings.chroma_port
        self._client = client
        self._collection: Any | None = None

    # ---------------- connection ----------------

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = get_client(self.host, self.port)
        return self._client

    def ping(self) -> bool:
        """True when the Chroma server answers a heartbeat."""
        try:
            self.client.heartbeat()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Chroma heartbeat failed: %s", exc)
            return False

    def require(self) -> None:
        """Raise :class:`VectorStoreUnavailable` unless Chroma is reachable."""
        if not self.ping():
            raise VectorStoreUnavailable(
                f"Chroma is not reachable at {self.host}:{self.port}. Start it "
                "with 'docker compose -f docker/docker-compose.yml up -d chroma'."
            )

    def server_version(self) -> str:
        try:
            return str(self.client.get_version())
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"could not read the Chroma version: {exc}") from exc

    # ---------------- collection ----------------

    def ensure_collection(self, metadata: Mapping[str, Any] | None = None) -> Any:
        """Create the collection if it does not exist, and return it.

        ``embedding_function=None`` is deliberate: the vectors are computed by
        :mod:`src.rag.embedder` and passed in explicitly, so Chroma must not
        try to embed anything itself (its default function would download a
        second, different model and silently produce an index whose vectors do
        not match the ones used at query time).
        """
        if self._collection is not None:
            return self._collection

        self.require()
        descriptor = {
            # hnsw:space is what makes the reported distance a cosine
            # distance; it can only be set at creation time.
            "hnsw:space": "cosine",
            **{str(k): v for k, v in (metadata or {}).items()},
        }
        try:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=descriptor,
                embedding_function=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(
                f"could not open collection {self.collection_name!r}: {exc}"
            ) from exc
        return self._collection

    @property
    def collection(self) -> Any:
        return self.ensure_collection()

    def drop(self) -> None:
        """Delete the whole collection, so the next run rebuilds it."""
        self.require()
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("delete_collection(%s): %s", self.collection_name, exc)
        self._collection = None

    # ---------------- writing ----------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Write one batch of chunks and their vectors; returns the count.

        The chunk *text* is stored next to the vector on purpose. Retrieval
        then needs a single round trip - a nearest-neighbour search that
        returns the passages themselves - instead of a search for ids
        followed by a second lookup in MongoDB for each hit.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        try:
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=[list(vector) for vector in embeddings],
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata() for chunk in chunks],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"upsert of {len(chunks)} chunk(s) failed: {exc}") from exc
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of one document.

        Called before a page is re-indexed: an upsert would overwrite chunks
        ``0..n`` but leave behind any chunk ``n+1..m`` from a longer earlier
        version of the same page, which would keep answering queries with
        text the page no longer contains.
        """
        try:
            self.collection.delete(where={"doc_id": {"$eq": doc_id}})
        except Exception as exc:  # noqa: BLE001
            logger.debug("delete_document(%s): %s", doc_id, exc)

    def delete_site(self, site: str) -> None:
        """Remove every chunk belonging to one site."""
        try:
            self.collection.delete(where={"site": {"$eq": site}})
        except Exception as exc:  # noqa: BLE001
            logger.debug("delete_site(%s): %s", site, exc)

    # ---------------- reading ----------------

    def count(self) -> int:
        """Number of vectors in the collection."""
        try:
            return int(self.collection.count())
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"could not count the collection: {exc}") from exc

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        site: str | None = None,
        doc_id: str | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Nearest-neighbour search; returns hits ordered by relevance.

        The optional ``site``/``doc_id`` filters are *metadata* filters
        evaluated by Chroma, so a scoped query still only reads the matching
        subset of the index.
        """
        conditions: list[dict[str, Any]] = []
        if site:
            conditions.append({"site": {"$eq": site}})
        if doc_id:
            conditions.append({"doc_id": {"$eq": doc_id}})
        if where:
            conditions.append(dict(where))

        if not conditions:
            where_clause = None
        elif len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = {"$and": conditions}

        try:
            result = self.collection.query(
                query_embeddings=[list(embedding)],
                n_results=max(1, top_k),
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"vector query failed: {exc}") from exc

        return _flatten_query_result(result)

    def get_document_chunks(self, doc_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """Every stored chunk of one document, in order."""
        try:
            result = self.collection.get(
                where={"doc_id": {"$eq": doc_id}},
                limit=limit,
                include=["documents", "metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"could not read chunks of {doc_id!r}: {exc}") from exc

        rows = _flatten_get_result(result)
        rows.sort(key=lambda row: row.get("chunk_index", 0))
        return rows

    def peek(self, limit: int = 3) -> list[dict[str, Any]]:
        """A few stored records, for the report and the demonstration."""
        try:
            return _flatten_get_result(
                self.collection.get(limit=limit, include=["documents", "metadatas"])
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"could not read the collection: {exc}") from exc

    def stats(self, *, sample: int = 10000) -> dict[str, Any]:
        """Index-level aggregates: vectors, documents and per-site counts.

        Chroma has no aggregation pipeline, so the metadata is pulled and
        counted here. ``sample`` caps that read - this is a report, not a
        query path.
        """
        self.require()
        total = self.count()

        try:
            result = self.collection.get(limit=sample, include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"could not read the collection metadata: {exc}") from exc

        metadatas = [row or {} for row in (result.get("metadatas") or [])]
        per_site: Counter[str] = Counter()
        documents: set[str] = set()
        words = 0

        for row in metadatas:
            per_site[str(row.get("site") or "unknown")] += 1
            if row.get("doc_id"):
                documents.add(str(row["doc_id"]))
            words += int(row.get("word_count") or 0)

        collection_metadata = {}
        try:
            collection_metadata = dict(self.collection.metadata or {})
        except Exception:  # noqa: BLE001 - metadata is a nicety, not a result
            pass

        return {
            "collection": self.collection_name,
            "endpoint": f"{self.host}:{self.port}",
            "vectors": total,
            "sampled": len(metadatas),
            "documents": len(documents),
            "chunks_per_document": round(len(metadatas) / len(documents), 2) if documents else 0.0,
            "words_indexed": words,
            "avg_words_per_chunk": round(words / len(metadatas), 1) if metadatas else 0.0,
            "per_site": [
                {"site": site, "chunks": count} for site, count in sorted(per_site.items())
            ],
            "metadata": collection_metadata,
        }


# ---------------------------------------------------------------------------
# Result shaping
#
# Chroma returns parallel lists ("ids": [[...]], "documents": [[...]], ...),
# one outer entry per query. Everything downstream wants one dict per hit, so
# the conversion happens once, here.
# ---------------------------------------------------------------------------


def _flatten_query_result(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[dict[str, Any]] = []
    for position, chunk_id in enumerate(ids):
        metadata = dict(metadatas[position] or {}) if position < len(metadatas) else {}
        distance = distances[position] if position < len(distances) else None
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": documents[position] if position < len(documents) else "",
                "distance": float(distance) if distance is not None else None,
                "score": relevance_from_distance(distance),
                "rank": position + 1,
                "source": "vector",
                **metadata,
            }
        )
    return hits


def _flatten_get_result(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    rows: list[dict[str, Any]] = []
    for position, chunk_id in enumerate(ids):
        metadata = dict(metadatas[position] or {}) if position < len(metadatas) else {}
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": documents[position] if position < len(documents) else "",
                **metadata,
            }
        )
    return rows


def iter_batches(items: Sequence[Any], size: int) -> Iterable[list[Any]]:
    """Yield ``items`` in lists of at most ``size`` (used by the indexer)."""
    for start in range(0, len(items), max(1, size)):
        yield list(items[start : start + size])
