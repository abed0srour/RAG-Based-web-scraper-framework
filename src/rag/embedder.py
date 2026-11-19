"""Stage 2 of the RAG module: turning text into vectors.

Retrieval in this framework is *semantic*: a query matches a passage when the
two mean the same thing, not when they share the same words. That requires an
embedding model - a function from text to a fixed-length vector whose geometry
encodes meaning, so that "how much does the blue widget cost" lands near "the
blue widget costs £9.99" even with no word in common.

**Why this model.** ``BAAI/bge-small-en-v1.5`` is run locally through
`fastembed <https://github.com/qdrant/fastembed>`_, which executes the model
as an ONNX graph on the CPU. That choice follows from three constraints of
this project:

* *Free.* A hosted embedding API would charge per token for every chunk of
  every page, on every re-index. This model costs nothing to run and needs no
  API key, which is what makes the whole indexing stage repeatable.
* *No heavyweight dependency.* fastembed pulls in ONNX Runtime (~15 MB), not
  PyTorch (~2.5 GB), so the worker image stays small and installs on Windows
  without a build toolchain.
* *Small vectors.* 384 dimensions, versus 1536 for the common hosted models.
  The index is a quarter of the size and nearest-neighbour search is
  correspondingly faster, at a quality level that is more than sufficient for
  retrieving the right page out of a few thousand.

**Normalised vectors.** The model's output is L2-normalised here, which makes
the cosine similarity between two vectors equal to their dot product and puts
every distance Chroma reports on the same ``0 .. 2`` scale. The retriever
converts those distances into a ``0 .. 1`` relevance score.

**Asymmetric encoding.** BGE models are trained with an instruction prefix on
the *query* side only: a question and the passage answering it are not the
same kind of text, so they are encoded slightly differently.
:meth:`Embedder.embed_query` adds that prefix, :meth:`Embedder.embed_texts`
does not.

The model is the quantised ONNX build (~33 MB of weights plus its
tokenizer), downloaded once into
:data:`~src.common.config.Settings.embedding_cache_dir` and reused from disk
afterwards, so only the very first run needs network access.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.common.config import settings

logger = logging.getLogger(__name__)

# Instruction prefix recommended by the BGE authors for the query side of an
# asymmetric search. Applied only to English BGE models.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding model cannot be loaded or executed."""


@dataclass
class EmbeddingStats:
    """How much work the embedder did, for the indexer's report."""

    texts: int = 0
    batches: int = 0
    chars: int = 0
    duration_ms: float = 0.0

    @property
    def texts_per_second(self) -> float:
        return (self.texts / (self.duration_ms / 1000.0)) if self.duration_ms else 0.0

    @property
    def ms_per_text(self) -> float:
        return (self.duration_ms / self.texts) if self.texts else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "texts": self.texts,
            "batches": self.batches,
            "chars": self.chars,
            "duration_ms": round(self.duration_ms, 2),
            "ms_per_text": round(self.ms_per_text, 3),
            "texts_per_second": round(self.texts_per_second, 1),
        }


def _prepare_download_environment() -> None:
    """Make the one-off model download work on a plain Windows machine.

    Two defaults of the Hugging Face downloader cause avoidable failures here,
    and both are set only when the operator has not already chosen a value:

    ``HF_HUB_DISABLE_XET``
        The newer Xet CDN is served from a host that several networks (and
        this project's own test machine) cannot resolve, which turns the
        download into a loop of read timeouts. Disabling it falls back to the
        standard CDN, which works.
    ``HF_HUB_DISABLE_SYMLINKS_WARNING``
        Windows without developer mode cannot create the cache's symlinks.
        The download still succeeds; only the warning is noise.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _load_model(model_name: str, cache_dir: str) -> Any:
    """Load (once per process) the fastembed model behind ``model_name``."""
    key = (model_name, cache_dir)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    _prepare_download_environment()

    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise EmbeddingUnavailable(
            "fastembed is not installed; run 'pip install -r requirements.txt'"
        ) from exc

    started = time.perf_counter()
    try:
        model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 - any load failure is fatal here
        raise EmbeddingUnavailable(
            f"could not load embedding model {model_name!r} "
            f"(cache_dir={cache_dir!r}): {exc}. The first run downloads about "
            "45 MB, so it needs network access once."
        ) from exc

    logger.info(
        "Loaded embedding model %s in %.1f s",
        model_name, time.perf_counter() - started,
    )
    _MODEL_CACHE[key] = model
    return model


def reset_model_cache() -> None:
    """Forget the loaded models (used by the tests)."""
    _MODEL_CACHE.clear()


def _normalise(vector: Sequence[float]) -> list[float]:
    """Return the L2-normalised copy of ``vector`` as plain floats."""
    total = sum(value * value for value in vector) ** 0.5
    if not total:
        return [float(value) for value in vector]
    return [float(value) / total for value in vector]


class Embedder:
    """A loaded embedding model, plus batching and statistics."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache_dir: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.cache_dir = cache_dir or str(settings.embedding_cache_dir)
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model: Any | None = None
        self._dimension: int | None = None
        self.stats = EmbeddingStats()

    # ---------------- model ----------------

    @property
    def model(self) -> Any:
        """The underlying fastembed model, loaded on first use."""
        if self._model is None:
            self._model = _load_model(self.model_name, self.cache_dir)
        return self._model

    @property
    def dimension(self) -> int:
        """The vector length, read from the model description.

        The vector store needs this before the first document is indexed, so
        it comes from the model registry rather than from an actual call.
        """
        if self._dimension is None:
            try:
                from fastembed import TextEmbedding

                for description in TextEmbedding.list_supported_models():
                    if description.get("model") == self.model_name:
                        self._dimension = int(description["dim"])
                        break
            except ImportError:  # pragma: no cover - dependency is pinned
                self._dimension = None
            if self._dimension is None:
                # Fall back to measuring it, which also loads the model.
                self._dimension = len(self.embed_query("dimension probe"))
        return int(self._dimension)

    @property
    def uses_query_prefix(self) -> bool:
        name = self.model_name.lower()
        return "bge" in name and "-en" in name

    def describe(self) -> dict[str, Any]:
        """Model identity, for the report and for the collection metadata."""
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "query_prefix": self.uses_query_prefix,
            "runtime": "onnxruntime (CPU)",
            "cache_dir": self.cache_dir,
        }

    # ---------------- embedding ----------------

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages (the document side of the index)."""
        if not texts:
            return []

        started = time.perf_counter()
        try:
            vectors = list(self.model.embed(list(texts), batch_size=self.batch_size))
        except EmbeddingUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingUnavailable(f"embedding failed: {exc}") from exc
        elapsed = (time.perf_counter() - started) * 1000.0

        self.stats.texts += len(texts)
        self.stats.batches += 1
        self.stats.chars += sum(len(text) for text in texts)
        self.stats.duration_ms += elapsed

        return [_normalise(vector.tolist()) for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed one query, with the model's query-side instruction prefix."""
        text = f"{_BGE_QUERY_PREFIX}{query}" if self.uses_query_prefix else query
        vectors = self.embed_texts([text])
        if not vectors:
            raise EmbeddingUnavailable("the embedding model returned no vector")
        return vectors[0]

    def embed_batches(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> Iterable[tuple[list[str], list[list[float]]]]:
        """Yield ``(texts, vectors)`` per batch, so callers can stream writes."""
        size = batch_size or self.batch_size
        for start in range(0, len(texts), size):
            window = list(texts[start : start + size])
            yield window, self.embed_texts(window)


_DEFAULT: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the process-wide embedder, so the model is loaded only once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Embedder()
    return _DEFAULT


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, used by the tests and the retrieval diagnostics."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)
