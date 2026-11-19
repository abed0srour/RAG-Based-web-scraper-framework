"""Stage 1 of the RAG module: splitting documents into retrievable chunks.

A whole web page is the wrong unit for retrieval. Pages in this corpus run
from a few dozen to several thousand words, and an embedding is a single
fixed-length vector: the longer the text behind it, the more topics are
averaged into the same point in the vector space, until the vector no longer
resembles any of them. Feeding whole pages to the LLM has the mirror problem -
most of the prompt is text that has nothing to do with the question.

So each normalised document is cut into **chunks**: passages of roughly
:data:`~src.common.config.Settings.chunk_words` words, which is what actually
gets embedded, stored and retrieved.

Three decisions matter here.

**Split on paragraph boundaries, not on a character count.** The cleaner from
Task 3 already emits one logical block per line, so paragraphs are free to
recover. A fixed character window would cut sentences in half and leave the
two halves embedded as fragments; packing whole paragraphs up to a word budget
keeps every chunk a readable passage.

**Overlap consecutive chunks.** The last
:data:`~src.common.config.Settings.chunk_overlap_words` words of a chunk are
repeated at the start of the next one. Without the overlap, a fact that
straddles a boundary belongs to neither chunk, and a query about it matches
neither well. The cost is ~20% more vectors, which is cheap compared with
missing an answer.

**Carry the page context into the embedded text.** A chunk taken from the
middle of a page often contains no clue which page it came from (a price with
no product name, a quote with no author). :attr:`Chunk.embedding_text`
therefore prefixes the site, the title and the nearest preceding heading to
the passage before it is embedded, so the vector reflects the passage *in its
context*. The stored ``text`` stays clean, because that is what is later shown
to the LLM and to the user.

Chunk identifiers are deterministic - ``<doc_id>:<nnnn>`` - which is what makes
re-indexing a page an upsert rather than a duplicate, exactly like the
``doc_id`` uniqueness in the MongoDB store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.common.config import settings

# A line that looks like a heading: the cleaner keeps headings on their own
# line, and the normaliser also stores them separately in ``headings``.
_WORD_RE = re.compile(r"\S+")

# Sentence boundary: terminal punctuation followed by whitespace and the start
# of something that looks like a new sentence. Deliberately conservative -
# scraped text is full of abbreviations and prices.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    """One retrievable passage, plus everything needed to cite it."""

    chunk_id: str
    doc_id: str
    index: int
    text: str
    url: str = ""
    site: str = ""
    title: str = ""
    heading: str = ""
    word_count: int = 0
    char_count: int = 0

    def __post_init__(self) -> None:
        if not self.word_count:
            self.word_count = len(_WORD_RE.findall(self.text))
        if not self.char_count:
            self.char_count = len(self.text)

    @property
    def embedding_text(self) -> str:
        """The text that is actually embedded: the passage in its context.

        The header is kept short and stable - a long header would start to
        dominate the vector of a short chunk.
        """
        header_parts = [part for part in (self.site, self.title, self.heading) if part]
        header = " | ".join(dict.fromkeys(header_parts))
        return f"{header}\n\n{self.text}" if header else self.text

    def metadata(self) -> dict[str, Any]:
        """The flat metadata record stored next to the vector.

        Chroma only accepts scalars in metadata, so everything here is a
        string, an int or a float - no nested documents, no ``None``.
        """
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.index,
            "url": self.url,
            "site": self.site,
            "title": self.title,
            "heading": self.heading,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }

    def as_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "text": self.text, **self.metadata()}


@dataclass
class ChunkStats:
    """Counters for one chunking run, used by the indexer's report."""

    documents: int = 0
    documents_skipped: int = 0
    chunks: int = 0
    words: int = 0
    sizes: list[int] = field(default_factory=list)

    @property
    def avg_words(self) -> float:
        return (sum(self.sizes) / len(self.sizes)) if self.sizes else 0.0

    @property
    def chunks_per_document(self) -> float:
        return (self.chunks / self.documents) if self.documents else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "documents_skipped": self.documents_skipped,
            "chunks": self.chunks,
            "words": self.words,
            "avg_words_per_chunk": round(self.avg_words, 1),
            "chunks_per_document": round(self.chunks_per_document, 2),
            "min_words": min(self.sizes) if self.sizes else 0,
            "max_words": max(self.sizes) if self.sizes else 0,
        }


def split_paragraphs(text: str) -> list[str]:
    """Split clean text into paragraph-sized blocks.

    The Task 3 cleaner emits one block per line and collapses runs of blank
    lines, so splitting on newlines recovers the original block structure.
    """
    blocks: list[str] = []
    for raw_block in re.split(r"\n\s*\n|\n", text or ""):
        block = raw_block.strip()
        if block:
            blocks.append(block)
    return blocks


def _looks_like_heading(block: str, headings: Sequence[str]) -> bool:
    """True when a block is one of the document's own headings."""
    if not block or len(block) > 300:
        return False
    return block in headings


def _tail_words(words: Sequence[str], count: int) -> list[str]:
    return list(words[-count:]) if count > 0 else []


def _split_long_block(block: str, budget: int) -> list[str]:
    """Break a paragraph that is longer than the budget into sub-blocks.

    The split is made at sentence boundaries, so the pieces are still
    readable passages. A "sentence" that is itself longer than the budget -
    a table flattened into one line, a list with no punctuation - has no
    boundary to use, so it is cut on a word window as a last resort.
    """
    words = _WORD_RE.findall(block)
    if len(words) <= budget:
        return [block]

    pieces: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in _SENTENCE_SPLIT_RE.split(block):
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_words = _WORD_RE.findall(sentence)
        if len(sentence_words) > budget:
            if current:
                pieces.append(" ".join(current))
                current, current_words = [], 0
            for start in range(0, len(sentence_words), budget):
                pieces.append(" ".join(sentence_words[start : start + budget]))
            continue

        if current and current_words + len(sentence_words) > budget:
            pieces.append(" ".join(current))
            current, current_words = [], 0

        current.append(sentence)
        current_words += len(sentence_words)

    if current:
        pieces.append(" ".join(current))
    return pieces or [block]


def split_text(
    text: str,
    *,
    words_per_chunk: int | None = None,
    overlap_words: int | None = None,
    min_words: int | None = None,
    headings: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Split one text into ``(chunk_text, heading)`` pairs.

    Paragraphs are packed until the word budget is reached; the next chunk
    starts with the overlap carried from the tail of the previous one.

    Two cases need care, and both are about not letting one paragraph decide
    the shape of the index:

    * A paragraph *longer* than the budget (a wall-of-text page with no
      markup) is first broken at sentence boundaries by
      :func:`_split_long_block`, so one such paragraph cannot produce a single
      enormous chunk whose vector is an average of everything it says.
    * A chunk *shorter* than ``min_words`` is not flushed just because the
      next paragraph would overflow the budget. It keeps accumulating and
      overshoots slightly instead, because a 15-word chunk is too small to
      retrieve usefully, whereas a 200-word chunk where 180 was asked for
      costs nothing.
    """
    budget = words_per_chunk or settings.chunk_words
    overlap = overlap_words if overlap_words is not None else settings.chunk_overlap_words
    floor = min_words if min_words is not None else settings.chunk_min_words

    # An overlap at or above the budget would never advance the window.
    overlap = max(0, min(overlap, budget // 2))

    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_words = 0
    carried = 0           # words of the current chunk that came from the overlap
    heading = ""
    chunk_heading = ""

    def flush() -> None:
        nonlocal current, current_words, carried, chunk_heading
        if not current:
            return
        body = "\n".join(current).strip()
        # Drop a chunk that is only the carried overlap plus noise: it adds a
        # vector without adding any new text.
        if body and (current_words - carried) > 0:
            chunks.append((body, chunk_heading))
        current = []
        current_words = 0
        carried = 0
        chunk_heading = ""

    blocks: list[str] = []
    for block in split_paragraphs(text):
        if _looks_like_heading(block, headings):
            # A heading labels what follows, so it is remembered as context
            # and never emitted as a chunk of its own. The marker keeps the
            # heading attached to the right position in the block stream.
            blocks.append(f"\x00{block}")
            continue
        # The sub-blocks are sized for the space that is actually free in a
        # chunk: every chunk after the first one starts with the carried
        # overlap, so ``budget - overlap`` is what remains for new text.
        blocks.extend(_split_long_block(block, budget - overlap))

    for block in blocks:
        if block.startswith("\x00"):
            heading = block[1:]
            continue

        block_words = _WORD_RE.findall(block)
        if not block_words:
            continue

        if (
            current
            and current_words >= floor
            and current_words + len(block_words) > budget
        ):
            tail = _tail_words(_WORD_RE.findall("\n".join(current)), overlap)
            flush()
            if tail:
                current = [" ".join(tail)]
                current_words = len(tail)
                carried = len(tail)

        if not chunk_heading:
            chunk_heading = heading
        current.append(block)
        current_words += len(block_words)

        if current_words >= budget:
            tail = _tail_words(_WORD_RE.findall("\n".join(current)), overlap)
            flush()
            if tail:
                current = [" ".join(tail)]
                current_words = len(tail)
                carried = len(tail)
                chunk_heading = heading

    flush()

    # A single short chunk is still worth keeping (a one-paragraph page is a
    # legitimate document); the floor only drops trailing scraps of a page
    # that already produced real chunks.
    if len(chunks) > 1:
        chunks = [
            (body, head)
            for body, head in chunks
            if len(_WORD_RE.findall(body)) >= floor
        ] or chunks[:1]

    return chunks


def chunk_document(
    document: Mapping[str, Any],
    *,
    words_per_chunk: int | None = None,
    overlap_words: int | None = None,
    min_words: int | None = None,
) -> list[Chunk]:
    """Cut one normalised MongoDB document into :class:`Chunk` objects.

    The document is the schema produced by
    :func:`src.processing.normalizer.normalize_page`, so this function is the
    seam between Task 3's output and Task 4's index.
    """
    doc_id = str(document.get("doc_id") or "")
    if not doc_id:
        raise ValueError("a document needs a doc_id before it can be chunked")

    text = document.get("text") or ""
    headings = [h for h in (document.get("headings") or []) if isinstance(h, str)]
    title = (document.get("title") or "").strip()
    description = (document.get("description") or "").strip()

    # The description is page-level metadata, not body text, but it is often
    # the single most descriptive sentence a page has, so it is prepended as
    # the first block when the body does not already repeat it.
    if description and description not in text:
        text = f"{description}\n{text}"

    pieces = split_text(
        text,
        words_per_chunk=words_per_chunk,
        overlap_words=overlap_words,
        min_words=min_words,
        headings=headings,
    )

    chunks: list[Chunk] = []
    for index, (body, heading) in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{index:04d}",
                doc_id=doc_id,
                index=index,
                text=body,
                url=str(document.get("url") or ""),
                site=str(document.get("site") or ""),
                title=title,
                heading=heading or "",
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    stats: ChunkStats | None = None,
    **kwargs: Any,
) -> Iterator[Chunk]:
    """Chunk a stream of documents, accumulating counters into ``stats``."""
    counters = stats if stats is not None else ChunkStats()

    for document in documents:
        chunks = chunk_document(document, **kwargs)
        if not chunks:
            counters.documents_skipped += 1
            continue

        counters.documents += 1
        for chunk in chunks:
            counters.chunks += 1
            counters.words += chunk.word_count
            counters.sizes.append(chunk.word_count)
            yield chunk
