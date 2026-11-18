"""Stage 2 of the data-processing module: normalisation to structured JSON.

Cleaning produces text. Normalisation produces a **document**: one JSON object
per page, with the same keys and the same types whatever site it came from.
That uniformity is the whole point of the stage - the RAG layer, the API and
the MongoDB indexes all address a single schema, and adding a new target
website cannot change the shape of the data they read.

The canonical document looks like this (abbreviated)::

    {
      "doc_id": "9f2c1e...",              # deterministic, derived from the URL
      "schema_version": "1.0",
      "url": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000",
      "site": "books.toscrape.com",
      "path": "/catalogue/a-light-in-the-attic_1000",
      "title": "A Light in the Attic | Books to Scrape",
      "description": "...",
      "language": "en",
      "headings": ["A Light in the Attic", "Product Description"],
      "text": "A Light in the Attic\\nIt's hard to imagine a world ...",
      "keywords": ["books", "light", "attic", ...],
      "fields": {"price": "£51.77", "rating": "Three", ...},
      "stats": {"word_count": 412, "char_count": 2671, "reading_time_sec": 112, ...},
      "source": {"raw_page_id": 17, "http_status": 200, "fetched_at": "...", ...},
      "processing": {"cleaner_version": "1.0", "processed_at": "...", ...}
    }

Three design decisions are worth stating:

* **The identifier is derived, not generated.** ``doc_id`` is a SHA-1 over the
  canonical URL, so re-processing the same page always produces the same key
  and the write into MongoDB is an idempotent upsert rather than a duplicate.
  A pipeline that is retried after a crash therefore converges.
* **Site-specific data is quarantined under one key.** Everything that differs
  per website lives in ``fields``; the top level is identical for every
  document. MongoDB is schema-less, which makes it tempting to let each site
  add its own top-level keys - that would push the per-site knowledge into
  every consumer of the data.
* **`content_hash` is computed over the clean text**, not the HTML, so a page
  whose markup changed but whose content did not is recognised as unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse

from src.processing.cleaner import (
    CLEANER_VERSION,
    CleanedText,
    clean_html,
    clean_value,
    normalise_characters,
    normalise_whitespace,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# Average adult reading speed, used for the ``reading_time_sec`` statistic.
WORDS_PER_MINUTE = 220

# Keys every document must carry, with the type each one must have. The
# validator below enforces this, so a malformed document is rejected *before*
# it reaches the database rather than being discovered by a reader later.
REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "doc_id": str,
    "schema_version": str,
    "url": str,
    "site": str,
    "title": (str, type(None)),
    "text": str,
    "headings": list,
    "keywords": list,
    "fields": dict,
    "stats": dict,
    "source": dict,
    "processing": dict,
}

_WORD = re.compile(r"[a-z][a-z'\-]{2,}")

# Small English stop-word list. A full NLP stack is deliberately avoided here:
# the keywords are a cheap summary for faceting and for the MongoDB text
# index, not a linguistic analysis.
STOPWORDS = frozenset(
    """
    the be to of and a in that have it for not on with he as you do at this but
    his by from they we say her she or an will my one all would there their what
    so up out if about who get which go me when make can like time no just him
    know take people into year your good some could them see other than then now
    look only come its over think also back after use two how our work first well
    way even new want because any these give day most us is are was were been has
    had does did doing i am more such very own same too each few own off here both
    through during before above below again further once while why where under
    """.split()
)


class NormalizationError(ValueError):
    """Raised when a document cannot be built or fails validation."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def canonical_url(url: str) -> str:
    """Normalise a URL so that the same page yields the same identifier.

    The scheme and host are lower-cased, the default port and the fragment are
    dropped, and a trailing slash is removed from non-root paths. Without this
    the same page reached as ``HTTP://Host/a/`` and ``http://host/a`` would be
    stored twice.
    """
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()
    if parsed.port and parsed.port not in (80, 443):
        host = f"{host}:{parsed.port}"
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse(
        ((parsed.scheme or "https").lower(), host, path, "", parsed.query, "")
    )


def document_id(url: str) -> str:
    """Deterministic primary key for a page: SHA-1 of its canonical URL."""
    return hashlib.sha1(canonical_url(url).encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """SHA-256 of the clean text - the change detector for a page."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def site_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def extract_keywords(text: str, limit: int = 15) -> list[str]:
    """Most frequent meaningful words, used for faceting and search boosting."""
    if not text:
        return []
    counts = Counter(
        word
        for word in _WORD.findall(text.lower())
        if word not in STOPWORDS and len(word) > 2
    )
    return [word for word, _count in counts.most_common(limit)]


def text_statistics(text: str, headings: list[str]) -> dict[str, Any]:
    """Size and shape measurements that make documents comparable."""
    words = text.split()
    lines = [line for line in text.splitlines() if line.strip()]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    word_count = len(words)
    unique = len({w.lower().strip(".,;:!?\"'()[]") for w in words if w})

    return {
        "word_count": word_count,
        "char_count": len(text),
        "line_count": len(lines),
        "sentence_count": len(sentences),
        "heading_count": len(headings),
        "unique_word_count": unique,
        # How repetitive the page is: 1.0 = every word different.
        "lexical_diversity": round(unique / word_count, 4) if word_count else 0.0,
        "avg_word_length": (
            round(sum(len(w) for w in words) / word_count, 2) if word_count else 0.0
        ),
        "reading_time_sec": (
            int(round(word_count / WORDS_PER_MINUTE * 60)) if word_count else 0
        ),
    }


def _as_iso(value: Any) -> str | None:
    """Render a timestamp as an ISO-8601 string, whatever form it arrives in."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _clean_field(value: Any, *, limit: int | None = None) -> str | None:
    """Normalise one short scalar field (a title, a description)."""
    if value is None:
        return None
    text = normalise_whitespace(normalise_characters(str(value)))
    text = " ".join(text.split())
    if not text:
        return None
    return text[:limit] if limit else text


def normalize_page(
    *,
    url: str,
    raw_html: str | None = None,
    cleaned: CleanedText | None = None,
    site: str | None = None,
    title: str | None = None,
    description: str | None = None,
    language: str | None = None,
    headings: list[str] | None = None,
    structured_data: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    processed_by: str | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON document for one page.

    Either ``raw_html`` (which is cleaned here) or an already-cleaned
    :class:`~src.processing.cleaner.CleanedText` must be given. Accepting both
    is what lets the benchmark measure the two stages separately while the
    pipeline runs them together.
    """
    started = time.perf_counter()

    if cleaned is None:
        if raw_html is None:
            raise NormalizationError("normalize_page needs raw_html or cleaned text")
        cleaned = clean_html(raw_html)

    if not url:
        raise NormalizationError("a document must have a URL")

    canonical = canonical_url(url)
    parsed_url = urlparse(canonical)

    clean_headings: list[str] = []
    for heading in headings or []:
        value = _clean_field(heading, limit=300)
        if value and value not in clean_headings:
            clean_headings.append(value)

    # The title is often missing from the markup but present as the first
    # heading, so fall back to it rather than storing a null.
    clean_title = _clean_field(title, limit=500) or (
        clean_headings[0] if clean_headings else None
    )

    fields = clean_value(dict(structured_data or {}))
    stats = text_statistics(cleaned.text, clean_headings)

    document: dict[str, Any] = {
        "doc_id": document_id(canonical),
        "schema_version": SCHEMA_VERSION,
        "url": canonical,
        "original_url": url if url != canonical else None,
        "site": (site or site_of(canonical)).lower().removeprefix("www."),
        "path": parsed_url.path or "/",
        "query": parsed_url.query or None,
        "title": clean_title,
        "description": _clean_field(description, limit=2000),
        "language": _clean_field(language, limit=16),
        "headings": clean_headings,
        "text": cleaned.text,
        "keywords": extract_keywords(cleaned.text),
        "fields": fields if isinstance(fields, dict) else {},
        "content_hash": content_hash(cleaned.text),
        "stats": stats,
        "source": {
            "raw_page_id": (source or {}).get("raw_page_id"),
            "job_id": (source or {}).get("job_id"),
            "depth": (source or {}).get("depth", 0),
            "http_status": (source or {}).get("http_status"),
            "content_type": (source or {}).get("content_type"),
            "fetched_at": _as_iso((source or {}).get("fetched_at")),
            "fetched_by": (source or {}).get("fetched_by"),
            "raw_length": cleaned.raw_length,
        },
        "processing": {
            "schema_version": SCHEMA_VERSION,
            "cleaner_version": CLEANER_VERSION,
            "processed_at": utcnow().isoformat(),
            "processed_by": processed_by,
            **cleaned.summary(),
        },
    }
    document["processing"]["normalize_ms"] = round(
        (time.perf_counter() - started) * 1000.0, 3
    )
    return document


def validate_document(document: Mapping[str, Any]) -> None:
    """Check the document against :data:`REQUIRED_FIELDS` and JSON-encodability.

    Raises :class:`NormalizationError` with every problem found, so a bad
    document produces one actionable message instead of a cascade of failures
    further down the pipeline.
    """
    problems: list[str] = []

    for key, expected in REQUIRED_FIELDS.items():
        if key not in document:
            problems.append(f"missing field {key!r}")
        elif not isinstance(document[key], expected):
            names = (
                expected.__name__
                if isinstance(expected, type)
                else "/".join(t.__name__ for t in expected)
            )
            problems.append(
                f"field {key!r} should be {names}, got "
                f"{type(document[key]).__name__}"
            )

    if not problems:
        if not document["doc_id"]:
            problems.append("doc_id is empty")
        if document["schema_version"] != SCHEMA_VERSION:
            problems.append(
                f"schema_version {document['schema_version']!r} != {SCHEMA_VERSION!r}"
            )
        try:
            json.dumps(document, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            problems.append(f"document is not JSON-serialisable: {exc}")

    if problems:
        raise NormalizationError("; ".join(problems))


def to_json(document: Mapping[str, Any], *, indent: int | None = 2) -> str:
    """Serialise a document, with datetimes rendered as ISO-8601 strings."""
    return json.dumps(document, ensure_ascii=False, indent=indent, default=_as_iso)
