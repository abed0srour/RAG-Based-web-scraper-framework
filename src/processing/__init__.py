"""Data-processing module (Task 3).

Turns the raw HTML stored by the scraping module into clean, normalised JSON
documents and indexes them in MongoDB:

    raw_pages (PostgreSQL)
        -> cleaner.clean_html        strip markup, scripts, boilerplate
        -> normalizer.normalize_page canonical JSON document
        -> store.DocumentStore       MongoDB, indexed for retrieval

``pipeline.ProcessingPipeline`` runs the three stages over a crawl, and
``benchmark`` measures their throughput over different data samples.
"""

from __future__ import annotations

from src.processing.cleaner import (
    CLEANER_VERSION,
    CleanedText,
    clean_html,
    clean_value,
    normalise_characters,
    normalise_whitespace,
    strip_tags,
)
from src.processing.normalizer import (
    SCHEMA_VERSION,
    NormalizationError,
    canonical_url,
    document_id,
    normalize_page,
    to_json,
    validate_document,
)
from src.processing.store import DocumentStore, StoreUnavailable

# ``pipeline`` and ``benchmark`` are deliberately NOT imported here: both are
# runnable as ``python -m src.processing.<module>``, and importing them from
# the package __init__ would make the interpreter load them twice.

__all__ = [
    "CLEANER_VERSION",
    "SCHEMA_VERSION",
    "CleanedText",
    "DocumentStore",
    "NormalizationError",
    "StoreUnavailable",
    "canonical_url",
    "clean_html",
    "clean_value",
    "document_id",
    "normalise_characters",
    "normalise_whitespace",
    "normalize_page",
    "strip_tags",
    "to_json",
    "validate_document",
]
