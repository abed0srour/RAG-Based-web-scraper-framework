"""The five router modules that make up the REST surface.

``raw``        the crawl exactly as it was fetched (PostgreSQL)
``documents``  the processed and RAG-enhanced documents (MongoDB)
``search``     retrieval over the indexed corpus, and the RAG answer endpoint
``fulltext``   full-text search with highlights and facets (Elasticsearch)
``system``     health, readiness, metrics and caller identity
"""

from src.api.routers import documents, fulltext, raw, search, system

__all__ = ["documents", "fulltext", "raw", "search", "system"]
