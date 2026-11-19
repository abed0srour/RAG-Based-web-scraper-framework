"""The four router modules that make up the REST surface.

``raw``        the crawl exactly as it was fetched (PostgreSQL)
``documents``  the processed and RAG-enhanced documents (MongoDB)
``search``     retrieval over the indexed corpus, and the RAG answer endpoint
``system``     health, readiness, metrics and caller identity
"""

from src.api.routers import documents, raw, search, system

__all__ = ["documents", "raw", "search", "system"]
