"""Full-text search over the processed documents (Task 6).

The module that backs the search box in the web UI. :mod:`src.search.index`
owns the Elasticsearch index - its mapping, the bulk write path from MongoDB
and the queries - and :mod:`src.search.report` prints what is in it.

Why a fourth store, when MongoDB already has a text index: the two answer
different questions. MongoDB's text index answers "which documents contain
these words". A search *interface* needs the ranked hits, the highlighted
snippet that shows why each hit matched, the facet counts that let a user
narrow by site, and the as-you-type suggestions - all in one round trip.
That is what Elasticsearch is built for, and ``src/search/index.py`` explains
each of those pieces where it builds them.
"""

from src.search.index import (
    SearchIndex,
    SearchUnavailable,
    get_search_index,
    reset_client,
)

__all__ = [
    "SearchIndex",
    "SearchUnavailable",
    "get_search_index",
    "reset_client",
]
