# Web UI

The React interface for the distributed RAG web-scraper framework (Task 6).

## Running it

The UI is a client of the API, so the API has to be up:

```bash
# 1. the backing services
docker compose -f ../docker/docker-compose.yml up -d

# 2. the full-text index, built from the processed documents
cd .. && python -m src.search.indexer && cd ui

# 3. the API  (port 8080)
cd .. && python -m src.api & cd ui

# 4. this UI   (port 5173)
npm install
npm run dev
```

Then open <http://localhost:5173>.

| Command | What it does |
|---------|--------------|
| `npm run dev` | Dev server with hot reload, proxying `/v1` and `/health` to `localhost:8080` |
| `npm run build` | Type-check and build a static bundle into `dist/` |
| `npm run preview` | Serve the built bundle |
| `npm run typecheck` | Type-check only |

## Configuration

Copy `.env.example` to `.env` to change either setting:

| Variable | Default | Meaning |
|----------|---------|---------|
| `VITE_API_BASE` | *(empty)* | Where the API is. Empty means same-origin, which the dev-server proxy forwards - the simplest setup. Set an absolute URL when the built bundle is served from elsewhere. |
| `VITE_API_KEY` | `dev-analyst-key` | The key sent as `X-API-Key`. It needs the `raw`, `documents`, `search` and `ask` scopes. |

A key can also be typed into the status bar at runtime, where it is kept in
`localStorage`. Note that **anything in a front-end bundle is public**: this
is a convenience for local use, not a secret store. A deployment that needs
per-user access puts a session-issuing service in front of the API rather
than shipping a key to the browser.

## Layout

```
src/
├── main.tsx                 Entry point
├── App.tsx                  Shell: three tabs, search state, paging
├── styles.css               Tokens on :root, redefined for dark mode
├── lib/
│   ├── api.ts               The one place that calls fetch
│   ├── types.ts             Mirrors src/api/schemas.py
│   └── highlight.tsx        Renders <mark> fragments without injecting HTML
├── hooks/
│   └── useDebouncedValue.ts
└── components/
    ├── SearchBar.tsx        Input, suggestions, keyboard navigation
    ├── Facets.tsx           Filter sidebar driven by the facet counts
    ├── ResultCard.tsx       One hit: marked title, snippet, metadata
    ├── DocumentDrawer.tsx   Detail panel (MongoDB) + related pages (Elastic)
    ├── AskPanel.tsx         The RAG tab, with citations
    ├── CrawlPanel.tsx       The raw PostgreSQL crawl record
    └── StatusBar.tsx        Readiness polling and the API key
```

## Two things to know before changing it

**Highlight fragments are not HTML.** Elasticsearch wraps matched terms in
`<mark>` inside text that came from a crawled page, and escapes nothing else
in it. `lib/highlight.tsx` splits on the markers and returns React nodes, so
the rest stays a text node that React escapes. Passing a fragment to
`dangerouslySetInnerHTML` would be stored cross-site scripting whose payload
is supplied by whichever site was scraped.

**Every request is abortable, and the search effect aborts the previous
one.** Typing is many state changes per second; without cancellation a slow
response for an earlier keystroke can arrive last and overwrite the current
results. An aborted request is reported as such rather than as a failure, so
a replaced keystroke never flashes an error.
