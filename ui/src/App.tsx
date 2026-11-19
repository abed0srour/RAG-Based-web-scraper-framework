import { useCallback, useEffect, useRef, useState } from "react";
import { AskPanel } from "./components/AskPanel";
import { CrawlPanel } from "./components/CrawlPanel";
import { DocumentDrawer } from "./components/DocumentDrawer";
import { Facets } from "./components/Facets";
import { ResultCard } from "./components/ResultCard";
import { SearchBar } from "./components/SearchBar";
import { StatusBar } from "./components/StatusBar";
import { useDebouncedValue } from "./hooks/useDebouncedValue";
import { ApiError, isAbort, search } from "./lib/api";
import type { FullTextResponse } from "./lib/types";

type Tab = "search" | "ask" | "crawl";

const PAGE_SIZE = 10;

const SORTS = [
  { value: "relevance", label: "Most relevant" },
  { value: "recent", label: "Newest first" },
  { value: "oldest", label: "Oldest first" },
  { value: "words", label: "Longest first" },
  { value: "title", label: "Title A–Z" },
];

/**
 * The application shell: three tabs over the same API.
 *
 * `search` is Elasticsearch, `ask` is the RAG read path, and `crawl` is the
 * raw PostgreSQL record. They are separate views because they answer
 * different questions over different stores, and keeping them separate is
 * what makes the architecture legible from the outside.
 *
 * Search runs as the query changes rather than on a button press. The
 * debounced query, the filters and the page are the inputs; one effect turns
 * them into a request and aborts the previous one, so the result on screen
 * is always the answer to the current inputs and never to a stale keystroke.
 */
export default function App() {
  const [tab, setTab] = useState<Tab>("search");

  const [input, setInput] = useState("");
  const query = useDebouncedValue(input, 250);

  const [site, setSite] = useState<string | undefined>();
  const [language, setLanguage] = useState<string | undefined>();
  const [minWords, setMinWords] = useState<number | undefined>();
  const [enriched, setEnriched] = useState<boolean | undefined>();
  const [sort, setSort] = useState("relevance");
  const [offset, setOffset] = useState(0);

  const [result, setResult] = useState<FullTextResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | string | null>(null);
  const [openDoc, setOpenDoc] = useState<string | null>(null);

  // Changing what is searched returns to the first page; changing the page
  // must not. Hence the separate effect rather than one dependency list.
  const filterKey = `${query}|${site}|${language}|${minWords}|${enriched}|${sort}`;
  const previousFilters = useRef(filterKey);
  useEffect(() => {
    if (previousFilters.current !== filterKey) {
      previousFilters.current = filterKey;
      setOffset(0);
    }
  }, [filterKey]);

  useEffect(() => {
    if (tab !== "search") return;

    const controller = new AbortController();
    setBusy(true);
    setError(null);

    search(
      { q: query, site, language, minWords, enriched, sort, limit: PAGE_SIZE, offset },
      controller.signal,
    )
      .then((value) => {
        setResult(value);
        setBusy(false);
      })
      .catch((cause) => {
        // An aborted request was replaced by a newer one: not an error, and
        // the newer one owns the busy flag now.
        if (isAbort(cause)) return;
        setError(cause instanceof ApiError ? cause : String(cause));
        setResult(null);
        setBusy(false);
      });

    return () => controller.abort();
  }, [tab, query, site, language, minWords, enriched, sort, offset]);

  const onSubmit = useCallback((value: string) => setInput(value), []);

  const total = result?.total ?? 0;
  const hits = result?.hits ?? [];
  const showingFrom = total === 0 ? 0 : offset + 1;
  const showingTo = Math.min(offset + PAGE_SIZE, total);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true" />
          <div>
            <h1>Scraped corpus</h1>
            <p>Distributed RAG web-scraper framework</p>
          </div>
        </div>

        <nav className="tabs" aria-label="Views">
          {([
            ["search", "Search"],
            ["ask", "Ask"],
            ["crawl", "Raw data"],
          ] as Array<[Tab, string]>).map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={tab === value ? "tab active" : "tab"}
              aria-current={tab === value}
              onClick={() => setTab(value)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {tab === "search" && (
        <>
          <div className="search-head">
            <SearchBar value={input} onChange={setInput} onSubmit={onSubmit} busy={busy} />
            <div className="search-controls">
              <label htmlFor="sort" className="sr-only">
                Sort order
              </label>
              <select
                id="sort"
                value={sort}
                onChange={(event) => setSort(event.target.value)}
              >
                {SORTS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <main className="layout">
            <Facets
              facets={result?.facets ?? {}}
              site={site}
              language={language}
              enriched={enriched}
              minWords={minWords}
              onSite={setSite}
              onLanguage={setLanguage}
              onEnriched={setEnriched}
              onMinWords={setMinWords}
            />

            <section className="results" aria-busy={busy}>
              <div className="results-head">
                <span className="muted">
                  {error
                    ? "—"
                    : total === 0
                      ? busy
                        ? "searching…"
                        : "no matches"
                      : `${total} match${total === 1 ? "" : "es"}` +
                        (total > PAGE_SIZE ? ` · showing ${showingFrom}–${showingTo}` : "")}
                </span>
                {result && !error && total > 0 && (
                  <span className="muted" title="Time inside Elasticsearch / full round trip">
                    {result.took_ms} ms in the cluster · {result.elapsed_ms} ms round trip
                  </span>
                )}
              </div>

              {error && (
                <div className="error-box">
                  <p className="error">
                    {error instanceof ApiError ? error.message : error}
                  </p>
                  {error instanceof ApiError && error.status === 403 && (
                    <p className="hint">
                      This key does not hold the <code>search</code> scope. Set a key that
                      does, at the bottom of the page.
                    </p>
                  )}
                  {error instanceof ApiError && error.type === "search_index_unavailable" && (
                    <p className="hint">
                      Elasticsearch is not reachable. Start it with
                      <code> docker compose -f docker/docker-compose.yml up -d elasticsearch</code>
                      and build the index with
                      <code> python -m src.search.indexer</code>.
                    </p>
                  )}
                  {error instanceof ApiError && error.requestId && (
                    <p className="hint">Request id: {error.requestId}</p>
                  )}
                </div>
              )}

              {result?.notes?.map((note) => (
                <p className="note" key={note}>
                  {note}
                </p>
              ))}

              {hits.map((hit) => (
                <ResultCard key={hit.doc_id} hit={hit} onOpen={setOpenDoc} />
              ))}

              {!busy && !error && total === 0 && (
                <p className="empty">
                  {query
                    ? `Nothing matched “${query}”.`
                    : "The index is empty, or nothing has been indexed yet."}
                </p>
              )}

              {total > PAGE_SIZE && (
                <nav className="pagination" aria-label="Result navigation">
                  <button
                    type="button"
                    disabled={offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                  >
                    Previous
                  </button>
                  <span className="muted">
                    page {Math.floor(offset / PAGE_SIZE) + 1} of{" "}
                    {Math.max(1, Math.ceil(total / PAGE_SIZE))}
                  </span>
                  <button
                    type="button"
                    disabled={offset + PAGE_SIZE >= total}
                    onClick={() => setOffset(offset + PAGE_SIZE)}
                  >
                    Next
                  </button>
                </nav>
              )}
            </section>
          </main>
        </>
      )}

      {tab === "ask" && (
        <main className="layout single">
          <AskPanel />
        </main>
      )}

      {tab === "crawl" && (
        <main className="layout single">
          <CrawlPanel site={site} />
        </main>
      )}

      {openDoc && (
        <DocumentDrawer
          docId={openDoc}
          onClose={() => setOpenDoc(null)}
          onOpen={setOpenDoc}
        />
      )}

      <StatusBar />
    </div>
  );
}
