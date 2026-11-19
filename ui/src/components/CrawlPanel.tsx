import { useEffect, useState } from "react";
import { ApiError, isAbort, rawPages } from "../lib/api";
import type { Page, RawPageSummary } from "../lib/types";

const PAGE_SIZE = 20;

/**
 * The raw-data tab: the crawl exactly as it was fetched, from PostgreSQL.
 *
 * This is the "fetching raw scraped data" endpoint given a face. It is
 * separate from search on purpose: these rows are the crawl record (status
 * code, content type, which worker fetched it, when), not the cleaned
 * content, and that is what makes them useful for checking a crawl rather
 * than for reading.
 */
export function CrawlPanel({ site }: { site?: string }) {
  const [page, setPage] = useState<Page<RawPageSummary> | null>(null);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    rawPages({ site, limit: PAGE_SIZE, offset }, controller.signal)
      .then(setPage)
      .catch((cause) => {
        if (isAbort(cause)) return;
        setError(cause instanceof ApiError ? cause.message : String(cause));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [site, offset]);

  const items = page?.items ?? [];
  const total = page?.meta.total ?? 0;

  return (
    <section className="crawl">
      <header className="crawl-head">
        <h2>Raw crawl data</h2>
        <span className="muted">
          {loading ? "loading…" : `${total} page${total === 1 ? "" : "s"} fetched`}
        </span>
      </header>

      {error && <p className="error">{error}</p>}

      {items.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>URL</th>
                <th>Status</th>
                <th>Type</th>
                <th>Size</th>
                <th>Depth</th>
                <th>Worker</th>
                <th>Fetched</th>
              </tr>
            </thead>
            <tbody>
              {items.map((row) => (
                <tr key={row.id}>
                  <td className="muted">{row.id}</td>
                  <td className="cell-url">
                    <a href={row.url} target="_blank" rel="noreferrer noopener">
                      {row.url}
                    </a>
                  </td>
                  <td>
                    <span
                      className={
                        row.http_status && row.http_status < 400 ? "ok-pill" : "bad-pill"
                      }
                    >
                      {row.http_status ?? "—"}
                    </span>
                  </td>
                  <td className="muted">{(row.content_type ?? "").split(";")[0] || "—"}</td>
                  <td className="muted">
                    {row.content_length ? `${Math.round(row.content_length / 1024)} kB` : "—"}
                  </td>
                  <td className="muted">{row.depth ?? "—"}</td>
                  <td className="muted">{row.fetched_by ?? "—"}</td>
                  <td className="muted">{(row.fetched_at ?? "").slice(0, 19).replace("T", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!loading && items.length === 0 && !error && (
        <p className="empty">No raw pages stored yet. Run a crawl first.</p>
      )}

      {total > PAGE_SIZE && (
        <nav className="pagination" aria-label="Raw page navigation">
          <button
            type="button"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
          >
            Previous
          </button>
          <span className="muted">
            {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}
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
  );
}
