import { useEffect, useState } from "react";
import { ApiError, getDocument, isAbort, similar } from "../lib/api";
import type { DocumentDetail, FullTextHit } from "../lib/types";

interface Props {
  docId: string;
  onClose: () => void;
  onOpen: (docId: string) => void;
}

/**
 * The detail panel for one document.
 *
 * Two requests, deliberately from two different stores: the document itself
 * comes from MongoDB through `/v1/documents/{id}`, and the "related pages"
 * list comes from Elasticsearch through `/v1/fulltext/similar/{id}`. Showing
 * them side by side is the point - it is where the split between the stores
 * becomes visible in the interface rather than only in the architecture.
 */
export function DocumentDrawer({ docId, onClose, onOpen }: Props) {
  const [document_, setDocument] = useState<DocumentDetail | null>(null);
  const [related, setRelated] = useState<FullTextHit[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    getDocument(docId, controller.signal)
      .then(setDocument)
      .catch((cause) => {
        if (isAbort(cause)) return;
        setError(cause instanceof ApiError ? cause.message : String(cause));
      })
      .finally(() => setLoading(false));

    // A missing "more like this" is not worth an error: the document is
    // still readable without it.
    similar(docId, controller.signal)
      .then(setRelated)
      .catch(() => setRelated([]));

    return () => controller.abort();
  }, [docId]);

  // Escape closes the panel, which is what a keyboard user will try first.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const stats = document_?.stats ?? {};

  return (
    <div className="drawer-backdrop" onClick={onClose} role="presentation">
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Document detail"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="drawer-head">
          <h2>{document_?.title ?? (loading ? "Loading…" : "Document")}</h2>
          <button type="button" className="close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        {error && <p className="error">{error}</p>}

        {document_ && (
          <div className="drawer-body">
            <a
              className="result-url"
              href={document_.url}
              target="_blank"
              rel="noreferrer noopener"
            >
              {document_.url}
            </a>

            <dl className="meta-grid">
              <div>
                <dt>Site</dt>
                <dd>{document_.site}</dd>
              </div>
              <div>
                <dt>Words</dt>
                <dd>{stats.word_count ?? "—"}</dd>
              </div>
              <div>
                <dt>Language</dt>
                <dd>{document_.language ?? "—"}</dd>
              </div>
              <div>
                <dt>Enhanced</dt>
                <dd>{document_.enriched ? "yes" : "no"}</dd>
              </div>
            </dl>

            {document_.summary && (
              <section>
                <h3>
                  AI summary <span className="badge">RAG</span>
                </h3>
                <p>{document_.summary}</p>
              </section>
            )}

            {document_.fields && Object.keys(document_.fields).length > 0 && (
              <section>
                <h3>Extracted fields</h3>
                <dl className="field-list">
                  {Object.entries(document_.fields).map(([key, value]) => (
                    <div key={key}>
                      <dt>{key}</dt>
                      <dd>{Array.isArray(value) ? value.join(", ") : String(value)}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            )}

            {document_.keywords && document_.keywords.length > 0 && (
              <section>
                <h3>Keywords</h3>
                <div className="chips">
                  {document_.keywords.map((word) => (
                    <span className="chip" key={word}>
                      {word}
                    </span>
                  ))}
                </div>
              </section>
            )}

            {related.length > 0 && (
              <section>
                <h3>Related pages</h3>
                <ul className="related">
                  {related.map((hit) => (
                    <li key={hit.doc_id}>
                      <button type="button" className="link" onClick={() => onOpen(hit.doc_id)}>
                        {hit.title ?? hit.url}
                      </button>
                      <span className="tag">{hit.site}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {document_.text && (
              <section>
                <h3>Clean text</h3>
                <pre className="text-block">{document_.text.slice(0, 4000)}</pre>
              </section>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}
