import { renderFragments, renderHighlight } from "../lib/highlight";
import type { FullTextHit } from "../lib/types";

interface Props {
  hit: FullTextHit;
  onOpen: (docId: string) => void;
}

function formatFields(fields: Record<string, unknown> | undefined): string[] {
  if (!fields) return [];
  return Object.entries(fields)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .slice(0, 4)
    .map(([key, value]) =>
      `${key}: ${Array.isArray(value) ? value.join(", ") : String(value)}`,
    );
}

/**
 * One search result.
 *
 * The title and the snippet are rendered from the highlight fragments, so
 * the words the user searched for are marked in place rather than the card
 * showing the first sentence of the page and leaving them to guess why it
 * matched. The relevance bar is the score as a fraction of the best hit in
 * this result set - an Elasticsearch score has no absolute meaning, so it is
 * shown as a comparison, not as a number out of ten.
 */
export function ResultCard({ hit, onOpen }: Props) {
  const titleFragments = hit.highlights?.title;
  const snippetFragments = hit.highlights?.text ?? hit.highlights?.summary;
  const fields = formatFields(hit.fields);

  return (
    <article className="result">
      <header>
        <h3>
          <button type="button" className="result-title" onClick={() => onOpen(hit.doc_id)}>
            {titleFragments && titleFragments.length > 0
              ? renderHighlight(titleFragments[0], `title-${hit.doc_id}`)
              : (hit.title ?? "(untitled)")}
          </button>
        </h3>
        <a className="result-url" href={hit.url} target="_blank" rel="noreferrer noopener">
          {hit.url}
        </a>
      </header>

      <p className="snippet">
        {snippetFragments && snippetFragments.length > 0
          ? renderFragments(snippetFragments, `snip-${hit.doc_id}`)
          : (hit.snippet ?? hit.description ?? "")}
      </p>

      {hit.enriched && hit.summary && (
        <p className="summary">
          <span className="badge">AI summary</span>
          {hit.summary}
        </p>
      )}

      {fields.length > 0 && (
        <ul className="fields">
          {fields.map((entry) => (
            <li key={entry}>{entry}</li>
          ))}
        </ul>
      )}

      <footer className="result-meta">
        {hit.site && <span className="tag">{hit.site}</span>}
        {typeof hit.word_count === "number" && <span>{hit.word_count} words</span>}
        {typeof hit.reading_time_sec === "number" && (
          <span>{Math.max(1, Math.round(hit.reading_time_sec / 60))} min read</span>
        )}
        {hit.processed_at && <span>{hit.processed_at.slice(0, 10)}</span>}
        <span className="relevance" title={`Elasticsearch score ${hit.score}`}>
          <span
            className="relevance-bar"
            style={{ width: `${Math.round((hit.relevance ?? 0) * 100)}%` }}
          />
        </span>
      </footer>
    </article>
  );
}
