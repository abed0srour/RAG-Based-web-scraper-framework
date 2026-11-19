import { useState } from "react";
import { ApiError, ask } from "../lib/api";
import type { AskResponse } from "../lib/types";

/**
 * The RAG tab: a question, an answer, and the sources it came from.
 *
 * The citations are the reason this is worth a tab of its own rather than
 * being folded into the search results. `[2]` in the answer is the second
 * source in the list, so every sentence can be traced back to a crawled URL
 * - and when the free offline provider generated it, the answer is
 * sentences copied from those pages, which the "grounded" badge says
 * explicitly.
 */
export function AskPanel() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (text.length < 3) return;

    setBusy(true);
    setError(null);
    try {
      setAnswer(await ask({ question: text, top_k: 5, max_sentences: 4 }));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
      setAnswer(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="ask">
      <form onSubmit={submit} className="ask-form">
        <label htmlFor="question">Ask a question about the crawled corpus</label>
        <div className="ask-row">
          <input
            id="question"
            type="text"
            value={question}
            placeholder="What does the corpus say about…?"
            onChange={(event) => setQuestion(event.target.value)}
          />
          <button type="submit" className="primary" disabled={busy || question.trim().length < 3}>
            {busy ? "Thinking…" : "Ask"}
          </button>
        </div>
        <p className="hint">
          The answer is generated only from the retrieved passages, and every claim is
          numbered to the source it came from.
        </p>
      </form>

      {error && <p className="error">{error}</p>}

      {answer && (
        <div className="answer">
          {answer.answer ? (
            <p className="answer-text">{answer.answer}</p>
          ) : (
            <p className="empty">
              Nothing in the index matched that question, so there is no grounded answer.
            </p>
          )}

          <div className="answer-meta">
            <span className="tag">{answer.provider}</span>
            <span>{answer.model}</span>
            {answer.grounded_by_construction && (
              <span className="badge" title="Every sentence was copied from a retrieved passage">
                grounded
              </span>
            )}
            {typeof answer.timings_ms.total === "number" && (
              <span>{Math.round(answer.timings_ms.total)} ms</span>
            )}
          </div>

          {answer.notes.length > 0 && (
            <ul className="notes">
              {answer.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}

          {answer.citations.length > 0 && (
            <section>
              <h3>Sources</h3>
              <ol className="citations">
                {answer.citations.map((citation) => (
                  <li key={`${citation.index}-${citation.chunk_id}`}>
                    <a href={citation.url} target="_blank" rel="noreferrer noopener">
                      {citation.title || citation.url}
                    </a>
                    <span className="tag">{citation.site}</span>
                  </li>
                ))}
              </ol>
            </section>
          )}
        </div>
      )}
    </section>
  );
}
