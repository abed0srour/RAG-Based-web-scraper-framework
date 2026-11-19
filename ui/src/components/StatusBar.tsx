import { useEffect, useState } from "react";
import { getApiKey, isAbort, readiness, setApiKey } from "../lib/api";
import type { ReadinessResponse } from "../lib/types";

const POLL_MS = 15000;

/**
 * The footer: which backing services are up, and the API key in use.
 *
 * It polls `/health/ready`, which reports each dependency separately, so a
 * partial outage is visible in the interface instead of only showing up as a
 * failed search. That matters here because the stores degrade independently:
 * Elasticsearch being down stops search while the RAG tab keeps working, and
 * the readiness line is what explains that to whoever is looking at it.
 */
export function StatusBar() {
  const [state, setState] = useState<ReadinessResponse | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const [key, setKey] = useState(getApiKey());
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    function poll() {
      readiness(controller.signal)
        .then((value) => {
          if (cancelled) return;
          setState(value);
          setUnreachable(false);
        })
        .catch((cause) => {
          if (cancelled || isAbort(cause)) return;
          // A 503 still has a body describing which store is down; only a
          // transport failure means the API itself is unreachable.
          setUnreachable(true);
        });
    }

    poll();
    const timer = window.setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  const checks = Object.entries(state?.checks ?? {});

  return (
    <footer className="statusbar">
      <div className="status-services">
        {unreachable && <span className="bad-pill">API unreachable</span>}
        {!unreachable && state && (
          <span className={state.status === "ready" ? "ok-pill" : "warn-pill"}>
            {state.status}
          </span>
        )}
        {checks.map(([name, check]) => (
          <span
            key={name}
            className={`dot-row ${check.status === "up" ? "up" : check.status === "down" ? "down" : "degraded"}`}
            title={check.detail ?? check.status}
          >
            <span className="dot" />
            {name}
          </span>
        ))}
      </div>

      <div className="status-key">
        {editing ? (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              setApiKey(key);
              setEditing(false);
              // The key is attached per request, so a reload is the simplest
              // way to re-run every query under the new one.
              window.location.reload();
            }}
          >
            <input
              type="password"
              value={key}
              aria-label="API key"
              onChange={(event) => setKey(event.target.value)}
              placeholder="X-API-Key"
            />
            <button type="submit" className="link">
              Save
            </button>
          </form>
        ) : (
          <button type="button" className="link" onClick={() => setEditing(true)}>
            API key: {key ? `${key.slice(0, 4)}…${key.slice(-4)}` : "not set"}
          </button>
        )}
      </div>
    </footer>
  );
}
