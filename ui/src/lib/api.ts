/**
 * The one place that talks to the FastAPI service.
 *
 * Everything the UI knows about HTTP is here: where the API is, how the key
 * is attached, how an error body becomes a thrown error, and how a request
 * is cancelled. Components call typed functions and never see a `fetch`,
 * which is what keeps "the search box re-queries on every keystroke" from
 * turning into request handling scattered across the tree.
 *
 * Two things worth pointing out:
 *
 * - **Every request is abortable.** The search box fires a request per
 *   keystroke, and without cancellation the answer to "widg" can arrive
 *   after the answer to "widgets" and overwrite it. Each call takes an
 *   `AbortSignal`, and an aborted request is reported as such rather than as
 *   a failure, so a cancelled keystroke does not flash an error.
 *
 * - **The API's error envelope is preserved.** The service returns
 *   `{error: {type, message, request_id, ...}}` for every failure, so the UI
 *   can tell "your key lacks the search scope" (403) from "Elasticsearch is
 *   down" (503) and show the request id, which is what ties a complaint to
 *   a log line.
 */

import type {
  AskResponse,
  DocumentDetail,
  FullTextResponse,
  Identity,
  Page,
  RawPageSummary,
  ReadinessResponse,
  SiteCount,
  Suggestion,
} from "./types";

/** Empty by default: requests go to the same origin and Vite proxies them. */
const API_BASE: string = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

const KEY_STORAGE = "rag-scraper.apiKey";

/**
 * The API key. Typed into the UI it lands in localStorage; otherwise the
 * build-time default is used. Reading localStorage is wrapped because it
 * throws in a private window with site data blocked.
 */
export function getApiKey(): string {
  try {
    const stored = window.localStorage.getItem(KEY_STORAGE);
    if (stored) return stored;
  } catch {
    /* storage unavailable: fall through to the build-time key */
  }
  return import.meta.env.VITE_API_KEY ?? "";
}

export function setApiKey(key: string): void {
  try {
    if (key) window.localStorage.setItem(KEY_STORAGE, key);
    else window.localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* nothing to do: the key still applies for this page load */
  }
}

/** An error carrying what the API said about it. */
export class ApiError extends Error {
  readonly status: number;
  readonly type: string;
  readonly requestId?: string;

  constructor(status: number, type: string, message: string, requestId?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.type = type;
    this.requestId = requestId;
  }

  /** Whether retrying unchanged could plausibly succeed. */
  get retryable(): boolean {
    return this.status === 503 || this.status === 429 || this.status >= 500;
  }
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

interface QueryParams {
  [key: string]: string | number | boolean | undefined | null;
}

function buildUrl(path: string, params?: QueryParams): string {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value === undefined || value === null || value === "") continue;
    url.searchParams.set(key, String(value));
  }
  return url.toString();
}

async function request<T>(
  path: string,
  options: { params?: QueryParams; signal?: AbortSignal; method?: string; body?: unknown } = {},
): Promise<T> {
  const { params, signal, method = "GET", body } = options;

  const headers: Record<string, string> = { Accept: "application/json" };
  const key = getApiKey();
  if (key) headers["X-API-Key"] = key;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const response = await fetch(buildUrl(path, params), {
    method,
    headers,
    signal,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  if (!response.ok) {
    let type = `http_${response.status}`;
    let message = `${response.status} ${response.statusText}`;
    let requestId: string | undefined = response.headers.get("X-Request-ID") ?? undefined;
    try {
      const payload = await response.json();
      const error = payload?.error;
      if (error) {
        type = error.type ?? type;
        message = error.message ?? message;
        requestId = error.request_id ?? requestId;
        // A 422 names the offending fields; showing them beats "invalid".
        if (Array.isArray(error.problems) && error.problems.length > 0) {
          message += `: ${error.problems
            .map((problem: { field?: string; message?: string }) =>
              [problem.field, problem.message].filter(Boolean).join(" "),
            )
            .join("; ")}`;
        }
      }
    } catch {
      /* not a JSON error envelope; the status line is all we have */
    }
    throw new ApiError(response.status, type, message, requestId);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// ---------------------------------------------------------------------------
// Full-text search (Elasticsearch)
// ---------------------------------------------------------------------------

export interface SearchParams {
  q: string;
  site?: string;
  language?: string;
  minWords?: number;
  enriched?: boolean;
  sort?: string;
  limit?: number;
  offset?: number;
}

export function search(params: SearchParams, signal?: AbortSignal): Promise<FullTextResponse> {
  return request<FullTextResponse>("/v1/fulltext", {
    signal,
    params: {
      q: params.q,
      site: params.site,
      language: params.language,
      min_words: params.minWords,
      enriched: params.enriched,
      sort: params.sort ?? "relevance",
      limit: params.limit ?? 10,
      offset: params.offset ?? 0,
    },
  });
}

export function suggest(prefix: string, signal?: AbortSignal): Promise<Suggestion[]> {
  return request<Suggestion[]>("/v1/fulltext/suggest", {
    signal,
    params: { q: prefix, limit: 8 },
  });
}

export function sites(signal?: AbortSignal): Promise<SiteCount[]> {
  return request<SiteCount[]>("/v1/fulltext/sites", { signal });
}

export function similar(docId: string, signal?: AbortSignal): Promise<FullTextResponse["hits"]> {
  return request<FullTextResponse["hits"]>(
    `/v1/fulltext/similar/${encodeURIComponent(docId)}`,
    { signal, params: { limit: 5 } },
  );
}

// ---------------------------------------------------------------------------
// Documents, raw pages, RAG and status
// ---------------------------------------------------------------------------

export function getDocument(docId: string, signal?: AbortSignal): Promise<DocumentDetail> {
  return request<DocumentDetail>(`/v1/documents/${encodeURIComponent(docId)}`, { signal });
}

export function rawPages(
  params: { site?: string; limit?: number; offset?: number },
  signal?: AbortSignal,
): Promise<Page<RawPageSummary>> {
  return request<Page<RawPageSummary>>("/v1/raw/pages", {
    signal,
    params: { site: params.site, limit: params.limit ?? 20, offset: params.offset ?? 0 },
  });
}

export function ask(
  payload: { question: string; top_k?: number; site?: string; max_sentences?: number },
  signal?: AbortSignal,
): Promise<AskResponse> {
  return request<AskResponse>("/v1/ask", { method: "POST", body: payload, signal });
}

export function readiness(signal?: AbortSignal): Promise<ReadinessResponse> {
  return request<ReadinessResponse>("/health/ready", { signal });
}

export function identity(signal?: AbortSignal): Promise<Identity> {
  return request<Identity>("/v1/me", { signal });
}
