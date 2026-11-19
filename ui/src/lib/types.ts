/**
 * The shapes the FastAPI service returns.
 *
 * These mirror the Pydantic models in `src/api/schemas.py`. They are written
 * out rather than inferred so that a change to the API contract shows up as
 * a TypeScript error here, which is the same reason the Pydantic models
 * exist on the other side.
 */

export interface FacetBucket {
  value: string | boolean | number;
  count: number;
}

export interface FullTextHit {
  doc_id: string;
  url: string;
  site?: string | null;
  path?: string | null;
  title?: string | null;
  description?: string | null;
  language?: string | null;
  keywords?: string[];
  headings?: string[];
  word_count?: number | null;
  reading_time_sec?: number | null;
  processed_at?: string | null;
  fetched_at?: string | null;
  http_status?: number | null;
  depth?: number | null;
  summary?: string | null;
  enriched?: boolean;
  quality_score?: number | null;
  /** Site-specific values; the shape differs per website. */
  fields?: Record<string, unknown>;
  rank: number;
  score: number;
  relevance?: number | null;
  /** Matched fragments per field, with <mark> around the terms. */
  highlights?: Record<string, string[]>;
  snippet?: string;
}

export interface FullTextResponse {
  query: string;
  sort: string;
  total: number;
  returned: number;
  took_ms: number;
  elapsed_ms: number;
  hits: FullTextHit[];
  facets: Record<string, FacetBucket[]>;
  notes: string[];
}

export interface Suggestion {
  text: string;
  doc_id?: string | null;
  url?: string | null;
  site?: string | null;
}

export interface SiteCount {
  value: string;
  count: number;
}

export interface Citation {
  index: number;
  url: string;
  title: string;
  site: string;
  doc_id: string;
  chunk_id: string;
  score: number;
}

export interface AskResponse {
  question: string;
  answer: string;
  provider: string;
  model: string;
  /** True when every sentence was copied from the retrieved passages. */
  grounded_by_construction: boolean;
  citations: Citation[];
  timings_ms: Record<string, number>;
  notes: string[];
  retrieval?: { chunks?: Array<Record<string, unknown>> } | null;
}

export interface DocumentDetail {
  doc_id: string;
  url: string;
  site: string;
  title?: string | null;
  description?: string | null;
  language?: string | null;
  keywords?: string[];
  headings?: string[];
  text?: string | null;
  enriched?: boolean;
  summary?: string | null;
  stats?: Record<string, number> | null;
  fields?: Record<string, unknown>;
  enrichment?: Record<string, unknown> | null;
}

export interface PageMeta {
  total: number;
  limit: number;
  offset: number;
  returned: number;
  next_offset?: number | null;
}

export interface Page<T> {
  meta: PageMeta;
  items: T[];
}

export interface RawPageSummary {
  id: number;
  job_id?: number | null;
  url: string;
  site: string;
  depth?: number | null;
  http_status?: number | null;
  content_type?: string | null;
  content_length?: number | null;
  fetched_at?: string | null;
  fetched_by?: string | null;
}

export interface ComponentHealth {
  status: string;
  detail?: string | null;
}

export interface ReadinessResponse {
  status: string;
  checks: Record<string, ComponentHealth>;
}

export interface Identity {
  name: string;
  key_id: string;
  scopes: string[];
  authenticated: boolean;
  rate_limit: Record<string, unknown>;
}
