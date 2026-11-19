import type { FacetBucket } from "../lib/types";

interface Props {
  facets: Record<string, FacetBucket[]>;
  site?: string;
  language?: string;
  enriched?: boolean;
  minWords?: number;
  onSite: (site?: string) => void;
  onLanguage: (language?: string) => void;
  onEnriched: (enriched?: boolean) => void;
  onMinWords: (words?: number) => void;
}

const LENGTHS = [
  { label: "Any length", value: undefined },
  { label: "100+ words", value: 100 },
  { label: "300+ words", value: 300 },
  { label: "1000+ words", value: 1000 },
];

/**
 * The filter sidebar, driven by the facet counts in the search response.
 *
 * The counts are computed over everything that matched, not over the page
 * being shown, which is what makes them worth displaying: "311 of your
 * matches are from this site" is a decision a user can act on, whereas "3 of
 * the 10 results on screen" is not.
 */
export function Facets({
  facets,
  site,
  language,
  enriched,
  minWords,
  onSite,
  onLanguage,
  onEnriched,
  onMinWords,
}: Props) {
  const sites = facets.sites ?? [];
  const languages = facets.languages ?? [];
  const keywords = (facets.keywords ?? []).slice(0, 10);

  const hasFilter =
    site !== undefined ||
    language !== undefined ||
    enriched !== undefined ||
    minWords !== undefined;

  return (
    <aside className="facets" aria-label="Filters">
      <div className="facet-head">
        <h2>Filters</h2>
        {hasFilter && (
          <button
            type="button"
            className="link"
            onClick={() => {
              onSite(undefined);
              onLanguage(undefined);
              onEnriched(undefined);
              onMinWords(undefined);
            }}
          >
            Clear all
          </button>
        )}
      </div>

      {sites.length > 0 && (
        <section className="facet">
          <h3>Site</h3>
          <ul>
            <li>
              <button
                type="button"
                className={site === undefined ? "facet-row active" : "facet-row"}
                onClick={() => onSite(undefined)}
              >
                <span>All sites</span>
              </button>
            </li>
            {sites.map((bucket) => {
              const value = String(bucket.value);
              return (
                <li key={value}>
                  <button
                    type="button"
                    className={site === value ? "facet-row active" : "facet-row"}
                    onClick={() => onSite(site === value ? undefined : value)}
                  >
                    <span className="facet-label">{value}</span>
                    <span className="count">{bucket.count}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      <section className="facet">
        <h3>Length</h3>
        <ul>
          {LENGTHS.map((option) => (
            <li key={option.label}>
              <button
                type="button"
                className={minWords === option.value ? "facet-row active" : "facet-row"}
                onClick={() => onMinWords(option.value)}
              >
                <span>{option.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section className="facet">
        <h3>AI summary</h3>
        <ul>
          <li>
            <button
              type="button"
              className={enriched === undefined ? "facet-row active" : "facet-row"}
              onClick={() => onEnriched(undefined)}
            >
              <span>Any</span>
            </button>
          </li>
          <li>
            <button
              type="button"
              className={enriched === true ? "facet-row active" : "facet-row"}
              onClick={() => onEnriched(enriched === true ? undefined : true)}
            >
              <span>Enhanced only</span>
            </button>
          </li>
          <li>
            <button
              type="button"
              className={enriched === false ? "facet-row active" : "facet-row"}
              onClick={() => onEnriched(enriched === false ? undefined : false)}
            >
              <span>Not yet enhanced</span>
            </button>
          </li>
        </ul>
      </section>

      {languages.length > 1 && (
        <section className="facet">
          <h3>Language</h3>
          <ul>
            {languages.map((bucket) => {
              const value = String(bucket.value);
              return (
                <li key={value}>
                  <button
                    type="button"
                    className={language === value ? "facet-row active" : "facet-row"}
                    onClick={() => onLanguage(language === value ? undefined : value)}
                  >
                    <span className="facet-label">{value}</span>
                    <span className="count">{bucket.count}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {keywords.length > 0 && (
        <section className="facet">
          <h3>Common terms</h3>
          <div className="chips">
            {keywords.map((bucket) => (
              <span className="chip" key={String(bucket.value)}>
                {String(bucket.value)}
                <span className="count">{bucket.count}</span>
              </span>
            ))}
          </div>
        </section>
      )}
    </aside>
  );
}
