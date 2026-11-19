import type { ReactNode } from "react";

/**
 * Render an Elasticsearch highlight fragment safely.
 *
 * The highlighter wraps matched terms in `<mark>` … `</mark>` and inserts
 * those tags into the *stored text*, which came from a scraped web page. It
 * does not escape anything else in that text. So a page whose clean text
 * contains the literal characters `<img onerror=...>` comes back in a
 * fragment with those characters intact, and handing the fragment to
 * `dangerouslySetInnerHTML` would execute it: a stored cross-site scripting
 * hole whose payload is supplied by whatever site was crawled.
 *
 * Rather than escaping the string and then un-escaping the two tags we want,
 * this splits on the markers and returns React nodes. Everything that is not
 * a marker stays a text node, which React escapes on render, so no HTML from
 * a crawled page can ever reach the DOM as markup.
 */
export function renderHighlight(fragment: string, keyPrefix = "h"): ReactNode[] {
  const parts = fragment.split(/(<mark>|<\/mark>)/g);
  const nodes: ReactNode[] = [];
  let marked = false;

  parts.forEach((part, index) => {
    if (part === "<mark>") {
      marked = true;
      return;
    }
    if (part === "</mark>") {
      marked = false;
      return;
    }
    if (!part) return;
    nodes.push(
      marked ? (
        <mark key={`${keyPrefix}-${index}`}>{part}</mark>
      ) : (
        <span key={`${keyPrefix}-${index}`}>{part}</span>
      ),
    );
  });

  return nodes;
}

/** Join several fragments with an ellipsis, each rendered safely. */
export function renderFragments(fragments: string[], keyPrefix = "f"): ReactNode[] {
  const nodes: ReactNode[] = [];
  fragments.forEach((fragment, index) => {
    if (index > 0) nodes.push(<span key={`${keyPrefix}-sep-${index}`}> … </span>);
    nodes.push(...renderHighlight(fragment, `${keyPrefix}-${index}`));
  });
  return nodes;
}
