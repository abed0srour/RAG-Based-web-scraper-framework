import { useEffect, useRef, useState } from "react";
import { isAbort, suggest } from "../lib/api";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import type { Suggestion } from "../lib/types";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  busy?: boolean;
}

/**
 * The search input, with suggestions from the index's edge n-grams.
 *
 * The suggestion request is debounced and abortable, so holding a key down
 * does not queue a request per repeat, and a slow answer for "wid" cannot
 * arrive after the answer for "widgets" and replace it.
 */
export function SearchBar({ value, onChange, onSubmit, busy }: Props) {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [open, setOpen] = useState(false);
  const [highlighted, setHighlighted] = useState(-1);
  const debounced = useDebouncedValue(value, 180);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (debounced.trim().length < 2) {
      setSuggestions([]);
      return;
    }
    const controller = new AbortController();
    suggest(debounced.trim(), controller.signal)
      .then((rows) => {
        setSuggestions(rows);
        setHighlighted(-1);
      })
      .catch((error) => {
        // A cancelled keystroke is not a failure, and a suggestion list is
        // not important enough to show an error for.
        if (!isAbort(error)) setSuggestions([]);
      });
    return () => controller.abort();
  }, [debounced]);

  // Clicking outside closes the list; without this it stays over the results.
  useEffect(() => {
    function onDocumentClick(event: MouseEvent) {
      if (!boxRef.current?.contains(event.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocumentClick);
    return () => document.removeEventListener("mousedown", onDocumentClick);
  }, []);

  function choose(text: string) {
    onChange(text);
    onSubmit(text);
    setOpen(false);
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      if (open && highlighted >= 0 && suggestions[highlighted]) {
        choose(suggestions[highlighted].text);
      } else {
        onSubmit(value);
        setOpen(false);
      }
      return;
    }
    if (event.key === "Escape") {
      setOpen(false);
      return;
    }
    if (event.key === "ArrowDown" && suggestions.length > 0) {
      event.preventDefault();
      setOpen(true);
      setHighlighted((index) => (index + 1) % suggestions.length);
      return;
    }
    if (event.key === "ArrowUp" && suggestions.length > 0) {
      event.preventDefault();
      setHighlighted((index) => (index <= 0 ? suggestions.length - 1 : index - 1));
    }
  }

  const showList = open && suggestions.length > 0;

  return (
    <div className="searchbar" ref={boxRef}>
      <div className="searchbar-input">
        <svg aria-hidden="true" viewBox="0 0 24 24" className="icon">
          <circle cx="11" cy="11" r="7" />
          <line x1="16.5" y1="16.5" x2="21" y2="21" />
        </svg>
        <input
          type="search"
          value={value}
          placeholder="Search the crawled corpus…"
          aria-label="Search the crawled corpus"
          autoComplete="off"
          role="combobox"
          aria-expanded={showList}
          aria-controls="suggestion-list"
          onChange={(event) => {
            onChange(event.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
        />
        {busy && <span className="spinner" aria-label="Searching" />}
        {value && !busy && (
          <button
            type="button"
            className="clear"
            aria-label="Clear the search"
            onClick={() => {
              onChange("");
              onSubmit("");
            }}
          >
            ×
          </button>
        )}
      </div>

      {showList && (
        <ul className="suggestions" id="suggestion-list" role="listbox">
          {suggestions.map((row, index) => (
            <li
              key={`${row.doc_id ?? row.text}-${index}`}
              role="option"
              aria-selected={index === highlighted}
              className={index === highlighted ? "active" : undefined}
              onMouseEnter={() => setHighlighted(index)}
              onMouseDown={(event) => {
                // mousedown, not click: the outside-click handler would
                // close the list before a click could land.
                event.preventDefault();
                choose(row.text);
              }}
            >
              <span className="suggestion-text">{row.text}</span>
              {row.site && <span className="suggestion-site">{row.site}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
