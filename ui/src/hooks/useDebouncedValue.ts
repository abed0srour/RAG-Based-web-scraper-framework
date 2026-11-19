import { useEffect, useState } from "react";

/**
 * A value that only updates once it has stopped changing for `delay` ms.
 *
 * Typing "widgets" is seven state updates. Without this each one would be a
 * request, six of which are obsolete before they return - and on the suggest
 * endpoint that is six against the rate limit for nothing. Debouncing the
 * value rather than the request keeps the input itself fully responsive: the
 * box shows every keystroke immediately, only the query lags behind.
 */
export function useDebouncedValue<T>(value: T, delay = 250): T {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);

  return debounced;
}
