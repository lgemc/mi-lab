import { useEffect, useRef, useState } from "react";

/** POST (or GET, with no body) to the atlas API and return the JSON. */
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* the status text will do */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

/**
 * Call the API whenever `body` changes, debounced, keeping the last good answer on screen
 * while the next one loads so a slider drag never flashes an empty chart.
 */
export function useApi<T>(path: string, body?: unknown, delay = 120) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const key = JSON.stringify(body ?? null);
  const seq = useRef(0);
  useEffect(() => {
    const mine = ++seq.current;
    setLoading(true);
    const t = setTimeout(() => {
      api<T>(path, body)
        .then((d) => {
          if (mine === seq.current) {
            setData(d);
            setError(null);
          }
        })
        .catch((e: Error) => mine === seq.current && setError(e.message))
        .finally(() => mine === seq.current && setLoading(false));
    }, delay);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, key]);
  return { data, error, loading };
}

export const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "—");
export const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;
