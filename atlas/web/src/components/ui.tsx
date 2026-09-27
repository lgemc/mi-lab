import { ReactNode } from "react";

export function Card({ title, caption, children, className = "" }: {
  title?: ReactNode; caption?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {title && <h3>{title}</h3>}
      {children}
      {caption && <div className="caption">{caption}</div>}
    </section>
  );
}

type CalloutKind = "note" | "takeaway" | "warn" | "real" | "try";
const LABELS: Record<CalloutKind, string> = {
  note: "Note", takeaway: "Takeaway", warn: "Careful", real: "In real models", try: "Try this",
};

export function Callout({ kind = "note", label, children }: { kind?: CalloutKind; label?: string; children: ReactNode }) {
  const cls = kind === "try" ? "" : kind;
  return (
    <div className={`callout ${cls}`}>
      <span className="label">{label ?? LABELS[kind]}</span>
      {children}
    </div>
  );
}

export function Slider({ label, value, min, max, step = 0.01, onChange, format }: {
  label: ReactNode; value: number; min: number; max: number; step?: number;
  onChange: (v: number) => void; format?: (v: number) => string;
}) {
  return (
    <label className="control">
      <span>
        {label} <span className="val">{format ? format(value) : value}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export function Seg<T extends string | number>({ label, value, options, onChange }: {
  label?: ReactNode; value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="control">
      {label && <span>{label}</span>}
      <div className="seg" role="radiogroup">
        {options.map((o) => (
          <button key={String(o.value)} className={o.value === value ? "on" : ""} role="radio"
            aria-checked={o.value === value} onClick={() => onChange(o.value)}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Select<T extends string>({ label, value, options, onChange }: {
  label: ReactNode; value: T; options: T[] | { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  const opts = (options as (T | { value: T; label: string })[]).map((o) =>
    typeof o === "string" ? { value: o, label: o } : o);
  return (
    <label className="control">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value as T)}>
        {opts.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

export function Stat({ k, v, d, color }: { k: ReactNode; v: ReactNode; d?: ReactNode; color?: string }) {
  return (
    <div className="stat">
      <span className="k">{k}</span>
      <span className="v" style={color ? { color } : undefined}>{v}</span>
      {d && <span className="d">{d}</span>}
    </div>
  );
}

export function Loading({ loading, error, children }: { loading: boolean; error: string | null; children: ReactNode }) {
  return (
    <div className={loading ? "loading" : ""}>
      {error && <div className="error">The server said: {error}</div>}
      {children}
    </div>
  );
}

export function Tokens({ tokens, highlight = {} }: { tokens: string[]; highlight?: Record<number, string> }) {
  return (
    <div className="tokens">
      {tokens.map((t, i) => (
        <span key={i} className="tok" style={highlight[i] ? { background: highlight[i] } : undefined}>
          <span className="muted small">{i} </span>{t}
        </span>
      ))}
    </div>
  );
}
