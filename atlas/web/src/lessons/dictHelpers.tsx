/*
 * Small drawing helpers shared by the four "features and dictionaries" lessons (superposition,
 * SAEs, transcoders, crosscoders). Only those lessons import this file.
 *
 * Note: styles.css sets `svg text { fill: var(--ink) }`, which beats a `fill` attribute, so text
 * colours here go through `style`.
 */
import { ReactNode } from "react";

export type Vec = [number, number];
export type PlaneArrow = {
  to: Vec; from?: Vec; color: string; label?: string; width?: number; dashed?: boolean; opacity?: number;
};

/** An arrowhead drawn as a polygon, so any CSS colour works (SVG markers cannot inherit a stroke). */
function head(x1: number, y1: number, x2: number, y2: number, size: number) {
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len;
  const bx = x2 - ux * size, by = y2 - uy * size;
  const px = -uy * size * 0.5, py = ux * size * 0.5;
  return `${x2},${y2} ${bx + px},${by + py} ${bx - px},${by - py}`;
}

/**
 * A square plane centred on the origin: an optional cloud of points (clickable) and arrows.
 * Used for feature directions, SAE decoder vectors and the data they come from.
 */
export function Plane({ points = [], pointColors, arrows, extent, size = 340, onPointClick, selected, unitCircle = false,
  ariaLabel = "A two-dimensional plot of vectors" }: {
  points?: Vec[]; pointColors?: string[]; arrows: PlaneArrow[]; extent: number; size?: number;
  onPointClick?: (i: number) => void; selected?: number | null; unitCircle?: boolean; ariaLabel?: string;
}) {
  const pad = 14;
  const sc = (size / 2 - pad) / extent;
  const X = (v: number) => size / 2 + v * sc;
  const Y = (v: number) => size / 2 - v * sc;
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width="100%" style={{ display: "block", maxWidth: size }} role="img" aria-label={ariaLabel}>
      <rect x={pad} y={pad} width={size - 2 * pad} height={size - 2 * pad} fill="none" stroke="var(--line)" />
      <line x1={pad} x2={size - pad} y1={Y(0)} y2={Y(0)} stroke="var(--line)" />
      <line y1={pad} y2={size - pad} x1={X(0)} x2={X(0)} stroke="var(--line)" />
      {unitCircle && <circle cx={X(0)} cy={Y(0)} r={sc} fill="none" stroke="var(--line)" strokeDasharray="3 4" />}
      {points.map(([px, py], i) => (
        <circle key={i} cx={X(px)} cy={Y(py)} r={selected === i ? 5 : 2.6} fill={pointColors?.[i] ?? "var(--ink-3)"}
          fillOpacity={selected === i ? 1 : 0.5} stroke={selected === i ? "var(--ink)" : "none"} strokeWidth={1.5} />
      ))}
      {arrows.map((a, i) => {
        const [fx, fy] = a.from ?? [0, 0];
        const x1 = X(fx), y1 = Y(fy), x2 = X(fx + a.to[0]), y2 = Y(fy + a.to[1]);
        const w = a.width ?? 2.4;
        const long = Math.hypot(x2 - x1, y2 - y1) > 6;
        const lx = X(fx + a.to[0] * 1.13), ly = Y(fy + a.to[1] * 1.13);
        return (
          <g key={i} opacity={a.opacity ?? 1}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={a.color} strokeWidth={w} strokeLinecap="round"
              strokeDasharray={a.dashed ? "5 4" : undefined} />
            {long && <polygon points={head(x1, y1, x2, y2, 6 + w * 1.4)} fill={a.color} />}
            {a.label && (
              <text x={lx} y={ly + 4} fontSize={11.5} fontWeight={600} textAnchor="middle"
                style={{ fill: a.color, paintOrder: "stroke", stroke: "var(--surface)", strokeWidth: 3 }}>{a.label}</text>
            )}
          </g>
        );
      })}
      {onPointClick && points.map(([px, py], i) => (
        <circle key={`h${i}`} cx={X(px)} cy={Y(py)} r={7} fill="transparent" style={{ cursor: "pointer" }}
          onClick={() => onPointClick(i)} />
      ))}
    </svg>
  );
}

/** A small x-y chart with numeric axes: points joined in order, each with an optional label. */
export function XYChart({ points, xLabel, yLabel, width = 460, height = 250, xMax, yMax, format = (v) => v.toFixed(2),
  highlight }: {
  points: { x: number; y: number; label?: string; color?: string }[];
  xLabel: string; yLabel: string; width?: number; height?: number; xMax?: number; yMax?: number;
  format?: (v: number) => string; highlight?: number;
}) {
  const pad = { l: 46, r: 48, t: 14, b: 38 };
  const xm = xMax ?? Math.max(1e-9, ...points.map((p) => p.x)) * 1.08;
  const ym = yMax ?? Math.max(1e-9, ...points.map((p) => p.y)) * 1.08;
  const pw = width - pad.l - pad.r, ph = height - pad.t - pad.b;
  const X = (v: number) => pad.l + (v / xm) * pw;
  const Y = (v: number) => pad.t + ph - (Math.min(v, ym) / ym) * ph;
  const ticks = [0, 0.5, 1];
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ display: "block", maxWidth: width }} role="img"
      aria-label={`${yLabel} against ${xLabel}`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={width - pad.r} y1={Y(t * ym)} y2={Y(t * ym)} stroke="var(--line)" />
          <text x={pad.l - 6} y={Y(t * ym) + 4} fontSize={10} textAnchor="end" style={{ fill: "var(--ink-3)" }}>{format(t * ym)}</text>
          <text x={X(t * xm)} y={height - 22} fontSize={10} textAnchor="middle" style={{ fill: "var(--ink-3)" }}>{format(t * xm)}</text>
        </g>
      ))}
      <text x={pad.l + pw / 2} y={height - 5} fontSize={10.5} textAnchor="middle" style={{ fill: "var(--ink-2)" }}>{xLabel}</text>
      <text x={11} y={pad.t + ph / 2} fontSize={10.5} textAnchor="middle" transform={`rotate(-90 11 ${pad.t + ph / 2})`}
        style={{ fill: "var(--ink-2)" }}>{yLabel}</text>
      <polyline fill="none" stroke="var(--ink-3)" strokeWidth={1.5} strokeDasharray="4 3"
        points={points.map((p) => `${X(p.x)},${Y(p.y)}`).join(" ")} />
      {points.map((p, i) => (
        <g key={i}>
          <circle cx={X(p.x)} cy={Y(p.y)} r={highlight === i ? 6.5 : 4.5} fill={p.color ?? "var(--accent)"}
            stroke={highlight === i ? "var(--ink)" : "var(--surface)"} strokeWidth={1.5} />
          {p.label && (
            <text x={X(p.x) + 8} y={Y(p.y) - 7} fontSize={10.5} className="mono"
              style={{ fill: "var(--ink-2)", paintOrder: "stroke", stroke: "var(--surface)", strokeWidth: 3 }}>{p.label}</text>
          )}
        </g>
      ))}
    </svg>
  );
}

/** A coloured square used in inline legends. */
export function Swatch({ color, children, dashed = false }: { color: string; children?: ReactNode; dashed?: boolean }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, marginRight: 12, whiteSpace: "nowrap" }}>
      <span style={{
        width: 14, height: dashed ? 0 : 10, borderRadius: 2, display: "inline-block",
        background: dashed ? "transparent" : color, borderTop: dashed ? `2px dashed ${color}` : undefined,
      }} />
      {children}
    </span>
  );
}

/** "reads a (+0.68), b (+0.64)": the entries of a weight row worth mentioning, largest first. */
export function describeRow(row: number[], names: string[], rel = 0.3) {
  const m = Math.max(1e-9, ...row.map(Math.abs));
  const parts = row.map((v, j) => ({ v, j })).filter((e) => Math.abs(e.v) >= rel * m && Math.abs(e.v) > 0.05)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  return parts.length ? parts.map((e) => `${names[e.j]} (${e.v > 0 ? "+" : "−"}${Math.abs(e.v).toFixed(2)})`).join(", ") : "nothing";
}

/** Preset buttons, in a wrapping row. */
export function Presets<T>({ items, onPick, active }: {
  items: { label: string; value: T }[]; onPick: (v: T) => void; active?: string;
}) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "0 0 12px" }}>
      {items.map((it) => (
        <button key={it.label} className={`chip ${active === it.label ? "on" : ""}`} onClick={() => onPick(it.value)}>{it.label}</button>
      ))}
    </div>
  );
}
