/*
 * Hand-rolled SVG charts. Four shapes cover every lesson: a heatmap (attention, patching grids,
 * weight matrices), bars (per-component effects, signed), lines (a quantity across layers or
 * training) and a scatter (activations, estimates against the truth). Colours come from the
 * CSS tokens so both themes work without a second palette.
 */
import { ReactNode, useState } from "react";

const POS = "var(--accent)";
const NEG = "var(--warm)";

/** A signed value -> a colour: blue for positive, orange for negative, transparent at zero. */
export function signedColor(v: number, max: number) {
  const a = Math.min(1, Math.abs(v) / (max || 1));
  const rgb = v >= 0 ? "var(--pos)" : "var(--neg)";
  return `rgba(${rgb}, ${(0.08 + 0.92 * a).toFixed(3)})`;
}

export function Heatmap({ values, rowLabels, colLabels, max, cell = 28, format = (v) => v.toFixed(2),
  signed = true, onCellClick, highlightRow, caption, rowLabelWidth = 64 }: {
  values: number[][]; rowLabels: string[]; colLabels: string[]; max?: number; cell?: number;
  format?: (v: number) => string; signed?: boolean; onCellClick?: (r: number, c: number) => void;
  highlightRow?: number; caption?: ReactNode; rowLabelWidth?: number;
}) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const m = max ?? Math.max(1e-9, ...values.flat().map((v) => Math.abs(v)));
  const top = 58;
  const w = rowLabelWidth + colLabels.length * cell + 4;
  const h = top + rowLabels.length * cell + 4;
  return (
    <div className="scroll-x">
      <svg width={w} height={h} role="img" style={{ display: "block" }}>
        {colLabels.map((c, j) => (
          <text key={j} x={rowLabelWidth + j * cell + cell / 2} y={top - 6} fontSize={11}
            transform={`rotate(-45 ${rowLabelWidth + j * cell + cell / 2} ${top - 6})`}>{c}</text>
        ))}
        {rowLabels.map((r, i) => (
          <text key={i} x={rowLabelWidth - 6} y={top + i * cell + cell / 2 + 4} fontSize={11} textAnchor="end"
            fontWeight={highlightRow === i ? 700 : 400}>{r}</text>
        ))}
        {values.map((row, i) => row.map((v, j) => (
          <rect key={`${i}-${j}`} x={rowLabelWidth + j * cell} y={top + i * cell} width={cell - 2} height={cell - 2} rx={3}
            fill={signed ? signedColor(v, m) : `rgba(var(--pos), ${(0.04 + 0.96 * Math.min(1, v / m)).toFixed(3)})`}
            stroke={hover && hover[0] === i && hover[1] === j ? "var(--ink)" : "none"}
            style={{ cursor: onCellClick ? "pointer" : "default" }}
            onMouseEnter={() => setHover([i, j])} onMouseLeave={() => setHover(null)}
            onClick={() => onCellClick?.(i, j)} />
        )))}
      </svg>
      <div className="caption small muted" style={{ minHeight: 20 }}>
        {hover ? <>{rowLabels[hover[0]]} → {colLabels[hover[1]]}: <b className="mono">{format(values[hover[0]][hover[1]])}</b></> : caption}
      </div>
    </div>
  );
}

export function Bars({ items, max, format = (v) => v.toFixed(2), width = 420, rowHeight = 26, labelWidth = 150, refLine }: {
  items: { label: ReactNode; value: number; color?: string; note?: ReactNode }[];
  max?: number; format?: (v: number) => string; width?: number; rowHeight?: number; labelWidth?: number;
  refLine?: { value: number; label: string };
}) {
  const m = max ?? Math.max(1e-9, ...items.map((i) => Math.abs(i.value)), refLine ? Math.abs(refLine.value) : 0);
  const hasNeg = items.some((i) => i.value < 0);
  const plotW = width - labelWidth - 60;
  const zero = labelWidth + (hasNeg ? plotW / 2 : 0);
  const scale = (hasNeg ? plotW / 2 : plotW) / m;
  const h = items.length * rowHeight + 8;
  return (
    <div className="scroll-x">
      <svg width={width} height={h} role="img" style={{ display: "block", maxWidth: "100%" }} viewBox={`0 0 ${width} ${h}`}>
        <line x1={zero} x2={zero} y1={0} y2={h} stroke="var(--line)" />
        {refLine && (
          <g>
            <line x1={zero + refLine.value * scale} x2={zero + refLine.value * scale} y1={0} y2={h}
              stroke="var(--ink-3)" strokeDasharray="3 3" />
          </g>
        )}
        {items.map((it, i) => {
          const y = i * rowHeight + 4;
          const len = Math.abs(it.value) * scale;
          const x = it.value >= 0 ? zero : zero - len;
          return (
            <g key={i}>
              <foreignObject x={0} y={y} width={labelWidth - 8} height={rowHeight}>
                <div style={{ fontSize: 12, textAlign: "right", lineHeight: `${rowHeight - 4}px`, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", color: "var(--ink)" }}>{it.label}</div>
              </foreignObject>
              <rect x={x} y={y + 3} width={Math.max(len, 1)} height={rowHeight - 10} rx={3}
                fill={it.color ?? (it.value >= 0 ? POS : NEG)} />
              <text x={it.value >= 0 ? x + len + 5 : x - 5} y={y + rowHeight / 2 + 2} fontSize={11}
                textAnchor={it.value >= 0 ? "start" : "end"} className="mono">{format(it.value)}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export type Series = { name: string; values: number[]; color: string; dashed?: boolean };

export function Lines({ series, xLabels, height = 220, width = 560, yMin, yMax, yLabel, format = (v) => v.toFixed(2), markX }: {
  series: Series[]; xLabels: string[]; height?: number; width?: number; yMin?: number; yMax?: number;
  yLabel?: string; format?: (v: number) => string; markX?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const all = series.flatMap((s) => s.values);
  const lo = yMin ?? Math.min(0, ...all);
  const hi = yMax ?? Math.max(...all, lo + 1e-9);
  const pad = { l: 44, r: 12, t: 12, b: 34 };
  const pw = width - pad.l - pad.r;
  const ph = height - pad.t - pad.b;
  const n = xLabels.length;
  const x = (i: number) => pad.l + (n <= 1 ? pw / 2 : (i / (n - 1)) * pw);
  const y = (v: number) => pad.t + ph - ((v - lo) / (hi - lo || 1)) * ph;
  const ticks = [lo, (lo + hi) / 2, hi];
  return (
    <div>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" style={{ display: "block", maxWidth: width }}
        onMouseLeave={() => setHover(null)}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} stroke="var(--line)" />
            <text x={pad.l - 6} y={y(t) + 4} fontSize={10} textAnchor="end" fill="var(--ink-3)">{format(t)}</text>
          </g>
        ))}
        {yLabel && <text x={12} y={pad.t + ph / 2} fontSize={10} transform={`rotate(-90 12 ${pad.t + ph / 2})`} textAnchor="middle" fill="var(--ink-3)">{yLabel}</text>}
        {xLabels.map((l, i) => (
          <text key={i} x={x(i)} y={height - 12} fontSize={10} textAnchor="middle" fill="var(--ink-3)">{l}</text>
        ))}
        {markX !== undefined && <line x1={x(markX)} x2={x(markX)} y1={pad.t} y2={pad.t + ph} stroke="var(--gold)" strokeDasharray="4 3" />}
        {series.map((s) => (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color} strokeWidth={2.2} strokeDasharray={s.dashed ? "5 4" : undefined}
              points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
            {s.values.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r={hover === i ? 4.5 : 3} fill={s.color} />)}
          </g>
        ))}
        {xLabels.map((_, i) => (
          <rect key={i} x={x(i) - pw / Math.max(1, n - 1) / 2} y={pad.t} width={pw / Math.max(1, n - 1)} height={ph}
            fill="transparent" onMouseEnter={() => setHover(i)} />
        ))}
      </svg>
      <div className="row small" style={{ gap: 14, marginTop: 4 }}>
        {series.map((s) => (
          <span key={s.name} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 14, height: 3, background: s.color, display: "inline-block", borderRadius: 2 }} />
            {s.name}{hover !== null && <b className="mono">{format(s.values[hover])}</b>}
          </span>
        ))}
        {hover !== null && <span className="muted">at {xLabels[hover]}</span>}
      </div>
    </div>
  );
}

export function Scatter({ points, size = 300, colors, arrows = [], domain, diagonal = false, xLabel, yLabel, radius = 2.6 }: {
  points: [number, number][]; size?: number; colors?: string[];
  arrows?: { to: [number, number]; color: string; label?: string; width?: number; dashed?: boolean }[];
  domain?: [number, number]; diagonal?: boolean; xLabel?: string; yLabel?: string; radius?: number;
}) {
  const ext = domain ?? (() => {
    const m = Math.max(1e-9, ...points.flat().map(Math.abs), ...arrows.flatMap((a) => a.to.map(Math.abs)));
    return [-m * 1.1, m * 1.1] as [number, number];
  })();
  const pad = 24;
  const s = (v: number) => pad + ((v - ext[0]) / (ext[1] - ext[0])) * (size - 2 * pad);
  const sy = (v: number) => size - s(v);
  return (
    <svg width="100%" viewBox={`0 0 ${size} ${size}`} style={{ display: "block", maxWidth: size }} role="img">
      <defs>
        {arrows.map((a, i) => (
          <marker key={i} id={`ah-${i}-${a.color.replace(/[^a-z0-9]/gi, "")}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={a.color} />
          </marker>
        ))}
      </defs>
      <rect x={pad} y={pad} width={size - 2 * pad} height={size - 2 * pad} fill="none" stroke="var(--line)" />
      {ext[0] < 0 && ext[1] > 0 && (
        <g stroke="var(--line)">
          <line x1={s(0)} x2={s(0)} y1={pad} y2={size - pad} />
          <line x1={pad} x2={size - pad} y1={sy(0)} y2={sy(0)} />
        </g>
      )}
      {diagonal && <line x1={s(ext[0])} y1={sy(ext[0])} x2={s(ext[1])} y2={sy(ext[1])} stroke="var(--ink-3)" strokeDasharray="4 4" />}
      {points.map(([px, py], i) => (
        <circle key={i} cx={s(px)} cy={sy(py)} r={radius} fill={colors?.[i] ?? "var(--ink-3)"} fillOpacity={0.55} />
      ))}
      {arrows.map((a, i) => (
        <g key={`a${i}`}>
          <line x1={s(0)} y1={sy(0)} x2={s(a.to[0])} y2={sy(a.to[1])} stroke={a.color} strokeWidth={a.width ?? 2.4}
            strokeDasharray={a.dashed ? "4 3" : undefined}
            markerEnd={`url(#ah-${i}-${a.color.replace(/[^a-z0-9]/gi, "")})`} />
          {a.label && <text x={s(a.to[0] * 1.1)} y={sy(a.to[1] * 1.1) + 4} fontSize={11} textAnchor="middle" fill={a.color}>{a.label}</text>}
        </g>
      ))}
      {xLabel && <text x={size / 2} y={size - 4} fontSize={10} textAnchor="middle" fill="var(--ink-3)">{xLabel}</text>}
      {yLabel && <text x={10} y={size / 2} fontSize={10} textAnchor="middle" fill="var(--ink-3)" transform={`rotate(-90 10 ${size / 2})`}>{yLabel}</text>}
    </svg>
  );
}

/** A categorical palette that reads in both themes. */
export const PALETTE = ["#2f7fc1", "#d9622b", "#2e9d5b", "#b8479b", "#c9a227", "#5b6bd6", "#1fa3a3", "#a0522d",
  "#7a8b99", "#e0457b", "#6a9f2a", "#8e5cc2", "#d4843e", "#3d9bd9", "#9c9c2a", "#c0392b"];
