/*
 * A causal graph, drawn. Positions are given in a 0..100 box so a lesson can lay a graph out by
 * hand (the layout of a causal diagram carries meaning -- causes on the left, time flowing right).
 * Nodes can be conditioned on (filled), latent (dashed), or be X / Y; edges can be cut (do-surgery)
 * or lit up as part of a path.
 */
import { ReactNode, useId } from "react";

export type DagNode = { id: string; x: number; y: number; latent?: boolean; label?: string };
export type DagEdge = [string, string];

export function Dag({ nodes, edges, width = 460, height = 260, given = [], role = {}, cut = [], lit = [],
  litColor = "var(--good)", onNodeClick, nodeValue, radius = 22, edgeLabels }: {
  nodes: DagNode[]; edges: DagEdge[]; width?: number; height?: number; given?: string[];
  role?: Record<string, "x" | "y" | "m">; cut?: DagEdge[]; lit?: DagEdge[]; litColor?: string;
  onNodeClick?: (id: string) => void; nodeValue?: Record<string, ReactNode>; radius?: number;
  edgeLabels?: Record<string, string>;
}) {
  // Marker ids are document-global: with a fixed id, every graph on a page would borrow the first
  // graph's arrowhead colours.
  const uid = useId().replace(/:/g, "");
  const pos = Object.fromEntries(nodes.map((n) => [n.id, { x: 30 + (n.x / 100) * (width - 60), y: 30 + (n.y / 100) * (height - 60) }]));
  const has = (list: DagEdge[], a: string, b: string) => list.some(([p, q]) => (p === a && q === b) || (p === b && q === a));
  const roleColor = (id: string) => role[id] === "x" ? "var(--accent)" : role[id] === "y" ? "var(--warm)" : role[id] === "m" ? "var(--gold)" : "var(--ink-2)";
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block", maxWidth: width }} role="img">
      <defs>
        {["ink", "lit", "cut"].map((k) => (
          <marker key={k} id={`dag-${uid}-${k}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={k === "lit" ? litColor : k === "cut" ? "var(--line)" : "var(--ink-3)"} />
          </marker>
        ))}
      </defs>
      {edges.map(([a, b]) => {
        const p = pos[a], q = pos[b];
        if (!p || !q) return null;
        const dx = q.x - p.x, dy = q.y - p.y, len = Math.hypot(dx, dy) || 1;
        const x1 = p.x + (dx / len) * radius, y1 = p.y + (dy / len) * radius;
        const x2 = q.x - (dx / len) * (radius + 3), y2 = q.y - (dy / len) * (radius + 3);
        const isCut = has(cut, a, b) && cut.some(([u, v]) => u === a && v === b);
        const isLit = has(lit, a, b);
        const label = edgeLabels?.[`${a}->${b}`];
        return (
          <g key={`${a}-${b}`}>
            <line x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={isCut ? "var(--line)" : isLit ? litColor : "var(--ink-3)"} strokeWidth={isLit ? 3.2 : 1.7}
              strokeDasharray={isCut ? "5 5" : undefined}
              markerEnd={`url(#dag-${uid}-${isCut ? "cut" : isLit ? "lit" : "ink"})`} />
            {isCut && <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 4} fontSize={14} textAnchor="middle" fill="var(--warm)">✂</text>}
            {label && !isCut && <text x={(x1 + x2) / 2 + 6} y={(y1 + y2) / 2 - 4} fontSize={10} fill="var(--ink-3)">{label}</text>}
          </g>
        );
      })}
      {nodes.map((n) => {
        const p = pos[n.id];
        const g = given.includes(n.id);
        return (
          <g key={n.id} onClick={() => onNodeClick?.(n.id)} style={{ cursor: onNodeClick ? "pointer" : "default" }}>
            <circle cx={p.x} cy={p.y} r={radius} fill={g ? "var(--ink-2)" : "var(--node)"}
              stroke={roleColor(n.id)} strokeWidth={role[n.id] ? 3 : 1.6} strokeDasharray={n.latent ? "4 3" : undefined} />
            <text x={p.x} y={p.y + 4} fontSize={13} fontWeight={600} textAnchor="middle"
              fill={g ? "var(--surface)" : "var(--ink)"}>{n.label ?? n.id}</text>
            {nodeValue?.[n.id] !== undefined && (
              <text x={p.x} y={p.y + radius + 14} fontSize={11} textAnchor="middle" fill="var(--ink-2)" className="mono">{nodeValue[n.id]}</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
