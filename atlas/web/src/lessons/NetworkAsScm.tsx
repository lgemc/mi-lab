import { useState } from "react";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Slider, Stat } from "../components/ui";
import { fmt, pct, useApi } from "../api";

/*
 * The seven-neuron network of api/atlas/mlp.py, drawn as a causal graph. The diagram component is
 * exported so the patching lesson can draw the same picture with its own colouring.
 */

export type NetEdge = { source: string; target: string; weight: number };
export type NetValues = Record<string, number>;
type Meta = {
  inputs: string[]; layer1: string[]; layer2: string[]; edges: NetEdge[];
  descriptions: Record<string, string>;
  truth_table: { a: number; b: number; c: number; prob: number; target: number }[];
};

/** Hand-placed positions in a 520 x 330 box: inputs on the left, the output on the right. */
const POS: Record<string, [number, number]> = {
  a: [48, 70], b: [48, 165], c: [48, 260],
  n1: [196, 42], n2: [196, 122], n3: [196, 208], n4: [196, 288],
  m1: [344, 110], m2: [344, 250],
  out: [468, 165],
};
export const HIDDEN = ["n1", "n2", "n3", "n4", "m1", "m2"];

/** The same edges the server sends, so the patching lesson can draw without a round trip. */
export const EDGES: NetEdge[] = [
  { source: "a", target: "n1", weight: 1 }, { source: "b", target: "n1", weight: 1 },
  { source: "a", target: "n2", weight: 1 }, { source: "b", target: "n2", weight: 1 },
  { source: "c", target: "n3", weight: 1 },
  { source: "a", target: "n4", weight: -1 }, { source: "b", target: "n4", weight: 1 },
  { source: "n1", target: "m1", weight: 1 }, { source: "n2", target: "m1", weight: 1 }, { source: "n3", target: "m1", weight: 1 },
  { source: "n4", target: "m2", weight: 1 },
  { source: "m1", target: "out", weight: 6 }, { source: "m2", target: "out", weight: -1 },
];

const R = 23;

export function Net({ edges = EDGES, values, cut = [], selected, onNodeClick, ring = {}, sub = {} }: {
  edges?: NetEdge[]; values?: NetValues; cut?: string[]; selected?: string | null;
  onNodeClick?: (id: string) => void; ring?: Record<string, string>; sub?: Record<string, string>;
}) {
  return (
    <svg viewBox="0 0 520 330" width="100%" style={{ display: "block", maxWidth: 560 }} role="img"
      aria-label="A network with inputs a, b, c, hidden neurons n1 to n4 and m1, m2, and one output">
      <defs>
        {["ink", "neg", "cut"].map((k) => (
          <marker key={k} id={`net-${k}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={k === "neg" ? "var(--warm)" : k === "cut" ? "var(--line)" : "var(--accent)"} />
          </marker>
        ))}
      </defs>
      {edges.map((e) => {
        const [x1, y1] = POS[e.source], [x2, y2] = POS[e.target];
        const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
        const ax = x1 + (dx / len) * R, ay = y1 + (dy / len) * R;
        const bx = x2 - (dx / len) * (R + 3), by = y2 - (dy / len) * (R + 3);
        const isCut = cut.includes(e.target);
        const col = isCut ? "var(--line)" : e.weight < 0 ? "var(--warm)" : "var(--accent)";
        const lx = ax + (bx - ax) * 0.72, ly = ay + (by - ay) * 0.72;
        return (
          <g key={`${e.source}-${e.target}`}>
            <line x1={ax} y1={ay} x2={bx} y2={by} stroke={col} strokeOpacity={isCut ? 1 : 0.75}
              strokeWidth={isCut ? 1.4 : 1 + Math.min(3, Math.abs(e.weight) * 0.6)}
              strokeDasharray={isCut ? "5 4" : undefined}
              markerEnd={`url(#net-${isCut ? "cut" : e.weight < 0 ? "neg" : "ink"})`} />
            {isCut
              ? <text x={lx} y={ly + 5} fontSize={14} textAnchor="middle" fill="var(--warm)">✂</text>
              : <text x={lx} y={ly + 4} fontSize={10.5} textAnchor="middle" className="mono"
                  style={{ paintOrder: "stroke", stroke: "var(--surface)", strokeWidth: 4, fill: col }}>
                  {e.weight > 0 ? `+${e.weight}` : e.weight}
                </text>}
          </g>
        );
      })}
      {Object.entries(POS).map(([id, [x, y]]) => {
        const v = values?.[id];
        const isInput = id === "a" || id === "b" || id === "c";
        const a = v === undefined ? 0 : Math.min(1, Math.abs(id === "out" ? (values?.prob ?? 0) : v) / (isInput ? 1 : 2));
        const fill = v === undefined || a < 0.01 ? "var(--node)"
          : `rgba(var(${v < 0 && id !== "out" ? "--neg" : "--pos"}), ${(0.12 + 0.45 * a).toFixed(3)})`;
        const clickable = !!onNodeClick;
        return (
          <g key={id} onClick={() => onNodeClick?.(id)} style={{ cursor: clickable ? "pointer" : "default" }}>
            {selected === id && <circle cx={x} cy={y} r={R + 6} fill="none" stroke="var(--accent)" strokeWidth={2} strokeDasharray="3 3" />}
            <circle cx={x} cy={y} r={R} fill={fill} stroke={ring[id] ?? (cut.includes(id) ? "var(--gold)" : "var(--ink-3)")}
              strokeWidth={ring[id] || cut.includes(id) ? 3.2 : 1.5} />
            <text x={x} y={v === undefined ? y + 4 : y - 3} fontSize={12.5} fontWeight={700} textAnchor="middle">{id}</text>
            {v !== undefined && (
              <text x={x} y={y + 12} fontSize={10.5} textAnchor="middle" className="mono">{fmt(v, isInput ? 0 : 2)}</text>
            )}
            {sub[id] && <text x={x} y={y + R + 13} fontSize={10.5} textAnchor="middle" fill="var(--ink-2)">{sub[id]}</text>}
          </g>
        );
      })}
    </svg>
  );
}

const EQUATIONS = String.raw`\begin{aligned}
n_1 &:= \mathrm{relu}(a + b - 1) && \text{a AND b}\\
n_2 &:= \mathrm{relu}(a + b - 1) && \text{the same AND, a backup copy}\\
n_3 &:= \mathrm{relu}(c) && \text{copies } c\\
n_4 &:= \mathrm{relu}(b - a) && \text{b but not a: a distractor}\\
m_1 &:= \mathrm{relu}(n_1 + n_2 + n_3) && \text{OR of the evidence}\\
m_2 &:= \mathrm{relu}(n_4 - 0.2) && \text{passes } n_4 \text{ on, weakly}\\
\text{out} &:= 6\,m_1 - m_2 - 3, \quad p = \sigma(\text{out}) && \text{logit and probability of "yes"}
\end{aligned}`;

const PRESETS: { label: string; x: number[]; set: Record<string, number>; note: string }[] = [
  { label: "Knock out n1", x: [1, 1, 0], set: { n1: 0 }, note: "n2 covers for it: the answer stays yes." },
  { label: "Knock out n1 and n2", x: [1, 1, 0], set: { n1: 0, n2: 0 }, note: "Now nothing carries a AND b, and the answer flips." },
  { label: "Force m1 = 1 with no evidence", x: [0, 0, 0], set: { m1: 1 }, note: "Every input is 0, yet the network says yes." },
  { label: "Turn the distractor up", x: [0, 1, 0], set: { n4: 3 }, note: "n4 only pushes a 'no' further down. It never decides an answer." },
];

export default function NetworkAsScm() {
  const [x, setX] = useState<number[]>([1, 1, 0]);
  const [set, setSet] = useState<Record<string, number>>({});
  const [sel, setSel] = useState<string | null>("n1");
  const [note, setNote] = useState<string | null>(null);
  const meta = useApi<Meta>("mlp/meta");
  const run = useApi<NetValues>("mlp/run", { x, set });
  const plain = useApi<NetValues>("mlp/run", { x, set: {} });

  const click = (id: string) => {
    setNote(null);
    const i = ["a", "b", "c"].indexOf(id);
    if (i >= 0) setX((old) => old.map((v, j) => (j === i ? 1 - v : v)));
    else if (id !== "out") setSel(id);
  };
  const intervened = Object.keys(set);
  const selOn = sel !== null && sel in set;
  const v = run.data;
  const target = (x[0] && x[1]) || x[2] ? 1 : 0;
  const descr = meta.data?.descriptions ?? {};

  return (
    <div className="prose-wide">
      <h2>Seven neurons, one Boolean function</h2>
      <div className="prose">
        <p>
          Here is a network small enough to hold in your head. It reads three bits <TeX>a, b, c</TeX> and should
          answer <b>yes</b> exactly when <TeX>{String.raw`(a \wedge b) \vee c`}</TeX>. Nobody trained it; the
          weights were set by hand so we know the ground truth, which is the luxury real interpretability never has.
        </p>
        <p>
          Now look at the picture again, but with the eyes of lesson 1. Each circle is a <b>variable</b>. Each arrow
          says "this variable is computed from that one". Each neuron's formula is a <b>structural equation</b>: the
          mechanism that sets its value from its parents. That is all a structural causal model is, so a neural
          network already is one. Nothing needs translating.
        </p>
      </div>

      <Card title="The network, live" caption="Click an input (a, b, c) to flip it. Click a hidden neuron to select it, then intervene on it below. Blue arrows are positive weights, orange negative; the fill shows how strongly each neuron is firing.">
        <div className="row">
          <div style={{ flex: "1 1 320px" }}>
            <Loading loading={run.loading} error={run.error}>
              <Net edges={meta.data?.edges} values={v ?? undefined} cut={intervened} selected={sel} onNodeClick={click}
                sub={v ? { out: `p = ${fmt(v.prob, 4)}` } : {}} />
            </Loading>
          </div>
          <div className="grow" style={{ flex: "1 1 240px" }}>
            <div className="control" style={{ marginBottom: 10 }}>
              <span>Inputs</span>
              <div className="row" style={{ gap: 6 }}>
                {["a", "b", "c"].map((n, i) => (
                  <button key={n} className={`chip ${x[i] ? "on" : ""}`} onClick={() => click(n)}>
                    {n} = {x[i]}
                  </button>
                ))}
              </div>
            </div>
            {sel && (
              <div style={{ borderTop: "1px solid var(--line)", paddingTop: 10 }}>
                <div className="small"><b className="mono">{sel}</b> <span className="muted">· {descr[sel] ?? ""}</span></div>
                <label className="small" style={{ display: "flex", gap: 6, alignItems: "center", margin: "6px 0" }}>
                  <input type="checkbox" checked={selOn} onChange={(e) => {
                    setNote(null);
                    setSet((old) => {
                      const d = { ...old };
                      if (e.target.checked) d[sel] = Math.round((plain.data?.[sel] ?? 0) * 100) / 100;
                      else delete d[sel];
                      return d;
                    });
                  }} />
                  intervene: <TeX>{`do(${sel.replace(/(\d)/, "_$1")} = v)`}</TeX>
                </label>
                {selOn && (
                  <Slider label={<>v</>} value={set[sel]} min={-1} max={3} step={0.05}
                    onChange={(val) => setSet((old) => ({ ...old, [sel]: val }))} format={(val) => val.toFixed(2)} />
                )}
              </div>
            )}
            <div className="row" style={{ marginTop: 10, gap: 10 }}>
              <Stat k="P(yes)" v={v ? pct(v.prob, 2) : "…"} color={v && (v.prob > 0.5) === !!target ? "var(--good)" : "var(--warm)"}
                d={`correct answer: ${target ? "yes" : "no"}`} />
              {intervened.length > 0 && plain.data && (
                <Stat k="without do()" v={pct(plain.data.prob, 2)} d="same inputs, untouched" />
              )}
            </div>
            {intervened.length > 0 && (
              <button className="btn" style={{ marginTop: 10 }} onClick={() => { setSet({}); setNote(null); }}>
                Release all interventions
              </button>
            )}
          </div>
        </div>
        <div className="row" style={{ gap: 6, marginTop: 12 }}>
          <span className="small muted" style={{ alignSelf: "center" }}>Try:</span>
          {PRESETS.map((p) => (
            <button key={p.label} className="btn" onClick={() => {
              setX(p.x); setSet(p.set); setSel(Object.keys(p.set)[0]); setNote(p.note);
            }}>{p.label}</button>
          ))}
        </div>
        {note && <p className="small" style={{ marginBottom: 0 }}><b>What happened:</b> {note}</p>}
      </Card>

      <h2>Weights are structural equations</h2>
      <div className="prose">
        <p>
          Written out, the arrows and their weights are just these seven assignments. The <TeX>{":="}</TeX> is
          deliberate: it means "is computed from", not "equals". Equality is symmetric; causation runs one way.
        </p>
      </div>
      <Card>
        <TeX block>{EQUATIONS}</TeX>
        <p className="small muted">
          <TeX>{String.raw`\mathrm{relu}(z) = \max(z, 0)`}</TeX> and <TeX>{String.raw`\sigma(z) = 1/(1+e^{-z})`}</TeX>.
          The bias <TeX>-1</TeX> in <TeX>n_1</TeX> is what turns "add <TeX>a</TeX> and <TeX>b</TeX>" into "both".
        </p>
      </Card>

      <div className="prose">
        <p>
          Two neurons are there on purpose, because they are small versions of what makes real networks hard to read:
        </p>
        <ul>
          <li>
            <b>n2 is a backup copy of n1.</b> Both compute <TeX>a \wedge b</TeX>, and <TeX>m_1</TeX> adds them.
            Knock out either one and the other still pushes the answer to yes. Real models are full of this
            redundancy; the next lesson shows how it makes two reasonable importance tests disagree.
          </li>
          <li>
            <b>n4 is a distractor.</b> It fires on "b without a", which correlates with the inputs but never changes an
            answer: when it fires, the answer is already no (unless <TeX>c</TeX> overrides everything). A neuron can be
            active and correlated with the task and still not matter.
          </li>
        </ul>
      </div>

      <h2>do() is overwriting an activation</h2>
      <div className="prose">
        <p>
          When you ticked "intervene" above, three things happened, and they are exactly Pearl's recipe:
        </p>
        <ol>
          <li>The neuron's own equation was thrown away. Its parents no longer matter, so the arrows into it are <b>cut</b> (the ✂ marks). That's graph surgery, as in lesson 1.</li>
          <li>The neuron was set to the constant <TeX>v</TeX>.</li>
          <li>Everything <em>downstream</em> was recomputed with the usual equations. Everything upstream stayed as it was.</li>
        </ol>
        <p>
          In a real model you do this with a <b>forward hook</b>: a function that runs when a layer produces its
          output and replaces part of it before the next layer sees it.
        </p>
      </div>
      <Card>
        <pre className="mono scroll-x" style={{ background: "var(--surface-2)", padding: "10px 12px", borderRadius: 8, margin: 0, fontSize: "0.8rem", lineHeight: 1.5 }}>
{`# do(n1 = 0): overwrite unit 0 of layer 1, then let the forward pass continue
def do_n1(module, inputs, output):
    output = output.clone()
    output[..., 0] = 0.0
    return output          # returning a value replaces the layer's output

handle = model.layer1.register_forward_hook(do_n1)
p = model(x)               # everything after layer1 sees the intervened value
handle.remove()`}
        </pre>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Libraries such as TransformerLens (hook points) and nnsight wrap exactly this pattern. Every
          method in the next lessons, patching, ablation, steering, mediation, is some choice of <em>which</em>{" "}
          activation to overwrite and <em>with what</em>.
        </p>
      </Card>

      <Card title="The truth table" caption="What the untouched network says for all eight inputs. The highlighted row is the input selected above.">
        <Loading loading={meta.loading} error={meta.error}>
          <div className="scroll-x">
            <table className="tbl" style={{ maxWidth: 480 }}>
              <thead><tr><th>a</th><th>b</th><th>c</th><th>(a ∧ b) ∨ c</th><th>P(yes)</th></tr></thead>
              <tbody>
                {(meta.data?.truth_table ?? []).map((r) => {
                  const here = r.a === x[0] && r.b === x[1] && r.c === x[2];
                  return (
                    <tr key={`${r.a}${r.b}${r.c}`} style={here ? { background: "var(--accent-soft)" } : undefined}>
                      <td className="mono">{r.a}</td><td className="mono">{r.b}</td><td className="mono">{r.c}</td>
                      <td>{r.target ? "yes" : "no"}</td>
                      <td className="mono" style={{ color: (r.prob > 0.5) === !!r.target ? "var(--good)" : "var(--warm)" }}>{fmt(r.prob, 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Loading>
      </Card>
      <div className="prose">
        <p>
          The network is right on all eight rows, and confidently so when the answer is yes: with both <TeX>n_1</TeX>{" "}
          and <TeX>n_2</TeX> firing the logit is <TeX>9</TeX> and <TeX>p \approx 0.9999</TeX>. That surplus of
          confidence (<b>saturation</b>) is the second thing, after redundancy, that the next lesson has to deal with.
        </p>
      </div>

      <Callout kind="takeaway">
        <p>
          A neural network is a structural causal model with neurons as variables and weighted sums plus
          nonlinearities as the structural equations. <TeX>do(n = v)</TeX> means: cut the neuron off from its inputs,
          pin it to <TeX>v</TeX>, and recompute everything downstream. In code that's a forward hook.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          The framing goes back to Geiger et al. (2021, "Causal abstraction of neural networks"), who treat a network
          and a hypothesised algorithm as two causal models and test whether interventions on one match
          interventions on the other; Vig et al. (2020) used it to run causal mediation on individual neurons and
          heads, and Meng et al. (2022, ROME) used activation interventions to locate factual recall in GPT. Real
          "variables" are rarely single neurons: they are directions in activation space, attention-head outputs or
          SAE features, which is why later lessons spend so long asking <em>which</em> variables to intervene on.
        </p>
      </Callout>
    </div>
  );
}
