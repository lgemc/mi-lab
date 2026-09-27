import { useState } from "react";
import { Bars, Heatmap, Lines } from "../components/charts";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Seg, Select, Stat, Tokens } from "../components/ui";
import { fmt, pct, useApi } from "../api";

/*
 * The model behind this page is six hand-written attention heads (api/atlas/ioi.py). Every chart
 * is a real forward pass on the server; nothing is precomputed. Two runs are kept apart on
 * purpose: a clean one for reading the circuit, and one under the reader's ablations.
 */

type Head = { name: string; layer: number; role: string; summary: string };
type Meta = { names: string[]; heads: Head[]; layout: Record<string, [number, number]>; d_model: number; example: string[] };
type Run = {
  tokens: string[];
  patterns: Record<string, number[][]>;
  dla: Record<string, number>;
  logits: number[];
  probs: number[];
  logit_diff: number;
  lens: { layer: number; probs: number[]; logit_diff: number }[];
  end_state: { dup: number[]; inhib: number[]; out: number[] };
  dup_flag: number[];
};
type Mode = "on" | "zero" | "mean";

const HEAD_NAMES = ["L0.0", "L0.1", "L1.0", "L1.1", "L2.0", "L2.1"];
const SHORT: Record<string, string> = {
  "L0.0": "prev-token", "L0.1": "duplicate", "L1.0": "S-inhibition",
  "L1.1": "copier", "L2.0": "name mover", "L2.1": "neg. mover",
};
const NAMES_FALLBACK = ["Mary", "John", "Alice", "Bob", "Sara", "Tom"];
const END = 14;
const S2 = 10;

/** Which position plays which role in the prompt, for labels and highlights. */
function roles(tokens: string[], io: string, s: string) {
  const ioPos = tokens.indexOf(io);
  const s1 = tokens.indexOf(s);
  return { ioPos, s1, s2: S2, end: END };
}

// ------------------------------------------------------------------- the circuit, drawn

type GEdge = { from: string; to: string; port?: "q" | "k" | "v"; label?: string; lit?: boolean; faint?: boolean; lx?: number; ly?: number };

const GX: Record<string, number> = { embed: 46, logits: 624 };
const GY: Record<string, number> = { embed: 130, logits: 130 };
for (const h of HEAD_NAMES) {
  GX[h] = 185 + 150 * Number(h[1]);
  GY[h] = h.endsWith(".0") ? 50 : 210;
}
const NW = 80, NH = 46;

const layerOf = (id: string) => (id === "embed" ? -1 : id === "logits" ? 3 : Number(id[1]));
/**
 * An edge between adjacent columns is one S-curve. An edge that skips a column runs along the
 * empty middle lane (y = 130) and rises to its port at the end, so it never passes behind a node.
 */
function edgePath(a: string, b: string, x1: number, y1: number, x2: number, y2: number, lane = 130) {
  if (layerOf(a) + 1 >= layerOf(b)) {
    const dx = (x2 - x1) / 2;
    return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
  }
  const b2 = 32, xb = x2 - 70;
  const start = y1 === lane ? `M${x1},${lane}` : `M${x1},${y1} C${x1 + b2},${y1} ${x1 + 50 - b2},${lane} ${x1 + 50},${lane}`;
  return `${start} L${xb},${lane} C${xb + b2},${lane} ${x2 - b2},${y2} ${x2},${y2}`;
}

function CircuitDiagram({ edges, selected, onPick, faded = [] }: {
  edges: GEdge[]; selected?: string; onPick?: (h: string) => void; faded?: string[];
}) {
  const portY = (id: string, port?: string) => GY[id] + (port === "q" ? -13 : port === "v" ? 13 : 0);
  const nodes = ["embed", ...HEAD_NAMES, "logits"];
  return (
    <svg viewBox="0 0 672 262" width="100%" style={{ display: "block", maxWidth: 700 }} role="img"
      aria-label="The IOI circuit: embed to duplicate head to S-inhibition head to name mover to logits">
      <defs>
        <marker id="ioi-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
        <marker id="ioi-ah-lit" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
        </marker>
      </defs>
      {edges.map((e, i) => {
        const x1 = GX[e.from] + NW / 2, y1 = GY[e.from];
        const x2 = GX[e.to] - NW / 2 - 1, y2 = portY(e.to, e.port);
        const d = edgePath(e.from, e.to, x1, y1, x2, y2, 130 + (e.port === "q" ? -8 : e.port === "v" ? 8 : 0));
        const color = e.lit ? "var(--accent)" : "var(--ink-3)";
        return (
          <g key={i} opacity={e.faint ? 0.45 : 1}>
            <path d={d} fill="none" stroke={color} strokeWidth={e.lit ? 2.6 : 1.4}
              strokeDasharray={e.faint ? "4 4" : undefined} markerEnd={`url(#ioi-ah${e.lit ? "-lit" : ""})`} />
            {e.label && (
              <text x={(x1 + x2) / 2 + (e.lx ?? 0)} y={(y1 + y2) / 2 - 6 + (e.ly ?? 0)} fontSize={11} textAnchor="middle"
                fill={e.lit ? "var(--accent)" : "var(--ink-3)"} style={{ paintOrder: "stroke", stroke: "var(--surface)", strokeWidth: 4 }}>
                {e.label}
              </text>
            )}
          </g>
        );
      })}
      {nodes.map((id) => {
        const isHead = id.startsWith("L");
        const sel = selected === id;
        const off = faded.includes(id);
        return (
          <g key={id} transform={`translate(${GX[id] - NW / 2},${GY[id] - NH / 2})`} opacity={off ? 0.45 : 1}
            onClick={() => isHead && onPick?.(id)} style={{ cursor: isHead && onPick ? "pointer" : "default" }}>
            <rect width={NW} height={NH} rx={9} fill={sel ? "var(--accent-soft)" : "var(--node)"}
              stroke={sel ? "var(--accent)" : "var(--ink-3)"} strokeWidth={sel ? 2.4 : 1.3}
              strokeDasharray={id === "L0.0" ? "4 3" : undefined} />
            <text x={NW / 2 + (isHead ? 4 : 0)} y={isHead ? 19 : 28} fontSize={13} fontWeight={700} textAnchor="middle">{id}</text>
            {isHead && <text x={NW / 2 + 4} y={35} fontSize={10} textAnchor="middle" fill="var(--ink-2)">{SHORT[id]}</text>}
            {isHead && ["q", "k", "v"].map((p, j) => (
              <text key={p} x={6} y={NH / 2 + (j - 1) * 13 + 3} fontSize={8.5} textAnchor="middle" fill="var(--ink-3)">{p}</text>
            ))}
          </g>
        );
      })}
    </svg>
  );
}

const CIRCUIT_EDGES: GEdge[] = [
  { from: "embed", to: "L0.0", port: "q", faint: true },
  { from: "embed", to: "L0.1", port: "k", label: "names", lit: true, lx: -14, ly: 18 },
  { from: "L0.1", to: "L1.0", port: "k", label: "dup flag", lit: true, lx: 34, ly: 34 },
  { from: "embed", to: "L1.0", port: "v" },
  { from: "L1.0", to: "L2.0", port: "q", label: "inhibit", lit: true, ly: -8 },
  { from: "L1.0", to: "L2.1", port: "q" },
  { from: "embed", to: "L2.0", port: "k" },
  { from: "embed", to: "L1.1", port: "v" },
  { from: "embed", to: "L2.1", port: "v" },
  { from: "L2.0", to: "logits", label: "+IO", lit: true, lx: 12, ly: -4 },
  { from: "L2.1", to: "logits", label: "−IO", lx: 12, ly: 14 },
  { from: "L1.1", to: "logits" },
];

// ------------------------------------------------------------------------------ the page

export default function Ioi() {
  const meta = useApi<Meta>("ioi/meta");
  const names = meta.data?.names ?? NAMES_FALLBACK;
  const heads = meta.data?.heads ?? [];
  const [io, setIo] = useState("Mary");
  const [s, setS] = useState("John");
  const [template, setTemplate] = useState<"ABBA" | "BABA">("ABBA");
  const [head, setHead] = useState("L2.0");
  const [ablate, setAblate] = useState<Record<string, Mode>>({});
  const [lensSrc, setLensSrc] = useState<"clean" | "ablated">("clean");

  const pickIo = (v: string) => { if (v === s) setS(io); setIo(v); };
  const pickS = (v: string) => { if (v === io) setIo(s); setS(v); };

  const clean = useApi<Run>("ioi/run", { io, s, template, ablate: {} });
  const activeAblate = Object.fromEntries(Object.entries(ablate).filter(([, m]) => m !== "on"));
  const abl = useApi<Run>("ioi/run", { io, s, template, ablate: activeAblate });

  const c = clean.data;
  const a = abl.data;
  const r = c ? roles(c.tokens, io, s) : null;
  const ioIdx = names.indexOf(io), sIdx = names.indexOf(s);
  const hl: Record<number, string> = r ? {
    [r.ioPos]: "var(--good-soft)", [r.s1]: "var(--warm-soft)", [r.s2]: "var(--warm-soft)", [r.end]: "var(--gold-soft)",
  } : {};
  const tokLabels = c ? c.tokens.map((t, i) => `${i} ${t}`) : [];
  const headInfo = heads.find((h) => h.name === head);
  const layout = meta.data?.layout;
  const nAblated = Object.keys(activeAblate).length;

  const dlaSum = c ? HEAD_NAMES.reduce((t, h) => t + c.dla[h], 0) : 0;

  return (
    <div className="prose-wide">
      <h2>The task</h2>
      <div className="prose">
        <p>
          Read this sentence and finish it: <em>“When Mary and John went to the store, John gave a drink to …”</em>.
          You said <b>Mary</b> without effort. Two names appeared, one of them appeared <em>twice</em>, and the
          answer is the one that did not repeat. The name that repeats is the <b>subject</b> (S); the other is the{" "}
          <b>indirect object</b> (IO). The task is called <b>indirect object identification</b>, or IOI.
        </p>
        <p>
          Wang et al. (2023), in <em>Interpretability in the Wild</em>, reverse-engineered how GPT-2 small does it.
          They found about 26 attention heads sorted into a handful of classes, each doing one legible step. It was the
          first circuit anyone had mapped end to end in a real language model on a natural task.
        </p>
        <p>
          The model on this page is a miniature of that circuit: three layers, two attention heads each, no MLPs and
          no layer norm. It was <b>written by hand</b>, one head per class, so that every weight has a meaning. It
          runs real softmax attention on real matrices, so everything you poke below is an actual forward pass.
        </p>
        <p>
          We measure the model with one number, the <b>logit difference</b>:
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`\Delta \;=\; \text{logit}(\text{IO}) - \text{logit}(\text{S})`}</TeX>
        <p className="small">
          Positive means the model prefers the right name. Zero means it cannot tell the two apart. Negative means it
          would say the subject, which is wrong. Everything else (probabilities, attributions, patching scores) is
          built on top of <TeX>\Delta</TeX>.
        </p>
      </Card>

      <h2>Six heads, four steps</h2>
      <div className="prose">
        <p>
          Here is the algorithm, one head per step. The blue path is the part that does the work. Each head reads
          three things from the residual stream: a <b>query</b> (q, “what am I looking for?”), a <b>key</b> (k,
          “what do I offer?”) and a <b>value</b> (v, “what do I pass on if you pick me?”). The small letters on each
          box's left edge are those three inputs, and each arrow lands on the one it feeds.
        </p>
      </div>
      <Card caption="Click a head to inspect its attention below. The dashed head is present but unused: IOI never reads what it writes.">
        <CircuitDiagram edges={CIRCUIT_EDGES} selected={head} onPick={setHead} faded={["L0.0"]} />
        <ol className="small" style={{ margin: "10px 0 0", paddingLeft: 20 }}>
          <li><b>L0.1 duplicate-token head.</b> At the second “John”, it finds the earlier “John” and raises a
            flag: <em>this name has been seen before</em>.</li>
          <li><b>L1.0 S-inhibition head.</b> At the last token, it looks for the flag and writes “John” into a
            separate subspace that means <em>do not say this name</em>.</li>
          <li><b>L2.0 name mover.</b> At the last token, its <em>query</em> reads that inhibition and steers its
            attention away from John, onto Mary. It copies what it finds into the output.</li>
          <li>The <b>unembedding</b> reads the output subspace and turns it into logits.</li>
        </ol>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Two supporting heads complicate the picture, as they do in GPT-2: <b>L1.1</b> copies a little of every name
          (a repeated name gets copied twice, so it leans slightly the wrong way) and <b>L2.1</b>, a{" "}
          <b>negative name mover</b>, attends like L2.0 but writes <em>against</em> the answer.
        </p>
      </Card>

      <h2>Pick a prompt</h2>
      <Card>
        <div className="controls">
          <Select label="Indirect object (the answer)" value={io} options={names} onChange={pickIo} />
          <Select label="Subject (repeats)" value={s} options={names} onChange={pickS} />
          <Seg label="Template" value={template} onChange={setTemplate} options={[
            { value: "ABBA", label: "ABBA: IO first" },
            { value: "BABA", label: "BABA: S first" },
          ]} />
        </div>
        <Loading loading={clean.loading} error={clean.error}>
          {c && r && (
            <>
              <Tokens tokens={c.tokens} highlight={hl} />
              <div className="small muted" style={{ marginTop: 6 }}>
                <span style={{ background: "var(--good-soft)", padding: "0 5px", borderRadius: 4 }}>IO</span>{" "}
                <span style={{ background: "var(--warm-soft)", padding: "0 5px", borderRadius: 4 }}>S1, S2</span>{" "}
                <span style={{ background: "var(--gold-soft)", padding: "0 5px", borderRadius: 4 }}>END</span>{" "}
                — position {END}, “to”, is where the model predicts the next token.
              </div>
              <div className="row" style={{ marginTop: 12 }}>
                <Stat k="Logit difference Δ" v={fmt(c.logit_diff, 2)} color={c.logit_diff > 0 ? "var(--good)" : "var(--warm)"} />
                <Stat k={`P(${io})`} v={pct(c.probs[ioIdx], 1)} d="the right answer" />
                <Stat k={`P(${s})`} v={pct(c.probs[sIdx], 1)} d="the subject" />
              </div>
            </>
          )}
        </Loading>
      </Card>

      <h2>Where each head looks</h2>
      <div className="prose">
        <p>
          An attention pattern is a table: row <TeX>i</TeX> says how position <TeX>i</TeX> splits its attention over
          positions <TeX>0 \dots i</TeX> (a head cannot look ahead, so the upper triangle is empty). Every row sums to 1.
          The row that matters most is the last one, <b>END</b>, because that is where the answer is computed.
        </p>
      </div>
      <Card>
        <Seg value={head} onChange={setHead} options={HEAD_NAMES.map((h) => ({ value: h, label: `${h} ${SHORT[h]}` }))} />
        {headInfo && <p className="small" style={{ marginBottom: 4 }}><b>{headInfo.name} · {headInfo.role}.</b> {headInfo.summary}</p>}
        <Loading loading={clean.loading} error={clean.error}>
          {c && (
            <div className="row" style={{ marginTop: 8 }}>
              <div style={{ flex: "0 1 auto", maxWidth: "100%" }}>
                <Heatmap values={c.patterns[head]} rowLabels={tokLabels} colLabels={tokLabels} max={1} signed={false}
                  cell={22} rowLabelWidth={74} highlightRow={END} format={(v) => pct(v, 1)}
                  caption="Rows: the position attending (query). Columns: the position attended to (key). Hover a cell." />
              </div>
              <div className="grow">
                <div className="small" style={{ marginBottom: 4 }}><b>The END row</b>: where position {END} looks</div>
                <Bars width={340} labelWidth={90} rowHeight={20} max={1} format={(v) => pct(v, 0)}
                  items={c.tokens.map((t, i) => ({
                    label: `${i} ${t}`, value: c.patterns[head][END][i],
                    color: i === r?.ioPos ? "var(--good)" : i === r?.s1 || i === S2 ? "var(--warm)" : "var(--ink-3)",
                  }))} />
              </div>
            </div>
          )}
        </Loading>
      </Card>
      <Callout kind="try">
        <p>
          Step through the heads in circuit order. <b>L0.1</b>: only row {S2} (the second {s}) has somewhere to go, and
          it goes to the first {s}; every other row falls back to <code>&lt;bos&gt;</code>, the “nothing to do here”
          sink. <b>L1.0</b>: the END row points at position {S2}, the flagged name. <b>L2.0</b>: the END row points
          at {io}, and almost nowhere else. Then swap the template to BABA and watch the name mover follow {io} to its new
          position.
        </p>
      </Callout>

      <h2>The residual stream, with names on it</h2>
      <div className="prose">
        <p>
          Heads do not talk to each other directly. Each one reads from a shared vector at every position, the{" "}
          <b>residual stream</b>, and adds its output back into it. In this toy the stream has{" "}
          {meta.data?.d_model ?? 65} dimensions, and we chose them so that each block of coordinates means one thing:
        </p>
      </div>
      <Card caption="The layout of the residual stream. Widths are to scale.">
        {layout && <LayoutStrip layout={layout} d={meta.data!.d_model} />}
      </Card>
      <Callout kind="warn" label="This basis is named on purpose">
        <p>
          A real model's residual stream has no labels. Its directions are learned, features share dimensions
          (superposition, later in this atlas), and finding a “duplicate” direction is itself a research project. We
          named the basis so you can watch the algorithm directly. Don't expect the same view of GPT-2.
        </p>
      </Callout>
      <Card title="What the heads leave behind">
        <Loading loading={clean.loading} error={clean.error}>
          {c && (
            <>
              <p className="small" style={{ marginTop: 0 }}>
                The <b>dup</b> coordinate at every position, after all layers. L0.1 raises it at exactly one place:
              </p>
              <Heatmap values={[c.dup_flag]} rowLabels={["dup"]} colLabels={tokLabels} max={1} signed={false}
                cell={24} rowLabelWidth={40} format={(v) => v.toFixed(3)} caption="Hover a cell for its value." />
              <div className="grid2" style={{ marginTop: 8 }}>
                <div>
                  <div className="small"><b>inhib</b> at END (written by L1.0)</div>
                  <Bars width={300} labelWidth={60} rowHeight={20} max={1} format={(v) => v.toFixed(2)}
                    items={names.map((n, i) => ({ label: n, value: c.end_state.inhib[i], color: n === s ? "var(--warm)" : "var(--ink-3)" }))} />
                </div>
                <div>
                  <div className="small"><b>out</b> at END (read by the unembedding)</div>
                  <Bars width={300} labelWidth={60} rowHeight={20} format={(v) => v.toFixed(2)}
                    items={names.map((n, i) => ({ label: n, value: c.end_state.out[i], color: n === io ? "var(--good)" : "var(--ink-3)" }))} />
                </div>
              </div>
              <p className="small muted" style={{ marginBottom: 0 }}>
                “Do not say {s}” sits in <b>inhib</b>; “say {io}” sits in <b>out</b>. The little bit of {s} in <b>out</b>{" "}
                is mostly L1.1, the copier, which picked up both names and {s} twice.
              </p>
            </>
          )}
        </Loading>
      </Card>

      <h2>Who wrote the answer? Direct logit attribution</h2>
      <div className="prose">
        <p>
          The final residual at END is the embedding plus every head's output at END. The logits are a linear
          readout of it. So the logit difference splits into one term per head, exactly:
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`\begin{aligned}\Delta &= \big(x_{\text{embed}} + \textstyle\sum_h o_h\big)\cdot(u_{\text{IO}} - u_{\text{S}}) \\ &= \sum_h \underbrace{o_h \cdot (u_{\text{IO}} - u_{\text{S}})}_{\text{DLA of head } h}\end{aligned}`}</TeX>
        <p className="small">
          Here <TeX>{String.raw`o_h`}</TeX> is head <TeX>h</TeX>'s output at END and <TeX>{String.raw`u_{\text{IO}}`}</TeX>,{" "}
          <TeX>{String.raw`u_{\text{S}}`}</TeX> are the unembedding columns of the two names. The embedding's term is zero
          because the unembedding reads only the <b>out</b> block. This is exact only because the toy has no layer norm
          and no MLPs. In a real model you freeze the final norm's scale, and the split becomes an approximation.
        </p>
        <Loading loading={clean.loading} error={clean.error}>
          {c && (
            <>
              <Bars width={400} labelWidth={130} format={(v) => (v >= 0 ? "+" : "") + v.toFixed(2)}
                items={HEAD_NAMES.map((h) => ({ label: `${h} ${SHORT[h]}`, value: c.dla[h] }))} />
              <div className="row" style={{ marginTop: 10 }}>
                <Stat k="Sum of the bars" v={fmt(dlaSum, 3)} />
                <Stat k="Model's Δ" v={fmt(c.logit_diff, 3)} d="the same number" />
              </div>
            </>
          )}
        </Loading>
      </Card>
      <Callout kind="warn" label="What DLA cannot see">
        <p>
          L0.1 and L1.0 score exactly <b>zero</b>, yet the circuit cannot work without them. DLA only counts what a head
          writes straight into the logits. A head that works by changing <em>another head's attention</em> is
          invisible to it. To see those heads you have to intervene.
        </p>
      </Callout>

      <h2>Knock heads out</h2>
      <div className="prose">
        <p>
          An <b>ablation</b> replaces a head's output with something uninformative and reruns the model. There are two
          common choices. <b>Zero ablation</b> writes zeros. <b>Mean ablation</b> writes the head's output averaged
          over a dataset (here: every ordered pair of names under both templates, 60 prompts), separately at each
          position. Mean ablation is usually thought of as gentler, because it keeps the head's “typical” output
          and only removes what is specific to this prompt.
        </p>
      </div>
      <Card>
        <div className="grid3">
          {HEAD_NAMES.map((h) => (
            <div key={h}>
              <div className="small"><b>{h}</b> <span className="muted">{SHORT[h]}</span></div>
              <Seg value={ablate[h] ?? "on"} onChange={(m) => setAblate((o) => ({ ...o, [h]: m }))} options={[
                { value: "on", label: "on" }, { value: "zero", label: "zero" }, { value: "mean", label: "mean" },
              ]} />
            </div>
          ))}
        </div>
        <div className="row" style={{ marginTop: 12, gap: 8 }}>
          <button className="btn" onClick={() => setAblate({ "L0.1": "mean" })}>Mean-ablate L0.1</button>
          <button className="btn" onClick={() => setAblate({ "L0.1": "zero" })}>Zero-ablate L0.1</button>
          <button className="btn" onClick={() => setAblate({ "L2.1": "zero" })}>Zero L2.1</button>
          <button className="btn" onClick={() => setAblate({})}>Reset</button>
        </div>
        <Loading loading={abl.loading} error={abl.error}>
          {a && c && (
            <>
              <div className="row" style={{ marginTop: 14 }}>
                <Stat k="Δ, all heads on" v={fmt(c.logit_diff, 2)} />
                <Stat k={nAblated ? "Δ, with your ablations" : "Δ (nothing ablated)"} v={fmt(a.logit_diff, 2)}
                  color={a.logit_diff > 0 ? "var(--good)" : "var(--warm)"}
                  d={nAblated ? `${a.logit_diff - c.logit_diff >= 0 ? "+" : ""}${fmt(a.logit_diff - c.logit_diff, 2)} vs clean` : undefined} />
                <Stat k="Model says" v={names[a.probs.indexOf(Math.max(...a.probs))]}
                  color={a.probs.indexOf(Math.max(...a.probs)) === ioIdx ? "var(--good)" : "var(--warm)"} />
              </div>
              <div className="grid2" style={{ marginTop: 12 }}>
                <div>
                  <div className="small"><b>Probabilities over the six names</b></div>
                  <Bars width={320} labelWidth={60} rowHeight={22} max={1} format={(v) => pct(v, 1)}
                    items={names.map((n, i) => ({ label: n, value: a.probs[i], color: n === io ? "var(--good)" : n === s ? "var(--warm)" : "var(--ink-3)" }))} />
                </div>
                <div>
                  <div className="small"><b>Direct logit attribution, ablated run</b></div>
                  <Bars width={320} labelWidth={60} rowHeight={22} format={(v) => (v >= 0 ? "+" : "") + v.toFixed(2)}
                    items={HEAD_NAMES.map((h) => ({ label: h, value: a.dla[h] }))} />
                </div>
              </div>
            </>
          )}
        </Loading>
      </Card>
      <div className="prose">
        <p>Three results are worth pausing on.</p>
        <p>
          <b>Zero-ablating L0.1 breaks the circuit; mean-ablating it does nothing at all.</b> Without the duplicate
          flag, L1.0 has nothing to find, nothing gets inhibited, and the name mover splits its attention between both
          names. The copier's extra vote for the repeated name then wins, and the model says {s}. But the mean ablation
          leaves Δ unchanged to the last digit. Why? In this fixed template the repeated name is <em>always</em> at
          position {S2}, so the duplicate head's output at each position is the same on every prompt. Its mean is
          its output. Mean ablation removes only what <em>varies</em> across the dataset, and here the head's
          information is carried by position, which never varies.
        </p>
        <p>
          <b>Zeroing L2.1 makes the model better.</b> The negative name mover writes against the answer, so removing
          it raises Δ from about 2.4 to about 3.8. A head can be part of the circuit, and matter a great deal, while
          pushing the wrong way.
        </p>
        <p>
          <b>Zeroing L0.0 changes nothing.</b> The previous-token head runs, writes, and nobody reads it. Being active
          is not the same as being used.
        </p>
      </div>
      <Callout kind="warn" label="Choose your ablation deliberately">
        <p>
          Zero and mean ablation ask different questions, and neither is “the” effect of a head. Zeros can push a
          model somewhere it never goes in normal use. Means can hide information that is constant across your
          dataset but still used. The next lesson swaps a head's output for its value on a <em>corrupted</em> prompt,
          which lets you choose exactly which information to remove.
        </p>
      </Callout>

      <h2>When does the answer appear?</h2>
      <div className="prose">
        <p>
          The <b>logit lens</b> decodes the residual stream at END after each layer as if the model stopped there:
          apply the unembedding to the partial sum and read the probabilities. (A later lesson is all about lenses.)
        </p>
      </div>
      <Card>
        <Seg value={lensSrc} onChange={setLensSrc} options={[
          { value: "clean", label: "Clean run" },
          { value: "ablated", label: `With your ablations${nAblated ? ` (${nAblated})` : ""}` },
        ]} />
        {(() => {
          const run = lensSrc === "clean" ? c : a;
          if (!run) return null;
          const xl = ["embed", "L0", "L1", "L2"];
          return (
            <div className="grid2" style={{ marginTop: 10 }}>
              <div>
                <div className="small"><b>Probability at END</b></div>
                <Lines width={340} height={200} xLabels={xl} yMin={0} yMax={1} format={(v) => pct(v, 0)}
                  series={[
                    { name: `P(${io})`, values: run.lens.map((l) => l.probs[ioIdx]), color: "var(--good)" },
                    { name: `P(${s})`, values: run.lens.map((l) => l.probs[sIdx]), color: "var(--warm)" },
                  ]} />
              </div>
              <div>
                <div className="small"><b>Logit difference at END</b></div>
                <Lines width={340} height={200} xLabels={xl} format={(v) => v.toFixed(1)}
                  series={[{ name: "Δ", values: run.lens.map((l) => l.logit_diff), color: "var(--accent)" }]} />
              </div>
            </div>
          );
        })()}
        <p className="small muted" style={{ marginBottom: 0 }}>
          After the embedding and after layer 0 the lens sees nothing: every name sits at 1/6. After layer 1 it sees
          only the copier's faint vote for the wrong name. The duplicate flag and the inhibition are real, but they
          live in subspaces the unembedding does not read. The answer shows up only after layer 2,
          when the name movers write into <b>out</b>. A readout sees only what is written in its own language.
        </p>
      </Card>

      <Callout kind="takeaway">
        <p>
          IOI is solved by a short algorithm: flag the repeated name, inhibit it, copy the other one. Each tool on
          this page sees a different slice of it. Attention patterns show where heads look. DLA shows who writes the
          answer, exactly, but misses heads that act through other heads. Ablations reveal those heads, but the
          answer depends on what you ablate <em>to</em>. The logit lens shows when the answer becomes readable, not
          when the work starts.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          In GPT-2 small, Wang et al. (2023) found the same classes with several heads each: duplicate-token and
          induction heads, S-inhibition heads, name movers, negative name movers, and <b>backup name movers</b>, which
          take over when the main ones are ablated. That redundancy, a form of self-repair (McGrath et al., 2023,
          “The Hydra Effect”), means a single-head ablation can understate how much a head matters. Wang et al. traced the wiring with path
          patching (Goldowsky-Dill et al., 2023) rather than plain ablation. Real residual
          streams are unlabeled, so the “subspaces” here correspond to directions that have to be discovered, for
          example with probes or sparse autoencoders. On choosing between zero, mean and resample ablations, see
          Chan et al. (2022, causal scrubbing) and Zhang &amp; Nanda (2024).
        </p>
      </Callout>
    </div>
  );
}

function LayoutStrip({ layout, d }: { layout: Record<string, [number, number]>; d: number }) {
  const DESC: Record<string, string> = {
    tok: "which token", pos: "which position", const: "always 1", isname: "is a name", bos: "is <bos>",
    dup: "seen before", prev: "previous token", inhib: "don't say", out: "say this",
  };
  const COLOR: Record<string, string> = {
    tok: "var(--surface-2)", pos: "var(--surface-2)", const: "var(--surface-2)", isname: "var(--surface-2)",
    bos: "var(--surface-2)", dup: "var(--warm-soft)", prev: "var(--surface-2)", inhib: "var(--gold-soft)", out: "var(--good-soft)",
  };
  const entries = Object.entries(layout);
  return (
    <div>
      <div style={{ display: "flex", height: 26, borderRadius: 6, overflow: "hidden", border: "1px solid var(--line)" }}>
        {entries.map(([k, [a, b]]) => (
          <div key={k} title={`${k}: ${b - a} dims`} style={{
            flex: `${b - a} 0 0`, background: COLOR[k], borderRight: "1px solid var(--line)", minWidth: 3,
          }} />
        ))}
      </div>
      <div className="grid3" style={{ marginTop: 10, gap: 6 }}>
        {entries.map(([k, [a, b]]) => (
          <div key={k} className="small">
            <span style={{ display: "inline-block", width: 10, height: 10, background: COLOR[k], border: "1px solid var(--ink-3)", borderRadius: 2, marginRight: 6 }} />
            <b className="mono">{k}</b> <span className="muted">({b - a})</span> {DESC[k] ?? ""}
          </div>
        ))}
      </div>
      <p className="small muted" style={{ marginBottom: 0 }}>
        {d} dimensions in total. The embedding fills tok, pos, const, isname and bos. L0.0 writes prev, L0.1 writes
        dup, L1.0 writes inhib, and the three movers write out. The unembedding reads out and nothing else.
      </p>
    </div>
  );
}
