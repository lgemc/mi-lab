import { useState } from "react";
import { Heatmap } from "../components/charts";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Slider, Stat } from "../components/ui";
import { fmt, pct, useApi } from "../api";
import { Presets, describeRow } from "./dictHelpers";

/*
 * A trained 5 -> 4 -> 5 MLP whose neurons are forced to share jobs, and a transcoder trained to
 * imitate it (api/atlas/transcoder.py). The first half compares what neurons and latents read and
 * write; the second draws the attribution graph the server computes for one input.
 */

type Tc = {
  in_features: string[]; out_features: string[];
  neuron_in: number[][]; neuron_out: number[][]; latent_in: number[][]; latent_out: number[][];
  latent_bias: number[]; alive: boolean[]; fvu: number; mlp_task_fvu: number; l0: number; neuron_l0: number;
};
type Graph = {
  latents: { id: number; activation: number }[];
  in_edges: { source: string; latent: number; weight: number }[];
  out_edges: { latent: number; target: string; weight: number }[];
  mlp_out: number[]; transcoder_out: number[]; bias_out: number[]; error: number[]; task_out: number[];
};

const IN = ["a", "b", "c", "d", "e"];
const OUT = ["a", "b", "a AND b", "c", "d NOT e"];

const INPUT_PRESETS: { label: string; value: number[] }[] = [
  { label: "a and b", value: [1, 1, 0, 0, 0] },
  { label: "a only", value: [1, 0, 0, 0, 0] },
  { label: "c only", value: [0, 0, 1, 0, 0] },
  { label: "d, no e", value: [0, 0, 0, 1, 0] },
  { label: "d and e", value: [0, 0, 0, 1, 1] },
  { label: "a, b, c, d", value: [1, 1, 1, 1, 0] },
];

export default function Transcoders() {
  const [latents, setLatents] = useState(8);
  const [l1, setL1] = useState(0.1);
  const [f, setF] = useState<number[]>([1, 1, 0, 0, 0]);
  const tc = useApi<Tc>("transcoder", { latents, l1 }, 250);
  const graph = useApi<Graph>("transcoder/graph", { latents, l1, features: f }, 250);
  const d = tc.data;
  const aliveIds = d ? d.alive.map((a, i) => (a ? i : -1)).filter((i) => i >= 0) : [];

  return (
    <div className="prose-wide">
      <h2>An MLP that has to share</h2>
      <div className="prose">
        <p>
          SAEs find the variables inside one activation. But a model <em>computes</em>: an MLP layer reads some
          features and writes others. To explain the computation we want to know which features feed which, and
          the MLP's neurons are the wrong place to look. Here is a small MLP that shows why. It gets five sparse
          input features and has to produce five outputs:
        </p>
      </div>
      <Card>
        <div className="scroll-x">
          <table className="tbl" style={{ maxWidth: 520 }}>
            <thead><tr><th>output</th><th>target</th><th>what it needs</th></tr></thead>
            <tbody>
              <tr><td>a</td><td className="mono">a</td><td>copy</td></tr>
              <tr><td>b</td><td className="mono">b</td><td>copy</td></tr>
              <tr><td>a AND b</td><td className="mono">min(a, b)</td><td>a nonlinearity</td></tr>
              <tr><td>c</td><td className="mono">c</td><td>copy</td></tr>
              <tr><td>d NOT e</td><td className="mono">relu(d − e)</td><td>a nonlinearity</td></tr>
            </tbody>
          </table>
        </div>
        <TeX block>{String.raw`y = W_2\,\mathrm{ReLU}(W_1 x + b_1) + b_2, \qquad x \in \mathbb{R}^5,\; 4 \text{ hidden neurons},\; y \in \mathbb{R}^5`}</TeX>
        <p className="small">
          Five jobs, four neurons, so there is no neuron-per-job solution. Features live on random orthonormal
          directions in the input and output spaces, so nothing is aligned with a neuron by accident. The MLP is
          trained by gradient descent. It gets the task mostly right: it leaves{" "}
          {d ? <b>{pct(d.mlp_task_fvu, 1)}</b> : "a few percent"} of the target's variance unexplained.
        </p>
      </Card>

      <Loading loading={tc.loading} error={tc.error}>
        {d && (
          <Card title="What each neuron reads and writes"
            caption="Left: how strongly each neuron's input weights line up with each input feature. Right: how much each neuron adds to each output feature per unit of activation.">
            <div className="grid2">
              <Heatmap values={d.neuron_in} rowLabels={["n1", "n2", "n3", "n4"]} colLabels={d.in_features} cell={34} max={1.2} rowLabelWidth={36}
                caption="reads from input features" />
              <Heatmap values={d.neuron_out} rowLabels={["n1", "n2", "n3", "n4"]} colLabels={d.out_features} cell={34} max={1.4} rowLabelWidth={36}
                caption="writes to output features" />
            </div>
            <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
              {d.neuron_in.map((row, i) => (
                <li key={i}><b>n{i + 1}</b> reads {describeRow(row, d.in_features)}; writes {describeRow(d.neuron_out[i], d.out_features)}</li>
              ))}
            </ul>
          </Card>
        )}
      </Loading>
      <div className="prose">
        <p>
          Most neurons are <b>polysemantic</b>: one reads <TeX>a</TeX> and <TeX>b</TeX> together and writes{" "}
          <TeX>a</TeX>, <TeX>b</TeX> and <TeX>a</TeX> AND <TeX>b</TeX>. Two others each mix <TeX>a</TeX>,{" "}
          <TeX>b</TeX> and <TeX>c</TeX> with different signs, so any single output comes out of partial
          cancellations among several neurons. Explaining "how does the MLP compute <TeX>c</TeX>?" in terms of
          neurons means tracing all of that.
          (One neuron does a single clean job, <TeX>d</TeX> NOT <TeX>e</TeX>, because nothing else it could share
          with.)
        </p>
      </div>

      <h2>A transcoder: an SAE that crosses the MLP</h2>
      <div className="prose">
        <p>
          An SAE on the hidden layer would decompose the four neurons' activations into sparse latents. That
          explains the <em>state</em> of the hidden layer, but not how it came from the input or where it goes.
          A <b>transcoder</b> is trained differently. It reads the MLP's <em>input</em> and predicts the MLP's{" "}
          <em>output</em> through many sparse latents:
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`z = \mathrm{ReLU}(W_{\text{enc}}\, x + b_{\text{enc}}), \qquad
\hat y = W_{\text{dec}}\, z + b_{\text{dec}}, \qquad
\mathcal L = \lVert y_{\text{MLP}}(x) - \hat y\rVert^2 + \lambda \sum_i z_i \lVert d_i\rVert`}</TeX>
        <div className="scroll-x">
          <table className="tbl" style={{ marginTop: 6 }}>
            <thead><tr><th></th><th>SAE</th><th>Transcoder</th></tr></thead>
            <tbody>
              <tr><td>reads</td><td>an activation <TeX>h</TeX></td><td>the MLP's input <TeX>x</TeX></td></tr>
              <tr><td>predicts</td><td>the same <TeX>h</TeX></td><td>the MLP's output <TeX>y</TeX></td></tr>
              <tr><td>can replace the MLP?</td><td>no, it only re-describes one activation</td><td>yes: <TeX>x \mapsto \hat y</TeX> is a stand-in for <TeX>x \mapsto y</TeX></td></tr>
            </tbody>
          </table>
        </div>
        <p className="small">
          Note the target: the transcoder imitates the <em>MLP</em>, not the task. When the MLP gets something
          slightly wrong, a good transcoder gets it wrong in the same way.
        </p>
      </Card>

      <Card title="Train the transcoder">
        <div className="controls">
          <Slider label="Latents" value={latents} min={2} max={16} step={1} onChange={setLatents} format={(v) => String(v)} />
          <Slider label="L1 penalty λ" value={l1} min={0} max={1} step={0.01} onChange={setL1} format={(v) => v.toFixed(2)} />
        </div>
        <Loading loading={tc.loading} error={tc.error}>
          {d && (
            <>
              <div className="row" style={{ gap: 10, marginBottom: 12 }}>
                <Stat k="FVU vs the MLP" v={pct(d.fvu, 1)} color={d.fvu > 0.1 ? "var(--warm)" : undefined} d="how well it imitates" />
                <Stat k="Latent L0" v={fmt(d.l0, 2)} d="active latents per input" />
                <Stat k="Neuron L0" v={fmt(d.neuron_l0, 2)} d="active neurons per input" />
                <Stat k="Alive latents" v={`${aliveIds.length} / ${latents}`} />
              </div>
              <div className="grid2">
                <Heatmap values={aliveIds.map((i) => d.latent_in[i])} rowLabels={aliveIds.map((i) => `L${i}`)} colLabels={d.in_features}
                  cell={30} max={1.2} rowLabelWidth={36} caption="each live latent reads from input features" />
                <Heatmap values={aliveIds.map((i) => d.latent_out[i])} rowLabels={aliveIds.map((i) => `L${i}`)} colLabels={d.out_features}
                  cell={30} max={1.4} rowLabelWidth={36} caption="…and writes to output features" />
              </div>
              <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
                {aliveIds.map((i) => (
                  <li key={i}><b>L{i}</b> reads {describeRow(d.latent_in[i], d.in_features)}; writes {describeRow(d.latent_out[i], d.out_features)}</li>
                ))}
              </ul>
            </>
          )}
        </Loading>
        <div className="caption">Dead latents (never active on 4,000 test inputs) are hidden.</div>
      </Card>
      <div className="prose">
        <p>
          At the default settings (8 latents, <TeX>\lambda = 0.1</TeX>) most live latents do one job each: one
          reads <TeX>a</TeX> and writes <TeX>a</TeX>, one does the same for <TeX>b</TeX>, one for <TeX>c</TeX>,
          and one reads <TeX>d - e</TeX> and writes <TeX>d</TeX> NOT <TeX>e</TeX>. Usually one or two stragglers
          fire rarely and patch up small errors. On average about one latent is active per input, against almost
          three neurons.
        </p>
        <p>
          The heatmaps also show something about the MLP. There's no "AND" latent. The <TeX>a</TeX> latent and
          the <TeX>b</TeX> latent each write about a third of a unit into "a AND b", so the MLP's AND is really
          "some of <TeX>a</TeX> plus some of <TeX>b</TeX>". That gives about 0.6 when both are on and about 0.25
          when only one is, instead of 1 and 0. Four neurons weren't enough for a proper AND, and the transcoder
          reports that faithfully.
        </p>
      </div>
      <Callout kind="try">
        <p>
          Drop to 4 latents: FVU jumps to about 50%. Only a couple of latents survive the penalty, and
          some jobs aren't done at all. Set <TeX>\lambda = 0</TeX> with 8 latents: imitation becomes near-perfect, but
          L0 rises to about 4.5 and the latents go back to being mixtures. Sparsity is what buys readability.
        </p>
      </Callout>

      <h2>The attribution graph</h2>
      <div className="prose">
        <p>
          Because the transcoder stands in for the MLP, its latents connect to the features on either side with
          weights that don't depend on the input. Latent <TeX>i</TeX> reads input feature <TeX>j</TeX> with
          weight <TeX>{String.raw`w^{\text{enc}}_i\!\cdot e_j`}</TeX> and writes output feature <TeX>k</TeX> with
          weight <TeX>{String.raw`u_k\!\cdot d_i`}</TeX>. For one particular input, multiply each edge by the
          activation flowing through it and you get an <b>attribution graph</b>: which features, through which
          latents, produced this output.
        </p>
      </div>
      <TeX block>{String.raw`\text{edge}(j \to i) = x_j\,\big(w^{\text{enc}}_i\!\cdot e_j\big), \qquad
\text{edge}(i \to k) = z_i\,\big(u_k\!\cdot d_i\big), \qquad
y_k = \sum_i \text{edge}(i\to k) + \text{bias}_k + \underbrace{\text{error}_k}_{y_{\text{MLP}} - \hat y}`}</TeX>

      <Card title="Build an input, read the graph">
        <Presets onPick={(v: number[]) => setF(v)} items={INPUT_PRESETS}
          active={INPUT_PRESETS.find((p) => p.value.every((v, i) => v === f[i]))?.label} />
        <div className="controls">
          {IN.map((name, j) => (
            <Slider key={name} label={`feature ${name}`} value={f[j]} min={0} max={1} step={0.05}
              onChange={(v) => setF((old) => old.map((o, k) => (k === j ? v : o)))} format={(v) => v.toFixed(2)} />
          ))}
        </div>
        <Loading loading={graph.loading} error={graph.error}>
          {graph.data && (
            <>
              <AttributionGraph g={graph.data} f={f} tc={d} />
              <div className="scroll-x" style={{ marginTop: 10 }}>
                <table className="tbl" style={{ maxWidth: 620 }}>
                  <thead><tr><th>output</th><th>task target</th><th>MLP</th><th>transcoder</th><th>error</th></tr></thead>
                  <tbody>
                    {OUT.map((name, k) => (
                      <tr key={name}>
                        <td>{name}</td>
                        <td className="mono">{fmt(graph.data!.task_out[k])}</td>
                        <td className="mono">{fmt(graph.data!.mlp_out[k])}</td>
                        <td className="mono">{fmt(graph.data!.transcoder_out[k])}</td>
                        <td className="mono" style={{ color: Math.abs(graph.data!.error[k]) > 0.1 ? "var(--warm)" : "var(--ink-3)" }}>{fmt(graph.data!.error[k])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </Loading>
        <div className="caption">
          Edge width is the size of the attribution; blue adds, orange subtracts. Edges smaller than 0.03 are
          hidden. The "bias" node is <TeX>{String.raw`b_{\text{dec}}`}</TeX>, and "error" is whatever the transcoder fails to
          reproduce of the real MLP's output, so the three sources always add up to the MLP.
        </div>
      </Card>

      <Callout kind="try">
        <p>
          Try <b>a only</b>: one latent fires and writes 1 to "a" and about 0.3 to "a AND b". The MLP makes the same
          mistake (task target 0) and the graph shows where it comes from. Try <b>d and e</b>: no latent fires at
          all, because the <TeX>d</TeX> NOT <TeX>e</TeX> latent reads <TeX>d - e = 0</TeX>. Slide <TeX>e</TeX>{" "}
          partway up with <TeX>d = 1</TeX> and watch that latent's activation, and its edge, shrink.
        </p>
      </Callout>

      <Callout kind="takeaway">
        <p>
          Neurons are a poor basis for explaining computation because they share jobs. A transcoder replaces the
          MLP with a wider, sparse layer trained to imitate it. Its latents read and write features cleanly, and
          because it is a replacement, the input-to-latent and latent-to-output weights form a circuit you can
          draw for any input. The error node keeps the account honest: it is the part of the real computation the
          replacement doesn't explain.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          Dunefsky, Chlenski and Nanda (2024), <em>Transcoders find interpretable LLM feature circuits</em>,
          introduced transcoders and showed their input-independent connections make circuit analysis in GPT-2
          much cleaner than SAEs on MLP activations. Anthropic's circuit tracing work (Ameisen et al., 2025,{" "}
          <em>Circuit Tracing</em>; Lindsey et al., 2025, <em>On the Biology of a Large Language Model</em>) scales
          the idea with <b>cross-layer transcoders</b>, where each latent reads one layer and writes to all later
          ones, replacing every MLP of Claude 3.5 Haiku. It builds attribution graphs for single prompts, with
          error nodes exactly like the one here, and prunes them to the few paths that matter. Real replacement
          models leave substantial error, and a graph describes the replacement, which is only as trustworthy as
          its match to the model.
        </p>
      </Callout>
    </div>
  );
}

/* ------------------------------------------------------------------------------------------------ */

function AttributionGraph({ g, f, tc }: { g: Graph; f: number[]; tc: Tc | null }) {
  const W = 600;
  const mids = [...g.latents.map((l) => ({ key: `L${l.id}`, id: l.id, act: l.activation })),
    { key: "bias", id: -1, act: NaN }, { key: "error", id: -2, act: NaN }];
  const rowH = 52;
  const H = Math.max(5, mids.length) * rowH + 30;
  const xIn = 60, xMid = W / 2, xOut = W - 90;
  const yCol = (i: number, n: number) => 24 + (H - 48) * (n <= 1 ? 0.5 : i / (n - 1));
  const yIn = (j: number) => yCol(j, 5);
  const yMid = (i: number) => yCol(i, mids.length);
  const midIndex = (key: string) => mids.findIndex((m) => m.key === key);
  const edges: { x1: number; y1: number; x2: number; y2: number; w: number }[] = [];
  g.in_edges.forEach((e) => edges.push({ x1: xIn + 20, y1: yIn(IN.indexOf(e.source)), x2: xMid - 30, y2: yMid(midIndex(`L${e.latent}`)), w: e.weight }));
  g.out_edges.forEach((e) => edges.push({ x1: xMid + 30, y1: yMid(midIndex(`L${e.latent}`)), x2: xOut - 22, y2: yIn(OUT.indexOf(e.target)), w: e.weight }));
  g.bias_out.forEach((v, k) => edges.push({ x1: xMid + 30, y1: yMid(midIndex("bias")), x2: xOut - 22, y2: yIn(k), w: v }));
  g.error.forEach((v, k) => edges.push({ x1: xMid + 30, y1: yMid(midIndex("error")), x2: xOut - 22, y2: yIn(k), w: v }));
  const label = (id: number) => {
    if (!tc) return "";
    const row = tc.latent_out[id];
    let best = 0;
    row.forEach((v, k) => { if (Math.abs(v) > Math.abs(row[best])) best = k; });
    return `writes ${tc.out_features[best]}`;
  };
  const textStyle = (fill: string) => ({ fill, paintOrder: "stroke" as const, stroke: "var(--surface)", strokeWidth: 3 });
  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", maxWidth: W, minWidth: 420 }} role="img"
        aria-label="Attribution graph from input features through active transcoder latents to output features">
        <text x={xIn} y={12} fontSize={11} textAnchor="middle" style={{ fill: "var(--ink-3)" }}>input features</text>
        <text x={xMid} y={12} fontSize={11} textAnchor="middle" style={{ fill: "var(--ink-3)" }}>active latents</text>
        <text x={xOut} y={12} fontSize={11} textAnchor="middle" style={{ fill: "var(--ink-3)" }}>output features</text>
        {edges.filter((e) => Math.abs(e.w) >= 0.03).map((e, i) => (
          <path key={i} d={`M${e.x1},${e.y1} C${(e.x1 + e.x2) / 2},${e.y1} ${(e.x1 + e.x2) / 2},${e.y2} ${e.x2},${e.y2}`}
            fill="none" stroke={e.w >= 0 ? "var(--accent)" : "var(--warm)"} strokeOpacity={0.7}
            strokeWidth={Math.min(9, 0.8 + 6 * Math.abs(e.w))} />
        ))}
        {IN.map((name, j) => {
          const on = f[j] > 0;
          return (
            <g key={name}>
              <circle cx={xIn} cy={yIn(j) + 2} r={21} fill={on ? `rgba(var(--pos), ${(0.15 + 0.5 * f[j]).toFixed(2)})` : "var(--node)"}
                stroke={on ? "var(--accent)" : "var(--line)"} strokeWidth={1.6} />
              <text x={xIn} y={yIn(j) + 1} fontSize={13} fontWeight={700} textAnchor="middle">{name}</text>
              <text x={xIn} y={yIn(j) + 14} fontSize={8.5} textAnchor="middle" className="mono" style={{ fill: "var(--ink-2)" }}>{f[j].toFixed(2)}</text>
            </g>
          );
        })}
        {mids.map((m, i) => {
          const y = yMid(i);
          const special = m.id < 0;
          return (
            <g key={m.key}>
              <rect x={xMid - 30} y={y - 17} width={60} height={34} rx={special ? 17 : 7}
                fill={m.id === -2 ? "var(--warm-soft)" : special ? "var(--surface-2)" : "var(--gold-soft)"}
                stroke={m.id === -2 ? "var(--warm)" : special ? "var(--line)" : "var(--gold)"} strokeWidth={1.5}
                strokeDasharray={special ? "4 3" : undefined} />
              <text x={xMid} y={special ? y + 4 : y - 1} fontSize={12} fontWeight={700} textAnchor="middle">{m.key}</text>
              {!special && <text x={xMid} y={y + 12} fontSize={9.5} textAnchor="middle" className="mono" style={{ fill: "var(--ink-2)" }}>z={m.act.toFixed(2)}</text>}
              {!special && <text x={xMid} y={y + 29} fontSize={9.5} textAnchor="middle" style={textStyle("var(--ink-3)")}>{label(m.id)}</text>}
            </g>
          );
        })}
        {g.latents.length === 0 && (
          <text x={xMid} y={yMid(0) - 26} fontSize={11} textAnchor="middle" style={{ fill: "var(--ink-3)" }}>no latent active</text>
        )}
        {OUT.map((name, k) => (
          <g key={name}>
            <rect x={xOut - 22} y={yIn(k) - 17} width={104} height={34} rx={7} fill="var(--node)" stroke="var(--ink-3)" strokeWidth={1.3} />
            <text x={xOut + 30} y={yIn(k) - 2} fontSize={11.5} fontWeight={700} textAnchor="middle">{name}</text>
            <text x={xOut + 30} y={yIn(k) + 12} fontSize={9.5} textAnchor="middle" className="mono" style={{ fill: "var(--ink-2)" }}>
              MLP {g.mlp_out[k].toFixed(2)}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
