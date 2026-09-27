import { useMemo, useState } from "react";
import { Bars, Heatmap, Scatter } from "../components/charts";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Seg, Select, Slider, Stat, Tokens } from "../components/ui";
import { fmt, useApi } from "../api";

/*
 * The same six-head IOI model as the previous lesson, now treated as a black box. Every number is
 * a forward pass on the server: 6 x 16 patched runs for the patching grid, and for the edges one
 * patched run, two for EAP and twenty for EAP-IG per edge, plus the ACDC loop.
 */

type Patch = {
  clean_tokens: string[]; corrupt_tokens: string[]; ld_clean: number; ld_corrupt: number; gap: number;
  per_head: Record<string, number>; per_position: Record<string, number[]>;
};
type EdgeRow = { sender: string; receiver: string; channel: Channel; patch: number; eap: number; eap_ig: number };
type Edges = { ld_clean: number; ld_corrupt: number; edges: EdgeRow[]; acdc_kept: [string, string, Channel][]; acdc_ld: number };
type Channel = "q" | "k" | "v" | "resid";

const HEAD_NAMES = ["L0.0", "L0.1", "L1.0", "L1.1", "L2.0", "L2.1"];
const NAMES = ["Mary", "John", "Alice", "Bob", "Sara", "Tom"];
const CH_COLOR: Record<Channel, string> = { q: "var(--warm)", k: "var(--gold)", v: "var(--good)", resid: "var(--accent)" };
const CH_NAME: Record<Channel, string> = { q: "query", k: "key", v: "value", resid: "direct to logits" };
const edgeName = (e: { sender: string; receiver: string; channel: string }) =>
  `${e.sender} → ${e.receiver}${e.channel === "resid" ? "" : `.${e.channel}`}`;

// ------------------------------------------------------------------ the circuit graph

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

function CircuitGraph({ edges }: { edges: [string, string, Channel][] }) {
  const used = new Set(edges.flatMap(([a, b]) => [a, b]));
  const nodes = ["embed", ...HEAD_NAMES, "logits"];
  const portY = (id: string, ch: Channel) => GY[id] + (ch === "q" ? -13 : ch === "v" ? 13 : 0);
  return (
    <svg viewBox="0 0 672 262" width="100%" style={{ display: "block", maxWidth: 700 }} role="img"
      aria-label={`The circuit ACDC keeps: ${edges.length} edges`}>
      <defs>
        {(Object.keys(CH_COLOR) as Channel[]).map((c) => (
          <marker key={c} id={`cd-ah-${c}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill={CH_COLOR[c]} />
          </marker>
        ))}
      </defs>
      {edges.map(([a, b, ch]) => {
        const x1 = GX[a] + NW / 2, y1 = GY[a];
        const x2 = GX[b] - NW / 2 - 1, y2 = portY(b, ch);
        return (
          <path key={`${a}-${b}-${ch}`} d={edgePath(a, b, x1, y1, x2, y2, 130 + (ch === "q" ? -8 : ch === "v" ? 8 : 0))}
            fill="none" stroke={CH_COLOR[ch]} strokeWidth={2.2} markerEnd={`url(#cd-ah-${ch})`} />
        );
      })}
      {nodes.map((id) => {
        const isHead = id.startsWith("L");
        const on = used.has(id);
        return (
          <g key={id} transform={`translate(${GX[id] - NW / 2},${GY[id] - NH / 2})`} opacity={on ? 1 : 0.35}>
            <rect width={NW} height={NH} rx={9} fill="var(--node)" stroke={on ? "var(--ink-2)" : "var(--ink-3)"}
              strokeWidth={on ? 1.6 : 1} strokeDasharray={on ? undefined : "4 3"} />
            <text x={NW / 2 + (isHead ? 4 : 0)} y={NH / 2 + 5} fontSize={13} fontWeight={700} textAnchor="middle">{id}</text>
            {isHead && ["q", "k", "v"].map((p, j) => (
              <text key={p} x={6} y={NH / 2 + (j - 1) * 13 + 3} fontSize={8.5} textAnchor="middle" fill="var(--ink-3)">{p}</text>
            ))}
          </g>
        );
      })}
    </svg>
  );
}

function ChannelLegend() {
  return (
    <div className="row small" style={{ gap: 14 }}>
      {(Object.keys(CH_COLOR) as Channel[]).map((c) => (
        <span key={c} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 5, background: CH_COLOR[c], display: "inline-block" }} />
          {c === "resid" ? "to logits" : `${c} (${CH_NAME[c]})`}
        </span>
      ))}
    </div>
  );
}

// --------------------------------------------------------------------------- the page

export default function CircuitDiscovery() {
  const [io, setIo] = useState("Mary");
  const [s, setS] = useState("John");
  const [template, setTemplate] = useState<"ABBA" | "BABA">("ABBA");
  const [corrupt, setCorrupt] = useState<"abc" | "swap">("abc");
  const [direction, setDirection] = useState<"denoise" | "noise">("noise");
  const [showAll, setShowAll] = useState(false);
  const [tau, setTau] = useState(0.1);

  const pickIo = (v: string) => { if (v === s) setS(io); setIo(v); };
  const pickS = (v: string) => { if (v === io) setIo(s); setS(v); };

  const patch = useApi<Patch>("ioi/patch", { io, s, template, corrupt, direction });
  const scored = useApi<Edges>("ioi/edges", { io, s, template, corrupt, tau: 0.1 });
  const acdc = useApi<Edges>("ioi/edges", { io, s, template, corrupt, tau }, 200);

  const p = patch.data;
  const e = scored.data;
  const k = acdc.data;

  const diffHl = useMemo(() => {
    if (!p) return {};
    const out: Record<number, string> = {};
    p.clean_tokens.forEach((t, i) => { if (t !== p.corrupt_tokens[i]) out[i] = "var(--warm-soft)"; });
    return out;
  }, [p]);

  const rows = useMemo(() => (e ? [...e.edges].sort((a, b) => Math.abs(b.patch) - Math.abs(a.patch)) : []), [e]);
  const dom = useMemo<[number, number]>(() => {
    const m = Math.max(1, ...rows.flatMap((r) => [Math.abs(r.patch), Math.abs(r.eap), Math.abs(r.eap_ig)]));
    return [-m * 1.1, m * 1.1];
  }, [rows]);
  const qk = rows.filter((r) => (r.channel === "q" || r.channel === "k") && Math.abs(r.patch) > 0.5);

  return (
    <div className="prose-wide">
      <h2>Pretend you did not write the model</h2>
      <div className="prose">
        <p>
          In the last lesson we knew the circuit because we built it. Real models don't come with a diagram. So take
          the same six-head IOI model and ask the question the way a researcher has to: <em>which parts of this
          network, connected how, produce the behaviour?</em>
        </p>
        <p>
          The workhorse answer is <b>activation patching</b>. Run the model twice: once on the <b>clean</b> prompt,
          once on a <b>corrupted</b> one that differs in a single controlled way. Then copy one component's activation
          from one run into the other and see how much of the behaviour moves with it. Unlike ablation, you choose
          exactly what information gets removed: whatever differs between the two prompts.
        </p>
      </div>

      <h2>Two runs, one difference</h2>
      <Card>
        <div className="controls">
          <Select label="IO (answer)" value={io} options={NAMES} onChange={pickIo} />
          <Select label="Subject" value={s} options={NAMES} onChange={pickS} />
          <Seg label="Template" value={template} onChange={setTemplate} options={[
            { value: "ABBA", label: "ABBA" }, { value: "BABA", label: "BABA" },
          ]} />
          <Seg label="Corruption" value={corrupt} onChange={setCorrupt} options={[
            { value: "abc", label: "ABC: new third name" },
            { value: "swap", label: "Swap the roles" },
          ]} />
        </div>
        <Loading loading={patch.loading} error={patch.error}>
          {p && (
            <>
              <div className="small muted">Clean</div>
              <Tokens tokens={p.clean_tokens} highlight={diffHl} />
              <div className="small muted" style={{ marginTop: 8 }}>Corrupted</div>
              <Tokens tokens={p.corrupt_tokens} highlight={diffHl} />
              <div className="row" style={{ marginTop: 12 }}>
                <Stat k="Δ clean" v={fmt(p.ld_clean, 2)} color="var(--good)" />
                <Stat k="Δ corrupted" v={fmt(p.ld_corrupt, 2)} color={p.ld_corrupt > 0.5 ? "var(--good)" : "var(--warm)"} />
                <Stat k="Gap to explain" v={fmt(p.gap, 2)} />
              </div>
            </>
          )}
        </Loading>
        <p className="small" style={{ marginBottom: 0 }}>
          <b>ABC</b> replaces the second {s} with a fresh name, so no name repeats and the model has no reason to prefer
          {" "}{io} (Δ ≈ 0). <b>Swap</b> exchanges the two names, so the repeated name is now {io} and the right answer flips
          (Δ turns negative). The corruption defines the question: patching finds the components that carry{" "}
          <em>the difference</em> between the two prompts, and nothing else.
        </p>
      </Card>

      <h2>Patch one head at a time</h2>
      <div className="prose">
        <p>
          There are two directions, and they ask different questions. <b>Denoising</b> starts from the corrupted run
          and restores one head's clean output: <em>is this head enough to bring the behaviour back?</em>{" "}
          <b>Noising</b> starts from the clean run and inserts one head's corrupted output: <em>does the behaviour
          need this head?</em> Both are scored as a fraction of the gap:
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`\text{denoise} = \frac{\Delta_{\text{patched}} - \Delta_{\text{corr}}}{\Delta_{\text{clean}} - \Delta_{\text{corr}}}`}</TeX>
        <TeX block>{String.raw`\text{noise} = \frac{\Delta_{\text{clean}} - \Delta_{\text{patched}}}{\Delta_{\text{clean}} - \Delta_{\text{corr}}}`}</TeX>
        <p className="small muted">1 means “this head alone carries the whole gap”, 0 means “no effect”, and a negative score means the head works against the answer.</p>
        <Seg value={direction} onChange={setDirection} options={[
          { value: "noise", label: "Noise: clean run, one head corrupted" },
          { value: "denoise", label: "Denoise: corrupted run, one head restored" },
        ]} />
        <Loading loading={patch.loading} error={patch.error}>
          {p && (
            <div style={{ marginTop: 12 }}>
              <Bars width={380} labelWidth={56} format={(v) => v.toFixed(2)}
                items={HEAD_NAMES.map((h) => ({ label: h, value: p.per_head[h] }))} refLine={{ value: 1, label: "whole gap" }} />
              <h3>Head × position</h3>
              <p className="small" style={{ marginTop: 0 }}>
                The same patch, but only at one position at a time. Blue helps the answer, orange hurts it.
              </p>
              <Heatmap values={HEAD_NAMES.map((h) => p.per_position[h])} rowLabels={HEAD_NAMES}
                colLabels={p.clean_tokens.map((t, i) => `${i} ${t}`)} cell={26} rowLabelWidth={48}
                max={Math.max(1, ...HEAD_NAMES.flatMap((h) => p.per_position[h].map(Math.abs)))}
                format={(v) => v.toFixed(2)} caption="Hover a cell. Almost everything is zero." />
            </div>
          )}
        </Loading>
      </Card>
      <div className="prose">
        <p>
          The grid is almost empty, and that is the point. The effects live in exactly two columns: position 10, the
          second subject (S2), and position 14, END. In a 15-token prompt, patching tells you not only <em>which</em>{" "}
          heads matter but <em>where</em> they do their work: the duplicate head at S2, everything else at END.
        </p>
        <p>Now play with the toggles. Three things change, and each one teaches something:</p>
        <ul>
          <li>
            <b>Noise vs denoise on L0.1 (ABC).</b> Noising the duplicate head costs more than the whole gap. Denoising
            it recovers nothing. Removing it breaks the circuit, but restoring it alone into the corrupted run is
            useless: its flag would point L1.0 at the fresh third name, which is not the subject. A component can be
            necessary without being sufficient.
          </li>
          <li>
            <b>Swap makes L0.1 vanish.</b> In the swapped prompt a name still repeats at the same position, so the
            duplicate head's output does not change at all and patching it does nothing. The head is still
            essential. Your corruption simply didn't touch what it computes.
          </li>
          <li>
            <b>L2.1 is negative in every setting.</b> The negative name mover fights the answer, so restoring it
            hurts and removing it helps.
          </li>
        </ul>
      </div>
      <Callout kind="warn" label="Corruption is a choice">
        <p>
          Patching measures the effect of <em>the difference between two prompts</em>. Pick a corruption that
          preserves some piece of information, and every component carrying it will look irrelevant. Heimersheim &amp;
          Nanda (2024) and Zhang &amp; Nanda (2024) discuss how the choice of corruption and metric changes the answer
          in real models.
        </p>
      </Callout>

      <h2>From heads to edges</h2>
      <div className="prose">
        <p>
          Knowing which heads matter doesn't tell you how they are wired. L1.0 might matter because of what it sends
          to L2.0's query, or to L2.1's, or straight to the logits. So we go one level finer. An <b>edge</b> is one
          sender's contribution to one input of one receiver: the embedding or an earlier head, feeding the q, k or v
          of a later head, or the final residual that the logits read.
        </p>
        <p>
          This model has 61 edges: 6 into layer 0 (embed → two heads × q, k, v), 18 into layer 1 (three senders), 30
          into layer 2 (five senders), and 7 into the logits. Patching an edge means the receiver reads the sender's
          corrupted output <em>on that one input only</em>, while every other path stays clean. That is the idea of{" "}
          <b>path patching</b> (Goldowsky-Dill et al., 2023): intervene on one route through the network rather
          than on a whole activation.
        </p>
        <p>
          Exact edge patching costs one forward pass per edge. That's fine for 61 edges. A real model has millions,
          so people estimate instead.
        </p>
      </div>

      <h2>Estimating every edge with one gradient</h2>
      <div className="prose">
        <p>
          <b>Edge attribution patching</b> (EAP; Syed et al., 2023) replaces each patched run with a first-order
          Taylor expansion. Take the change in the sender's output, and multiply it by the gradient of Δ with
          respect to the receiver's input:
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`\widehat{\text{effect}}_{s\to r} \;=\; \big(z_s^{\text{corr}} - z_s^{\text{clean}}\big)^{\!\top}\,\frac{\partial \Delta}{\partial x_r}\bigg|_{\text{clean}}`}</TeX>
        <p className="small">
          One forward and one backward pass give the gradient at every receiver at once, so every edge in the model
          is scored for the price of about two runs. <b>EAP-IG</b> (Hanna et al., 2024) averages the gradient along
          the straight path from clean to corrupted instead of reading it at one end:
        </p>
        <TeX block>{String.raw`\begin{aligned}\widehat{\text{effect}}^{\,\text{IG}}_{s\to r} \;=\; &\big(z_s^{\text{corr}} - z_s^{\text{clean}}\big)^{\!\top}\\ &\cdot\frac{1}{m}\sum_{k=1}^{m}\frac{\partial \Delta}{\partial x_r}\bigg|_{z_s^{\text{clean}} + \alpha_k (z_s^{\text{corr}} - z_s^{\text{clean}})}\end{aligned}`}</TeX>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Here <TeX>m = 10</TeX>. For a linear path the two are identical to the true effect. They come apart
          exactly where the model bends.
        </p>
      </Card>

      <Card title="Estimates against the truth">
        <Loading loading={scored.loading} error={scored.error}>
          {e && (
            <>
              <ChannelLegend />
              <div className="grid2" style={{ marginTop: 8 }}>
                <div>
                  <div className="small"><b>EAP</b> (y) vs true patch effect (x)</div>
                  <Scatter points={rows.map((r) => [r.patch, r.eap])} colors={rows.map((r) => CH_COLOR[r.channel])}
                    domain={dom} diagonal size={300} radius={5} xLabel="true effect on Δ" yLabel="EAP estimate" />
                </div>
                <div>
                  <div className="small"><b>EAP-IG</b> (y) vs true patch effect (x)</div>
                  <Scatter points={rows.map((r) => [r.patch, r.eap_ig])} colors={rows.map((r) => CH_COLOR[r.channel])}
                    domain={dom} diagonal size={300} radius={5} xLabel="true effect on Δ" yLabel="EAP-IG estimate" />
                </div>
              </div>
              <p className="small muted">
                Each dot is one of the 61 edges; the dashed line is “estimate = truth”. Axes run from{" "}
                {fmt(dom[0], 1)} to {fmt(dom[1], 1)}, crossing at zero. Most edges sit at the origin: noising them
                changes nothing.
              </p>
            </>
          )}
        </Loading>
      </Card>
      <div className="prose">
        <p>
          The EAP plot has a row of dots lying flat on the horizontal axis. They are{" "}
          {qk.length ? <>the {qk.length} query and key edges ({qk.slice(0, 3).map(edgeName).join(", ")}{qk.length > 3 ? ", …" : ""})</> : "the query and key edges"}.
          Patching them is catastrophic: it can flip the answer. EAP scores them at nearly zero.
        </p>
        <p>
          The reason is <b>saturated attention</b>. A query or key edge changes <em>where</em> a head looks, and
          that goes through a softmax. When a head attends almost entirely to one token, the softmax is flat
          there:
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`a_j = \frac{e^{\beta s_j}}{\sum_i e^{\beta s_i}}
`}</TeX>
        <TeX block>{String.raw`\frac{\partial a_j}{\partial s_j} = \beta\, a_j (1 - a_j) \;\approx\; 0 \text{ when } a_j \approx 0 \text{ or } 1`}</TeX>
        <p className="small" style={{ marginBottom: 0 }}>
          At the clean point the name mover puts about 100% of its attention on {io}. A tiny nudge to its query
          doesn't move that at all, so the gradient is about zero. A full corruption moves it all the way to the other
          name. The first-order estimate sees the nudge; patching sees the jump. Value and logit edges are linear in
          this model, so EAP gets them exactly right. EAP-IG takes gradients at points <em>between</em> clean and
          corrupted, where the softmax is mid-switch and steep, and recovers the jump.
        </p>
      </Card>
      <Card title="All the edges, by true effect">
        <Loading loading={scored.loading} error={scored.error}>
          {e && (
            <>
              <div className="scroll-x">
                <table className="tbl" style={{ minWidth: 420 }}>
                  <thead><tr><th>Edge</th><th>Patch (truth)</th><th>EAP</th><th>EAP-IG</th></tr></thead>
                  <tbody>
                    {(showAll ? rows : rows.slice(0, 12)).map((r) => (
                      <tr key={edgeName(r)}>
                        <td>
                          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 4, background: CH_COLOR[r.channel], marginRight: 6 }} />
                          <span className="mono">{edgeName(r)}</span>
                        </td>
                        {[r.patch, r.eap, r.eap_ig].map((v, i) => (
                          <td key={i} className="mono" style={{ color: Math.abs(v) < 0.005 ? "var(--ink-3)" : v < 0 ? "var(--warm)" : "var(--accent)" }}>
                            {v >= 0 ? "+" : ""}{v.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <button className="btn" style={{ marginTop: 10 }} onClick={() => setShowAll((x) => !x)}>
                {showAll ? "Show the top 12" : `Show all ${rows.length} edges`}
              </button>
            </>
          )}
        </Loading>
        <div className="caption">Change in Δ when the edge is noised (clean run, one edge corrupted). Negative means the edge carries the answer.</div>
      </Card>

      <h2>Pruning the graph: ACDC</h2>
      <div className="prose">
        <p>
          Scores rank edges; they don't say where to stop. <b>ACDC</b> (Automatic Circuit DisCovery; Conmy et al.,
          2023) turns patching into a pruning loop:
        </p>
        <ol>
          <li>Start with every edge in place and the clean run.</li>
          <li>Visit receivers from the logits backwards. For each incoming edge, patch it to its corrupted value.</li>
          <li>If Δ moves by less than a threshold <TeX>\tau</TeX>, leave it patched (the edge is pruned). Otherwise
            restore it.</li>
          <li>What survives is the circuit.</li>
        </ol>
        <p>Drag <TeX>\tau</TeX> and watch the circuit shrink.</p>
      </div>
      <Card>
        <Slider label={<>Threshold <TeX>\tau</TeX></>} value={tau} min={0} max={4.5} step={0.05}
          onChange={setTau} format={(v) => v.toFixed(2)} />
        <Loading loading={acdc.loading} error={acdc.error}>
          {k && (
            <>
              <div className="row" style={{ margin: "8px 0" }}>
                <Stat k="Edges kept" v={`${k.acdc_kept.length} / ${k.edges.length}`} />
                <Stat k="Δ of the circuit" v={fmt(k.acdc_ld, 2)} color={k.acdc_ld > 1 ? "var(--good)" : "var(--warm)"} />
                <Stat k="Δ of the full model" v={fmt(k.ld_clean, 2)} d={`corrupted: ${fmt(k.ld_corrupt, 2)}`} />
              </div>
              <CircuitGraph edges={k.acdc_kept} />
              <ChannelLegend />
            </>
          )}
        </Loading>
      </Card>
      <div className="prose">
        <p>
          With the ABC corruption: at <TeX>\tau = 0</TeX> nothing is pruned, since every patch changes Δ by
          at least a rounding error. Any small positive <TeX>\tau</TeX> drops the graph to 11 edges, and they are
          the circuit from the previous lesson: embed → L0.1's query, L0.1 → L1.0's key, L1.0 → the name movers'
          queries, the name edges into keys and values, and the movers into the logits. The unused previous-token
          head disappears along with every one of its edges.
        </p>
        <p>
          Past about 0.17 the copier L1.1 goes. Past about 1.4 the negative name mover goes, and the pruned circuit's
          Δ <em>rises above the full model's</em>, because what was removed was fighting the answer. Around 3.5 the
          circuit breaks: pruning is greedy and order-dependent, and once the early decisions are made, even
          load-bearing edges fall under the threshold. By 4 nothing is left. The threshold is the knob that trades
          size for faithfulness, and no single value is the right one. (Switch the corruption to Swap and the numbers
          change, because the gap you are explaining is twice as large.)
        </p>
      </div>
      <Callout kind="note" label="Why ACDC and not EAP?">
        <p>
          ACDC uses real patched runs, so saturated attention does not fool it. But it pays one forward pass per edge
          per visit, which on GPT-2 small took hours. EAP ranks every edge with two passes and EAP-IG with a few dozen,
          and you keep the top edges. On real models the practical recipe is to rank with an estimate and then check
          the resulting circuit with actual patching.
        </p>
      </Callout>

      <Callout kind="takeaway">
        <p>
          Circuit discovery is activation patching, scaled up. Patch heads to find <em>what</em> matters and positions
          to find <em>where</em>. Patch edges to find the wiring. Gradient estimates make that affordable, but a
          first-order estimate is blind wherever the model is saturated, and attention usually is. Integrated
          gradients fix much of that. Whatever the scoring, the answer depends on the corruption you chose and the
          threshold you stop at.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          Wang et al. (2023) mapped the IOI circuit in GPT-2 small largely by hand with path patching
          (Goldowsky-Dill et al., 2023). Conmy et al. (2023) automated it with ACDC and recovered most of the
          hand-found circuit. Syed et al. (2023) showed that attribution patching approximates edge patching at a
          tiny fraction of the cost. Hanna et al. (2024), “Have Faith in Faithfulness”, showed that EAP's circuits can
          overlap well with the true ones and still be unfaithful, and that EAP-IG closes much of that gap. Later
          methods learn a mask over edges directly (Bhaskar et al., 2024, edge pruning). In real models, backup heads
          and self-repair mean that a single patch can understate an edge's importance, so the circuits all these
          methods find are hypotheses to test, not final answers.
        </p>
      </Callout>
    </div>
  );
}
