import { useState } from "react";
import { Bars } from "../components/charts";
import { Dag } from "../components/Dag";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Seg, Select, Stat } from "../components/ui";
import { fmt, pct, useApi } from "../api";
import { HIDDEN, Net } from "./NetworkAsScm";

type Metric = "logit" | "prob";
type Row = {
  node: string; clean: number; corrupt: number; denoise: number; noise: number;
  recovery: number; damage: number; attribution: number; attribution_frac: number;
  integrated: number; true_effect: number;
};
type PatchOut = {
  clean: Record<string, number>; corrupt: Record<string, number>;
  metric_clean: number; metric_corrupt: number; rows: Row[];
};
type Med = {
  te: number; nde: number; nie: number; interaction: number;
  mediator_before: number; mediator_after: number; y_before: number; y_after: number;
};

const sub = (n: string) => n.replace(/(\d)/, "_$1");
const bits = (x: number[]) => `(${x.join(", ")})`;

/** Three toggles for one input triple. */
function BitPicker({ label, x, onChange, color }: { label: string; x: number[]; onChange: (x: number[]) => void; color: string }) {
  return (
    <div className="control">
      <span><b style={{ color }}>{label}</b> <span className="mono">{bits(x)}</span></span>
      <div className="row" style={{ gap: 6 }}>
        {["a", "b", "c"].map((n, i) => (
          <button key={n} className={`chip ${x[i] ? "on" : ""}`} onClick={() => onChange(x.map((v, j) => (j === i ? 1 - v : v)))}>
            {n} = {x[i]}
          </button>
        ))}
      </div>
    </div>
  );
}

/** Per-node groups of three horizontal bars: the true effect and its two estimates. */
function Grouped({ rows, fmtv }: { rows: Row[]; fmtv: (v: number) => string }) {
  const series = [
    { key: "true_effect" as const, name: "true effect (patch it)", color: "var(--ink-2)" },
    { key: "attribution" as const, name: "attribution patching", color: "var(--warm)" },
    { key: "integrated" as const, name: "integrated gradients", color: "var(--accent)" },
  ];
  const W = 460, L = 44, bh = 14, gap = 12;
  const max = Math.max(1e-9, ...rows.flatMap((r) => series.map((s) => Math.abs(r[s.key]))));
  const hasNeg = rows.some((r) => series.some((s) => r[s.key] < -1e-9));
  const plot = W - L - 58;
  const zero = L + (hasNeg ? plot / 2 : 0);
  const scale = (hasNeg ? plot / 2 : plot) / max;
  const gh = series.length * bh + gap;
  const H = rows.length * gh + 4;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", maxWidth: W }} role="img">
        <line x1={zero} x2={zero} y1={0} y2={H} stroke="var(--line)" />
        {rows.map((r, i) => (
          <g key={r.node} transform={`translate(0 ${i * gh + 4})`}>
            <text x={L - 8} y={(series.length * bh) / 2 + 4} fontSize={12} fontWeight={600} textAnchor="end">{r.node}</text>
            {series.every((s) => Math.abs(r[s.key]) < 5e-4) ? (
              <text x={zero + 6} y={(series.length * bh) / 2 + 4} fontSize={10.5} fill="var(--ink-3)">0 by every measure</text>
            ) : series.map((s, j) => {
              const v = r[s.key], len = Math.abs(v) * scale;
              return (
                <g key={s.key}>
                  <rect x={v >= 0 ? zero : zero - len} y={j * bh + 1} width={Math.max(len, 1)} height={bh - 4} rx={2} fill={s.color} />
                  <text x={v >= 0 ? zero + len + 4 : zero - len - 4} y={j * bh + bh - 4} fontSize={10} className="mono"
                    textAnchor={v >= 0 ? "start" : "end"} fill="var(--ink-2)">{fmtv(v)}</text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <div className="row small" style={{ gap: 14, marginTop: 4 }}>
        {series.map((s) => (
          <span key={s.key} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 12, height: 10, background: s.color, display: "inline-block", borderRadius: 2 }} />{s.name}
          </span>
        ))}
      </div>
    </div>
  );
}

/**
 * The metric along the straight line from the corrupted to the clean value of one node, with the
 * tangent that attribution patching extrapolates. The curve is recomputed here from the same
 * equations as the server (see the previous lesson), so the picture can be smooth.
 */
function forwardLocal(x: number[], set: Record<string, number>) {
  const g = (k: string, v: number) => (k in set ? set[k] : v);
  const relu = (z: number) => Math.max(z, 0);
  const [a, b, c] = x;
  const n1 = g("n1", relu(a + b - 1)), n2 = g("n2", relu(a + b - 1)), n3 = g("n3", relu(c)), n4 = g("n4", relu(b - a));
  const m1 = g("m1", relu(n1 + n2 + n3)), m2 = g("m2", relu(n4 - 0.2));
  const out = 6 * m1 - m2 - 3;
  return { out, prob: 1 / (1 + Math.exp(-out)) };
}

function PathPlot({ row, corrupt, metric, m0 }: { row: Row; corrupt: number[]; metric: Metric; m0: number }) {
  const W = 460, H = 230, pad = { l: 46, r: 110, t: 14, b: 34 };
  const n = 60;
  const ys = Array.from({ length: n + 1 }, (_, i) => {
    const a = i / n;
    const v = forwardLocal(corrupt, { [row.node]: row.corrupt + a * (row.clean - row.corrupt) });
    return (metric === "prob" ? v.prob : v.out) - m0;
  });
  const lo = Math.min(0, ...ys, row.attribution), hi = Math.max(1e-9, ...ys, row.attribution, row.integrated);
  const px = (a: number) => pad.l + a * (W - pad.l - pad.r);
  const py = (v: number) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * (H - pad.t - pad.b);
  const end = (y: number, color: string, label: string, dy = 0) => (
    <g>
      <circle cx={px(1)} cy={py(y)} r={4} fill={color} />
      <text x={px(1) + 8} y={py(y) + 4 + dy} fontSize={11} fill={color}>{label} {fmt(y, 2)}</text>
    </g>
  );
  const near = Math.abs(py(row.integrated) - py(row.true_effect)) < 14;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", maxWidth: W }} role="img">
      <line x1={pad.l} x2={px(1)} y1={py(0)} y2={py(0)} stroke="var(--line)" />
      <line x1={pad.l} x2={pad.l} y1={pad.t} y2={H - pad.b} stroke="var(--line)" />
      {[lo, hi].map((t, i) => <text key={i} x={pad.l - 6} y={py(t) + 4} fontSize={10} textAnchor="end" fill="var(--ink-3)">{fmt(t, 2)}</text>)}
      <text x={pad.l} y={H - 12} fontSize={10} fill="var(--ink-3)">corrupted value</text>
      <text x={px(1)} y={H - 12} fontSize={10} textAnchor="end" fill="var(--ink-3)">clean value</text>
      <text x={(pad.l + px(1)) / 2} y={H - 12} fontSize={10} textAnchor="middle" fill="var(--ink-3)">{row.node} →</text>
      <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(row.attribution)} stroke="var(--warm)" strokeWidth={2} strokeDasharray="5 4" />
      <polyline fill="none" stroke="var(--ink-2)" strokeWidth={2.4} points={ys.map((y, i) => `${px(i / n)},${py(y)}`).join(" ")} />
      {end(row.true_effect, "var(--ink-2)", "true", near ? -7 : 0)}
      {end(row.integrated, "var(--accent)", "IG", near ? 7 : 0)}
      {end(row.attribution, "var(--warm)", "attrib.")}
    </svg>
  );
}

const MED_NODES = [
  { id: "T", x: 0, y: 80, label: "T" },
  { id: "M", x: 50, y: 0, label: "M" },
  { id: "Y", x: 100, y: 80, label: "Y" },
];

export default function Patching() {
  const [clean, setClean] = useState([1, 1, 0]);
  const [corrupt, setCorrupt] = useState([1, 0, 0]);
  const [metric, setMetric] = useState<Metric>("prob");
  const [pathNode, setPathNode] = useState("m1");
  const patch = useApi<PatchOut>("mlp/patch", { clean, corrupt, metric });

  const [treat, setTreat] = useState<"a" | "b" | "c">("b");
  const [base, setBase] = useState([1, 0, 0]);
  const [mediator, setMediator] = useState("n1");
  const [medMetric, setMedMetric] = useState<Metric>("prob");
  const ti = ["a", "b", "c"].indexOf(treat);
  const changed = 1 - base[ti];
  const med = useApi<Med>("mlp/mediation", { treatment: treat, mediator, base, changed, metric: medMetric });

  const d = patch.data;
  const span = d ? d.metric_clean - d.metric_corrupt : 0;
  const same = clean.every((v, i) => v === corrupt[i]);
  const flat = d && Math.abs(span) < 1e-6;
  const fm = (v: number) => (metric === "prob" ? v.toFixed(3) : v.toFixed(2));
  const rowOf = (n: string) => d?.rows.find((r) => r.node === n);
  const n1 = rowOf("n1");
  const pathRow = rowOf(pathNode);
  const x1 = base.map((v, i) => (i === ti ? changed : v));
  const mfm = (v: number) => (medMetric === "prob" ? fmt(v, 3) : fmt(v, 2));
  const m = med.data;
  const Mname = sub(mediator);

  return (
    <div className="prose-wide">
      <h2>Two runs and a difference</h2>
      <div className="prose">
        <p>
          Take the network from the last lesson and feed it two inputs that differ in one bit. On the{" "}
          <b style={{ color: "var(--accent)" }}>clean</b> input <TeX>(1,1,0)</TeX> the answer is yes; on the{" "}
          <b style={{ color: "var(--warm)" }}>corrupted</b> input <TeX>(1,0,0)</TeX> it is no. Somewhere between the
          input and the output, some neurons carry that difference. <b>Activation patching</b> finds them by moving
          one neuron's value from one run into the other and watching the output.
        </p>
        <p>
          There are two directions to move it, and they answer different questions:
        </p>
        <ul>
          <li>
            <b>Denoising</b> runs the <em>corrupted</em> input and restores one neuron to its clean value. If the answer
            comes back, that neuron is <b>sufficient</b> (given the rest of the corrupted run) to carry the
            difference.
          </li>
          <li>
            <b>Noising</b> runs the <em>clean</em> input and breaks one neuron by giving it its corrupted value. If
            the answer goes away, that neuron is <b>necessary</b>.
          </li>
        </ul>
        <p>Both are <TeX>do()</TeX> interventions. We score them on the same scale, as a fraction of the clean–corrupt gap in some metric <TeX>M</TeX>:</p>
      </div>
      <Card>
        <TeX block>{String.raw`\text{recovery}(n)=\frac{M\big(\text{corr};\,do(n=n^{\text{clean}})\big)-M(\text{corr})}{M(\text{clean})-M(\text{corr})}
\qquad
\text{damage}(n)=\frac{M(\text{clean})-M\big(\text{clean};\,do(n=n^{\text{corr}})\big)}{M(\text{clean})-M(\text{corr})}`}</TeX>
        <p className="small muted" style={{ marginBottom: 0 }}>
          1 means "this neuron alone accounts for the whole gap"; 0 means "it makes no difference".
        </p>
      </Card>

      <Card title="Patch every neuron, both ways">
        <div className="controls">
          <BitPicker label="clean" x={clean} onChange={setClean} color="var(--accent)" />
          <BitPicker label="corrupted" x={corrupt} onChange={setCorrupt} color="var(--warm)" />
          <Seg label="metric M" value={metric} onChange={setMetric} options={[
            { value: "prob", label: "probability p" }, { value: "logit", label: "logit" },
          ]} />
        </div>
        {same ? (
          <p className="error">Clean and corrupted inputs are the same, so there is no difference to explain. Flip a bit.</p>
        ) : (
          <Loading loading={patch.loading} error={patch.error}>
            {d && (
              <>
                <div className="row">
                  <div style={{ flex: "1 1 300px" }}>
                    <Net values={d.clean}
                      ring={Object.fromEntries(HIDDEN.filter((n) => Math.abs(d.clean[n] - d.corrupt[n]) > 1e-9).map((n) => [n, "var(--warm)"]))}
                      sub={Object.fromEntries([...HIDDEN, "out"].filter((n) => Math.abs(d.clean[n] - d.corrupt[n]) > 1e-9)
                        .map((n) => [n, `corr: ${fmt(d.corrupt[n], n === "out" ? 1 : 2)}`]))} />
                    <div className="caption">Values in the clean run. Orange rings mark neurons whose value differs in the corrupted run (shown beneath); only these can matter.</div>
                  </div>
                  <div className="grow" style={{ flex: "0 1 220px" }}>
                    <div className="row" style={{ gap: 10 }}>
                      <Stat k={`M(clean)`} v={fm(d.metric_clean)} color="var(--accent)" />
                      <Stat k={`M(corrupted)`} v={fm(d.metric_corrupt)} color="var(--warm)" />
                    </div>
                  </div>
                </div>
                {flat ? (
                  <p className="error">Both inputs give the same {metric}, so there is no gap to recover. Pick inputs with different answers.</p>
                ) : (
                  <div className="grid2" style={{ marginTop: 10 }}>
                    <div>
                      <h3 style={{ marginTop: 0 }}>Denoising: recovery <span className="muted small">(sufficiency)</span></h3>
                      <Bars width={340} labelWidth={50} max={1} format={pct}
                        items={d.rows.map((r) => ({ label: r.node, value: r.recovery, color: "var(--accent)" }))} />
                    </div>
                    <div>
                      <h3 style={{ marginTop: 0 }}>Noising: damage <span className="muted small">(necessity)</span></h3>
                      <Bars width={340} labelWidth={50} max={1} format={pct}
                        items={d.rows.map((r) => ({ label: r.node, value: r.damage, color: "var(--warm)" }))} />
                    </div>
                  </div>
                )}
              </>
            )}
          </Loading>
        )}
      </Card>

      {n1 && !flat && !same && (
        <div className="prose">
          <p>
            With these inputs and the {metric === "prob" ? "probability" : "logit"} metric, look at <b>n1</b>: restoring
            it into the corrupted run recovers <b>{pct(n1.recovery)}</b> of the gap, and breaking it in the clean run
            does <b>{pct(n1.damage)}</b> damage.{" "}
            {n1.recovery - n1.damage > 0.3
              ? <>Denoising says "crucial", noising says "barely matters". Both are right.</>
              : <>Here the two tests roughly agree. (With the defaults, (1,1,0) against (1,0,0) and the probability
                metric, they disagree by 90 points.)</>}
          </p>
        </div>
      )}
      <div className="prose">
        <p>
          Two things conspire. <b>Redundancy</b>: n2 computes the same AND, so when you break n1 in the clean run, n2
          still carries the answer; and when you restore n1 into the corrupted run, it doesn't need n2's help.
          <b> Saturation</b>: in the clean run the logit is 9, far into the flat top of the sigmoid. Losing n1 drops the
          logit to 3, which is still <TeX>p = 0.95</TeX>. The probability barely notices.
        </p>
        <p>
          Now switch the metric to <b>logit</b>. The logit is linear in <TeX>m_1</TeX>, so there is no saturation, and
          n1 scores 50% both ways: each copy contributes half of the 12-unit gap. Same network, same patches, a
          different story. The metric is part of the question you're asking.
        </p>
      </div>
      <Callout kind="warn" label="Metric choice matters">
        <p>
          Zhang &amp; Nanda (2024, "Towards best practices of activation patching") show on real language models that
          probability-based metrics hide components (exactly the saturation effect above) and recommend logit
          differences; they also find the choice of corrupted input changes the result. Heimersheim &amp; Nanda (2024,
          "How to use and interpret activation patching") explain the denoising/noising asymmetry the same way this
          page does: denoising finds components that are <em>sufficient</em>, noising those that are{" "}
          <em>necessary</em>, and backup components make the two disagree.
        </p>
      </Callout>
      <Callout kind="try">
        <p>
          Set the corrupted input to <TeX>(0,1,0)</TeX>. Now the distractor n4 differs between the runs (it fires on
          "b without a") and patching hands it a sliver of credit: active, different between runs, and nearly
          irrelevant. Then try clean <TeX>(0,0,1)</TeX> against corrupted <TeX>(1,0,0)</TeX>: the answer runs through
          n3 alone, with no backup, and denoising and noising agree.
        </p>
      </Callout>

      <h2>Attribution patching: patching by gradient</h2>
      <div className="prose">
        <p>
          Patching costs one forward pass per neuron. A real model has millions of neurons (or thousands of heads and
          edges), so people approximate it. <b>Attribution patching</b> (Nanda 2023; Syed, Rager &amp; Conmy 2023)
          takes a first-order Taylor expansion around the corrupted run: the effect of moving <TeX>n</TeX> is roughly
          its change times the gradient.
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`\underbrace{M\big(\text{corr};\,do(n=n^{\text{clean}})\big)-M(\text{corr})}_{\text{true effect}}
\;\approx\;
\underbrace{\big(n^{\text{clean}}-n^{\text{corr}}\big)\,\frac{\partial M}{\partial n}\Big|_{\text{corr}}}_{\text{attribution patching}}`}</TeX>
        <p className="small">
          One backward pass gives the gradient for every neuron at once. The catch is the word "first-order": the
          estimate trusts the slope at the corrupted point all the way to the clean point. On a curved metric
          that's wrong. <b>Integrated gradients</b> (Sundararajan, Taly &amp; Yan 2017) averages the slope along the
          whole straight path instead, which sums to the true total change:
        </p>
        <TeX block>{String.raw`\text{IG}(n)=\Delta n\int_0^1 \frac{\partial M}{\partial n}\Big|_{n^{\text{corr}}+\alpha\,\Delta n}\,d\alpha`}</TeX>
        <p className="small muted">with <TeX>{String.raw`\Delta n = n^{\text{clean}}-n^{\text{corr}}`}</TeX>.</p>
        <p className="small muted" style={{ marginBottom: 0 }}>
          The integral is estimated here with 10 points, so IG costs about 10 backward passes instead of one.
        </p>
      </Card>

      <Card title="True effect against its estimates" caption={`Denoising effect of each neuron, in ${metric === "prob" ? "probability" : "logit"} units, for the inputs chosen above. Change the metric in the patching card.`}>
        {d && !same && <Loading loading={patch.loading} error={patch.error}><Grouped rows={d.rows} fmtv={fm} /></Loading>}
      </Card>

      <Card title="Why the tangent misses">
        <div className="controls">
          <Seg label="neuron" value={pathNode} onChange={setPathNode} options={HIDDEN.map((n) => ({ value: n, label: n }))} />
        </div>
        {pathRow && !same && (Math.abs(pathRow.clean - pathRow.corrupt) > 1e-9 ? (
          <PathPlot row={pathRow} corrupt={corrupt} metric={metric} m0={d!.metric_corrupt} />
        ) : (
          <p className="small muted">{pathNode} has the same value in both runs, so there is nothing to move along.</p>
        ))}
        <div className="caption">
          The grey curve is the metric as {pathNode} slides from its corrupted to its clean value (with the rest of the
          corrupted run fixed). Its height at the right is the true effect. The dashed orange line is the tangent at the
          left: attribution patching's guess. IG is the average slope of the curve times the distance.
        </div>
      </Card>
      <div className="prose">
        <p>
          With the <b>probability</b> metric, the corrupted run sits on the flat bottom tail of the sigmoid
          (<TeX>p \approx 0.05</TeX>) where the slope is small, and the curve then shoots up. The tangent badly
          underestimates: attribution patching gives m1 about half its true effect. IG follows the curve and lands on
          the truth.
        </p>
        <p>
          With the <b>logit</b> metric the output is linear in <TeX>m_1</TeX>, so for m1 attribution patching is exact.
          But n1 and n2 get <b>zero</b>, even though restoring either one moves the logit by 6. The reason is a{" "}
          <em>kink</em>: in the corrupted run <TeX>m_1 = 0</TeX>, right at the corner of its ReLU. The slope there is 0
          from the left and 1 from the right, and autograd (PyTorch, JAX, and this page, which copies them) takes it as
          0. So attribution patching calls n1 and n2 useless. A downstream neuron that is switched off in the corrupted
          run gives every neuron upstream of it no gradient to see with. That is a real failure mode, not a toy
          artifact, and IG does not have it because it takes slopes along the whole path, not just at the corner.
        </p>
      </div>

      <h2>Causal mediation: direct and indirect effects</h2>
      <div className="prose">
        <p>
          Patching asks about one neuron at a time. <b>Mediation analysis</b> asks a sharper question about an
          input <TeX>T</TeX> and a neuron <TeX>M</TeX>: how much of <TeX>T</TeX>'s effect on the output{" "}
          <TeX>Y</TeX> flows <em>through</em> <TeX>M</TeX>, and how much goes around it? Pearl (2001) defined the
          pieces with nested interventions:
        </p>
      </div>
      <Card>
        <div className="row">
          <div style={{ flex: "0 0 220px" }}>
            <Dag nodes={MED_NODES} edges={[["T", "M"], ["M", "Y"], ["T", "Y"]]} width={220} height={140}
              role={{ T: "x", Y: "y", M: "m" }} edgeLabels={{ "T->Y": "direct" }} />
          </div>
          <div className="grow">
            <TeX block>{String.raw`\begin{aligned}
\text{TE}  &= Y(t') - Y(t)\\
\text{NDE} &= Y\big(t',\,M(t)\big) - Y(t)\\
\text{NIE} &= Y\big(t,\,M(t')\big) - Y(t)
\end{aligned}`}</TeX>
          </div>
        </div>
        <p className="small" style={{ marginBottom: 0 }}>
          The <b>total effect</b> changes the input from <TeX>t</TeX> to <TeX>t'</TeX> and lets everything follow.
          The <b>natural direct effect</b> changes the input but pins the mediator where it was (a noising patch on{" "}
          <TeX>M</TeX>). The <b>natural indirect effect</b> keeps the input and moves only the mediator to where{" "}
          <TeX>t'</TeX> would have put it (a denoising patch). "Direct" here means every path that avoids{" "}
          <TeX>M</TeX>, not literally one arrow.
        </p>
      </Card>

      <Card title="Mediation in the network">
        <div className="controls">
          <Seg label="treatment T (flipped)" value={treat} onChange={setTreat} options={[
            { value: "a", label: "a" }, { value: "b", label: "b" }, { value: "c", label: "c" },
          ]} />
          <BitPicker label="base input t" x={base} onChange={setBase} color="var(--ink-2)" />
          <Select label="mediator M" value={mediator} onChange={setMediator} options={HIDDEN} />
          <Seg label="metric" value={medMetric} onChange={setMedMetric} options={[
            { value: "prob", label: "probability" }, { value: "logit", label: "logit" },
          ]} />
        </div>
        <Loading loading={med.loading} error={med.error}>
          {m && (
            <>
              <p className="small">
                Flip <b>{treat}</b> from {base[ti]} to {changed}: input <span className="mono">{bits(base)}</span> →{" "}
                <span className="mono">{bits(x1)}</span>. The mediator {mediator} goes from {fmt(m.mediator_before)} to{" "}
                {fmt(m.mediator_after)}.
              </p>
              <div className="small">
                <TeX block>{String.raw`\text{TE} = ${mfm(m.y_after)} - ${mfm(m.y_before)} = \mathbf{${mfm(m.te)}}`}</TeX>
                <TeX block>{String.raw`\text{NDE} = Y(t',\,${Mname}{=}${fmt(m.mediator_before)}) - Y(t) = \mathbf{${mfm(m.nde)}}`}</TeX>
                <TeX block>{String.raw`\text{NIE} = Y(t,\,${Mname}{=}${fmt(m.mediator_after)}) - Y(t) = \mathbf{${mfm(m.nie)}}`}</TeX>
                <TeX block>{String.raw`\text{TE} - \text{NDE} - \text{NIE} = \mathbf{${mfm(m.interaction)}}`}</TeX>
              </div>
              <Bars width={460} labelWidth={130} format={mfm}
                max={1.6 * Math.max(1e-9, Math.abs(m.te), Math.abs(m.nde), Math.abs(m.nie), Math.abs(m.interaction))}
                items={[
                  { label: "total effect", value: m.te, color: "var(--ink-2)" },
                  { label: "natural direct", value: m.nde, color: "var(--accent)" },
                  { label: "natural indirect", value: m.nie, color: "var(--gold)" },
                  { label: "interaction", value: m.interaction, color: "var(--warm)" },
                ]} />
            </>
          )}
        </Loading>
      </Card>
      <div className="prose">
        <p>
          In a linear model the direct and indirect parts add up to the total. Here, with the defaults (flip{" "}
          <TeX>b</TeX> from <TeX>(1,0,0)</TeX>, mediator n1, probability), both NDE and NIE are about 0.9 while the
          total is 0.95. The leftover, <TeX>{String.raw`\text{TE} - \text{NDE} - \text{NIE}`}</TeX>, is the{" "}
          <b>interaction</b>, and it is large and negative. It's the same redundancy again: the path through n1
          alone is enough to flip the answer, the path through n2 alone is enough too, and "enough" can't be counted
          twice once the sigmoid saturates.
        </p>
        <p>
          Switch the metric to logit and the interaction vanishes: 6 + 6 = 12. Pick m1 as the mediator and the NIE is
          the whole effect with NDE = 0: every path from <TeX>b</TeX> to the output goes through m1, so m1{" "}
          <em>fully mediates</em> it.
        </p>
      </div>

      <Callout kind="takeaway">
        <p>
          Denoising measures sufficiency and noising measures necessity; redundancy makes them disagree, and a
          saturating metric widens the gap. Attribution patching is a cheap first-order guess at patching that fails
          wherever the model is curved or kinked; integrated gradients pays more passes to follow the curve.
          Mediation splits an effect into direct and indirect parts, and an interaction term tells you when the parts
          don't add.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          Vig et al. (2020, "Investigating gender bias in language models using causal mediation analysis") applied
          Pearl's NDE/NIE to individual neurons and attention heads in GPT-2. Activation patching became the workhorse
          of circuit work through ROME (Meng et al. 2022) and the IOI circuit (Wang et al. 2022). Attribution patching
          (Nanda 2023) and its edge version EAP (Syed, Rager &amp; Conmy 2023) scale it to whole models; Hanna,
          Pezzelle &amp; Belinkov (2024) and Marks et al. (2024) use integrated-gradient variants because plain
          gradients miss saturated components. Backup heads that switch on when a primary head is ablated were found
          in IOI (Wang et al. 2022) and studied as "self-repair" by McGrath et al. (2023), the real version of n2.
        </p>
      </Callout>
    </div>
  );
}
