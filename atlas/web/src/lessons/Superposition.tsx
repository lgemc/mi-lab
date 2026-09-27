import { useEffect, useState } from "react";
import { Bars, Heatmap, Lines } from "../components/charts";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Seg, Slider, Stat } from "../components/ui";
import { api, fmt, pct, useApi } from "../api";
import { Plane, Presets, Swatch } from "./dictHelpers";

/*
 * Elhage et al.'s toy model of superposition (api/atlas/superposition.py), trained on the server
 * per request. n features, 2 hidden dimensions. The page drags the sparsity and watches the
 * columns of W rearrange; everything else (interference, the one-feature test) is arithmetic on
 * the returned W and b, done here.
 */

type Result = {
  W: number[][]; b: number[]; norms: number[]; importance: number[]; dims_per_feature: number[];
  features_represented: number; gram: number[][]; history: { step: number; loss: number }[];
};

/** Most important feature orange, least important blue, blended in between. */
const impColor = (i: number, n: number) => {
  const t = n <= 1 ? 0 : i / (n - 1);
  return `color-mix(in oklab, var(--warm) ${Math.round(100 - 100 * t)}%, var(--accent))`;
};

const SWEEP = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99];

export default function Superposition() {
  const [n, setN] = useState(5);
  const [sparsity, setSparsity] = useState(0);
  const [decay, setDecay] = useState(0.9);
  const [seed, setSeed] = useState(0);
  const [probe, setProbe] = useState(0);
  const { data, error, loading } = useApi<Result>("superposition", { n, sparsity, decay, seed }, 250);

  // the phase curve: one small model per sparsity, fetched one after another (the server caches them)
  const [sweep, setSweep] = useState<{ key: string; rows: { s: number; k: number; d: number }[] } | null>(null);
  const sweepKey = `${n}-${decay}-${seed}`;
  useEffect(() => {
    let live = true;
    const t = setTimeout(async () => {
      const rows: { s: number; k: number; d: number }[] = [];
      for (const s of SWEEP) {
        try {
          const r = await api<Result>("superposition", { n, sparsity: s, decay, seed });
          const rep = r.norms.map((v, i) => [v, r.dims_per_feature[i]]).filter(([v]) => v > 0.5);
          rows.push({ s, k: r.features_represented, d: rep.length ? rep.reduce((a, [, d]) => a + d, 0) / rep.length : 0 });
        } catch { return; }
        if (!live) return;
      }
      if (live) setSweep({ key: sweepKey, rows });
    }, 600);
    return () => { live = false; clearTimeout(t); };
  }, [n, decay, seed, sweepKey]);

  const names = Array.from({ length: n }, (_, i) => `x${i + 1}`);
  const cols = data ? data.W[0].map((_, i) => [data.W[0][i], data.W[1][i]] as [number, number]) : [];
  const ext = data ? Math.max(1.2, ...data.norms) * 1.18 : 1.4;
  const p = Math.min(probe, n - 1);
  const rep = data ? data.norms.map((v, i) => ({ v, d: data.dims_per_feature[i] })).filter((e) => e.v > 0.5) : [];
  const meanDims = rep.length ? rep.reduce((a, e) => a + e.d, 0) / rep.length : 0;

  return (
    <div className="prose-wide">
      <h2>More concepts than neurons</h2>
      <div className="prose">
        <p>
          A layer with <TeX>m</TeX> neurons has room for <TeX>m</TeX> perpendicular directions. If each neuron stood
          for one concept, a layer could know at most <TeX>m</TeX> things. Language models seem to know far more
          things than they have neurons. How?
        </p>
        <p>
          Elhage and colleagues built the smallest model that can answer this. There are <TeX>n</TeX> input
          features and only <TeX>m = 2</TeX> hidden dimensions. The model has to squeeze the features into the
          plane and read them back out:
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`h = W x \quad (W \in \mathbb{R}^{2\times n}), \qquad
x' = \mathrm{ReLU}\!\left(W^\top h + b\right), \qquad
\mathcal{L} = \sum_i I_i\,(x_i - x'_i)^2`}</TeX>
        <p className="small">
          Each feature <TeX>x_i</TeX> is zero with probability <TeX>S</TeX> (the <b>sparsity</b>) and otherwise
          uniform on <TeX>[0, 1]</TeX>. Feature <TeX>i</TeX> matters with weight <TeX>{String.raw`I_i = \text{decay}^{\,i-1}`}</TeX>,
          so <TeX>x_1</TeX> is the most important. Column <TeX>i</TeX> of <TeX>W</TeX> is the arrow in the plane
          where feature <TeX>i</TeX> is stored. The same arrow is used to read it back.
        </p>
      </Card>

      <Card title="Drag the sparsity">
        <Presets onPick={(s: number) => setSparsity(s)} items={[
          { label: "Dense (S = 0)", value: 0 },
          { label: "In between (S = 0.7)", value: 0.7 },
          { label: "Sparse (S = 0.9)", value: 0.9 },
          { label: "Very sparse (S = 0.99)", value: 0.99 },
        ]} />
        <div className="controls">
          <Slider label="Sparsity S" value={sparsity} min={0} max={0.99} step={0.01} onChange={setSparsity} format={(v) => v.toFixed(2)} />
          <Slider label="Features n" value={n} min={2} max={8} step={1} onChange={setN} format={(v) => String(v)} />
          <Slider label="Importance decay" value={decay} min={0.5} max={1} step={0.05} onChange={setDecay} format={(v) => v.toFixed(2)} />
          <Seg label="Seed" value={seed} onChange={setSeed} options={[0, 1, 2].map((s) => ({ value: s, label: String(s) }))} />
        </div>
        <Loading loading={loading} error={error}>
          {data && (
            <div className="row">
              <div style={{ flex: "1 1 300px", maxWidth: 360 }}>
                <Plane extent={ext} unitCircle ariaLabel="The columns of W drawn as arrows in the plane"
                  arrows={cols.map((c, i) => ({ to: c, color: impColor(i, n), label: names[i], width: 2.2 + 1.6 * data.importance[i], opacity: data.norms[i] > 0.5 ? 1 : 0.35 }))} />
                <div className="small muted" style={{ marginTop: 4 }}>
                  <Swatch color={impColor(0, n)}>most important</Swatch>
                  <Swatch color={impColor(n - 1, n)}>least important</Swatch>
                  <span>dashed circle: length 1</span>
                </div>
              </div>
              <div className="grow">
                <div className="row" style={{ gap: 10 }}>
                  <Stat k="Features represented" v={`${data.features_represented} / ${n}`} d="columns with length > 0.5" />
                  <Stat k="Dims per feature" v={fmt(meanDims, 2)} d="mean over represented" />
                  <Stat k="P(two given features both on)" v={pct((1 - sparsity) ** 2, sparsity > 0.9 ? 2 : 0)} d={<TeX>{"(1-S)^2"}</TeX>} />
                </div>
                <div className="small" style={{ margin: "12px 0 4px" }}>Length of each arrow, <TeX>{String.raw`\lVert W_i \rVert`}</TeX></div>
                <Bars width={360} labelWidth={50} max={1.3} refLine={{ value: 0.5, label: "0.5" }}
                  items={data.norms.map((v, i) => ({ label: names[i], value: v, color: impColor(i, n) }))} />
                <div className="small" style={{ margin: "10px 0 4px" }}>Bias <TeX>b_i</TeX></div>
                <Bars width={360} labelWidth={50} max={0.6}
                  items={data.b.map((v, i) => ({ label: names[i], value: v, color: v < 0 ? "var(--warm)" : "var(--accent)" }))} />
              </div>
            </div>
          )}
        </Loading>
        <div className="caption">
          Each drag trains a fresh model on the server (3,000 steps, a fraction of a second). The arrows are the
          columns of <TeX>W</TeX>; faded arrows are features the model gave up on.
        </div>
      </Card>

      <Callout kind="try">
        <p>
          Start at <b>S = 0</b>. The model keeps two features on perpendicular axes and drops the rest, exactly
          what PCA would do. Move to about <b>0.7</b>: here the default model stores four features as two{" "}
          <em>antipodal pairs</em>, pointing in opposite directions on the same line. Past about <b>0.9</b> all five
          fan out into a pentagon, each about 72° from its neighbours, and the biases turn negative.
        </p>
      </Callout>

      <Card title="The phase change, all at once" caption="One model per sparsity value, for the current n, decay and seed. The number of features stored jumps rather than creeping up.">
        {sweep && sweep.key === sweepKey ? (
          <div className="grid2">
            <div>
              <div className="small" style={{ marginBottom: 4 }}>Features represented</div>
              <Lines height={200} width={420} yMin={0} yMax={n % 2 ? n + 1 : n} format={(v) => v.toFixed(0)}
                xLabels={sweep.rows.map((r) => String(r.s))} markX={SWEEP.indexOf(sparsity) >= 0 ? SWEEP.indexOf(sparsity) : undefined}
                series={[{ name: "features stored", values: sweep.rows.map((r) => r.k), color: "var(--accent)" }]} />
            </div>
            <div>
              <div className="small" style={{ marginBottom: 4 }}>Dimensions per stored feature</div>
              <Lines height={200} width={420} yMin={0} yMax={1.05}
                xLabels={sweep.rows.map((r) => String(r.s))}
                series={[{ name: "dims / feature", values: sweep.rows.map((r) => r.d), color: "var(--warm)" }]} />
            </div>
          </div>
        ) : <p className="small muted">Training one model per sparsity value…</p>}
        <p className="small">
          "Dimensions per feature" is Elhage et al.'s measure of how much of the space a feature gets: 1 for a
          feature alone on its axis, 1/2 for an antipodal pair, 2/5 for a pentagon of five features in two
          dimensions.
        </p>
      </Card>

      <h2>Interference, and why sparsity makes it cheap</h2>
      <div className="prose">
        <p>
          Five arrows in a plane cannot all be perpendicular. When feature <TeX>i</TeX> is on alone, the model
          reads output <TeX>j</TeX> as <TeX>{String.raw`W_j \cdot W_i\, x_i`}</TeX>: a bit of feature <TeX>i</TeX>{" "}
          leaks into feature <TeX>j</TeX>. All these dot products sit in the <b>Gram matrix</b>{" "}
          <TeX>{String.raw`W^\top W`}</TeX>. The diagonal is how strongly each feature is stored; everything off the
          diagonal is interference.
        </p>
      </div>
      <Loading loading={loading} error={error}>
        {data && (
          <div className="grid2">
            <Card title={<>Gram matrix <TeX>{String.raw`W^\top W`}</TeX></>}>
              <Heatmap values={data.gram} rowLabels={names} colLabels={names} cell={34} max={1.4}
                caption="Hover a cell. Blue is positive overlap, orange negative." />
            </Card>
            <Card title="Turn one feature on">
              <Seg value={p} onChange={setProbe} options={names.map((nm, i) => ({ value: i, label: nm }))} />
              <div className="scroll-x" style={{ marginTop: 10 }}>
                <table className="tbl">
                  <thead><tr><th>output</th><th><TeX>{String.raw`W_j\!\cdot\! W_i`}</TeX></th><th>+ bias</th><th>ReLU</th></tr></thead>
                  <tbody>
                    {names.map((nm, j) => {
                      const pre = data.gram[j][p] + data.b[j];
                      return (
                        <tr key={j} style={j === p ? { fontWeight: 700 } : undefined}>
                          <td>{nm}{j === p ? " (on)" : ""}</td>
                          <td className="mono">{fmt(data.gram[j][p])}</td>
                          <td className="mono">{fmt(pre)}</td>
                          <td className="mono" style={{ color: j !== p && pre > 0.005 ? "var(--warm)" : undefined }}>{fmt(Math.max(0, pre))}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="small muted">
                Input: <TeX>{`x_{${p + 1}} = 1`}</TeX>, everything else 0. The target output is 1 for{" "}
                <TeX>{`x_{${p + 1}}`}</TeX> and 0 elsewhere; orange numbers are leaks that survive.
              </p>
            </Card>
          </div>
        )}
      </Loading>
      <div className="prose">
        <p>
          Look at the pentagon (S ≥ 0.9). The two features <em>opposite</em> <TeX>x_i</TeX> have negative overlap,
          so the ReLU zeroes them for free. The two <em>neighbours</em> have positive overlap, and that's what the
          negative bias is for: it subtracts a little from every output so small leaks fall below zero and vanish.
          The bias doesn't remove all of it here; the remaining leak is the price of storing five things in two
          dimensions.
        </p>
        <p>
          Why is that price worth paying only when features are sparse? Storing an extra feature earns its loss
          reduction every time the feature is on, which is proportional to <TeX>1-S</TeX>. The ReLU-and-bias trick
          cleans up interference as long as only one feature is on at a time. The leak it <em>can't</em> clean up
          happens when two overlapping features are on together, which is proportional to <TeX>(1-S)^2</TeX>. As{" "}
          <TeX>S \to 1</TeX>, the cost shrinks faster than the benefit, and superposition wins. For dense
          features every feature is on at once, the interference never goes away, and the model is better off
          with the two most important features on clean axes.
        </p>
      </div>

      <h2>Why this breaks "one neuron, one concept"</h2>
      <div className="prose">
        <p>
          Now read the same matrix by rows. Row <TeX>k</TeX> of <TeX>W</TeX> says how much hidden neuron{" "}
          <TeX>k</TeX> responds to each feature.
        </p>
      </div>
      <Loading loading={loading} error={error}>
        {data && (
          <Card title="What each hidden neuron responds to" caption="Rows of W. In the dense regime each neuron carries about one feature. In the pentagon, each responds to most of them.">
            <Heatmap values={data.W} rowLabels={["neuron 1", "neuron 2"]} colLabels={names} cell={36} max={1.2} rowLabelWidth={74} />
          </Card>
        )}
      </Loading>
      <div className="prose">
        <p>
          In the sparse regime each neuron is active for several unrelated features. That's{" "}
          <b>polysemanticity</b>, and it follows directly from superposition. If you listen to one neuron you hear
          a mixture. The model's real variables are the five <em>directions</em>, and none of them lines up with a
          neuron.
        </p>
        <p>
          One honest caveat: this <TeX>h</TeX> has no nonlinearity, so any rotation of the pentagon is an equally
          good solution. Nothing ties features to neurons at all, and seeds give different rotations. Elhage et al.
          also give the hidden layer its own ReLU, which does make individual neurons special. Superposition still
          appears there, which is closer to what happens in real models.
        </p>
      </div>

      <Callout kind="takeaway">
        <p>
          When features are sparse, a network can store many more of them than it has dimensions by giving each
          one a direction and tolerating a little interference. The interference is rare because the features are
          rarely on together, and a ReLU with a negative bias filters much of it out. The cost is that neurons stop
          meaning one thing, so to find the model's variables we have to look for directions. The next lesson
          does that.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          Elhage et al. (2022), <em>Toy Models of Superposition</em>, map the phase diagram of this model
          (importance against sparsity), find the antipodal pairs and the pentagons, and connect superposition
          to polysemantic neurons. Polysemantic neurons were documented earlier in vision models (Olah et al.,
          2017, 2020). The superposition hypothesis, that a layer uses many more near-orthogonal directions than it
          has neurons, is what motivates the dictionary-learning methods in the rest of this part.
        </p>
      </Callout>
    </div>
  );
}
