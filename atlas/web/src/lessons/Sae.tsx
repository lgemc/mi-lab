import { useState } from "react";
import { Bars, Heatmap, PALETTE } from "../components/charts";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Seg, Slider, Stat } from "../components/ui";
import { fmt, pct, useApi } from "../api";
import { Plane, Presets, Swatch, XYChart } from "./dictHelpers";

/*
 * A sparse autoencoder trained (api/atlas/sae.py) on 2-D points built from k known feature
 * directions, so every latent can be graded against the truth. The server trains per request;
 * the page draws the cloud, the true arrows and the learned decoder directions, and works out the
 * shrinkage and the per-point breakdown from the returned activations.
 */

type Result = {
  true: number[][]; decoder: number[][]; encoder: number[][]; b_enc: number[]; b_dec: number[];
  alive: boolean[]; cos: number[][]; best_match: number[]; l0: number; fvu: number; dead: number;
  resampled: number; points: number[][]; point_latents: number[][]; point_features: number[][];
  history: { step: number; mse: number; penalty: number; l0: number }[];
};
type Sweep = { rows: { knob: number; l0: number; fvu: number; dead: number }[] };
type Cfg = { kTrue: number; latents: number; mode: "relu" | "topk"; l1: number; topk: number; p: number; seed: number };

const DEFAULT: Cfg = { kTrue: 5, latents: 5, mode: "relu", l1: 0.3, topk: 1, p: 0.15, seed: 0 };
const PRESETS: { label: string; value: Partial<Cfg> }[] = [
  { label: "Just right", value: {} },
  { label: "Too few latents", value: { latents: 3 } },
  { label: "Too many latents", value: { latents: 12 } },
  { label: "Weak L1", value: { l1: 0.01 } },
  { label: "Strong L1", value: { l1: 1 } },
  { label: "TopK, k = 1", value: { mode: "topk", topk: 1 } },
  { label: "TopK, k = 2", value: { mode: "topk", topk: 2 } },
];

const fcol = (i: number) => PALETTE[i % PALETTE.length];

export default function Sae() {
  const [cfg, setCfg] = useState<Cfg>(DEFAULT);
  const [preset, setPreset] = useState("Just right");
  const [sel, setSel] = useState<number | null>(null);
  const set = (patch: Partial<Cfg>) => { setCfg((c) => ({ ...c, ...patch })); setPreset(""); };
  const { data, error, loading } = useApi<Result>("sae", {
    k_true: cfg.kTrue, latents: cfg.latents, p: cfg.p, l1: cfg.l1, mode: cfg.mode, topk: cfg.topk, seed: cfg.seed,
  }, 250);
  const sweep = useApi<Sweep>("sae/sweep", { k_true: cfg.kTrue, latents: cfg.latents, p: cfg.p, mode: cfg.mode }, 400);

  // per-latent: which true feature it matches best, and how well
  const match = data ? data.decoder.map((_, j) => {
    let best = 0;
    for (let i = 1; i < data.true.length; i++) if (data.cos[i][j] > data.cos[best][j]) best = i;
    return { f: best, cos: data.cos[best][j] };
  }) : [];
  const latColor = (j: number) => (match[j] && match[j].cos > 0.9 ? fcol(match[j].f) : "var(--ink-2)");

  const recon = (i: number) => {
    if (!data) return [0, 0];
    const z = data.point_latents[i];
    return [0, 1].map((d) => data.b_dec[d] + z.reduce((s, zj, j) => s + zj * data.decoder[j][d], 0));
  };
  // Shrinkage: on points built from one feature, how long is the reconstruction compared with the input?
  let shrink = NaN;
  if (data) {
    let num = 0, den = 0;
    data.points.forEach((x, i) => {
      const on = data.point_features[i].filter((v) => v > 0).length;
      if (on !== 1) return;
      const xh = recon(i);
      num += xh[0] * x[0] + xh[1] * x[1];
      den += x[0] * x[0] + x[1] * x[1];
    });
    shrink = den > 0 ? num / den : NaN;
  }

  const pointColors = data ? data.point_features.map((f) => {
    const m = Math.max(...f);
    return m > 0 ? fcol(f.indexOf(m)) : "var(--ink-3)";
  }) : [];
  const ext = data ? Math.min(2.2, Math.max(1.25, ...data.points.flat().map(Math.abs)) * 1.08) : 1.4;
  const firstRich = data ? data.point_features.findIndex((f) => f.filter((v) => v > 0).length >= 2) : -1;
  const s = sel !== null && data && sel < data.points.length ? sel : firstRich >= 0 ? firstRich : null;
  const knob = cfg.mode === "topk" ? cfg.topk : cfg.l1;

  return (
    <div className="prose-wide">
      <h2>Superposition, run backwards</h2>
      <div className="prose">
        <p>
          The last lesson squeezed known features into two dimensions. In a real model we have the opposite
          problem: we can see the activations but not the features. If superposition is right, every activation
          is a sum of a <em>few</em> feature directions taken from a much larger set. Recovering that set is{" "}
          <b>dictionary learning</b>, and a <b>sparse autoencoder</b> (SAE) is the version that scales to language
          models.
        </p>
        <p>
          Here is a cloud built exactly that way. There are five true feature directions in the plane. Each one is
          on with probability <TeX>p</TeX>, at a random strength between 0.3 and 1. A point is the sum of whichever
          features are on, so the cloud is a star. The SAE sees only the points. Can it find the arrows?
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`z = \mathrm{act}\big(W_{\text{enc}}(x - b_{\text{dec}}) + b_{\text{enc}}\big), \qquad
\hat x = W_{\text{dec}}\, z + b_{\text{dec}} = b_{\text{dec}} + \sum_j z_j\, d_j`}</TeX>
        <TeX block>{String.raw`\text{ReLU + L1:}\;\; \mathcal L = \lVert x-\hat x\rVert^2 + \lambda \sum_j z_j \lVert d_j\rVert
\qquad\qquad
\text{TopK:}\;\; \mathcal L = \lVert x-\hat x\rVert^2,\;\; \text{keep the } k \text{ largest } z_j`}</TeX>
        <p className="small">
          Each latent <TeX>j</TeX> has a decoder direction <TeX>d_j</TeX>, a column of{" "}
          <TeX>{String.raw`W_{\text{dec}}`}</TeX>. The hope is that each <TeX>d_j</TeX> lands on one true feature
          and <TeX>z_j</TeX> reports how strongly that feature is on. There are two ways to force sparsity. You
          can charge a price <TeX>\lambda</TeX> for activity (the L1 penalty, weighted by decoder norm as in
          Anthropic's 2024 recipe), or you can simply keep the <TeX>k</TeX> largest activations and zero the rest
          (TopK).
        </p>
      </Card>

      <Card title="Train an SAE on the star">
        <Presets active={preset} onPick={(v: Partial<Cfg>) => {
          setCfg({ ...DEFAULT, ...v });
          setPreset(PRESETS.find((p) => p.value === v)?.label ?? "");
        }} items={PRESETS} />
        <div className="controls">
          <Seg label="Sparsity by" value={cfg.mode} onChange={(mode) => set({ mode })}
            options={[{ value: "relu", label: "ReLU + L1" }, { value: "topk", label: "TopK" }]} />
          {cfg.mode === "relu"
            ? <Slider label="L1 penalty λ" value={cfg.l1} min={0} max={2} step={0.01} onChange={(l1) => set({ l1 })} format={(v) => v.toFixed(2)} />
            : <Slider label="k (active latents)" value={cfg.topk} min={1} max={4} step={1} onChange={(topk) => set({ topk })} format={(v) => String(v)} />}
          <Slider label="Latents" value={cfg.latents} min={2} max={16} step={1} onChange={(latents) => set({ latents })} format={(v) => String(v)} />
        </div>
        <div className="controls">
          <Slider label="True features" value={cfg.kTrue} min={2} max={8} step={1} onChange={(kTrue) => set({ kTrue })} format={(v) => String(v)} />
          <Slider label="P(feature on) p" value={cfg.p} min={0.02} max={0.5} step={0.01} onChange={(p) => set({ p })} format={(v) => v.toFixed(2)} />
          <Seg label="Seed" value={cfg.seed} onChange={(seed) => set({ seed })} options={[0, 1, 2, 3].map((v) => ({ value: v, label: String(v) }))} />
        </div>
        <Loading loading={loading} error={error}>
          {data && (
            <>
              <div className="row">
                <div style={{ flex: "1 1 300px", maxWidth: 380 }}>
                  <Plane points={data.points as [number, number][]} pointColors={pointColors} extent={ext} size={380}
                    selected={s} onPointClick={setSel} ariaLabel="The data cloud with true feature arrows and learned decoder directions"
                    arrows={[
                      ...data.true.map((t, i) => ({ to: t as [number, number], color: fcol(i), dashed: true, width: 1.6, label: `f${i + 1}` })),
                      ...data.decoder.map((d, j) => {
                        const nrm = Math.hypot(d[0], d[1]) || 1;
                        return data.alive[j]
                          ? { to: [0.72 * d[0] / nrm, 0.72 * d[1] / nrm] as [number, number], color: latColor(j), width: 3.4, label: `L${j}` }
                          : null;
                      }).filter((a): a is NonNullable<typeof a> => a !== null),
                    ]} />
                  <div className="small muted" style={{ marginTop: 4 }}>
                    <Swatch color="var(--ink-2)" dashed>true feature <i>f</i></Swatch>
                    <Swatch color="var(--ink-2)">learned latent <i>L</i> (direction)</Swatch>
                    <br />Points are coloured by the strongest true feature in them (grey: none on). Click a point.
                  </div>
                </div>
                <div className="grow">
                  <div className="row" style={{ gap: 10 }}>
                    <Stat k="L0" v={fmt(data.l0, 2)} d="active latents per input" />
                    <Stat k="FVU" v={pct(data.fvu, 1)} d="variance unexplained" color={data.fvu > 0.2 ? "var(--warm)" : undefined} />
                    <Stat k="Dead latents" v={`${data.dead} / ${cfg.latents}`} d={`${data.resampled} resamples`} color={data.dead ? "var(--warm)" : undefined} />
                    <Stat k="Shrinkage" v={Number.isFinite(shrink) ? pct(shrink) : "—"} d="recon. length, one-feature inputs" />
                  </div>
                  <div className="small" style={{ margin: "12px 0 4px" }}>
                    Best cosine between each true feature and any live latent (1 = found exactly)
                  </div>
                  <Bars width={380} labelWidth={40} max={1} format={(v) => v.toFixed(2)} refLine={{ value: 0.9, label: "0.9" }}
                    items={data.best_match.map((v, i) => ({ label: `f${i + 1}`, value: v, color: v > 0.9 ? fcol(i) : "var(--ink-3)" }))} />
                  <PointPanel data={data} i={s} match={match} recon={s !== null ? recon(s) : null} />
                </div>
              </div>
              <div style={{ marginTop: 10 }}>
                <Heatmap values={data.cos.map((row) => row.map((v, j) => (data.alive[j] ? v : 0)))}
                  rowLabels={data.true.map((_, i) => `f${i + 1}`)} rowLabelWidth={34} cell={30} max={1}
                  colLabels={data.decoder.map((_, j) => (data.alive[j] ? `L${j}` : `L${j} dead`))}
                  caption="Cosine between each true feature (rows) and each latent's decoder direction (columns). A clean dictionary has one strong blue cell per row and per column." />
              </div>
            </>
          )}
        </Loading>
      </Card>

      <Callout kind="try">
        <p>
          Work through the presets. <b>Just right</b>: five latents for five features, and every best-match bar
          reaches 1. <b>Too few latents</b>: three latents for five features, so some latents sit <em>between</em>{" "}
          two true arrows and stand for a blend of them. <b>Too many latents</b>: twelve latents and nothing
          finer to find, so several features get two near-identical latents and a few latents die. Watch the
          cosine grid for rows with two blue cells.
        </p>
      </Callout>

      <h2>Sparsity against fidelity</h2>
      <div className="prose">
        <p>
          The L1 penalty pulls two ways. A larger <TeX>\lambda</TeX> makes fewer latents fire (lower L0) and forces
          each input to be explained by one clean latent. It also makes reconstruction worse. Every SAE sits
          somewhere on this trade-off, and papers compare SAEs by the <b>frontier</b> they trace: fraction of
          variance unexplained (FVU) against L0.
        </p>
      </div>
      <Card title={cfg.mode === "relu" ? "One SAE per λ" : "One SAE per k"}
        caption={`Each point is an SAE with ${cfg.latents} latents on the current data, trained for 1,500 steps, half as long as the main one, so the numbers differ slightly. Lower-left is better. The circled point is the setting nearest the slider.`}>
        <Loading loading={sweep.loading} error={sweep.error}>
          {sweep.data ? (() => {
            const rows = sweep.data.rows;
            let hi = 0;
            rows.forEach((r, i) => { if (Math.abs(Math.log(r.knob + 1e-3) - Math.log(knob + 1e-3)) < Math.abs(Math.log(rows[hi].knob + 1e-3) - Math.log(knob + 1e-3))) hi = i; });
            return (
              <div className="row">
                <div style={{ flex: "1 1 320px", maxWidth: 480 }}>
                  <XYChart xLabel="L0 (active latents per input)" yLabel="FVU" yMax={1.05} format={(v) => v.toFixed(1)} highlight={hi}
                    points={rows.map((r) => ({ x: r.l0, y: r.fvu, label: cfg.mode === "relu" ? `λ=${r.knob}` : `k=${r.knob}`, color: r.dead ? "var(--warm)" : "var(--accent)" }))} />
                </div>
                <div className="grow scroll-x">
                  <table className="tbl">
                    <thead><tr><th>{cfg.mode === "relu" ? "λ" : "k"}</th><th>L0</th><th>FVU</th><th>dead</th></tr></thead>
                    <tbody>{rows.map((r) => (
                      <tr key={r.knob}><td className="mono">{r.knob}</td><td className="mono">{fmt(r.l0, 2)}</td>
                        <td className="mono">{pct(r.fvu, 1)}</td><td className="mono" style={{ color: r.dead ? "var(--warm)" : undefined }}>{r.dead}</td></tr>
                    ))}</tbody>
                  </table>
                  <p className="small muted">Orange points have dead latents.</p>
                </div>
              </div>
            );
          })() : <p className="small muted">Training the sweep (a few seconds the first time)…</p>}
        </Loading>
      </Card>
      <div className="prose">
        <p>
          The L1 penalty has a side effect called <b>shrinkage</b>. It charges for the <em>size</em> of every
          activation, so the SAE learns to report features a little weaker than they are and accepts some
          reconstruction error in exchange for a smaller bill. The "Shrinkage" stat above measures it on inputs
          made of a single feature. At <TeX>\lambda = 0.3</TeX> reconstructions come out noticeably short, and at{" "}
          <TeX>\lambda = 1</TeX> much shorter. TopK charges nothing for size, so it reports about 100%. Push{" "}
          <TeX>\lambda</TeX> toward 2 and shutting latents off becomes the cheapest option: L0 falls toward
          zero and FVU climbs past 90%.
        </p>
      </div>

      <h2>A perfect autoencoder can be a useless dictionary</h2>
      <div className="prose">
        <p>
          Now pick <b>TopK, k = 2</b>. FVU drops to essentially zero, far better than any L1 setting. Look at the
          arrows, though. Some true features have no latent of their own, and the best-match bars fall to around
          0.8.
        </p>
        <p>
          The reason is simple geometry. The data is two-dimensional, and <em>any</em> two non-parallel directions
          can add up to any point in the plane. With two latents allowed per input, the SAE doesn't need the
          features. A rough basis of four directions reconstructs everything perfectly. Reconstruction was never
          the goal. We wanted the variables that generated the data, and a loss that only checks reconstruction
          can't tell those apart from any other spanning set. With <TeX>k = 1</TeX> the SAE <em>has</em> to explain
          each point with one arrow, and it finds the features.
        </p>
      </div>
      <Callout kind="warn">
        <p>
          Real SAEs never have ground truth, so they are judged on the frontier (FVU or downstream loss against
          L0) plus human or automated interpretability ratings. This toy shows why low FVU alone proves nothing:
          it can be achieved with no features found at all. Here the two-dimensional data makes this extreme. In
          a real model the risk is subtler, for example latents that absorb or merge features.
        </p>
      </Callout>

      <h2>Dead latents</h2>
      <div className="prose">
        <p>
          A latent whose encoder never produces a positive activation gets no gradient and never recovers. It's{" "}
          <b>dead</b>, a wasted slot. This trainer uses the fix from Bricken et al.: every 250 steps during the first
          70% of training, each latent that hasn't fired is re-pointed at an input the SAE currently reconstructs
          badly. The "resamples" count under the dead-latents stat says how often that happened. The latents
          still dead at the end are ones for which there was genuinely nothing left to explain.
        </p>
      </div>

      <Callout kind="takeaway">
        <p>
          An SAE rewrites each activation as a sparse sum of learned directions. When the data really is built
          that way and the dictionary is the right size, it recovers the true features. It can also fail
          quietly: with too few latents it merges features, with too many it duplicates them or lets them die,
          L1 shrinks every activation, and a loose sparsity budget lets it reconstruct perfectly without finding
          anything. Low reconstruction error is necessary, but it doesn't show that the latents are the model's
          variables.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          Bricken et al. (2023), <em>Towards Monosemanticity</em>, trained SAEs on a one-layer transformer's MLP
          and found thousands of interpretable features, together with feature splitting as the dictionary
          grows and dead-latent resampling. Templeton et al. (2024), <em>Scaling Monosemanticity</em>, scaled this
          to Claude 3 Sonnet with millions of latents and the decoder-norm-weighted L1 used here. Gao et al.
          (2024), <em>Scaling and evaluating sparse autoencoders</em>, replaced L1 with TopK to remove shrinkage
          and fix L0 directly, and studied the sparsity–fidelity frontier across sizes. Whether SAE latents are the
          model's own variables is still an open question. Later lessons return to it with transcoders and
          crosscoders.
        </p>
      </Callout>
    </div>
  );
}

function PointPanel({ data, i, match, recon }: {
  data: Result; i: number | null; match: { f: number; cos: number }[]; recon: number[] | null;
}) {
  if (i === null || !recon) {
    return <p className="small muted" style={{ marginTop: 12 }}>Click any point in the cloud to see which true features built it and which latents fire for it.</p>;
  }
  const feats = data.point_features[i].map((v, k) => ({ v, k })).filter((e) => e.v > 0);
  const lats = data.point_latents[i].map((v, j) => ({ v, j })).filter((e) => e.v > 0);
  const x = data.points[i];
  return (
    <div style={{ marginTop: 12 }}>
      <div className="small">
        Point <span className="mono">({fmt(x[0])}, {fmt(x[1])})</span> — reconstructed as{" "}
        <span className="mono">({fmt(recon[0])}, {fmt(recon[1])})</span>
      </div>
      <div className="grid2" style={{ gap: 10, marginTop: 6 }}>
        <div>
          <div className="small muted">Built from (truth)</div>
          {feats.length ? (
            <Bars width={220} labelWidth={40} max={1.2} rowHeight={22}
              items={feats.map((e) => ({ label: `f${e.k + 1}`, value: e.v, color: fcol(e.k) }))} />
          ) : <div className="small">no feature on: just noise near the origin</div>}
        </div>
        <div>
          <div className="small muted">Latents that fire (SAE)</div>
          {lats.length ? (
            <Bars width={220} labelWidth={96} max={Math.max(1.2, ...lats.map((e) => e.v))} rowHeight={22}
              items={lats.map((e) => ({
                label: `L${e.j} ≈ f${match[e.j].f + 1} (${match[e.j].cos.toFixed(2)})`, value: e.v,
                color: match[e.j].cos > 0.9 ? fcol(match[e.j].f) : "var(--ink-3)",
              }))} />
          ) : <div className="small">nothing fires</div>}
        </div>
      </div>
      <div className="small muted">Latent labels show the best-matching true feature and its cosine.</div>
    </div>
  );
}
