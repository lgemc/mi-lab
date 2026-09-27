import { useState } from "react";
import { Bars, Scatter } from "../components/charts";
import { Dag } from "../components/Dag";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Slider, Stat } from "../components/ui";
import { fmt, pct, useApi } from "../api";

type Steer = { name: string; direction: number[]; cos_used: number; cos_spurious: number; effect: number };
type ProbeOut = {
  probe_accuracy: number;
  single_direction_accuracy: { used: number; spurious: number };
  baseline_output: number;
  steering: Steer[];
  points: [number, number][];
  labels: number[];
  probe_2d: [number, number];
};
type Params = { correlation: number; spurious_scale: number; noise: number; strength: number };

const DEFAULTS: Params = { correlation: 1, spurious_scale: 1.5, noise: 0.5, strength: 1 };

const COLOR: Record<string, string> = {
  "probe": "var(--gold)",
  "difference of means": "var(--good)",
  "true (used) direction": "var(--accent)",
  "spurious direction": "var(--warm)",
  "random": "var(--ink-3)",
};

const SHORT: Record<string, string> = {
  "probe": "probe", "difference of means": "diff. of means", "true (used) direction": "used direction u",
  "spurious direction": "spurious direction s", "random": "random",
};

const GRAPH = [
  { id: "c", x: 0, y: 50, label: "c" },
  { id: "u", x: 45, y: 0, label: "h·u" },
  { id: "s", x: 45, y: 100, label: "h·s" },
  { id: "y", x: 100, y: 0, label: "out" },
];

export default function Probing() {
  const [p, setP] = useState<Params>(DEFAULTS);
  const set = (k: keyof Params) => (v: number) => setP((o) => ({ ...o, [k]: v }));
  const q = useApi<ProbeOut>("probe", p);
  const d = q.data;
  const steer = (name: string) => d?.steering.find((s) => s.name === name);
  const probe = steer("probe"), truth = steer("true (used) direction"), dom = steer("difference of means");
  const ext = d ? Math.max(1e-9, ...d.points.flat().map(Math.abs)) : 1;
  const L = ext * 0.85;

  return (
    <div className="prose-wide">
      <h2>A concept, written twice</h2>
      <div className="prose">
        <p>
          A <b>probe</b> is a small classifier, usually linear, trained to read a concept off a model's hidden
          state. If a probe can tell positive reviews from negative ones using layer 12's activations, we say layer 12
          "represents sentiment". It's the most common tool in the box, and this lesson is about what it does{" "}
          <em>not</em> tell you.
        </p>
        <p>
          Here is a toy "model" with a 6-dimensional hidden state <TeX>h</TeX>. A binary concept <TeX>c</TeX> (say,
          sentiment) is written into it along two directions:
        </p>
        <ul>
          <li>
            a <b style={{ color: "var(--accent)" }}>used direction</b> <TeX>u</TeX>, the only thing the rest of the
            model reads: the output is <TeX>{String.raw`\sigma\big(6\,h\cdot u\big)`}</TeX>;
          </li>
          <li>
            a <b style={{ color: "var(--warm)" }}>spurious direction</b> <TeX>s</TeX>, carrying a correlate of{" "}
            <TeX>c</TeX> that nothing downstream reads. Think of exclamation marks: they travel with positive reviews,
            but the verdict doesn't depend on them.
          </li>
        </ul>
        <p>
          How often the correlate agrees with <TeX>c</TeX> is the <b>correlation</b> slider (at 1 it always does, as
          if every positive review had an exclamation mark), and it can be written more loudly than the real thing
          (the <b>loudness</b> slider). The used direction is also noisier.
          In causal terms:
        </p>
      </div>
      <Card>
        <div className="row">
          <div style={{ flex: "0 0 240px" }}>
            <Dag nodes={GRAPH} edges={[["c", "u"], ["c", "s"], ["u", "y"]]} width={240} height={150} radius={20}
              role={{ u: "x", y: "y" }} />
          </div>
          <p className="small grow">
            Both projections are caused by the concept, so both <em>correlate</em> with it, and a probe can decode it
            from either. Only <TeX>h\cdot u</TeX> has an arrow to the output. A probe sees the left half of this
            graph; only an intervention can see the missing arrow from <TeX>h \cdot s</TeX>.
          </p>
        </div>
      </Card>

      <Card title="Train a probe, then push along it">
        <div className="controls">
          <Slider label="correlation of the correlate" value={p.correlation} min={0} max={1} step={0.05} onChange={set("correlation")} format={(v) => v.toFixed(2)} />
          <Slider label="loudness of the spurious direction" value={p.spurious_scale} min={0} max={4} step={0.1} onChange={set("spurious_scale")} format={(v) => v.toFixed(1)} />
          <Slider label="noise" value={p.noise} min={0.05} max={1} step={0.05} onChange={set("noise")} format={(v) => v.toFixed(2)} />
          <Slider label="steering strength" value={p.strength} min={0} max={3} step={0.1} onChange={set("strength")} format={(v) => v.toFixed(1)} />
        </div>
        <button className="btn" onClick={() => setP(DEFAULTS)}>Reset</button>
        <Loading loading={q.loading} error={q.error}>
          {d && (
            <div className="row" style={{ marginTop: 14 }}>
              <div style={{ flex: "1 1 280px", maxWidth: 360 }}>
                <Scatter points={d.points} size={340} radius={2.8}
                  colors={d.labels.map((l) => (l ? "var(--accent)" : "var(--warm)"))}
                  xLabel="h · u  (used)" yLabel="h · s  (spurious)"
                  arrows={[
                    { to: [d.probe_2d[0] * L, d.probe_2d[1] * L], color: "var(--gold)", label: "probe", width: 3 },
                    { to: [L * 0.75, 0], color: "var(--accent)", label: "u", width: 2, dashed: true },
                  ]} />
                <div className="small muted">
                  <span style={{ color: "var(--accent)" }}>● c = 1</span> · <span style={{ color: "var(--warm)" }}>● c = 0</span> ·
                  300 held-out examples projected on the two directions. The gold arrow is the probe's
                  direction; the dashed one is <TeX>u</TeX>, the direction the model reads.
                </div>
              </div>
              <div className="grow">
                <div className="row" style={{ gap: 10 }}>
                  <Stat k="Probe accuracy" v={pct(d.probe_accuracy, 1)} d="full 6-D logistic probe" color="var(--gold)" />
                  <Stat k="Using h·u only" v={pct(d.single_direction_accuracy.used, 1)} d="the direction that matters" />
                  <Stat k="Using h·s only" v={pct(d.single_direction_accuracy.spurious, 1)} d="the direction nobody reads" />
                </div>
                <h3>Steering: push the negative examples along each direction</h3>
                <Bars width={420} labelWidth={140} format={(v) => (v >= 0 ? "+" : "") + v.toFixed(3)}
                  items={d.steering.map((s) => ({
                    label: SHORT[s.name] ?? s.name,
                    value: s.effect, color: COLOR[s.name],
                  }))} />
                <div className="caption">
                  Cosine with <TeX>u</TeX>:{" "}
                  {d.steering.map((s, i) => <span key={s.name}>{i ? " · " : ""}{SHORT[s.name]} {fmt(s.cos_used, 2)}</span>)}.
                  Change in the model's average P(positive) on c = 0 examples after adding{" "}
                  {p.strength.toFixed(1)} × (unit direction) to <TeX>h</TeX>. Baseline P(positive): {fmt(d.baseline_output, 3)}.
                </div>
              </div>
            </div>
          )}
        </Loading>
      </Card>

      {d && probe && truth && (
        <div className="prose">
          <p>
            At these settings the probe is <b>{pct(d.probe_accuracy)}</b> accurate. By the usual reading, it has
            found "the sentiment direction". But steering along it moves the output by {fmt(probe.effect, 2)}, while
            the direction the model actually reads moves it by {fmt(truth.effect, 2)}. The probe's direction is
            partly the spurious one (cosine {fmt(probe.cos_spurious, 2)} with <TeX>s</TeX>), and that part does
            nothing at all.
          </p>
        </div>
      )}
      <div className="prose">
        <p>
          Why does the probe lean the wrong way? A probe's only job is to separate the classes. It prefers whichever
          direction separates them most cleanly, and here the correlate is louder and less noisy than the real
          signal, so on its own it decodes the concept <em>better</em> than the direction the model uses (compare the
          two single-direction accuracies). The probe has no way to know that nothing reads that direction. That
          information isn't in the activations; it is in the <em>weights downstream</em>, which the probe never looks
          at.
        </p>
      </div>
      <Callout kind="try">
        <p>
          Turn loudness up to 3. The probe gets <em>more</em> accurate and its steering effect <em>shrinks</em>.
          Now set correlation to 0: the correlate is noise, the probe has to use <TeX>u</TeX>, and probe and truth
          coincide. High accuracy was never the evidence; the correlation structure of the data decided what the
          probe found.
        </p>
      </Callout>

      <h2>Decodable is not the same as used</h2>
      <div className="prose">
        <p>
          In the language of lesson 1: a probe estimates something like <TeX>P(c \mid h)</TeX>, a "seeing"
          quantity. Whether the model <em>uses</em> a direction is a "doing" question: what happens to the output
          under <TeX>{String.raw`do(h \leftarrow h + \alpha\, d)`}</TeX>? The two come apart whenever the hidden state
          contains information the model doesn't rely on, which, in a large model, is most information.
        </p>
        <p>
          There are two standard causal checks. <b>Steering</b> adds a multiple of the direction, as above, and asks
          whether the behaviour shifts. <b>Ablation</b> removes the direction (projects it out) and asks whether the
          behaviour breaks. Both are interventions on the activation, the same <TeX>do()</TeX> as in the
          previous two lessons.
        </p>
        <p>
          <b>Difference of means</b> is the simplest possible "probe": the mean activation of the positive class minus
          the mean of the negative class, no training. Marks &amp; Tegmark (2023) found that on real models these
          directions were more causally effective than logistic-probe directions, arguably because logistic
          regression hunts for any separating feature, correlates included, while the mean difference is dominated by
          what changes most on average. In this toy it is not a free win.
          At the current settings it steers {dom && probe ? <>{fmt(dom.effect / Math.max(1e-9, probe.effect), 1)}×</> : "about"}{" "}
          as hard as the probe (at the defaults, about twice); but set correlation to 0.9, loudness to 3 and noise to
          0.3 and it falls behind: with the correlate loud and imperfect, the mean difference is dominated by the
          correlate too. Neither is a substitute for checking.
        </p>
      </div>

      <Callout kind="takeaway">
        <p>
          A probe measures whether a concept is <em>decodable</em> from an activation, not whether the model{" "}
          <em>uses</em> it. High accuracy can come from a correlate that nothing downstream reads. To claim use, you
          intervene: steer along the direction or ablate it, and see whether behaviour changes.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          Belinkov (2022, "Probing classifiers: promises, shortcomings, and advances") reviews exactly this gap.
          Hewitt &amp; Liang (2019) introduced control tasks: a probe expressive enough can learn random labels too,
          so accuracy must be compared against what the probe could memorise. Ravfogel et al. (2020, INLP) remove a
          concept by repeatedly projecting out probe directions, and Elazar et al. (2021, "amnesic probing") use that
          removal as the causal test: if the model's behaviour survives the concept being erased, the model wasn't
          using it, however decodable it was. Marks &amp; Tegmark (2023, "The geometry of truth") compared probe and
          difference-of-means directions by intervening with them, and representation-engineering and steering work
          since (Turner et al. 2023; Zou et al. 2023; Arditi et al. 2024) treats "does adding it change behaviour?" as
          the test that counts.
        </p>
      </Callout>
    </div>
  );
}
