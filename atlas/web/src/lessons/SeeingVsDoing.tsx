import { useMemo, useState } from "react";
import { Bars } from "../components/charts";
import { Dag } from "../components/Dag";
import { TeX } from "../components/tex";
import { Callout, Card, Seg, Slider, Stat } from "../components/ui";
import { fmt, pct } from "../api";

/*
 * Everything on this page is arithmetic on seven numbers, so it runs in the browser: the point is
 * that the reader can check every step with a calculator.
 */

type Model = { pz: number; pxz: [number, number]; py: [[number, number], [number, number]] };
// py[x][z]: P(recovery | treatment x, stone size z); x = 1 is treatment A, z = 1 is a large stone.

// Charig et al. (1986), the kidney-stone study behind the textbook example.
const KIDNEY = {
  small: { A: [81, 87], B: [234, 270] },
  large: { A: [192, 263], B: [55, 80] },
};
const kidneyModel = (): Model => {
  const nSmall = KIDNEY.small.A[1] + KIDNEY.small.B[1];
  const nLarge = KIDNEY.large.A[1] + KIDNEY.large.B[1];
  return {
    pz: nLarge / (nSmall + nLarge),
    pxz: [KIDNEY.small.A[1] / nSmall, KIDNEY.large.A[1] / nLarge],
    py: [
      [KIDNEY.small.B[0] / KIDNEY.small.B[1], KIDNEY.large.B[0] / KIDNEY.large.B[1]],
      [KIDNEY.small.A[0] / KIDNEY.small.A[1], KIDNEY.large.A[0] / KIDNEY.large.A[1]],
    ],
  };
};

function analyse(m: Model) {
  const pz = [1 - m.pz, m.pz];
  const px1 = pz[0] * m.pxz[0] + pz[1] * m.pxz[1]; // P(X = A)
  const pxGivenZ = (x: number, z: number) => (x === 1 ? m.pxz[z] : 1 - m.pxz[z]);
  const px = [1 - px1, px1];
  // P(z | x) by Bayes: what seeing x tells you about z
  const pzGivenX = (z: number, x: number) => (pxGivenZ(x, z) * pz[z]) / px[x];
  const seeing = [0, 1].map((x) => [0, 1].reduce((s, z) => s + m.py[x][z] * pzGivenX(z, x), 0));
  const doing = [0, 1].map((x) => [0, 1].reduce((s, z) => s + m.py[x][z] * pz[z], 0));
  return { pz, px, pzGivenX, seeing, doing };
}

/** A small deterministic generator, so the simulated population does not reshuffle on every render. */
function mulberry(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function Population({ m, mode }: { m: Model; mode: "see" | "do" }) {
  const people = useMemo(() => {
    const r = mulberry(42);
    return Array.from({ length: 400 }, () => {
      const z = r() < m.pz ? 1 : 0;
      const x = mode === "see" ? (r() < m.pxz[z] ? 1 : 0) : (r() < 0.5 ? 1 : 0);
      const y = r() < m.py[x][z] ? 1 : 0;
      return { z, x, y };
    });
  }, [m, mode]);
  const rate = (x: number) => {
    const g = people.filter((p) => p.x === x);
    return g.length ? g.filter((p) => p.y).length / g.length : 0;
  };
  const cols = 40, s = 11;
  return (
    <div>
      <svg viewBox={`0 0 ${cols * s} ${(400 / cols) * s}`} width="100%" style={{ maxWidth: 460, display: "block" }} role="img">
        {people.map((p, i) => {
          const cx = (i % cols) * s + s / 2, cy = Math.floor(i / cols) * s + s / 2;
          const fill = p.x === 1 ? "var(--accent)" : "var(--warm)";
          return p.z === 1
            ? <rect key={i} x={cx - 4} y={cy - 4} width={8} height={8} fill={p.y ? fill : "none"} stroke={fill} strokeWidth={1.3} />
            : <circle key={i} cx={cx} cy={cy} r={4} fill={p.y ? fill : "none"} stroke={fill} strokeWidth={1.3} />;
        })}
      </svg>
      <div className="small muted" style={{ marginTop: 6 }}>
        <span style={{ color: "var(--accent)" }}>■ treatment A</span> · <span style={{ color: "var(--warm)" }}>■ treatment B</span> ·
        ● small stone · ■ large stone · filled = recovered, hollow = did not.
      </div>
      <div className="row" style={{ marginTop: 10 }}>
        <Stat k="Recovered with A" v={pct(rate(1))} />
        <Stat k="Recovered with B" v={pct(rate(0))} />
        <Stat k="A − B in this sample" v={fmt(rate(1) - rate(0), 3)} color={rate(1) - rate(0) >= 0 ? "var(--good)" : "var(--warm)"} />
      </div>
    </div>
  );
}

const NODES = [
  { id: "Z", x: 50, y: 0, label: "Z" },
  { id: "X", x: 5, y: 100, label: "X" },
  { id: "Y", x: 95, y: 100, label: "Y" },
];

export default function SeeingVsDoing() {
  const [m, setM] = useState<Model>(kidneyModel);
  const [mode, setMode] = useState<"see" | "do">("see");
  const a = analyse(m);
  const set = (f: (d: Model) => void) => setM((old) => { const d = structuredClone(old); f(d); return d; });
  const diffSee = a.seeing[1] - a.seeing[0];
  const diffDo = a.doing[1] - a.doing[0];

  return (
    <div className="prose-wide">
      <h2>A treatment that is better in every group and worse overall</h2>
      <div className="prose">
        <p>
          In 1986 a hospital compared two treatments for kidney stones. Treatment <b>A</b> was open surgery and
          treatment <b>B</b> was a newer, less invasive procedure. Here are the recovery rates:
        </p>
      </div>
      <Card>
        <div className="scroll-x">
          <table className="tbl" style={{ maxWidth: 560 }}>
            <thead><tr><th></th><th>Treatment A</th><th>Treatment B</th></tr></thead>
            <tbody>
              <tr><td>Small stones</td><td><b>93%</b> <span className="muted">(81/87)</span></td><td>87% <span className="muted">(234/270)</span></td></tr>
              <tr><td>Large stones</td><td><b>73%</b> <span className="muted">(192/263)</span></td><td>69% <span className="muted">(55/80)</span></td></tr>
              <tr><td>Everyone</td><td>78% <span className="muted">(273/350)</span></td><td><b>83%</b> <span className="muted">(289/350)</span></td></tr>
            </tbody>
          </table>
        </div>
        <div className="caption">A wins for small stones and for large stones, and loses overall. That's Simpson's paradox.</div>
      </Card>
      <div className="prose">
        <p>
          Nothing is miscounted. Doctors gave the invasive treatment A mostly to <em>hard</em> cases (large stones),
          and large stones recover less whatever you do. So A's patients were sicker to begin with. Seeing that
          someone got A tells you something about their stone, and that bad news drags A's overall number down.
        </p>
        <p>
          The graph says it in three arrows. Stone size <TeX>Z</TeX> affects which treatment <TeX>X</TeX> a doctor
          picks <em>and</em> whether the patient recovers (<TeX>Y</TeX>). So <TeX>Z</TeX> is a <b>confounder</b>:
        </p>
      </div>

      <Card title="Two questions that look the same">
        <div className="row">
          <div style={{ flex: "0 0 260px" }}>
            <Dag nodes={NODES} edges={[["Z", "X"], ["Z", "Y"], ["X", "Y"]]} width={260} height={170}
              role={{ X: "x", Y: "y" }} cut={mode === "do" ? [["Z", "X"]] : []} />
          </div>
          <div className="grow">
            <Seg value={mode} onChange={setMode} options={[
              { value: "see", label: "Seeing: P(y | x)" },
              { value: "do", label: "Doing: P(y | do(x))" },
            ]} />
            {mode === "see" ? (
              <p className="small">
                <b>Seeing</b> means filtering the records you already have down to the people who got A. Those people
                were chosen <em>because</em> of their stones, so the arrow <TeX>Z \to X</TeX> is still in force.
              </p>
            ) : (
              <p className="small">
                <b>Doing</b> means imagining that <em>we</em> assign the treatment, by coin flip or by decree,
                whatever the stone. That cuts the arrow into <TeX>X</TeX>. This is Pearl's{" "}
                <b>graph surgery</b>, and <TeX>do(x)</TeX> is the name for it.
              </p>
            )}
          </div>
        </div>
      </Card>

      <h2>The calculation, by hand</h2>
      <div className="prose">
        <p>
          Both quantities average the same per-group recovery rates <TeX>P(y \mid x, z)</TeX>. They differ only in how
          the groups are <em>weighted</em>:
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`\underbrace{P(y \mid x)=\sum_z P(y\mid x,z)\,P(z \mid x)}_{\text{seeing: weights from people who got }x}
\qquad
\underbrace{P(y \mid do(x))=\sum_z P(y\mid x,z)\,P(z)}_{\text{doing: weights from everyone}}`}</TeX>
        <p className="small">
          The right-hand formula is the <b>back-door adjustment</b>. It's valid because <TeX>Z</TeX> blocks the only
          back-door path <TeX>X \leftarrow Z \to Y</TeX> (the next lesson makes "blocks" precise). With the numbers
          from the sliders below:
        </p>
        <div>
          {[1, 0].map((x) => (
            <div key={x}>
              <b>{x === 1 ? "Treatment A" : "Treatment B"}</b>
              <TeX block>{String.raw`\begin{aligned}
P(y\mid ${x ? "A" : "B"}) &= ${fmt(m.py[x][0], 3)}\cdot ${fmt(a.pzGivenX(0, x), 3)} + ${fmt(m.py[x][1], 3)}\cdot ${fmt(a.pzGivenX(1, x), 3)} = \mathbf{${fmt(a.seeing[x], 3)}}\\
P(y\mid do(${x ? "A" : "B"})) &= ${fmt(m.py[x][0], 3)}\cdot ${fmt(a.pz[0], 3)} + ${fmt(m.py[x][1], 3)}\cdot ${fmt(a.pz[1], 3)} = \mathbf{${fmt(a.doing[x], 3)}}
\end{aligned}`}</TeX>
            </div>
          ))}
        </div>
        <p className="small muted">
          Each term is (recovery rate in the group) × (weight of the group); the first term is small stones, the
          second large. <TeX>{String.raw`P(\text{large}\mid A) = ${fmt(a.pzGivenX(1, 1), 2)}`}</TeX> while{" "}
          <TeX>{String.raw`P(\text{large}) = ${fmt(a.pz[1], 2)}`}</TeX>. Seeing A tells you the stone is probably
          large.
        </p>
        <div className="row">
          <Stat k="Seeing: A − B" v={fmt(diffSee, 3)} color={diffSee >= 0 ? "var(--good)" : "var(--warm)"} d="what the records say" />
          <Stat k="Doing: A − B" v={fmt(diffDo, 3)} color={diffDo >= 0 ? "var(--good)" : "var(--warm)"} d="what assigning A would do" />
        </div>
      </Card>

      <Card title="Change the world" caption="Every number above updates. Try making treatment assignment ignore the stone: set both 'P(A | …)' sliders equal.">
        <div className="controls">
          <Slider label="P(large stone)" value={m.pz} min={0.05} max={0.95} onChange={(v) => set((d) => { d.pz = v; })} format={(v) => v.toFixed(2)} />
          <Slider label="P(A | small)" value={m.pxz[0]} min={0.01} max={0.99} onChange={(v) => set((d) => { d.pxz[0] = v; })} format={(v) => v.toFixed(2)} />
          <Slider label="P(A | large)" value={m.pxz[1]} min={0.01} max={0.99} onChange={(v) => set((d) => { d.pxz[1] = v; })} format={(v) => v.toFixed(2)} />
        </div>
        <div className="controls">
          {([[1, 0, "recover | A, small"], [1, 1, "recover | A, large"], [0, 0, "recover | B, small"], [0, 1, "recover | B, large"]] as const).map(([x, z, label]) => (
            <Slider key={label} label={`P(${label})`} value={m.py[x][z]} min={0.01} max={0.99}
              onChange={(v) => set((d) => { d.py[x][z] = v; })} format={(v) => v.toFixed(2)} />
          ))}
        </div>
        <button className="btn" onClick={() => setM(kidneyModel())}>Reset to the 1986 data</button>
        <div style={{ marginTop: 14 }}>
          <Bars width={460} labelWidth={170} format={(v) => v.toFixed(3)} max={1}
            items={[
              { label: "P(recover | A)", value: a.seeing[1], color: "var(--accent)" },
              { label: "P(recover | B)", value: a.seeing[0], color: "var(--warm)" },
              { label: "P(recover | do(A))", value: a.doing[1], color: "var(--accent)" },
              { label: "P(recover | do(B))", value: a.doing[0], color: "var(--warm)" },
            ]} />
        </div>
      </Card>

      <h2>Run the experiment instead</h2>
      <div className="prose">
        <p>
          Here are 400 simulated patients drawn from the model. In <b>seeing</b> mode the doctor picks the treatment
          based on the stone, as in the hospital records. In <b>doing</b> mode a coin picks it, which is a randomized
          trial. The trial's A − B lands close to the do-quantity above, up to sampling noise from only 400 people.
          The records' A − B lands close to the seeing quantity.
        </p>
      </div>
      <Card>
        <Seg value={mode} onChange={setMode} options={[
          { value: "see", label: "Doctor chooses (observational)" },
          { value: "do", label: "Coin chooses (randomized)" },
        ]} />
        <div style={{ marginTop: 12 }}><Population m={m} mode={mode} /></div>
      </Card>

      <Callout kind="takeaway">
        <p>
          <TeX>P(y\mid x)</TeX> and <TeX>P(y \mid do(x))</TeX> are different quantities that can point in opposite
          directions. The graph tells you which variables to adjust for so that data you merely <em>observed</em>{" "}
          answers the question "what if we <em>acted</em>?"
        </p>
      </Callout>
      <Callout kind="real" label="Why this is the first lesson of an interpretability course">
        <p>
          Every causal method in mechanistic interpretability, including activation patching, ablation, steering and
          causal mediation, computes a <TeX>do()</TeX> quantity: <em>set this activation to that value and see what
          the output does</em>. Correlational methods like probes and attention weights compute a "seeing" quantity.
          When the two disagree, the disagreement is a finding, and part two of this atlas is built around exactly
          that.
        </p>
      </Callout>
    </div>
  );
}
