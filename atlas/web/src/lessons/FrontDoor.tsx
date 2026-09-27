import { useState } from "react";
import { Bars } from "../components/charts";
import { Dag, DagEdge, DagNode } from "../components/Dag";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Seg, Slider, Stat } from "../components/ui";
import { fmt, useApi } from "../api";

/*
 * Numbers come from POST /api/causal/frontdoor (the exact joint of a four-variable model); each
 * step of the derivation is checked by POST /api/causal/rule, which returns the mutilated graph it
 * tested d-separation in and any path it found open.
 */

type FD = {
  p_x: number[]; p_m_given_x: number[][]; p_y_given_xm: number[][]; inner_sum: number[];
  naive: number[]; truth: number[]; frontdoor: number[];
  effects: { naive: number; truth: number; frontdoor: number };
};
type Triple = { node: string; kind: string; blocked: boolean; reason: string };
type RuleR = { holds: boolean; graph: string; edges: string[][]; open_paths: { nodes: string[]; triples: Triple[] }[] };
type Analysis = { backdoor_sets: string[][]; frontdoor_sets: string[][] };

const NODES: DagNode[] = [
  { id: "U", x: 50, y: 0, latent: true },
  { id: "X", x: 0, y: 100 },
  { id: "M", x: 50, y: 100 },
  { id: "Y", x: 100, y: 100 },
];
const EDGES: DagEdge[] = [["U", "X"], ["U", "Y"], ["X", "M"], ["M", "Y"]];
const GRAPH = { nodes: NODES.map((n) => ({ id: n.id, latent: !!n.latent })), edges: EDGES };

type Params = { pu: number; px_u: [number, number]; pm_x: [number, number]; py_mu: [[number, number], [number, number]] };
const WORLDS: Record<string, { label: string; p: Params; note: string }> = {
  harmful: {
    label: "Tar causes cancer",
    p: { pu: 0.5, px_u: [0.2, 0.8], pm_x: [0.05, 0.95], py_mu: [[0.1, 0.3], [0.2, 0.6]] },
    note: "Tar raises cancer risk, and the gene raises both smoking and cancer. The naive comparison doubles the real effect.",
  },
  fisher: {
    label: "Fisher was right",
    p: { pu: 0.5, px_u: [0.15, 0.85], pm_x: [0.05, 0.95], py_mu: [[0.1, 0.5], [0.1, 0.5]] },
    note: "Tar does nothing; the gene does everything. Smokers still get more cancer, and the front door correctly says the effect is zero.",
  },
  protective: {
    label: "Tar protects",
    p: { pu: 0.5, px_u: [0.2, 0.8], pm_x: [0.05, 0.95], py_mu: [[0.3, 0.7], [0.15, 0.55]] },
    note: "A deliberately strange world, in the spirit of Pearl's own example: the data show smokers get more cancer, yet smoking lowers it.",
  },
};

/* --------------------------------------------------------------------------- one checked step */

function RuleCheck({ rule, y, x = [], z, w = [], cond }: {
  rule: 1 | 2 | 3; y: string[]; x?: string[]; z: string[]; w?: string[]; cond: string;
}) {
  const { data, error, loading } = useApi<RuleR>("causal/rule", { ...GRAPH, rule, y, x, z, w }, 0);
  const kept = new Set((data?.edges ?? []).map((e) => e.join(">")));
  const cut = data ? EDGES.filter((e) => !kept.has(e.join(">"))) : [];
  const open = data?.open_paths[0];
  const lit: DagEdge[] = open ? open.nodes.slice(1).map((n, i) => [open.nodes[i], n]) : [];
  const role: Record<string, "x" | "y" | "m"> = {};
  z.forEach((n) => { role[n] = "m"; });
  y.forEach((n) => { role[n] = "y"; });
  return (
    <Loading loading={loading} error={error}>
      <div className="row" style={{ alignItems: "center", gap: 12 }}>
        <div style={{ flex: "0 1 230px" }}>
          <Dag nodes={NODES} edges={EDGES} cut={cut} lit={lit} litColor="var(--warm)" given={[...x, ...w]}
            role={role} width={230} height={130} radius={17} />
        </div>
        <div className="grow small">
          <div>Rule {rule} needs <TeX>{cond}</TeX></div>
          {data && (
            <>
              <div className="muted">
                checked in the graph with {cut.length ? cut.map((e) => e.join("→")).join(", ") : "nothing"} cut (dashed ✂)
                {rule === 3 && w.length === 0 ? "; with W empty, Z(W) is all of Z" : ""}
              </div>
              <div style={{ marginTop: 4, fontWeight: 700, color: data.holds ? "var(--good)" : "var(--warm)" }}>
                {data.holds ? "✓ holds: every path is blocked" : "✗ fails: an open path remains"}
              </div>
              {open && (
                <div style={{ color: "var(--warm)" }}>
                  Open path <span className="mono">{open.nodes.join(" – ")}</span>:{" "}
                  {open.triples.map((t) => t.reason).join("; ")}.
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </Loading>
  );
}

function Step({ n, children, check }: { n: number | string; children: React.ReactNode; check?: React.ReactNode }) {
  return (
    <div style={{ borderTop: "1px solid var(--line)", padding: "10px 0" }}>
      <div className="row" style={{ gap: 10, alignItems: "baseline" }}>
        <span className="mono muted" style={{ minWidth: 22 }}>{n}</span>
        <div className="grow" style={{ minWidth: 0 }}>{children}</div>
      </div>
      {check && <div style={{ marginLeft: 32, marginTop: 6 }}>{check}</div>}
    </div>
  );
}

function Derivation() {
  const [dropX, setDropX] = useState(false);
  return (
    <Card title="Deriving the front-door formula, one checked step at a time">
      <p className="small muted" style={{ marginTop: 0 }}>
        Each rule application below is sent to the server, which cuts the graph the rule asks for and runs
        d-separation on it. In the small graphs, a <span style={{ color: "var(--gold)" }}>gold ring</span> marks the
        variable being swapped (<TeX>Z</TeX> in the rule), an <span style={{ color: "var(--warm)" }}>orange ring</span>{" "}
        the outcome, and filled nodes are conditioned on.
      </p>
      <Step n={1}>
        Sum over the mediator, inside the world where we set <TeX>X</TeX>. This is just the law of total probability:
        <TeX block>{String.raw`P(y\mid do(x)) = \sum_m P(y\mid do(x), m)\,P(m\mid do(x))`}</TeX>
      </Step>
      <Step n={2} check={<RuleCheck rule={2} y={["M"]} z={["X"]} cond={String.raw`(M \perp\!\!\!\perp X)_{G_{\underline{X}}}`} />}>
        <b>Setting X is the same as seeing X, as far as M is concerned.</b> The only back-door path from X to M is{" "}
        <TeX>X \leftarrow U \to Y \leftarrow M</TeX>, and Y is a collider on it.
        <TeX block>{String.raw`P(m\mid do(x)) = P(m\mid x)`}</TeX>
      </Step>
      <Step n={3} check={<RuleCheck rule={2} y={["Y"]} x={["X"]} z={["M"]} cond={String.raw`(Y \perp\!\!\!\perp M \mid X)_{G_{\overline{X}\,\underline{M}}}`} />}>
        <b>Swap the observation of M for an action on M</b> (rule 2 read right to left). With X already set, nothing
        confounds M and Y.
        <TeX block>{String.raw`P(y\mid do(x), m) = P(y\mid do(x), do(m))`}</TeX>
      </Step>
      <Step n={4} check={<RuleCheck rule={3} y={["Y"]} x={["M"]} z={["X"]} cond={String.raw`(Y \perp\!\!\!\perp X \mid M)_{G_{\overline{M}\,\overline{X}}}`} />}>
        <b>Once M is set, setting X does nothing to Y.</b> X only reaches Y through M, and we have fixed M.
        <TeX block>{String.raw`P(y\mid do(x), do(m)) = P(y\mid do(m))`}</TeX>
      </Step>
      <Step n={5}>
        Now we need <TeX>P(y\mid do(m))</TeX>. M's back door runs through X, and X <em>is</em> observed, so sum over it:
        <TeX block>{String.raw`P(y\mid do(m)) = \sum_{x'} P(y\mid do(m), x')\,P(x'\mid do(m))`}</TeX>
      </Step>
      <Step n={6} check={
        <>
          <Seg value={dropX ? "no" : "yes"} onChange={(v) => setDropX(v === "no")} options={[
            { value: "yes", label: "condition on X (w = X)" },
            { value: "no", label: "forget X (w = ∅)" },
          ]} />
          <div style={{ marginTop: 6 }}>
            {dropX
              ? <RuleCheck key="nox" rule={2} y={["Y"]} z={["M"]} cond={String.raw`(Y \perp\!\!\!\perp M)_{G_{\underline{M}}}`} />
              : <RuleCheck key="x" rule={2} y={["Y"]} z={["M"]} w={["X"]} cond={String.raw`(Y \perp\!\!\!\perp M \mid X)_{G_{\underline{M}}}`} />}
          </div>
        </>
      }>
        <b>With X held fixed, seeing M is as good as setting it.</b> Conditioning on X blocks the back door{" "}
        <TeX>M \leftarrow X \leftarrow U \to Y</TeX>. Toggle it off to watch the rule fail.
        <TeX block>{String.raw`P(y\mid do(m), x') = P(y\mid m, x')`}</TeX>
      </Step>
      <Step n={7} check={<RuleCheck rule={3} y={["X"]} z={["M"]} cond={String.raw`(X \perp\!\!\!\perp M)_{G_{\overline{M}}}`} />}>
        <b>Setting M does not change X.</b> M comes after X, so an action on it cannot reach back.
        <TeX block>{String.raw`P(x'\mid do(m)) = P(x')`}</TeX>
      </Step>
      <Step n={8}>
        <b>Put it together.</b> Substitute 2, 3–4 and 6–7 into 1:
        <TeX block>{String.raw`P(y\mid do(x)) = \sum_m P(m\mid x)\sum_{x'} P(y\mid m, x')\,P(x')`}</TeX>
        Every term on the right is a plain conditional probability of observed variables. U never appears.
      </Step>
    </Card>
  );
}

/* --------------------------------------------------------------------------- lesson */

export default function FrontDoor() {
  const [world, setWorld] = useState("harmful");
  const [p, setP] = useState<Params>(WORLDS.harmful.p);
  const set = (f: (d: Params) => void) => setP((old) => { const d = structuredClone(old); f(d); return d; });
  const pickWorld = (w: string) => { setWorld(w); setP(structuredClone(WORLDS[w].p)); };
  const { data, error, loading } = useApi<FD>("causal/frontdoor", p);
  const an = useApi<Analysis>("causal/analyze", { ...GRAPH, x: "X", y: "Y", given: [] }, 0);
  const f2 = (v: number) => fmt(Math.abs(v) < 5e-7 ? 0 : v, 3);

  return (
    <div className="prose-wide">
      <h2>When you can't measure the confounder</h2>
      <div className="prose">
        <p>
          In the 1950s the evidence that smoking causes lung cancer was mostly observational: smokers got more cancer.
          R. A. Fisher, one of the founders of statistics, objected. Perhaps a gene makes people both crave cigarettes
          and develop cancer. If so, the correlation would appear even if cigarettes were harmless.
        </p>
        <p>
          In graph form the gene is a hidden common cause <TeX>U</TeX> (dashed, because nobody measured it). The
          back-door path <TeX>X \leftarrow U \to Y</TeX> can only be blocked by conditioning on <TeX>U</TeX>, and we
          can't. So the back-door adjustment from the last two lessons is not available.
        </p>
      </div>
      <Card>
        <div className="row">
          <div style={{ flex: "0 1 300px" }}>
            <Dag nodes={NODES.map((n) => (n.id === "U" ? n : { ...n, y: 78 }))} edges={EDGES} width={300} height={180} role={{ X: "x", Y: "y", M: "m" }}
              nodeValue={{ X: "smoking", M: "tar", Y: "cancer" }} />
            <div className="small muted" style={{ textAlign: "center" }}>U = genotype (hidden)</div>
          </div>
          <div className="grow">
            <Loading loading={an.loading} error={an.error}>
              {an.data && (
                <div className="row" style={{ gap: 10 }}>
                  <Stat k="Back-door sets" v={an.data.backdoor_sets.length ? an.data.backdoor_sets.map((s) => `{${s.join(",")}}`).join(" ") : "none"}
                    color="var(--warm)" d="U can't be used: it's unobserved" />
                  <Stat k="Front-door sets" v={an.data.frontdoor_sets.map((s) => `{${s.join(",")}}`).join(" ") || "none"}
                    color="var(--good)" d="found by the server" />
                </div>
              )}
            </Loading>
            <p className="small">
              Pearl's suggestion (1995) was to use a <b>mediator</b>. Suppose smoking harms you only by depositing tar
              in your lungs (<TeX>M</TeX>), and the gene affects tar only through smoking. Then two pieces of the
              effect can each be estimated on their own:
            </p>
            <ul className="small">
              <li><b>X → M.</b> There's no open back door from smoking to tar, because the only route passes <em>into</em> Y and Y is a collider on it.</li>
              <li><b>M → Y.</b> Tar's back door <TeX>M \leftarrow X \leftarrow U \to Y</TeX> is blocked by conditioning on X, which we observe.</li>
            </ul>
            <p className="small">Chain the two pieces and you have the effect of X on Y, without ever touching U.</p>
          </div>
        </div>
      </Card>

      <h2>The front-door formula</h2>
      <div className="prose">
        <p>Written out, "how much tar smoking produces" times "how much cancer tar produces" becomes:</p>
      </div>
      <Card>
        <TeX block>{String.raw`P(y\mid do(x)) \;=\; \sum_m \underbrace{P(m\mid x)}_{\text{effect of }X\text{ on }M}\;\underbrace{\sum_{x'} P(y\mid m,x')\,P(x')}_{\text{effect of }M\text{ on }Y\text{, back-door adjusted for }X}`}</TeX>
        <p className="small">
          The inner sum is the back-door formula for <TeX>{String.raw`P(y\mid do(m))`}</TeX>, adjusting for X. The
          outer sum weights it by how likely each tar level is when you smoke. It uses only X, M and Y.
        </p>
      </Card>

      <Card title="A world you control" caption="The server builds the exact joint distribution of U, X, M, Y, hides U, and computes all three answers.">
        <Seg value={world} onChange={pickWorld} options={Object.entries(WORLDS).map(([k, w]) => ({ value: k, label: w.label }))} />
        <p className="small muted">{WORLDS[world].note}</p>
        <div className="controls">
          <Slider label="P(gene)" value={p.pu} min={0.05} max={0.95} onChange={(v) => set((d) => { d.pu = v; })} format={(v) => v.toFixed(2)} />
          <Slider label="P(smoke | no gene)" value={p.px_u[0]} min={0.02} max={0.98} onChange={(v) => set((d) => { d.px_u[0] = v; })} format={(v) => v.toFixed(2)} />
          <Slider label="P(smoke | gene)" value={p.px_u[1]} min={0.02} max={0.98} onChange={(v) => set((d) => { d.px_u[1] = v; })} format={(v) => v.toFixed(2)} />
          <Slider label="P(tar | non-smoker)" value={p.pm_x[0]} min={0.02} max={0.98} onChange={(v) => set((d) => { d.pm_x[0] = v; })} format={(v) => v.toFixed(2)} />
          <Slider label="P(tar | smoker)" value={p.pm_x[1]} min={0.02} max={0.98} onChange={(v) => set((d) => { d.pm_x[1] = v; })} format={(v) => v.toFixed(2)} />
        </div>
        <div className="controls">
          {([[0, 0, "no tar, no gene"], [0, 1, "no tar, gene"], [1, 0, "tar, no gene"], [1, 1, "tar, gene"]] as const).map(([m, u, label]) => (
            <Slider key={label} label={`P(cancer | ${label})`} value={p.py_mu[m][u]} min={0.02} max={0.98}
              onChange={(v) => set((d) => { d.py_mu[m][u] = v; })} format={(v) => v.toFixed(2)} />
          ))}
        </div>
        <Loading loading={loading} error={error}>
          {data && (
            <>
              <div className="row">
                <Stat k="Naive: P(c|smoke) − P(c|no)" v={f2(data.effects.naive)} color="var(--warm)" d="what the records say" />
                <Stat k="Truth: do(smoke) − do(no)" v={f2(data.effects.truth)} d="needs U; no one has this" />
                <Stat k="Front-door estimate" v={f2(data.effects.frontdoor)} color="var(--good)" d="uses only X, M, Y" />
              </div>
              <div style={{ marginTop: 14 }}>
                <Bars width={480} labelWidth={200} max={1} format={f2} items={[
                  { label: "P(cancer | smoker)", value: data.naive[1], color: "var(--warm)" },
                  { label: "P(cancer | non-smoker)", value: data.naive[0], color: "var(--warm)" },
                  { label: "P(cancer | do(smoke)), truth", value: data.truth[1], color: "var(--ink-3)" },
                  { label: "P(cancer | do(no)), truth", value: data.truth[0], color: "var(--ink-3)" },
                  { label: "front door, do(smoke)", value: data.frontdoor[1], color: "var(--good)" },
                  { label: "front door, do(no)", value: data.frontdoor[0], color: "var(--good)" },
                ]} />
              </div>
            </>
          )}
        </Loading>
      </Card>

      <Callout kind="try">
        <p>
          Drag any slider. The grey bars (truth) and green bars (front door) always agree to every decimal, while the
          orange bars (naive) drift with the strength of the gene. Set <b>P(smoke | gene)</b> equal to{" "}
          <b>P(smoke | no gene)</b> and the gene stops confounding, so all three agree.
        </p>
      </Callout>

      <h2>The calculation, by hand</h2>
      {data && (
        <Card>
          <div className="grid2">
            <div className="scroll-x">
              <table className="tbl">
                <thead><tr><th>observed table</th><th>x = 0</th><th>x = 1</th></tr></thead>
                <tbody>
                  <tr><td><TeX>P(x)</TeX></td><td>{f2(data.p_x[0])}</td><td>{f2(data.p_x[1])}</td></tr>
                  <tr><td><TeX>{String.raw`P(m{=}1\mid x)`}</TeX></td><td>{f2(data.p_m_given_x[0][1])}</td><td>{f2(data.p_m_given_x[1][1])}</td></tr>
                  <tr><td><TeX>{String.raw`P(y{=}1\mid x, m{=}0)`}</TeX></td><td>{f2(data.p_y_given_xm[0][0])}</td><td>{f2(data.p_y_given_xm[1][0])}</td></tr>
                  <tr><td><TeX>{String.raw`P(y{=}1\mid x, m{=}1)`}</TeX></td><td>{f2(data.p_y_given_xm[0][1])}</td><td>{f2(data.p_y_given_xm[1][1])}</td></tr>
                </tbody>
              </table>
            </div>
            <p className="small" style={{ margin: 0 }}>
              These four rows are everything an observer with records of smoking, tar and cancer can compute. The gene
              is nowhere in them.
            </p>
          </div>
          <p className="small"><b>Inner sum</b>: the effect of setting tar, adjusted for smoking.</p>
          <TeX block>{String.raw`\begin{aligned}
\textstyle\sum_{x'} P(y\mid m{=}0,x')P(x') &= ${f2(data.p_y_given_xm[0][0])}\cdot${f2(data.p_x[0])} + ${f2(data.p_y_given_xm[1][0])}\cdot${f2(data.p_x[1])} = \mathbf{${f2(data.inner_sum[0])}}\\
\textstyle\sum_{x'} P(y\mid m{=}1,x')P(x') &= ${f2(data.p_y_given_xm[0][1])}\cdot${f2(data.p_x[0])} + ${f2(data.p_y_given_xm[1][1])}\cdot${f2(data.p_x[1])} = \mathbf{${f2(data.inner_sum[1])}}
\end{aligned}`}</TeX>
          <p className="small"><b>Outer sum</b>: weight by how much tar each choice produces.</p>
          <TeX block>{String.raw`\begin{aligned}
P(y\mid do(x{=}1)) &= ${f2(data.p_m_given_x[1][0])}\cdot${f2(data.inner_sum[0])} + ${f2(data.p_m_given_x[1][1])}\cdot${f2(data.inner_sum[1])} = \mathbf{${f2(data.frontdoor[1])}}\\
P(y\mid do(x{=}0)) &= ${f2(data.p_m_given_x[0][0])}\cdot${f2(data.inner_sum[0])} + ${f2(data.p_m_given_x[0][1])}\cdot${f2(data.inner_sum[1])} = \mathbf{${f2(data.frontdoor[0])}}
\end{aligned}`}</TeX>
          <p className="small muted">
            Compare with the truth, computed with U: {f2(data.truth[1])} and {f2(data.truth[0])}. And with the naive
            conditionals: {f2(data.naive[1])} and {f2(data.naive[0])}.
          </p>
        </Card>
      )}

      <h2>Why it works: the three rules of do-calculus</h2>
      <div className="prose">
        <p>
          The front-door formula looks like a trick. It isn't: it follows from three rules that let you rewrite
          expressions containing <TeX>do()</TeX>, each licensed by a d-separation check in a modified graph (Pearl
          1995; Pearl 2009, ch. 3.4). Notation: <TeX>{String.raw`G_{\overline{X}}`}</TeX> is the graph with arrows{" "}
          <em>into</em> X deleted (what <TeX>do(x)</TeX> does), and <TeX>{String.raw`G_{\underline{Z}}`}</TeX> is the
          graph with arrows <em>out of</em> Z deleted (leaving only Z's back doors).
        </p>
      </div>
      <Card>
        <p className="small" style={{ marginTop: 0 }}><b>Rule 1, add or remove an observation.</b> If Z tells you nothing about Y once X is set and W is known:</p>
        <TeX block>{String.raw`P(y\mid do(x), z, w) = P(y\mid do(x), w) \quad\text{if}\quad (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}}}`}</TeX>
        <p className="small"><b>Rule 2, swap an action for an observation.</b> If Z has no open back door to Y, seeing Z and setting Z are the same:</p>
        <TeX block>{String.raw`P(y\mid do(x), do(z), w) = P(y\mid do(x), z, w) \quad\text{if}\quad (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}\,\underline{Z}}}`}</TeX>
        <p className="small"><b>Rule 3, add or remove an action.</b> If setting Z can't reach Y:</p>
        <TeX block>{String.raw`P(y\mid do(x), do(z), w) = P(y\mid do(x), w) \quad\text{if}\quad (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}\,\overline{Z(W)}}}`}</TeX>
        <p className="small muted">
          <TeX>Z(W)</TeX> is the part of Z that is not an ancestor of any W in <TeX>{String.raw`G_{\overline{X}}`}</TeX>.
          When W is empty it is all of Z. The rules are complete: any effect that can be identified from a graph can be
          identified by some sequence of them (Huang &amp; Valtorta 2006; Shpitser &amp; Pearl 2006).
        </p>
      </Card>

      <Derivation />

      <Callout kind="warn">
        <p>
          The formula is only as good as the graph. It needs <em>all</em> of smoking's effect to flow through tar, and
          the gene to have no direct line to tar. Real lungs don't promise either. The toy shows that the effect{" "}
          <em>can</em> be identified without seeing the confounder, not that tar data settled the smoking debate.
        </p>
      </Callout>

      <Callout kind="takeaway">
        <p>
          An unobserved confounder does not automatically make an effect unknowable. If a mediator carries the whole
          effect and is itself unconfounded, the front-door formula recovers <TeX>P(y\mid do(x))</TeX> from
          observational data. Do-calculus turns the question "is this identifiable?" into a series of d-separation
          checks that a program can run.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          Mechanistic interpretability usually gets to intervene directly, so it rarely needs identification formulas.
          The front-door idea still appears as <b>mediation</b>: splitting a component's total effect on the output into
          the part that flows through a chosen intermediate (an attention head, an SAE feature) and the rest. Path
          patching and causal mediation analysis in language models (Vig et al. 2020; Wang et al. 2023) are
          interventional versions of the same chain rule: the effect of X on M, then the effect of M on Y with X held
          fixed. See Pearl (2009, ch. 3) and Pearl, Glymour &amp; Jewell (2016, ch. 3.4).
        </p>
      </Callout>
    </div>
  );
}
