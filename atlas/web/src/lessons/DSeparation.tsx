import { useMemo, useState } from "react";
import { Scatter } from "../components/charts";
import { Dag, DagEdge, DagNode } from "../components/Dag";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Select, Slider, Stat } from "../components/ui";
import { fmt, useApi } from "../api";

/*
 * The verdicts on this page come from the server (POST /api/causal/analyze), which enumerates every
 * path and reports the triple that decided it. The page only draws graphs and lays out the answer.
 * The three one-triple demos at the top are simple enough to decide in the browser.
 */

type Triple = { node: string; kind: "chain" | "fork" | "collider"; blocked: boolean; reason: string };
type PathR = {
  nodes: string[]; arrows: ("->" | "<-")[]; open: boolean; backdoor: boolean; directed: boolean; triples: Triple[];
};
type Analysis = {
  paths: PathR[]; d_separated: boolean; backdoor_sets: string[][]; frontdoor_sets: string[][];
  given_is_backdoor: boolean; descendants_of_x: string[];
};

type Preset = {
  name: string; nodes: DagNode[]; edges: DagEdge[]; x: string; y: string; height: number;
  legend?: Record<string, string>; blurb: string;
};

const PRESETS: Record<string, Preset> = {
  chain: {
    name: "Chain", height: 120, x: "X", y: "Y",
    nodes: [{ id: "X", x: 0, y: 50 }, { id: "M", x: 50, y: 50 }, { id: "Y", x: 100, y: 50 }],
    edges: [["X", "M"], ["M", "Y"]],
    legend: { X: "fire", M: "smoke", Y: "alarm" },
    blurb: "Fire causes smoke, and smoke sets off the alarm. Condition on M to block the only path.",
  },
  fork: {
    name: "Fork", height: 170, x: "X", y: "Y",
    nodes: [{ id: "Z", x: 50, y: 0 }, { id: "X", x: 5, y: 100 }, { id: "Y", x: 95, y: 100 }],
    edges: [["Z", "X"], ["Z", "Y"]],
    legend: { Z: "hot weather", X: "ice-cream sales", Y: "drownings" },
    blurb: "Hot days sell ice cream and send people swimming. Sales and drownings move together, and neither causes the other.",
  },
  collider: {
    name: "Collider", height: 170, x: "T", y: "L",
    nodes: [{ id: "T", x: 5, y: 0 }, { id: "L", x: 95, y: 0 }, { id: "S", x: 50, y: 100 }],
    edges: [["T", "S"], ["L", "S"]],
    legend: { T: "talent", L: "looks", S: "film star" },
    blurb: "Talent and looks are independent in the population. Either can make you a star. Condition on S and watch the path open.",
  },
  colliderDesc: {
    name: "Collider + descendant", height: 230, x: "T", y: "L",
    nodes: [{ id: "T", x: 5, y: 0 }, { id: "L", x: 95, y: 0 }, { id: "S", x: 50, y: 55 }, { id: "C", x: 50, y: 100 }],
    edges: [["T", "S"], ["L", "S"], ["S", "C"]],
    legend: { T: "talent", L: "looks", S: "film star", C: "on a magazine cover" },
    blurb: "Only stars make magazine covers. Conditioning on C is conditioning on a noisy copy of S, so it opens the collider too.",
  },
  mbias: {
    name: "M-bias", height: 220, x: "X", y: "Y",
    nodes: [{ id: "A", x: 5, y: 0 }, { id: "B", x: 95, y: 0 }, { id: "M", x: 50, y: 45 }, { id: "X", x: 5, y: 100 }, { id: "Y", x: 95, y: 100 }],
    edges: [["A", "X"], ["A", "M"], ["B", "M"], ["B", "Y"], ["X", "Y"]],
    blurb: "M happens before X and is correlated with both X and Y, so it looks like a confounder. It is a collider. Adjusting for it creates bias instead of removing it.",
  },
  confMed: {
    name: "Confounder + mediator", height: 180, x: "X", y: "Y",
    nodes: [{ id: "Z", x: 50, y: 0 }, { id: "X", x: 0, y: 100 }, { id: "M", x: 50, y: 100 }, { id: "Y", x: 100, y: 100 }],
    edges: [["Z", "X"], ["Z", "Y"], ["X", "M"], ["M", "Y"]],
    blurb: "Z confounds, M mediates. Conditioning on Z closes the back door. Conditioning on M closes the front door, which is the effect you wanted to measure.",
  },
  pearl: {
    name: "Pearl's six-node graph", height: 260, x: "X", y: "Y",
    nodes: [{ id: "Z1", x: 5, y: 0 }, { id: "Z2", x: 95, y: 0 }, { id: "Z3", x: 50, y: 42 },
      { id: "X", x: 5, y: 100 }, { id: "W", x: 50, y: 100 }, { id: "Y", x: 95, y: 100 }],
    edges: [["Z1", "Z3"], ["Z2", "Z3"], ["Z1", "X"], ["Z3", "X"], ["Z3", "Y"], ["Z2", "Y"], ["X", "W"], ["W", "Y"]],
    blurb: "The textbook back-door example (Pearl, Glymour & Jewell 2016, ch. 3). Z3 is a confounder and also a collider. Conditioning on it alone is not enough.",
  },
  sprinkler: {
    name: "Sprinkler", height: 300, x: "Sp", y: "R",
    nodes: [{ id: "Se", x: 50, y: 0 }, { id: "Sp", x: 5, y: 45 }, { id: "R", x: 95, y: 45 }, { id: "W", x: 50, y: 72 }, { id: "Sl", x: 50, y: 100 }],
    edges: [["Se", "Sp"], ["Se", "R"], ["Sp", "W"], ["R", "W"], ["W", "Sl"]],
    legend: { Se: "season", Sp: "sprinkler on", R: "rain", W: "grass wet", Sl: "path slippery" },
    blurb: "Pearl's classic. Season drives both the sprinkler and the rain, and either one wets the grass.",
  },
};

/* --------------------------------------------------------------------------- the three rules */

function MiniTriple({ kind }: { kind: "chain" | "fork" | "collider" }) {
  const [given, setGiven] = useState(false);
  const nodes: DagNode[] = kind === "chain"
    ? [{ id: "A", x: 0, y: 50 }, { id: "B", x: 50, y: 50 }, { id: "C", x: 100, y: 50 }]
    : kind === "fork"
      ? [{ id: "A", x: 0, y: 100 }, { id: "B", x: 50, y: 0 }, { id: "C", x: 100, y: 100 }]
      : [{ id: "A", x: 0, y: 0 }, { id: "B", x: 50, y: 100 }, { id: "C", x: 100, y: 0 }];
  const edges: DagEdge[] = kind === "chain" ? [["A", "B"], ["B", "C"]] : kind === "fork" ? [["B", "A"], ["B", "C"]] : [["A", "B"], ["C", "B"]];
  const open = kind === "collider" ? given : !given;
  const title = kind === "chain" ? "Chain  A → B → C" : kind === "fork" ? "Fork  A ← B → C" : "Collider  A → B ← C";
  return (
    <div className="card" style={{ margin: 0 }}>
      <b className="small">{title}</b>
      <Dag nodes={nodes} edges={edges} width={220} height={kind === "chain" ? 80 : 130} radius={18}
        given={given ? ["B"] : []} onNodeClick={(id) => id === "B" && setGiven((g) => !g)}
        lit={open ? edges : []} litColor="var(--good)" />
      <div className="small" style={{ marginTop: 4 }}>
        <label style={{ cursor: "pointer" }}>
          <input type="checkbox" checked={given} onChange={() => setGiven((g) => !g)} /> condition on B
        </label>
        {" · "}
        <b style={{ color: open ? "var(--good)" : "var(--warm)" }}>{open ? "open" : "blocked"}</b>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------------- Berkson */

function mulberry(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r: () => number) {
  return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
}
function corr(pts: [number, number][]) {
  const n = pts.length;
  if (n < 3) return NaN;
  const mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (const [a, b] of pts) { sxy += (a - mx) * (b - my); sxx += (a - mx) ** 2; syy += (b - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
}

function Berkson() {
  const [bar, setBar] = useState(1.6);
  const people = useMemo(() => {
    const r = mulberry(7);
    return Array.from({ length: 360 }, () => [gauss(r), gauss(r)] as [number, number]);
  }, []);
  const star = people.map(([t, l]) => t + l > bar);
  const stars = people.filter((_, i) => star[i]);
  const rAll = corr(people), rStar = corr(stars);
  return (
    <div className="row">
      <div style={{ flex: "0 1 300px" }}>
        <Scatter points={people} size={300} domain={[-3.2, 3.2]} radius={3}
          colors={star.map((s) => (s ? "var(--warm)" : "var(--ink-3)"))} xLabel="talent" yLabel="looks" />
      </div>
      <div className="grow">
        <Slider label="How good you must be to become a star (talent + looks >)" value={bar} min={-1} max={3} step={0.05}
          onChange={setBar} format={(v) => v.toFixed(2)} />
        <p className="small">
          Every dot is a person. Talent and looks were drawn independently, and a person becomes a
          <span style={{ color: "var(--warm)" }}> star (orange)</span> when the two add up to more than the bar.
        </p>
        <div className="row">
          <Stat k="Correlation, everyone" v={fmt(rAll, 2)} d="talent vs looks" />
          <Stat k="Correlation, stars only" v={fmt(rStar, 2)} color="var(--warm)" d={`${stars.length} stars`} />
        </div>
        <p className="small">
          Among stars, the less talented ones must be better looking, or they would not have made it. Selecting on
          the collider manufactures a negative correlation out of nothing. This is <b>Berkson's paradox</b>{" "}
          (Berkson 1946 saw it in hospital patients).
        </p>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------------- explorer */

const arrow = (a: "->" | "<-") => (a === "->" ? "→" : "←");
const texId = (v: string) => (/\d/.test(v) ? v.replace(/(\d)/, "_$1") : v.length > 1 ? `\\mathrm{${v}}` : v);
const setTeX = (s: string[]) => (s.length ? `\\{${s.map(texId).join(", ")}\\}` : "\\varnothing");

function PathText({ p }: { p: PathR }) {
  const blockers = new Set(p.triples.filter((t) => t.blocked).map((t) => t.node));
  return (
    <span className="mono" style={{ fontSize: "0.92rem" }}>
      {p.nodes.map((n, i) => (
        <span key={i}>
          {i > 0 && <span className="muted"> {arrow(p.arrows[i - 1])} </span>}
          <span style={blockers.has(n) ? { color: "var(--warm)", fontWeight: 700, textDecoration: "underline" } : undefined}>{n}</span>
        </span>
      ))}
    </span>
  );
}

function Explorer() {
  const [key, setKey] = useState<keyof typeof PRESETS>("sprinkler");
  const preset = PRESETS[key];
  const [x, setX] = useState(preset.x);
  const [y, setY] = useState(preset.y);
  const [given, setGiven] = useState<string[]>([]);
  const [sel, setSel] = useState(0);

  const pick = (k: keyof typeof PRESETS) => {
    setKey(k); setX(PRESETS[k].x); setY(PRESETS[k].y); setGiven([]); setSel(0);
  };
  const toggle = (id: string) => {
    if (id === x || id === y) return;
    setGiven((g) => (g.includes(id) ? g.filter((v) => v !== id) : [...g, id]));
  };
  const ids = preset.nodes.map((n) => n.id);
  const body = {
    nodes: preset.nodes.map((n) => ({ id: n.id, latent: !!n.latent })),
    edges: preset.edges, x, y, given: given.filter((g) => g !== x && g !== y),
  };
  const { data, error, loading } = useApi<Analysis>("causal/analyze", body);
  const paths = data?.paths ?? [];
  const cur = paths[Math.min(sel, Math.max(0, paths.length - 1))];
  const lit: DagEdge[] = cur ? cur.nodes.slice(1).map((n, i) => [cur.nodes[i], n]) : [];
  const badDesc = data ? given.filter((g) => data.descendants_of_x.includes(g)) : [];
  const openBack = paths.filter((p) => p.backdoor && p.open);

  return (
    <Card title="The d-separation explorer">
      <div className="controls">
        <Select label="Graph" value={key} onChange={pick}
          options={Object.entries(PRESETS).map(([k, p]) => ({ value: k, label: p.name }))} />
        <Select label="X (cause)" value={x} onChange={(v) => { setX(v); setGiven((g) => g.filter((n) => n !== v)); setSel(0); }}
          options={ids.filter((i) => i !== y)} />
        <Select label="Y (effect)" value={y} onChange={(v) => { setY(v); setGiven((g) => g.filter((n) => n !== v)); setSel(0); }}
          options={ids.filter((i) => i !== x)} />
        <button className="btn" onClick={() => setGiven([])} disabled={!given.length}>Clear conditioning</button>
      </div>
      <p className="small muted" style={{ marginTop: 0 }}>{preset.blurb}</p>
      <div className="row">
        <div style={{ flex: "1 1 300px", maxWidth: 420 }}>
          <Dag nodes={preset.nodes} edges={preset.edges} width={400} height={preset.height} given={given}
            role={{ [x]: "x", [y]: "y" }} onNodeClick={toggle} lit={lit}
            litColor={cur?.open ? "var(--good)" : "var(--warm)"} />
          <div className="small muted">
            Click a node to condition on it (filled = conditioned).{" "}
            <span style={{ color: "var(--accent)" }}>Blue ring = X</span>,{" "}
            <span style={{ color: "var(--warm)" }}>orange ring = Y</span>. The highlighted path is the one selected
            below: green if open, orange if blocked.
          </div>
          {preset.legend && (
            <div className="small" style={{ marginTop: 6 }}>
              {Object.entries(preset.legend).map(([k, v]) => <span key={k} style={{ marginRight: 10 }}><b>{k}</b> = {v}</span>)}
            </div>
          )}
        </div>
        <div className="grow">
          <Loading loading={loading} error={error}>
            {data && (
              <>
                <div className="row" style={{ gap: 10 }}>
                  <Stat k="Verdict" v={data.d_separated ? "d-separated" : "d-connected"}
                    color={data.d_separated ? "var(--good)" : "var(--warm)"}
                    d={<TeX>{`${texId(x)} ${data.d_separated ? "\\perp\\!\\!\\!\\perp" : "\\not\\!\\perp\\!\\!\\!\\perp"} ${texId(y)} \\mid ${setTeX(given)}`}</TeX>} />
                  <Stat k="Open paths" v={`${paths.filter((p) => p.open).length} / ${paths.length}`} />
                </div>
                <div className="small" style={{ marginTop: 10 }}>
                  <b>Minimal back-door sets</b> for the effect of {x} on {y}:{" "}
                  {data.backdoor_sets.length
                    ? data.backdoor_sets.map((s, i) => <span key={i}>{i > 0 && ", "}<TeX>{setTeX(s)}</TeX></span>)
                    : <span className="muted">none exist (up to size 3)</span>}
                </div>
                <div className="small" style={{ marginTop: 6 }}>
                  <b>Is your conditioning set <TeX>{setTeX(given)}</TeX> a valid back-door set?</b>{" "}
                  {data.given_is_backdoor
                    ? <span style={{ color: "var(--good)" }}>Yes. Adjusting for it gives <TeX>P(y\mid do(x))</TeX>.</span>
                    : <span style={{ color: "var(--warm)" }}>
                      No.{" "}
                      {badDesc.length > 0 && <>It contains {badDesc.join(", ")}, a descendant of {x}. </>}
                      {openBack.length > 0 && <>{openBack.length} back-door path{openBack.length > 1 ? "s stay" : " stays"} open. </>}
                      {badDesc.length === 0 && openBack.length === 0 && <>It includes a node the criterion forbids. </>}
                    </span>}
                </div>
              </>
            )}
          </Loading>
        </div>
      </div>

      <h3>Every path from {x} to {y}</h3>
      <p className="small muted" style={{ marginTop: 0 }}>
        Click a path to light it up on the graph. Underlined orange nodes are where the path is blocked.
      </p>
      <div style={{ display: "grid", gap: 8 }}>
        {paths.map((p, i) => (
          <div key={p.nodes.join("-")} onClick={() => setSel(i)} role="button" tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && setSel(i)}
            style={{
              cursor: "pointer", padding: "8px 12px", borderRadius: 8, background: "var(--surface)",
              border: `1px solid ${cur === p ? (p.open ? "var(--good)" : "var(--warm)") : "var(--line)"}`,
              boxShadow: cur === p ? `inset 3px 0 0 ${p.open ? "var(--good)" : "var(--warm)"}` : undefined,
            }}>
            <div className="row" style={{ gap: 8, alignItems: "center" }}>
              <PathText p={p} />
              <span className="chip" style={{ cursor: "inherit", color: p.open ? "var(--good)" : "var(--warm)", borderColor: "currentColor" }}>
                {p.open ? "open" : "blocked"}
              </span>
              <span className="small muted">
                {p.directed ? "causal path" : p.backdoor ? "back-door path" : "non-causal path"}
              </span>
            </div>
            {p.triples.length === 0
              ? <div className="small muted">A direct edge: nothing in the middle, nothing can block it.</div>
              : (
                <ul className="small" style={{ margin: "4px 0 0", paddingLeft: 18 }}>
                  {p.triples.map((t, j) => (
                    <li key={j} style={{ color: t.blocked ? "var(--warm)" : "var(--ink-2)" }}>{t.reason}.</li>
                  ))}
                </ul>
              )}
          </div>
        ))}
      </div>
    </Card>
  );
}

/* --------------------------------------------------------------------------- quiz */

const QUIZ: { preset: keyof typeof PRESETS; x: string; y: string; given: string[]; hint: string }[] = [
  { preset: "chain", x: "X", y: "Y", given: ["M"], hint: "One path, and its middle node is conditioned." },
  { preset: "collider", x: "T", y: "L", given: [], hint: "Nothing is conditioned. What kind of node sits between them?" },
  { preset: "colliderDesc", x: "T", y: "L", given: ["C"], hint: "C is not on the path. Does that matter?" },
  { preset: "sprinkler", x: "Sp", y: "R", given: ["Se"], hint: "Two paths: one through the season, one through the wet grass." },
  { preset: "sprinkler", x: "Sp", y: "R", given: ["Se", "Sl"], hint: "Same, but now we also know the path is slippery." },
  { preset: "pearl", x: "Z1", y: "Y", given: ["Z3", "X"], hint: "Conditioning on Z3 blocks one path and opens another." },
];

function Quiz() {
  const [i, setI] = useState(0);
  const [guess, setGuess] = useState<boolean | null>(null);
  const [score, setScore] = useState<boolean[]>([]);
  const q = QUIZ[i];
  const p = PRESETS[q.preset];
  const { data, error } = useApi<Analysis>("causal/analyze", {
    nodes: p.nodes.map((n) => ({ id: n.id })), edges: p.edges, x: q.x, y: q.y, given: q.given,
  }, 0);
  const answer = data?.d_separated;
  const openPath = data?.paths.find((pp) => pp.open);
  const choose = (g: boolean) => {
    if (guess !== null || answer === undefined) return;
    setGuess(g);
    setScore((s) => [...s, g === answer]);
  };
  const next = () => { setI((i + 1) % QUIZ.length); setGuess(null); if (i + 1 === QUIZ.length) setScore([]); };
  return (
    <Card title={`Predict before you click (${i + 1} of ${QUIZ.length})`}>
      <div className="row">
        <div style={{ flex: "0 1 300px" }}>
          <Dag nodes={p.nodes} edges={p.edges} width={300} height={Math.min(p.height, 240)} given={q.given}
            role={{ [q.x]: "x", [q.y]: "y" }}
            lit={guess !== null && openPath ? openPath.nodes.slice(1).map((n, k) => [openPath.nodes[k], n] as DagEdge) : []} />
        </div>
        <div className="grow">
          <p style={{ marginTop: 0 }}>
            In the <b>{p.name}</b> example, conditioning on <TeX>{setTeX(q.given)}</TeX> (filled), are{" "}
            <b>{q.x}</b> and <b>{q.y}</b> d-separated?
          </p>
          <p className="small muted">Hint: {q.hint}</p>
          <div className="row" style={{ gap: 8 }}>
            <button className="btn" onClick={() => choose(true)} disabled={guess !== null}>Separated (no open path)</button>
            <button className="btn" onClick={() => choose(false)} disabled={guess !== null}>Connected (some open path)</button>
          </div>
          {error && <div className="error">The server said: {error}</div>}
          {guess !== null && data && (
            <div style={{ marginTop: 10 }}>
              <b style={{ color: guess === answer ? "var(--good)" : "var(--warm)" }}>
                {guess === answer ? "Right." : "Not quite."} They are {answer ? "d-separated" : "d-connected"}.
              </b>
              <ul className="small" style={{ paddingLeft: 18 }}>
                {data.paths.map((pp) => (
                  <li key={pp.nodes.join()}>
                    <PathText p={pp} />: {pp.open ? "open" : (pp.triples.find((t) => t.blocked)?.reason ?? "blocked")}
                    {pp.open && pp.triples.some((t) => t.kind === "collider") && ` (${pp.triples.find((t) => t.kind === "collider")!.reason})`}.
                  </li>
                ))}
              </ul>
              <button className="btn primary" onClick={next}>{i + 1 === QUIZ.length ? "Start over" : "Next question"}</button>
            </div>
          )}
          {score.length > 0 && <div className="small muted" style={{ marginTop: 8 }}>Score so far: {score.filter(Boolean).length} / {score.length}</div>}
        </div>
      </div>
    </Card>
  );
}

/* --------------------------------------------------------------------------- lesson */

export default function DSeparation() {
  return (
    <div className="prose-wide">
      <h2>Information flows along paths</h2>
      <div className="prose">
        <p>
          In the last lesson, stone size made treatment and recovery look related even apart from any effect of
          the treatment. That kind of association travels along <b>paths</b> in the graph. A path is any route
          between two nodes along edges, ignoring which way the arrows point.
        </p>
        <p>
          Every path is built from overlapping triples of three nodes, and there are only three shapes a triple can
          take. Learn what each one does when you condition on its middle node, and you can read any graph. Click
          the middle node <b>B</b> in each:
        </p>
      </div>
      <div className="grid3">
        <MiniTriple kind="chain" />
        <MiniTriple kind="fork" />
        <MiniTriple kind="collider" />
      </div>
      <div className="prose">
        <ul>
          <li>
            <b>Chain</b> <TeX>A \to B \to C</TeX>: fire causes smoke, smoke sets off the alarm. Knowing fire makes an
            alarm more likely. But if you already <em>know</em> whether there is smoke, learning about the fire adds
            nothing. Conditioning on the middle <b>blocks</b> the chain.
          </li>
          <li>
            <b>Fork</b> <TeX>A \leftarrow B \to C</TeX>: heat drives both ice-cream sales and drownings, so the two
            are correlated. Compare only days with the same temperature and the correlation disappears.
            Conditioning on the common cause <b>blocks</b> the fork.
          </li>
          <li>
            <b>Collider</b> <TeX>A \to B \leftarrow C</TeX>: talent and looks both help make a film star. In the
            whole population they are unrelated, so the collider is <b>blocked on its own</b>. Now look only at
            stars. A star with little talent must have great looks. Conditioning on the collider, or on anything it
            causes, <b>opens</b> the path.
          </li>
        </ul>
        <p>The collider is the one that surprises people. You can watch it happen:</p>
      </div>
      <Card title="Berkson's paradox: conditioning on a collider">
        <Berkson />
      </Card>

      <h2>The rule</h2>
      <div className="prose">
        <p>
          A path is <b>blocked</b> by a conditioning set <TeX>Z</TeX> if some triple on it is blocked: a chain or
          fork whose middle is in <TeX>Z</TeX>, or a collider where neither the collider nor any of its descendants
          is in <TeX>Z</TeX>. Otherwise the path is <b>open</b>.
        </p>
        <p>
          <TeX>X</TeX> and <TeX>Y</TeX> are <b>d-separated</b> given <TeX>Z</TeX>, written{" "}
          <TeX>{String.raw`X \perp\!\!\!\perp Y \mid Z`}</TeX>, when <em>every</em> path between them is
          blocked. If the data came from the graph, d-separation guarantees that <TeX>X</TeX> and <TeX>Y</TeX> are
          independent given <TeX>Z</TeX> (Pearl 1988; Pearl 2009, ch. 1.2). One open path is enough to make them
          (generally) dependent.
        </p>
        <p>
          The <b>back-door criterion</b> uses the same rule. A set <TeX>Z</TeX> is a valid adjustment set for the effect
          of <TeX>X</TeX> on <TeX>Y</TeX> when (1) no node in <TeX>Z</TeX> is a descendant of <TeX>X</TeX>, and (2){" "}
          <TeX>Z</TeX> blocks every path that starts with an arrow <em>into</em> <TeX>X</TeX>. Then{" "}
          <TeX>{String.raw`P(y\mid do(x))=\sum_z P(y\mid x,z)P(z)`}</TeX>, which is the formula from the kidney-stone
          lesson.
        </p>
      </div>

      <Explorer />

      <Callout kind="try">
        <p>
          In <b>M-bias</b>, check that the empty set is a valid back-door set, then condition on M. A pre-treatment
          variable that correlates with both X and Y was the thing to <em>leave out</em>. In <b>Pearl's six-node
          graph</b>, condition on Z3 alone: it blocks the fork through Z3 but opens the collider Z1 → Z3 ← Z2, so you
          must add Z1 or Z2. In <b>confounder + mediator</b>, condition on M and see the causal path close.
        </p>
      </Callout>

      <Quiz />

      <Callout kind="warn">
        <p>
          d-separation is a statement about the <em>graph</em>. It promises independence when paths are blocked. When
          a path is open, the variables are dependent for almost every choice of numbers, but a coincidence of
          parameters can cancel two paths exactly. The graph can't see cancellations like that.
        </p>
      </Callout>

      <Callout kind="takeaway">
        <p>
          Chains and forks carry association unless you condition on their middle. Colliders block it unless you
          condition on them or their descendants. Every adjustment you make is a choice about which paths to close,
          and conditioning on the wrong variable can open a path that was closed.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          A transformer's computational graph is a DAG, so the same reading applies. The residual stream is a long
          chain with many forks: every layer writes into it and every later layer reads from it. When you
          <em> condition</em> on an activation by picking only prompts where a feature fires, you are selecting on a
          node that many upstream components collide into. Correlations you then find between those components can
          be Berkson artifacts. Interventions sidestep the problem, which is why circuit work relies on patching
          rather than conditioning (Pearl 2009, <em>Causality</em>; Pearl, Glymour &amp; Jewell 2016,{" "}
          <em>Causal Inference in Statistics: A Primer</em>, ch. 2–3).
        </p>
      </Callout>
    </div>
  );
}
