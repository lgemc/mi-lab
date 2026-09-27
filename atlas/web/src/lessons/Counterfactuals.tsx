import { useState } from "react";
import { Lines } from "../components/charts";
import { Dag, DagEdge, DagNode } from "../components/Dag";
import { TeX } from "../components/tex";
import { Callout, Card, Slider, Stat } from "../components/ui";
import { fmt } from "../api";

/*
 * A three-variable structural causal model, small enough to run in the browser:
 *   X = U_x,   M = b·X + U_m,   Y = a·X + c·M + (1 + d·X)·U_y
 * With d = 0 it is linear and every unit has the same effect; d > 0 lets the effect of X depend on
 * the unit's own U_y, which is what makes a counterfactual differ from an average.
 */

type Coef = { a: number; b: number; c: number; d: number };
type U = { ux: number; um: number; uy: number };

const solve = (k: Coef, u: U, x: number) => {
  const m = k.b * x + u.um;
  const y = k.a * x + k.c * m + (1 + k.d * x) * u.uy;
  return { x, m, y };
};
/** Abduction: the noise values that make the model reproduce what we saw. */
const abduce = (k: Coef, x: number, m: number, y: number): U => ({
  ux: x, um: m - k.b * x, uy: (y - k.a * x - k.c * m) / (1 + k.d * x),
});
/** E[Y | do(x)]: average over units, and every U has mean zero. */
const average = (k: Coef, x: number) => k.a * x + k.c * k.b * x;

const NODES: DagNode[] = [
  { id: "Ux", x: 0, y: 0, latent: true, label: "Uₓ" },
  { id: "Um", x: 50, y: 0, latent: true, label: "Uₘ" },
  { id: "Uy", x: 100, y: 0, latent: true, label: "Uᵧ" },
  { id: "X", x: 0, y: 88 }, { id: "M", x: 50, y: 88 }, { id: "Y", x: 100, y: 88 },
];
const EDGES: DagEdge[] = [["Ux", "X"], ["Um", "M"], ["Uy", "Y"], ["X", "M"], ["M", "Y"], ["X", "Y"]];

// A class of six students. U_m and U_y each sum to zero, so their average counterfactual is exactly
// the interventional mean -- the point of the table.
const CLASS: (U & { name: string })[] = [
  { name: "Asha", ux: 0.5, um: 0.6, uy: 1.2 },
  { name: "Ben", ux: 1.0, um: -0.4, uy: -0.5 },
  { name: "Chen", ux: 1.5, um: 0.2, uy: 0.3 },
  { name: "Dara", ux: 2.0, um: -0.8, uy: -1.5 },
  { name: "Eli", ux: 2.5, um: 0.3, uy: 0.8 },
  { name: "Femi", ux: 0.5, um: 0.1, uy: -0.3 },
];

function StepPanel({ n, title, children, dag }: { n: number; title: string; children: React.ReactNode; dag: React.ReactNode }) {
  return (
    <div className="card" style={{ margin: 0 }}>
      <div className="small" style={{ color: "var(--accent)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em" }}>
        Step {n}
      </div>
      <b>{title}</b>
      <div style={{ margin: "6px 0" }}>{dag}</div>
      <div className="small">{children}</div>
    </div>
  );
}

export default function Counterfactuals() {
  const [k, setK] = useState<Coef>({ a: 1, b: 1.5, c: 2, d: 0.5 });
  const [obs, setObs] = useState({ x: 1, m: 2, y: 6 });
  const [xp, setXp] = useState(3);
  const setC = (key: keyof Coef) => (v: number) => setK((o) => ({ ...o, [key]: v }));
  const setO = (key: "x" | "m" | "y") => (v: number) => setObs((o) => ({ ...o, [key]: v }));

  const u = abduce(k, obs.x, obs.m, obs.y);
  const cf = solve(k, u, xp);
  const avg = average(k, xp);
  const f = (v: number) => fmt(v, 2);
  const fs = (v: number) => (v >= 0 ? `+${fmt(v, 2)}` : fmt(v, 2));

  const grid = Array.from({ length: 13 }, (_, i) => i * 0.25);
  const markX = Math.round(obs.x / 0.25);

  return (
    <div className="prose-wide">
      <h2>Three rungs</h2>
      <div className="prose">
        <p>
          Pearl describes a <b>ladder of causation</b> with three rungs (Pearl &amp; Mackenzie 2018). Each rung asks a
          question the one below can't answer:
        </p>
        <ol>
          <li><b>Seeing</b>: <TeX>P(y\mid x)</TeX>. Among students who studied 3 hours, what do scores look like?</li>
          <li><b>Doing</b>: <TeX>P(y\mid do(x))</TeX>. If we made every student study 3 hours, what would the average score be?</li>
          <li>
            <b>Imagining</b>: <TeX>{String.raw`P(y_{x'}\mid x, y)`}</TeX>. Ana studied 1 hour and scored 6. What would{" "}
            <em>Ana</em> have scored if she had studied 3?
          </li>
        </ol>
        <p>
          The third question is about one person, and it conflicts with the facts: Ana did <em>not</em> study 3 hours. We
          can't run that experiment, because she has already taken the test. What we can do is use what we saw of Ana to
          learn what is particular about her, and then replay the model with that particular person in it.
        </p>
      </div>

      <h2>A model with noise terms</h2>
      <div className="prose">
        <p>
          The previous lessons only needed the graph and the probabilities. Counterfactuals need a full{" "}
          <b>structural causal model</b>: an equation for each variable, plus an unobserved noise term{" "}
          <TeX>U</TeX> that holds everything about the unit the equation leaves out. Here <TeX>X</TeX> is hours
          studied, <TeX>M</TeX> is practice problems solved (in tens), and <TeX>Y</TeX> is the exam score:
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`X = U_x,\qquad M = b\,X + U_m,\qquad Y = a\,X + c\,M + (1 + d\,X)\,U_y`}</TeX>
        <p className="small">
          <TeX>U_y</TeX> is the student's aptitude or luck on the day. The <TeX>d</TeX> term lets aptitude change how
          much studying helps: with <TeX>d &gt; 0</TeX>, students with high <TeX>U_y</TeX> gain more per hour. Across the
          class each <TeX>U</TeX> averages zero. This is a toy: real exam scores don't follow three tidy equations.
        </p>
        <div className="controls">
          <Slider label={<>direct effect <TeX>a</TeX></>} value={k.a} min={-1} max={3} step={0.1} onChange={setC("a")} format={f} />
          <Slider label={<><TeX>X \to M</TeX> <TeX>b</TeX></>} value={k.b} min={0} max={3} step={0.1} onChange={setC("b")} format={f} />
          <Slider label={<><TeX>M \to Y</TeX> <TeX>c</TeX></>} value={k.c} min={-1} max={3} step={0.1} onChange={setC("c")} format={f} />
          <Slider label={<>aptitude × study <TeX>d</TeX></>} value={k.d} min={0} max={1} step={0.05} onChange={setC("d")} format={f} />
        </div>
      </Card>

      <h2>Abduction, action, prediction</h2>
      <div className="prose">
        <p>
          Every counterfactual is computed in the same three steps (Pearl 2009, ch. 7.1). First describe what you saw
          about the unit:
        </p>
      </div>
      <Card title="What we observed about Ana">
        <div className="controls">
          <Slider label="hours studied x" value={obs.x} min={0} max={3} step={0.25} onChange={setO("x")} format={f} />
          <Slider label="problems solved m" value={obs.m} min={-2} max={8} step={0.1} onChange={setO("m")} format={f} />
          <Slider label="score y" value={obs.y} min={-4} max={20} step={0.1} onChange={setO("y")} format={f} />
          <Slider label="counterfactual hours x′" value={xp} min={0} max={3} step={0.25} onChange={setXp} format={f} />
        </div>
      </Card>
      <div className="grid3">
        <StepPanel n={1} title="Abduction: who is Ana?"
          dag={<Dag nodes={NODES} edges={EDGES} width={240} height={150} radius={17} given={["X", "M", "Y"]}
            lit={[["Ux", "X"], ["Um", "M"], ["Uy", "Y"]]} litColor="var(--gold)"
            nodeValue={{ X: f(obs.x), M: f(obs.m), Y: f(obs.y) }} />}>
          <p style={{ marginTop: 0 }}>Run each equation backwards to find the noise that produced what we saw.</p>
          <TeX block>{String.raw`\begin{aligned}
U_x &= ${f(obs.x)}\\
U_m &= ${f(obs.m)} - ${f(k.b)}\cdot ${f(obs.x)} = \mathbf{${f(u.um)}}\\
U_y &= \dfrac{y - a\,x - c\,m}{1 + d\,x}\\
&= \dfrac{${f(obs.y)} - ${f(k.a)}\cdot${f(obs.x)} - ${f(k.c)}\cdot${f(obs.m)}}{1 + ${f(k.d)}\cdot${f(obs.x)}}\\
&= \mathbf{${f(u.uy)}}
\end{aligned}`}</TeX>
          <p className="muted" style={{ marginBottom: 0 }}>
            {u.uy > 0.05 ? "Ana did better than the model predicts for her hours: high aptitude." : u.uy < -0.05 ? "Ana did worse than the model predicts: low aptitude or a bad day." : "Ana is exactly average."}
          </p>
        </StepPanel>
        <StepPanel n={2} title="Action: set X = x′"
          dag={<Dag nodes={NODES} edges={EDGES} width={240} height={150} radius={17} cut={[["Ux", "X"]]}
            role={{ X: "x" }} nodeValue={{ X: f(xp), M: "?", Y: "?" }} />}>
          <div style={{ color: "var(--gold)", marginBottom: 6 }}>
            <TeX>{`U_x=${f(u.ux)},\\; U_m=${f(u.um)},\\; U_y=${f(u.uy)}`}</TeX>
          </div>
          <p style={{ marginTop: 0 }}>
            Keep Ana's noise exactly as inferred. Cut the arrow into <TeX>X</TeX> and set <TeX>{`X = ${f(xp)}`}</TeX>. This
            is the same surgery as <TeX>do(x)</TeX>, done on one person instead of a population.
          </p>
          <TeX block>{String.raw`X \leftarrow ${f(xp)}\quad (\text{was } ${f(obs.x)})`}</TeX>
        </StepPanel>
        <StepPanel n={3} title="Prediction: replay the model"
          dag={<Dag nodes={NODES} edges={EDGES} width={240} height={150} radius={17} cut={[["Ux", "X"]]}
            role={{ X: "x", Y: "y" }} lit={[["X", "M"], ["M", "Y"], ["X", "Y"], ["Um", "M"], ["Uy", "Y"]]} litColor="var(--gold)"
            nodeValue={{ X: f(xp), M: f(cf.m), Y: f(cf.y) }} />}>
          <p style={{ marginTop: 0 }}>Run the equations forward with the new X and Ana's own noise.</p>
          <TeX block>{String.raw`\begin{aligned}
M_{x'} &= ${f(k.b)}\cdot${f(xp)} + ${f(u.um)} = \mathbf{${f(cf.m)}}\\
Y_{x'} &= ${f(k.a)}\cdot${f(xp)} + ${f(k.c)}\cdot${f(cf.m)}\\
&\quad + (1+${f(k.d)}\cdot${f(xp)})\cdot${f(u.uy)}\\
&= \mathbf{${f(cf.y)}}
\end{aligned}`}</TeX>
        </StepPanel>
      </div>

      <Card title="Ana versus the average">
        <div className="row">
          <Stat k="Ana actually scored" v={f(obs.y)} d={`with ${f(obs.x)} hours`} />
          <Stat k={`Ana, had she studied ${f(xp)} h`} v={f(cf.y)} color="var(--accent)" d={`counterfactual: ${fs(cf.y - obs.y)}`} />
          <Stat k={`E[Y | do(X = ${f(xp)})]`} v={f(avg)} color="var(--warm)" d="the class average" />
        </div>
        <div style={{ marginTop: 12 }}>
          <Lines width={560} height={230} xLabels={grid.map((g) => g.toFixed(2))} markX={markX} format={f} yLabel="score"
            series={[
              { name: "Ana: Y_x′ (counterfactual)", values: grid.map((g) => solve(k, u, g).y), color: "var(--accent)" },
              { name: "everyone: E[Y | do(x′)]", values: grid.map((g) => average(k, g)), color: "var(--warm)", dashed: true },
            ]} />
        </div>
        <p className="small">
          The solid line is Ana's personal response curve: what she would score at each number of hours. It passes
          through what she actually scored at <TeX>{`x = ${f(obs.x)}`}</TeX> (the marker), because the counterfactual
          must agree with the facts where they overlap. The dashed line is the interventional average. It is the
          answer to rung two, and it forgets everything we learned about Ana.
        </p>
        <p className="small">
          With <TeX>d = 0</TeX> the two lines are parallel: every student gains the same{" "}
          <TeX>{`a + bc = ${f(k.a + k.b * k.c)}`}</TeX> per hour, and only the starting level is personal. With{" "}
          <TeX>d &gt; 0</TeX> the slopes differ too, so Ana's gain from extra study is not the class's gain.
        </p>
      </Card>

      <Callout kind="try">
        <p>
          Raise Ana's observed score while keeping hours and problems fixed. Abduction concludes she has high aptitude,
          and with <TeX>d &gt; 0</TeX> her counterfactual gain from extra study grows. The average doesn't move at all.
          Then set <TeX>d = 0</TeX> and watch her gain become the same as everyone's.
        </p>
      </Callout>

      <h2>The average of the counterfactuals is the intervention</h2>
      <div className="prose">
        <p>
          Here is a class of six students, each with their own noise. For every one we can compute what they would have
          scored with <TeX>{`x' = ${f(xp)}`}</TeX> hours. Their noise terms average to zero, so the mean of their
          counterfactual scores equals <TeX>{String.raw`E[Y\mid do(x')]`}</TeX> exactly. Rung three determines rung two.
          The reverse fails: the average alone can't tell you any one student's row.
        </p>
      </div>
      <Card>
        <div className="scroll-x">
          <table className="tbl">
            <thead>
              <tr><th>student</th><th>x</th><th>m</th><th>y</th><th><TeX>U_y</TeX></th><th><TeX>{`Y_{x'=${f(xp)}}`}</TeX></th><th>gain</th></tr>
            </thead>
            <tbody>
              {CLASS.map((s) => {
                const o = solve(k, s, s.ux), c = solve(k, s, xp);
                return (
                  <tr key={s.name}>
                    <td>{s.name}</td><td>{f(o.x)}</td><td>{f(o.m)}</td><td>{f(o.y)}</td><td>{f(s.uy)}</td>
                    <td><b>{f(c.y)}</b></td>
                    <td style={{ color: c.y - o.y >= 0 ? "var(--good)" : "var(--warm)" }}>{fs(c.y - o.y)}</td>
                  </tr>
                );
              })}
              <tr>
                <td><b>mean</b></td><td colSpan={4} className="muted small">compare with <TeX>{String.raw`E[Y\mid do(x')]`}</TeX> = {f(avg)}</td>
                <td><b>{f(CLASS.reduce((sum, s) => sum + solve(k, s, xp).y, 0) / CLASS.length)}</b></td><td />
              </tr>
            </tbody>
          </table>
        </div>
        <p className="small muted">
          Each row is abduction, action and prediction for one student. The gain column is the student's own causal
          effect of moving from their actual hours to <TeX>x'</TeX>. It differs from row to row because their hours
          differ and, when <TeX>d &gt; 0</TeX>, because their aptitude does too.
        </p>
      </Card>

      <Callout kind="warn">
        <p>
          Abduction needs the equations, not only the graph and the probabilities. Two models with the same{" "}
          <TeX>P(x, m, y)</TeX> and the same <TeX>P(y\mid do(x))</TeX> can disagree about Ana's counterfactual. In this
          page we also observed M. If M were hidden and <TeX>d &gt; 0</TeX>, we couldn't separate Ana's <TeX>U_m</TeX>{" "}
          from her <TeX>U_y</TeX>, and her counterfactual would no longer be pinned down. That's why rung three is the
          hardest to reach from data.
        </p>
      </Callout>

      <Callout kind="takeaway">
        <p>
          A counterfactual asks what would have happened to <em>this</em> unit. To answer it, infer the unit's hidden
          noise from what you saw (abduction), cut and set the variable (action), and run the model forward with that
          same noise (prediction). The interventional average is the mean of these answers over units, and it can differ
          from any one of them.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          A neural network is a structural causal model with no noise: once the input is fixed, every activation is
          determined. So abduction is free, because the input <em>is</em> the unit, and rung three is directly
          computable. <b>Activation patching</b> is exactly this three-step recipe on one prompt: run the clean input and
          cache it (abduction), overwrite one activation with its value from another input (action), and let the rest of
          the forward pass run (prediction). That is why patching results are per-example counterfactuals, and averaging
          them over a dataset gives the interventional quantity. Causal mediation analysis (Vig et al. 2020) and
          interchange interventions (Geiger et al. 2021) are built on this. See Pearl (2009, ch. 7) and Pearl, Glymour
          &amp; Jewell (2016, ch. 4).
        </p>
      </Callout>
    </div>
  );
}
