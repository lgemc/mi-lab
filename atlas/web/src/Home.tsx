import { Link, useNavigate } from "react-router-dom";
import { Callout } from "./components/ui";
import { LESSONS, PARTS } from "./lessons/registry";

/*
 * The map: the root question interpretability is trying to answer, the proxy questions it is
 * broken into, and the techniques that attack each -- each technique a link to its lesson.
 */
const QUESTIONS = [
  { id: "q1", text: "Which parts of the input explain this output?", x: 8 },
  { id: "q2", text: "What is encoded, and where?", x: 22 },
  { id: "q3", text: "What algorithm does it implement?", x: 36 },
  { id: "q4", text: "How do we scale circuit discovery?", x: 50 },
  { id: "q5", text: "What are the right variables?", x: 64 },
  { id: "q6", text: "How are features computed from features?", x: 78 },
  { id: "q7", text: "What is it thinking without saying it?", x: 92 },
];

const TECHNIQUES: { label: string; q: string[]; slug: string; x: number }[] = [
  { label: "Causal graphs · do()", q: ["q1", "q2"], slug: "seeing-vs-doing", x: 6 },
  { label: "Patching · IG", q: ["q1"], slug: "patching", x: 16 },
  { label: "Probing", q: ["q2"], slug: "probing", x: 26 },
  { label: "IOI · path patching", q: ["q3"], slug: "ioi", x: 37 },
  { label: "EAP · ACDC", q: ["q4"], slug: "circuit-discovery", x: 49 },
  { label: "Superposition · SAEs", q: ["q5", "q7"], slug: "sae", x: 61 },
  { label: "Transcoders · graphs", q: ["q6"], slug: "transcoders", x: 72 },
  { label: "Crosscoders", q: ["q5", "q6"], slug: "crosscoders", x: 82 },
  { label: "Logit · J · tuned lens", q: ["q7"], slug: "lenses", x: 93 },
];

function QuestionMap() {
  const nav = useNavigate();
  const W = 960, H = 380;
  const X = (p: number) => (p / 100) * W;
  const rootY = 40, qY = 170, tY = 320;
  return (
    <div className="card" style={{ padding: 10 }}>
      <div className="scroll-x">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 760, display: "block" }} role="img"
          aria-label="Map of interpretability questions and techniques">
          {QUESTIONS.map((q) => (
            <line key={q.id} x1={X(50)} y1={rootY + 22} x2={X(q.x)} y2={qY - 26} stroke="var(--line)" strokeWidth={1.5} />
          ))}
          {TECHNIQUES.flatMap((t) => t.q.map((qid) => {
            const q = QUESTIONS.find((z) => z.id === qid)!;
            return <line key={`${t.slug}-${qid}`} x1={X(t.x)} y1={tY - 18} x2={X(q.x)} y2={qY + 26} stroke="var(--line)" strokeWidth={1.5} />;
          }))}
          <foreignObject x={X(50) - 300} y={rootY - 22} width={600} height={46}>
            <div style={{ background: "var(--accent)", color: "var(--surface)", borderRadius: 10, padding: "8px 12px", textAlign: "center", fontWeight: 700, fontSize: 14, lineHeight: 1.3 }}>
              What will the model do in situations we have not seen — and why?
            </div>
          </foreignObject>
          {QUESTIONS.map((q) => (
            <foreignObject key={q.id} x={X(q.x) - 62} y={qY - 26} width={124} height={54}>
              <div style={{ background: "var(--accent-soft)", border: "1px solid var(--accent)", borderRadius: 8, padding: "4px 6px", fontSize: 11.5, lineHeight: 1.25, textAlign: "center", color: "var(--ink)", height: 52, display: "flex", alignItems: "center", justifyContent: "center" }}>
                {q.text}
              </div>
            </foreignObject>
          ))}
          {TECHNIQUES.map((t) => (
            <foreignObject key={t.slug} x={X(t.x) - 50} y={tY - 22} width={100} height={48}>
              <button onClick={() => nav(`/${t.slug}`)} style={{ width: "100%", height: 46, padding: "2px 4px", background: "var(--gold-soft)", border: "1px solid var(--gold)", borderRadius: 8, color: "var(--ink)", cursor: "pointer", fontFamily: "inherit", fontSize: 11.5, lineHeight: 1.2 }}>
                {t.label}
              </button>
            </foreignObject>
          ))}
        </svg>
      </div>
      <div className="caption small muted" style={{ padding: "0 8px 4px" }}>
        The question interpretability wants to answer (top), the proxy questions it is broken into, and the
        techniques that attack them. Click a technique to open its lesson.
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <div>
      <div className="hero">
        <div className="kicker">mi-lab · atlas</div>
        <h1>Interpretability you can poke</h1>
        <p className="lede">
          From Judea Pearl's do-calculus to circuits, lenses, sparse autoencoders, transcoders and crosscoders, with
          every idea running live on a model small enough to see all of. Drag the sliders. Break things on purpose.
        </p>
      </div>
      <QuestionMap />
      <Callout kind="note" label="How to read this">
        <p>
          Every model here is a toy, and on purpose: a graph with a probability table, or a network with a handful of
          neurons, wired by hand or trained in under a second. That means each lesson's idea is <em>exactly</em> true in
          its model, so you can see what an instrument would find if the story were true, and where it fails even then.
          The <b>In real models</b> boxes say what changes at scale.
        </p>
        <p>
          The first part needs no neural networks at all. The rest builds on it: activation patching is an intervention,
          mediation is a causal quantity, and a circuit is a causal claim about a graph of components.
        </p>
      </Callout>
      {PARTS.map((part) => (
        <section key={part}>
          <h2>{part}</h2>
          <div className="lesson-grid">
            {LESSONS.map((l, i) => l.part === part && (
              <Link key={l.slug} to={`/${l.slug}`} className="lesson-card">
                <span className="n">{String(i + 1).padStart(2, "0")}</span>
                <b>{l.title}</b>
                <span>{l.question}</span>
              </Link>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
