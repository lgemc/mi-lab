import { Suspense, useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation, useParams } from "react-router-dom";
import Home from "./Home";
import { LESSONS, PARTS, bySlug } from "./lessons/registry";

function useTheme() {
  const [theme, setTheme] = useState<string>(() => {
    try { return localStorage.getItem("atlas-theme") ?? "auto"; } catch { return "auto"; }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try { localStorage.setItem("atlas-theme", theme); } catch { /* private mode: fine */ }
  }, [theme]);
  const next = theme === "auto" ? "dark" : theme === "dark" ? "light" : "auto";
  return { theme, cycle: () => setTheme(next) };
}

function Nav({ onPick }: { onPick: () => void }) {
  return (
    <nav className="nav">
      <NavLink to="/" end onClick={onPick}><span className="num">◆</span>The map</NavLink>
      {PARTS.map((part) => (
        <div key={part}>
          <div className="nav-part">{part}</div>
          {LESSONS.map((l, i) => l.part === part && (
            <NavLink key={l.slug} to={`/${l.slug}`} onClick={onPick}>
              <span className="num">{String(i + 1).padStart(2, "0")}</span>{l.title}
            </NavLink>
          ))}
        </div>
      ))}
    </nav>
  );
}

function LessonPage() {
  const { slug = "" } = useParams();
  const i = bySlug(slug);
  if (i < 0) return <div><h1>Not here</h1><p>No lesson called “{slug}”. <Link to="/">Back to the map.</Link></p></div>;
  const lesson = LESSONS[i];
  const prev = LESSONS[i - 1];
  const next = LESSONS[i + 1];
  const Body = lesson.component;
  return (
    <article>
      <div className="kicker">{String(i + 1).padStart(2, "0")} · {lesson.part}</div>
      <h1>{lesson.title}</h1>
      <p className="lede">{lesson.lede}</p>
      <span className="question">Answers: {lesson.question}</span>
      <Suspense fallback={<p className="muted">Loading…</p>}>
        <Body />
      </Suspense>
      <div className="pager">
        {prev ? <Link to={`/${prev.slug}`}><small>← Previous</small>{prev.title}</Link> : <span />}
        {next ? <Link to={`/${next.slug}`} style={{ textAlign: "right" }}><small>Next →</small>{next.title}</Link> : <span />}
      </div>
    </article>
  );
}

export default function App() {
  const [open, setOpen] = useState(false);
  const { theme, cycle } = useTheme();
  const loc = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [loc.pathname]);
  useEffect(() => {
    const i = bySlug(loc.pathname.slice(1));
    document.title = i >= 0 ? `${LESSONS[i].title} · Interp Atlas` : "Interp Atlas";
  }, [loc.pathname]);
  return (
    <div className="shell">
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <Link to="/" className="brand" onClick={() => setOpen(false)}>
          <img src="/favicon.svg" alt="" />
          <span>Interp Atlas<small>interpretability you can poke</small></span>
        </Link>
        <Nav onPick={() => setOpen(false)} />
        <button className="theme-btn" onClick={cycle}>Theme: {theme}</button>
        <p className="small muted" style={{ marginTop: 16 }}>
          Part of <a href="https://github.com/lgemc/mi-lab">mi-lab</a>. Every model here is a toy, built so the idea is exactly true in it.
        </p>
      </aside>
      <div className="main">
        <div className="topbar">
          <button className="btn" onClick={() => setOpen(!open)} aria-label="Menu">☰ Lessons</button>
          <Link to="/" style={{ fontWeight: 700, color: "var(--ink)" }}>Interp Atlas</Link>
        </div>
        <main className="content" onClick={() => open && setOpen(false)}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/:slug" element={<LessonPage />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
