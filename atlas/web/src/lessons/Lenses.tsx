import { useState } from "react";
import { Lines } from "../components/charts";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Select, Slider, Stat } from "../components/ui";
import { pct, useApi } from "../api";

/*
 * A six-block residual network with a silent middle step, by construction (api/atlas/lens.py).
 * The lenses are fitted on the server from a sampled corpus: the J-lens is a mean Jacobian taken
 * by central differences through the real blocks, the tuned lens a least-squares affine map.
 */

type LensKey = "logit" | "jlens" | "tuned";
type Read = { token: string; p: number };
type Layer = {
  layer: number; after: string; top: Record<LensKey, Read[]>;
  tracked: Record<LensKey, Record<string, number>>; workspace: Read[]; pending: number;
};
type Probe = { subject: string; country: string; capital: string; layers: Layer[]; answer: Read[]; tracked: string[] };
type Meta = {
  subjects: string[]; countries: string[]; capitals: string[]; vocab: string[];
  blocks: { name: string; does: string }[]; d_model: number; corpus_size: number; corpus_mixes: number[];
};

const SUBJECTS = ["Curie", "Turing", "Hokusai", "Kafka", "Gandhi"];
const COUNTRIES = ["Poland", "England", "Japan", "Czechia", "India"];
const CAPITALS = ["Warsaw", "London", "Tokyo", "Prague", "Delhi"];
const MIXES = [0, 0.03, 0.08, 0.25];
const LENS_NAME: Record<LensKey, string> = { logit: "Logit lens", jlens: "J-lens", tuned: "Tuned lens" };

/** Colour a token by what kind of thing it is, so the grid reads at a glance. */
function kind(t: string): { bg: string; label: string } {
  if (SUBJECTS.includes(t)) return { bg: "var(--surface-2)", label: "person" };
  if (COUNTRIES.includes(t)) return { bg: "var(--gold-soft)", label: "country" };
  if (CAPITALS.includes(t)) return { bg: "var(--good-soft)", label: "capital" };
  return { bg: "transparent", label: "" };
}

function Chip({ r, muted = false }: { r: Read; muted?: boolean }) {
  const k = kind(r.token);
  return (
    <span style={{
      display: "inline-block", padding: "2px 7px", borderRadius: 6, background: k.bg,
      border: "1px solid var(--line)", whiteSpace: "nowrap", color: muted ? "var(--ink-3)" : "var(--ink)",
    }}>
      <b>{r.token}</b> <span className="mono muted">{muted ? "" : pct(r.p, 0)}</span>
    </span>
  );
}

function Legend() {
  return (
    <div className="row small" style={{ gap: 10, marginTop: 8 }}>
      {[["Curie", "person"], ["Poland", "country"], ["Warsaw", "capital"]].map(([t, l]) => (
        <span key={l} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 12, height: 12, borderRadius: 3, background: kind(t).bg, border: "1px solid var(--line)" }} />{l}
        </span>
      ))}
    </div>
  );
}

function LensGrid({ data, lenses = ["logit", "jlens", "tuned"], editLayer }: {
  data: Probe; lenses?: LensKey[]; editLayer?: number;
}) {
  return (
    <div className="scroll-x">
      <table className="tbl" style={{ minWidth: 520 }}>
        <thead>
          <tr>
            <th>After</th>
            <th style={{ textAlign: "left" }}>Actually in W</th>
            <th>p</th>
            {lenses.map((l) => <th key={l} style={{ textAlign: "left" }}>{LENS_NAME[l]}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.layers.map((L) => (
            <tr key={L.layer} style={editLayer === L.layer ? { background: "var(--warm-soft)" } : undefined}>
              <td className="small">
                <span className="mono muted">{L.layer}</span> {L.after}{editLayer === L.layer ? " ✎" : ""}
              </td>
              <td style={{ textAlign: "left" }}>
                {L.workspace[0].p > 0.3 ? <Chip r={L.workspace[0]} muted /> : <span className="muted small">empty</span>}
              </td>
              <td className="mono small">{L.pending > 0.5 ? "on" : "off"}</td>
              {lenses.map((l) => (
                <td key={l} style={{ textAlign: "left" }}><Chip r={L.top[l][0]} /></td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TrackedLines({ data, lenses, extra = false }: { data: Probe; lenses: LensKey[]; extra?: boolean }) {
  const xl = data.layers.map((l) => String(l.layer));
  const [subj, country, capital, pc, pcap] = data.tracked;
  return (
    <div className="grid3" style={{ marginTop: 10 }}>
      {lenses.map((l) => (
        <div key={l}>
          <div className="small"><b>{LENS_NAME[l]}</b></div>
          <Lines width={300} height={180} xLabels={xl} yMin={0} yMax={1} format={(v) => pct(v, 0)}
            series={[
              { name: subj, values: data.layers.map((L) => L.tracked[l][subj]), color: "var(--ink-3)" },
              { name: country, values: data.layers.map((L) => L.tracked[l][country]), color: "var(--gold)" },
              { name: capital, values: data.layers.map((L) => L.tracked[l][capital]), color: "var(--good)" },
              ...(extra && pc ? [
                { name: pc, values: data.layers.map((L) => L.tracked[l][pc]), color: "var(--gold)", dashed: true },
                { name: pcap, values: data.layers.map((L) => L.tracked[l][pcap]), color: "var(--good)", dashed: true },
              ] : []),
            ]} />
        </div>
      ))}
    </div>
  );
}

export default function Lenses() {
  const meta = useApi<Meta>("lens/meta");
  const subjects = meta.data?.subjects ?? SUBJECTS;
  const [subject, setSubject] = useState("Curie");
  const [mixI, setMixI] = useState(2);
  const [patchLayer, setPatchLayer] = useState(2);
  const i = SUBJECTS.indexOf(subject);
  const others = COUNTRIES.filter((c) => c !== COUNTRIES[i]);
  const [patchCountry, setPatchCountry] = useState("Japan");
  const pc = others.includes(patchCountry) ? patchCountry : others[0];

  const base = useApi<Probe>("lens/probe", { subject, two_hop: 0 });
  const mixed = useApi<Probe>("lens/probe", { subject, two_hop: MIXES[mixI] });
  const edited = useApi<Probe>("lens/probe", { subject, two_hop: 0, patch_country: pc, patch_layer: patchLayer });

  const country = COUNTRIES[i], capital = CAPITALS[i];
  const pcCapital = CAPITALS[COUNTRIES.indexOf(pc)];
  const ans = edited.data?.answer[0]?.token;
  const regime = patchLayer <= 2 ? "early" : patchLayer <= 4 ? "middle" : "late";

  return (
    <div className="prose-wide">
      <h2>A thought the model never says</h2>
      <div className="prose">
        <p>
          Ask: <em>“The capital of the country where Curie was born is …”</em>. To answer <b>Warsaw</b>, you first
          have to get to <b>Poland</b>. But “Poland” is never the output. It is an intermediate, used and then
          dropped. If a model does the same, can we see “Poland” in its activations, and at which layer?
        </p>
        <p>
          A <b>lens</b> is a way of reading a hidden layer in the model's own vocabulary: take the residual stream
          partway through and ask “if the model had to speak now, what would it say?”. This lesson compares three
          lenses on a toy model that has a silent intermediate step <em>by construction</em>, so we know the right
          answer at every layer.
        </p>
      </div>

      <h2>The toy</h2>
      <div className="prose">
        <p>
          The residual stream at the last position has four parts. {meta.data && <>It has {meta.data.d_model} dimensions in total.</>}
        </p>
      </div>
      <Card>
        <div style={{ display: "flex", borderRadius: 6, overflow: "hidden", border: "1px solid var(--line)", fontSize: 13 }}>
          {[
            ["T", "input token", "var(--surface-2)", 17],
            ["W", "workspace (rotated)", "var(--gold-soft)", 17],
            ["O", "output basis", "var(--good-soft)", 17],
            ["p", "pending", "var(--warm-soft)", 3],
          ].map(([k, d, c, w]) => (
            <div key={k as string} style={{ flex: `${w} 0 0`, background: c as string, padding: "6px 8px", borderRight: "1px solid var(--line)", minWidth: 0 }}>
              <b className="mono">{k}</b> <span className="small muted" style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d}</span>
            </div>
          ))}
        </div>
        <ul className="small" style={{ paddingLeft: 20, marginBottom: 0 }}>
          <li><b>T</b> holds the input token (the subject, already moved to the last position by attention we don't model).</li>
          <li><b>W</b> is a <b>workspace</b>: whatever the model is currently thinking about. It is stored in a private,
            randomly rotated basis <TeX>{String.raw`R\,e_x`}</TeX>, so nothing about it looks like a token from outside.</li>
          <li><b>O</b> is the output basis. The unembedding reads O loudly, T faintly (an echo of the input), and W not at all.</li>
          <li><b>p</b> is one flag: “a hop is still pending”.</li>
        </ul>
      </Card>
      <Card title="Six blocks, each a handful of ReLUs">
        <div className="scroll-x">
          <table className="tbl" style={{ maxWidth: 640 }}>
            <thead><tr><th>Block</th><th style={{ textAlign: "left" }}>What it does</th></tr></thead>
            <tbody>
              {(meta.data?.blocks ?? []).map((b, j) => (
                <tr key={j}><td><span className="mono muted">{j + 1}</span> {b.name}</td><td style={{ textAlign: "left" }}>{b.does}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="caption">
          So Poland sits in W after block 1, Warsaw replaces it after block 3, and only block 5 copies anything into O.
          The noise blocks keep the model from being unrealistically clean.
        </div>
      </Card>

      <h2>Three ways to read a layer</h2>
      <div className="prose">
        <p>
          Let <TeX>h_\ell</TeX> be the residual after block <TeX>\ell</TeX>, <TeX>h_L</TeX> the final one, and{" "}
          <TeX>{String.raw`U\,\mathrm{norm}(\cdot)`}</TeX> the model's own last step (a norm, then the unembedding). The
          three lenses differ only in what they do to <TeX>h_\ell</TeX> before that last step.
        </p>
      </div>
      <Card>
        <p className="small" style={{ marginTop: 0 }}>
          <b>Logit lens</b> (nostalgebraist, 2020): pretend the layer is the last one.
        </p>
        <TeX block>{String.raw`p_\ell = \mathrm{softmax}\big(U\,\mathrm{norm}(h_\ell)\big)`}</TeX>
        <p className="small">
          <b>J-lens</b>, the Jacobian lens (Anthropic, 2026, “Verbalizable representations form a global workspace in
          language models”): first carry <TeX>h_\ell</TeX> to the last layer with the Jacobian of the rest of the
          network, averaged over a corpus <TeX>\mathcal D</TeX>.
        </p>
        <TeX block>{String.raw`J_\ell = \mathbb{E}_{x\sim\mathcal D}\!\left[\frac{\partial h_L}{\partial h_\ell}\right]`}</TeX>
        <TeX block>{String.raw`p_\ell = \mathrm{softmax}\big(U\,\mathrm{norm}(J_\ell\, h_\ell)\big)`}</TeX>
        <p className="small">
          <b>Tuned lens</b> (Belrose et al., 2023): learn a map per layer. The original trains an affine translator to
          match the final distribution; here it's a least-squares affine map from <TeX>h_\ell</TeX> to{" "}
          <TeX>h_L</TeX> on the same corpus.
        </p>
        <TeX block>{String.raw`(A_\ell, b_\ell) = \arg\min_{A,b} \sum_{x\in\mathcal D} \big\|A\,h_\ell + b - h_L\big\|^2`}</TeX>
        <TeX block>{String.raw`p_\ell = \mathrm{softmax}\big(U\,\mathrm{norm}(A_\ell h_\ell + b_\ell)\big)`}</TeX>
        <p className="small muted" style={{ marginBottom: 0 }}>
          The corpus here is {meta.data?.corpus_size ?? 400} sampled residual states: most of the time the model has
          something in W and simply says it (pending off). The rest of the time it has nothing in mind. We'll add
          two-hop questions to the corpus later.
        </p>
      </Card>

      <h2>What each lens sees</h2>
      <Card>
        <div className="controls">
          <Select label="Person" value={subject} options={subjects} onChange={setSubject} />
        </div>
        <div className="small" style={{ marginBottom: 8 }}>
          The capital of the country where <b>{subject}</b> was born: {subject} → <b>{country}</b> → <b>{capital}</b>.
          Lenses fitted on a corpus with no two-hop questions.
        </div>
        <Loading loading={base.loading} error={base.error}>
          {base.data && (
            <>
              <LensGrid data={base.data} />
              <Legend />
              <p className="small muted">Each cell shows the lens's top token and its probability. “Actually in W” is ground truth, read with the private basis R that no lens is given.</p>
              <TrackedLines data={base.data} lenses={["logit", "jlens", "tuned"]} />
            </>
          )}
        </Loading>
      </Card>
      <div className="prose">
        <p>
          <b>The logit lens is stuck on “{subject}” until layer 5.</b> It can only see what is already in the output
          basis. Until the verbalizer copies W into O, the only thing the unembedding picks up is the faint echo of
          the input token. {country} (layers 1 and 2) and then {capital} (from layer 3) are sitting in the residual
          stream; the logit lens just has no way to read them.
        </p>
        <p>
          <b>The J-lens reads “{country}” at layers 1 and 2</b>, exactly when {country} is what the model is holding.
          Why can it see into W when the unembedding can't? Because of the corpus. In most of the states it was
          averaged over, the pending flag is off and the verbalizer is live, so the average Jacobian from layer{" "}
          <TeX>\ell</TeX> to the output <em>contains the verbalizer</em>, the map <TeX>{String.raw`R^\top`}</TeX> from
          workspace to output. The J-lens reports what the model would say if it said what it is holding. On this
          question the model is not about to say it (pending is on), but the lens reads the content anyway.
        </p>
        <p>
          The tuned lens, fitted on the same corpus, learns much the same map from data and also reads {country}.
        </p>
      </div>

      <h2>A lens is a model of the model</h2>
      <div className="prose">
        <p>
          Both the J-lens and the tuned lens are <em>fitted</em>. They only know the model through the corpus they
          were fitted on. So change the corpus: mix in a share of two-hop questions and refit both.
        </p>
      </div>
      <Card>
        <Slider label="Share of two-hop questions in the fitting corpus" value={mixI} min={0} max={3} step={1}
          onChange={setMixI} format={(v) => pct(MIXES[v], 0)} />
        <Loading loading={mixed.loading} error={mixed.error}>
          {mixed.data && (
            <>
              <LensGrid data={mixed.data} lenses={["jlens", "tuned"]} />
              <TrackedLines data={mixed.data} lenses={["jlens", "tuned"]} />
            </>
          )}
        </Loading>
      </Card>
      <div className="prose">
        <p>
          At <b>8%</b> the tuned lens stops reporting {country}. It reads <b>{capital}</b> from layer 0, before a
          single block has run, when the only thing in the residual is “{subject}” and a pending flag. The model has
          not computed {capital}. The lens has: a least-squares fit that has seen enough two-hop questions learns that
          “{subject}, pending” <em>ends up as</em> {capital}, and predicts it. The J-lens drifts the same way, more slowly: at 8% it already puts about half its weight on {capital} at layer 0, and at 25% it reads {capital} at layers 1 and 2 as well. (At
          3% the tuned lens even reports a confident wrong capital at layer 0: a fit extrapolating from too few
          examples.)
        </p>
        <p>
          None of this is a bug. A lens trained to predict the final output from layer <TeX>\ell</TeX> is rewarded
          for predicting the final output, whether or not layer <TeX>\ell</TeX> has computed it yet. It reports
          computation that has not happened.
        </p>
      </div>
      <Callout kind="warn" label="Reading ahead">
        <p>
          A lens that shows the answer early may be showing you its own prediction, not the model's state. The more
          a lens learns, the more it can “know” that the model doesn't. That's one argument for lenses with little
          freedom, like the logit lens (none) or a Jacobian averaged over neutral text.
        </p>
      </Callout>

      <h2>Reading is not using</h2>
      <div className="prose">
        <p>
          The J-lens says {country} is in the workspace at layer 2. Is the model <em>using</em> it, or does the lens
          just happen to decode something there? The only way to find out is to intervene: overwrite the workspace
          with a different country partway through and see whether the answer follows.
        </p>
      </div>
      <Card>
        <div className="controls">
          <Select label="Write this country into W" value={pc} options={others} onChange={setPatchCountry} />
          <Slider label="…right after block" value={patchLayer} min={1} max={6} step={1} onChange={setPatchLayer}
            format={(v) => `${v} (${(meta.data?.blocks[v - 1]?.name) ?? ""})`} />
        </div>
        <Loading loading={edited.loading} error={edited.error}>
          {edited.data && (
            <>
              <div className="row" style={{ marginBottom: 10 }}>
                <Stat k="Unedited answer" v={capital} />
                <Stat k="Answer after the edit" v={ans ?? "—"}
                  color={ans === capital ? "var(--ink-2)" : "var(--warm)"}
                  d={edited.data.answer[0] ? pct(edited.data.answer[0].p, 0) : undefined} />
              </div>
              <LensGrid data={edited.data} lenses={["logit", "jlens"]} editLayer={patchLayer} />
              <TrackedLines data={edited.data} lenses={["jlens"]} extra />
              <p className="small" style={{ marginBottom: 0 }}>
                {regime === "early" && <>Edited before hop 2: the model reads {pc} out of W, looks up its capital, and
                  says <b>{pcCapital}</b>. The workspace content is causally used.</>}
                {regime === "middle" && <>Edited after hop 2: the pending flag is already off, so no further lookup
                  happens. The verbalizer copies whatever is in W, and the model says <b>{pc}</b>, the country itself.
                  Same edit, different answer, because the computation has moved on.</>}
                {regime === "late" && <>Edited after the verbalizer has spoken: {capital} is already in the output basis,
                  so rewriting W changes nothing the unembedding reads. The answer stays <b>{capital}</b>.</>}
              </p>
            </>
          )}
        </Loading>
      </Card>
      <div className="prose">
        <p>
          Drag the layer slider through all six positions. The same edit has three different effects depending on
          when it happens, and that is how an intervention tells you what a representation is <em>for</em>. The lens
          said {country} was there at layer 2; the edit shows that the model reads it from there and nowhere else.
          Only both together make the claim “the model thinks {country} on the way to {capital}”.
        </p>
      </div>

      <h2>Beyond the Jacobian</h2>
      <div className="prose">
        <p>
          The J-lens averages plain gradients, and in a real model gradients through attention and nonlinearities can
          be noisy or saturated (the same problem that fooled EAP in the previous lesson). The <b>R-lens</b> replaces
          the ordinary backward pass with <b>RelP</b>, a relevance-propagation backward pass that uses LRP-style rules
          for layer norms, nonlinearities and attention, and then decodes through that modified Jacobian. It is not
          implemented in this toy, whose blocks are simple enough that the plain Jacobian works.
        </p>
        <p>
          How do you compare lenses fairly? <b>WorkspaceBench</b> poses questions whose answers require a silent
          intermediate, like the country here, and asks whether a readout finds that intermediate in the model's
          hidden layers without the model ever saying it.
        </p>
      </div>

      <Callout kind="takeaway">
        <p>
          The logit lens reads only what is already in the output basis, so it misses intermediate thoughts
          entirely. The J-lens and the tuned lens can see into the workspace because they are fitted to the model's
          behaviour on a corpus. For the same reason, they can report what the model is <em>about to</em> compute
          instead of what it holds. A lens is a model of the model. Treat its readings as hypotheses, and confirm them
          by intervening.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          The logit lens (nostalgebraist, 2020) works surprisingly well in GPT-2's later layers and poorly in early
          layers and in other model families. The tuned lens (Belrose et al., 2023) fixes much of that with a learned
          affine translator per layer, trained to match the final distribution. The Jacobian lens (Anthropic, 2026)
          reports that verbalizable content is held in a shared “workspace” readable across layers. Its averaged
          Jacobian is fitted on generic text, which bounds how far it can read ahead. The R-lens swaps in a RelP
          backward pass. WorkspaceBench scores readouts on finding silent intermediates. Real two-hop reasoning is
          messier than this toy: models often skip the hop, shortcut through memorized pairs, or hold the bridge
          entity only weakly. Real workspaces are not a clean named block either.
        </p>
      </Callout>
    </div>
  );
}
