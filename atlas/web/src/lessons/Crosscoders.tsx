import { useState } from "react";
import { TeX } from "../components/tex";
import { Callout, Card, Loading, Seg, Slider, Stat } from "../components/ui";
import { fmt, pct, useApi } from "../api";
import { Presets, Swatch } from "./dictHelpers";

/*
 * A crosscoder trained on synthetic "base" and "chat" activations with known features
 * (api/atlas/crosscoder.py): 6 shared, 2 base-only, 2 chat-only, in 8 dimensions. The page plots
 * each live latent's relative decoder norm, coloured by the kind of feature it matched, and lets
 * the reader look for the spurious chat-only latents that the L1 objective can produce.
 */

type Latent = {
  id: number; alive: boolean; rel: number; norm_base: number; norm_chat: number;
  match: number; match_kind: "shared" | "base" | "chat"; match_cos: number; frequency: number;
};
type Result = { latents: Latent[]; kinds: string[]; true_rel: number[]; fvu: number; l0: number; spurious_chat_only: number };
type Cfg = { mode: "l1" | "batchtopk"; latents: number; l1: number; k: number; drift: number };

const DEFAULT: Cfg = { mode: "l1", latents: 16, l1: 0.2, k: 2, drift: 0.2 };
const PRESETS: { label: string; value: Cfg }[] = [
  { label: "Defaults (L1)", value: DEFAULT },
  { label: "Defaults (BatchTopK)", value: { ...DEFAULT, mode: "batchtopk" } },
  { label: "L1, 24 latents, λ 0.5, drift 0.5", value: { mode: "l1", latents: 24, l1: 0.5, k: 2, drift: 0.5 } },
  { label: "Same, BatchTopK", value: { mode: "batchtopk", latents: 24, l1: 0.5, k: 2, drift: 0.5 } },
  { label: "L1, drift 1", value: { ...DEFAULT, drift: 1 } },
];

const KIND_COLOR: Record<string, string> = { base: "var(--warm)", shared: "var(--accent)", chat: "var(--good)" };
const isSpurious = (l: Latent) => l.alive && l.rel > 0.9 && l.match_kind === "shared";

/** "shared 3 (louder in chat)" from the flat feature index the server uses. */
function featureName(i: number, kinds: string[], trueRel: number[]) {
  const kind = kinds[i];
  const idx = kinds.slice(0, i).filter((k) => k === kind).length + 1;
  return `${kind} ${idx}${kind === "shared" && trueRel[i] > 0.52 ? " (louder in chat)" : ""}`;
}

export default function Crosscoders() {
  const [cfg, setCfg] = useState<Cfg>(DEFAULT);
  const set = (patch: Partial<Cfg>) => setCfg((c) => ({ ...c, ...patch }));
  const { data, error, loading } = useApi<Result>("crosscoder", {
    mode: cfg.mode, latents: cfg.latents, l1: cfg.l1, k: cfg.k, drift: cfg.drift,
  }, 250);
  const alive = data ? data.latents.filter((l) => l.alive).sort((a, b) => a.rel - b.rel) : [];
  const activePreset = PRESETS.find((p) => (Object.keys(p.value) as (keyof Cfg)[])
    .every((k) => (k === "l1" && cfg.mode !== "l1") || (k === "k" && cfg.mode !== "batchtopk") || p.value[k] === cfg[k]))?.label;

  return (
    <div className="prose-wide">
      <h2>What did fine-tuning change?</h2>
      <div className="prose">
        <p>
          A chat model starts life as a base model and is then fine-tuned. Most of what it knows it inherited. A
          natural interpretability question is what's <em>new</em>, and what was lost. You could train one SAE on
          each model and try to match their features, but the two dictionaries come out in different orders,
          split differently, and matching them is guesswork.
        </p>
        <p>
          A <b>crosscoder</b> learns a single dictionary for both models at once. Each latent has one encoder
          that reads <em>both</em> models' activations on the same input, and one decoder <em>per model</em>:
        </p>
      </div>
      <Card>
        <TeX block>{String.raw`z = \mathrm{act}\big(W^{\text{base}} x^{\text{base}} + W^{\text{chat}} x^{\text{chat}} + b\big), \qquad
\hat x^{\text{base}} = \sum_i z_i\, d_i^{\text{base}} + c^{\text{base}}, \qquad
\hat x^{\text{chat}} = \sum_i z_i\, d_i^{\text{chat}} + c^{\text{chat}}`}</TeX>
        <p className="small">
          A latent that stands for a concept both models use needs both decoders. A latent for something only the
          chat model represents can do its job with <TeX>{String.raw`d^{\text{base}}_i \approx 0`}</TeX>. So the
          decoder norms tell you which model a latent belongs to, and one number summarizes them:
        </p>
        <TeX block>{String.raw`r_i = \frac{\lVert d_i^{\text{chat}}\rVert}{\lVert d_i^{\text{base}}\rVert + \lVert d_i^{\text{chat}}\rVert}
\qquad\begin{cases} r \approx 0 & \text{base only}\\ r \approx 0.5 & \text{shared} \\ r \approx 1 & \text{chat only}\end{cases}`}</TeX>
      </Card>
      <div className="prose">
        <p>
          Our two "models" are synthetic 8-dimensional activations built from ten known features: six{" "}
          <b style={{ color: KIND_COLOR.shared }}>shared</b>, two <b style={{ color: KIND_COLOR.base }}>base-only</b>{" "}
          (the fine-tune dropped them), and two <b style={{ color: KIND_COLOR.chat }}>chat-only</b> (the fine-tune
          added them). Shared features point in slightly different directions in the two models, by an amount
          set by <b>drift</b>. Three of the six are written 1.3× louder by chat, so their true{" "}
          <TeX>r</TeX> is <TeX>1.3/2.3 \approx 0.57</TeX> rather than 0.5. Each latent is graded by the true
          feature whose (base, chat) decoder pair it matches best.
        </p>
      </div>

      <Card title="Train a crosscoder, read the decoder norms">
        <Presets active={activePreset} onPick={(v: Cfg) => setCfg(v)} items={PRESETS} />
        <div className="controls">
          <Seg label="Sparsity by" value={cfg.mode} onChange={(mode) => set({ mode })}
            options={[{ value: "l1", label: "L1 penalty" }, { value: "batchtopk", label: "BatchTopK" }]} />
          {cfg.mode === "l1"
            ? <Slider label="L1 penalty λ" value={cfg.l1} min={0} max={2} step={0.05} onChange={(l1) => set({ l1 })} format={(v) => v.toFixed(2)} />
            : <Slider label="k (latents per input, on average)" value={cfg.k} min={1} max={6} step={1} onChange={(k) => set({ k })} format={(v) => String(v)} />}
          <Slider label="Latents" value={cfg.latents} min={4} max={24} step={1} onChange={(latents) => set({ latents })} format={(v) => String(v)} />
          <Slider label="Drift between models" value={cfg.drift} min={0} max={1} step={0.05} onChange={(drift) => set({ drift })} format={(v) => v.toFixed(2)} />
        </div>
        <Loading loading={loading} error={error}>
          {data && (
            <>
              <div className="row" style={{ gap: 10, marginBottom: 12 }}>
                <Stat k="FVU" v={pct(data.fvu, 1)} d="both models together" color={data.fvu > 0.2 ? "var(--warm)" : undefined} />
                <Stat k="L0" v={fmt(data.l0, 2)} d="active latents per input" />
                <Stat k="Alive" v={`${alive.length} / ${cfg.latents}`} />
                <Stat k="Spurious chat-only" v={String(data.spurious_chat_only)}
                  d="r > 0.9 but matches a shared feature" color={data.spurious_chat_only > 0 ? "var(--warm)" : "var(--good)"} />
              </div>
              <RelHistogram latents={alive} />
              <div className="small muted" style={{ marginTop: 4 }}>
                Colour = kind of the matched true feature:{" "}
                <Swatch color={KIND_COLOR.base}>base-only</Swatch>
                <Swatch color={KIND_COLOR.shared}>shared</Swatch>
                <Swatch color={KIND_COLOR.chat}>chat-only</Swatch>
                Hollow dots are weak matches (cosine &lt; 0.9); a dark ring marks a spurious chat-only latent.
              </div>
            </>
          )}
        </Loading>
      </Card>

      <Loading loading={loading} error={error}>
        {data && (
          <Card title="Every live latent" caption="Sorted by r. 'Match' is the true feature whose concatenated (base, chat) decoder pair is closest in cosine; 'fires' is the fraction of test inputs on which the latent is active.">
            <div className="scroll-x">
              <table className="tbl">
                <thead><tr><th>latent</th><th>r</th><th>‖d base‖</th><th>‖d chat‖</th><th>match</th><th>cos</th><th>fires</th></tr></thead>
                <tbody>
                  {alive.map((l) => (
                    <tr key={l.id} style={isSpurious(l) ? { background: "var(--warm-soft)" } : undefined}>
                      <td>L{l.id}{isSpurious(l) && <b style={{ color: "var(--warm)" }}> spurious</b>}</td>
                      <td className="mono">{fmt(l.rel, 2)}</td>
                      <td className="mono">{fmt(l.norm_base, 2)}</td>
                      <td className="mono">{fmt(l.norm_chat, 2)}</td>
                      <td style={{ textAlign: "right", color: KIND_COLOR[l.match_kind], whiteSpace: "nowrap" }}>{featureName(l.match, data.kinds, data.true_rel)}</td>
                      <td className="mono" style={{ color: l.match_cos < 0.9 ? "var(--ink-3)" : undefined }}>{fmt(l.match_cos, 2)}</td>
                      <td className="mono">{pct(l.frequency, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </Loading>

      <div className="prose">
        <p>
          At the defaults the picture is what you'd hope for. The two base-only features get latents at{" "}
          <TeX>r \approx 0</TeX>, the chat-only ones at <TeX>r \approx 1</TeX>, and the shared ones cluster at 0.5
          and 0.57. The leftover latents are weak partial copies (hollow dots), spread in between. If this were a
          real pair of models, the latents near 1 are where you'd start reading to learn what chat training added.
        </p>
      </div>

      <h2>When the objective invents differences</h2>
      <div className="prose">
        <p>
          Minder et al. (2025) looked closely at the chat-only latents of an L1-trained crosscoder and found that
          many weren't chat-specific at all. The L1 penalty charges each latent{" "}
          <TeX>{String.raw`\lambda\, z_i \big(\lVert d_i^{\text{base}}\rVert + \lVert d_i^{\text{chat}}\rVert\big)`}</TeX>. That's
          an L1 norm <em>across the two models</em>, and like any L1 penalty it rewards pushing entries all the
          way to zero. They describe two ways this goes wrong:
        </p>
        <ul>
          <li>
            <b>Complete shrinkage.</b> A concept the base model does have, only weakly, gets its base decoder
            pushed to exactly zero. Keeping a small base decoder cost more than the reconstruction error it saved.
            The latent looks chat-only.
          </li>
          <li>
            <b>Latent decoupling.</b> One shared concept ends up spread over two latents: one leaning base, one
            with <TeX>r \approx 1</TeX>. Together they reconstruct the concept in both models, but the second
            looks like a brand-new chat feature.
          </li>
        </ul>
        <p>
          Their fix is to train with <b>BatchTopK</b> instead. There's no L1 penalty: each batch keeps its{" "}
          <TeX>{String.raw`k \times \text{batch size}`}</TeX> largest activations, and the decoder is normalized. Nothing then
          rewards zeroing one model's decoder. They also introduce <b>Latent Scaling</b>, a check that measures
          how much each latent actually explains in each model.
        </p>
      </div>

      <Callout kind="try" label="Check whether it happens here">
        <p>
          The "spurious chat-only" counter above flags live latents with <TeX>r &gt; 0.9</TeX> whose best match
          is a <em>shared</em> feature. At the defaults it reads 0: this toy does not show the failure there. Now
          pick <b>L1, 24 latents, λ 0.5, drift 0.5</b>. The counter goes to 3. In the table you'll find shared
          features covered twice, once by a latent with small <TeX>r</TeX> and once by a "chat-only" latent that
          fires about as often as a real feature. That's latent decoupling. Then pick <b>Same, BatchTopK</b>: the
          counter goes back to 0.
        </p>
      </Callout>
      <div className="prose">
        <p>
          How common is it here? Across a grid of 48 settings per method (8, 16 or 24 latents; drift 0, 0.2, 0.5
          or 1; four values of <TeX>\lambda</TeX> or of <TeX>k</TeX>), the L1 crosscoder produced at least one
          spurious chat-only latent in 11 settings. BatchTopK produced none. It shows up mostly with more latents
          and more drift, which are the conditions where a spare latent is available and a shared feature's
          two versions differ enough to be worth splitting.
        </p>
      </div>
      <Callout kind="warn">
        <p>
          BatchTopK isn't free here either. Look at its histogram at the defaults: the base-only features no
          longer sit cleanly at <TeX>r = 0</TeX>. They drift toward the middle (about 0.4) with weaker matches, and FVU is essentially
          zero at L0 = 2. That's the same "reconstructs without finding" effect as TopK in the SAE lesson, now in
          16 dimensions (8 per model). A toy this small can't settle which objective is better for real models.
          What it does show is that the relative decoder norm is only as trustworthy as the objective that
          produced it.
        </p>
      </Callout>

      <Callout kind="takeaway">
        <p>
          A crosscoder gives two models one shared dictionary, and each latent's decoder norms say which model
          uses it: base-only near 0, shared near 0.5, chat-only near 1. That makes model diffing a matter of
          reading the latents at the ends of the histogram. The training objective shapes those norms, though.
          An L1 penalty can manufacture "chat-only" latents out of shared concepts, so before believing a
          difference, check it with a method that doesn't share the same bias.
        </p>
      </Callout>
      <Callout kind="real">
        <p>
          Lindsey et al. (2024), <em>Sparse Crosscoders for Cross-Layer Features and Model Diffing</em>{" "}
          (Anthropic), introduced crosscoders, both across layers of one model and across models, and used the
          relative decoder norm to find features specific to a fine-tuned model. Minder, Dumas et al. (2025),{" "}
          <em>Robustly identifying concepts introduced during chat fine-tuning using crosscoders</em>, showed
          that many L1 crosscoder "chat-only" latents on Gemma 2 2B are artefacts of complete shrinkage and latent
          decoupling. They proposed Latent Scaling to detect these and BatchTopK training to avoid them, and
          found genuinely chat-specific latents such as ones related to refusal and to the chat template.
        </p>
      </Callout>
    </div>
  );
}

/** A dot histogram of r over [0, 1]: one dot per live latent, stacked within 0.05-wide bins. */
function RelHistogram({ latents }: { latents: Latent[] }) {
  const W = 600, bins = 20, dot = 13;
  const counts = new Array(bins).fill(0);
  const placed = latents.map((l) => {
    const b = Math.min(bins - 1, Math.floor(l.rel * bins));
    return { l, b, level: counts[b]++ };
  });
  const maxStack = Math.max(4, ...counts);
  const top = 12, plotH = maxStack * dot + 8;
  const H = top + plotH + 44;
  const pad = 24;
  const X = (v: number) => pad + v * (W - 2 * pad);
  const base = top + plotH;
  const refs = [{ v: 0, t: "base-only" }, { v: 0.5, t: "shared" }, { v: 1.3 / 2.3, t: "" }, { v: 1, t: "chat-only" }];
  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", maxWidth: W, minWidth: 340 }} role="img"
        aria-label="Histogram of relative decoder norms of the live latents">
        {refs.map((r, i) => (
          <line key={i} x1={X(r.v)} x2={X(r.v)} y1={top - 4} y2={base} stroke="var(--line)" strokeDasharray="3 3" />
        ))}
        <line x1={pad} x2={W - pad} y1={base} y2={base} stroke="var(--ink-3)" />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={base} y2={base + 4} stroke="var(--ink-3)" />
            <text x={X(t)} y={base + 16} fontSize={10} textAnchor="middle" style={{ fill: "var(--ink-3)" }}>{t}</text>
          </g>
        ))}
        {refs.filter((r) => r.t).map((r) => (
          <text key={r.t} x={X(r.v)} y={base + 32} fontSize={10.5} textAnchor={r.v === 0 ? "start" : r.v === 1 ? "end" : "middle"}
            style={{ fill: "var(--ink-2)" }}>{r.t}</text>
        ))}
        {placed.map(({ l, b, level }) => {
          const cx = X((b + 0.5) / bins);
          const cy = base - dot / 2 - 2 - level * dot;
          const col = KIND_COLOR[l.match_kind];
          const weak = l.match_cos < 0.9;
          return (
            <g key={l.id}>
              <title>{`L${l.id}: r = ${l.rel.toFixed(2)}, matches ${l.match_kind} feature (cos ${l.match_cos.toFixed(2)})`}</title>
              {isSpurious(l) && <circle cx={cx} cy={cy} r={dot / 2 + 1.5} fill="none" stroke="var(--ink)" strokeWidth={2} />}
              <circle cx={cx} cy={cy} r={dot / 2 - 1.5} fill={weak ? "var(--surface)" : col} stroke={col} strokeWidth={2} />
            </g>
          );
        })}
      </svg>
    </div>
  );
}
