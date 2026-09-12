# **Proposal: circuit overlap as a predictor of interference**

| Field          | Value                                                                                     |
| :------------- | :---------------------------------------------------------------------------------------- |
| **Status**     | Proposed — not implemented                                                                 |
| **Date**       | September 2026                                                                             |
| **Reference**  | `methods/circuits/edges.py` (exists), `domains/lm/data/translation.py` (needs the pair as a parameter), new `domains/lm/experiments/overlap.py` |
| **Enables**    | the first claim in this repository that is about *training* rather than about a frozen model |
| **Supersedes** | nothing. The ES→EN study keeps its artifacts; this makes it the first of four              |

---

## I. The claim

> A circuit is currently the output of this repository. This proposal makes it an
> *input*: run the same discovery on four language pairs, and ask whether the **overlap**
> between two pairs' circuits predicts how much fine-tuning on one damages the other.

The unit of analysis stops being "the circuit of a task" and becomes "the overlap structure
of circuits across a family of tasks". That is a different object, it is cheap to build
from what is already here, and it carries a claim nobody appears to be making.

---

## II. Why now, and why this repository specifically

Three things landed that make it a small piece of work rather than a programme.

| What exists | Where | What it removes from this proposal |
| :--- | :--- | :--- |
| `eap`, `eap_ig`, `eap_gp` over edges | `methods/circuits/edges.py` | the entire discovery method. Nothing to build |
| `--edges-only`, 14,140 edge gates on Qwen3-1.7B in place of 1.409e9 | 2026-09-04 | the capacity failure that made the weight sweep unreadable |
| ES→EN at 26.3% of edges: held-out 1.000, train 1.000, first token 0.624 against the model's 0.629, random-edge control passing | 2026-09-04 | the "does this work on translation at all" question |

So the marginal cost of the *first* claim below is four runs of a method that is already
written, against data this repository already knows how to load.

---

## III. The two claims, in order of cost

### III.1 Claim A — the staged story has an edge-level signature

The multilingual line (Wendler 2402.10588, Dumas 2411.08745, Tang 2402.16438, Zhang
2502.11806) describes translation as **staged**: a language-agnostic concept space in the
middle layers, language identity separable from content, language-specific encode/decode
concentrated in the first and last layers. Every one of those results is stated over
*representations* — probes, subspace-confined path patching, neuron sets, transcoder
features.

If that description is true of the *computation* and not only of the representations, it
has a signature in an edge graph:

> Edges shared across ES/FR/DE/ZH→EN concentrate in the middle layers; pair-exclusive
> edges concentrate at the first and last.

Four EAP-IG runs and one histogram over layer index settle it. Either outcome pays:

- **Match** — a CLT/SAE-derived finding replicated on a base that costs two passes per
  prompt and needs no dictionary. That is a result about method cost, which is the kind
  of result this repository is already producing.
- **Mismatch** — evidence that the representational claim does not cash out as
  computational structure. This is the more interesting outcome and it is close in spirit
  to Sheng & Fu (Jul 2026): circuit claims move with what is extracted.

### III.2 Claim B — per-layer overlap predicts interference

Fine-tune on pair B; measure degradation on pair A; ask whether degradation is predicted
by the overlap of A's and B's circuits. Four pairs give twelve ordered pairs, and
typological distance (ES/FR romance, DE germanic, ZH sinitic) gives an independent prior
on the ordering, so the relation can be wrong in a visible way.

This is the claim. The continual-learning literature allocates masks by capacity (PackNet,
Piggyback) or by sparsity structure (SSDE, 2503.05246), never by a measured circuit; the
circuits literature measures overlap between **methods** (Hanna, Pezzelle & Belinkov,
2403.17806) and reads it as validation, not as a prediction about training.

---

## IV. Why this route and not the completeness route

The obvious bridge from this repository to continual learning is DiscoGP's `complete`
term: a mask whose complement fails has *localised* a behaviour rather than merely found
somewhere it survives, and that is the property that should predict non-interference.

That route is blocked, and the measurements are this repository's own.

| Obstacle | Receipt |
| :--- | :--- |
| `complete` is satisfiable by margin collapse | Across all five points of the GPT-2/IOI sparsity sweep (2026-09-02) the term sat pinned at its ln 2 floor, 0.704–0.707, while complement *accuracy* wandered 0.378–0.465. Cross-entropy to a uniform target is minimised by tying the logits, not by making the complement ignorant |
| The complement measures destruction, not localisation | Complement accuracy 0.425–0.527 against a chance floor of 0.477. Zeroing 85% of a model lands at chance whichever 85% you choose |
| Redundancy says a healthy complement *should* work | The Hydra effect (McGrath 2307.15771). A completeness term rewards breaking shared machinery, and redundancy grows with scale |

Measuring interference **behaviourally** — pair A's score after training on pair B —
sidesteps all three. No complement is ablated, so no destruction is measured, so no
reference distribution has to be defended (0003's lesson, and
`ablation-reference-distribution`'s). Completeness demotes from prerequisite to optional
hypothesis: once an overlap↔interference relation exists, ask separately whether a
completeness-style quantity adds predictive power over raw overlap.

---

## V. What has to be built

Small, and mostly parametrisation.

### V.1 The language pair becomes a parameter

`domains/lm/data/translation.py` is Spanish-to-English in its constants, not only in its
docstring:

```python
WORD_FRAME   = "Spanish:{source}\nEnglish:{answer}"
FEW_SHOT_FORM = "Spanish: {source}\nEnglish:"
```

The frame becomes `"{src_lang}:{source}\n{tgt_lang}:{answer}"` with the pair carried on
the task, and `load_word_pairs` / `load_pairs` take a pair identifier. The counterfactual
form (`Text:`/`Nothing:`, shot targets rotated off their sources) is pair-independent
already and does not move.

**The constraint that binds:** the word-level task's corruption is a swap of the source
word inside one frame, which is what makes the clean/corrupt alignment hold. That property
has to survive the change of pair, and for ZH it may not — a Chinese source word need not
occupy one token. **Verify tokenisation alignment per pair before committing to four**,
and drop to three if ZH cannot be aligned.

### V.2 Overlap as an artifact

An `EdgeSet` comparison: intersection over union, and the same quantity bucketed by the
layer of the edge's *source*. It belongs beside `methods/circuits/comparison.py`, which
already compares circuits, and it is the only genuinely new computation in the proposal.

### V.3 The interference run

Sequential fine-tune, held-out score on every other pair before and after. This is the
first thing in the repository that trains the model rather than a mask, so it needs its
own artifact shape and its own place in the measurement contract — the degradation number
is a `Score` like any other, and it should go through `core/readout.py` rather than beside it.

---

## VI. What would kill it

Recorded now, because three of the four are cheap to check and two of them come before any
fine-tuning compute is spent.

| # | Kill condition | When it is visible | Cost to check |
| :--- | :--- | :--- | :--- |
| 1 | Overlap is near-total across every pair — no range in the independent variable | after §V.2, before any training | four EAP-IG runs |
| 2 | Interference is uniform — every pair damages every other equally | after the first two fine-tunes | two fine-tunes |
| 3 | Graphs are not comparable across pairs: different prompt geometry, different chance floors, so "overlap" is an artefact | at design time | matched skeleton + per-pair `recovered` normalisation |
| 4 | Twelve ordered pairs is too few to separate a relation from noise | always | treat it as a shape, not a coefficient — the same caveat that sank the five-point sparsity sweep |

And one that is not a kill but a warning. **ES→EN at 26.3% of edges scored held-out 1.000.**
A metric at its ceiling has no room to show degradation, so the interference measurement in
§V.3 cannot use it. The quality side (sentence-level BLEU/chrF/COMET, `TextQuality` from
0003) is where degradation has to be read, and the word-level task stays what the *circuit*
is discovered on. Those are two different numbers on two different data shapes and the
proposal needs both.

---

## VII. Steps

| # | Step | Gate |
| :--- | :--- | :--- |
| 1 | Language pair as a parameter in `domains/lm/data/translation.py`; tokenisation alignment verified per pair | ZH may be dropped here |
| 2 | EAP-IG edge circuits for ES/FR/DE(/ZH)→EN, same skeleton, same shot count, `recovered` units, random-edge control each | the existing ES→EN run is step 2a and is done |
| 3 | `EdgeSet` overlap, scalar and per-source-layer | kill condition 1 |
| 4 | Claim A: overlap-by-layer against the staged prediction | first publishable outcome, either way |
| 5 | Sequential fine-tune, quality-side degradation matrix over ordered pairs | kill condition 2 |
| 6 | Claim B: per-layer overlap against degradation; typological distance as the control variable | the contribution |
| 7 | Crosscoder / SAE read-out *only* on pairs that measurably interfere — what was lost, in feature terms (Masip 2601.22012: capacity vs. readout) | optional, and last |

Steps 1–4 stand alone as a result. Steps 5–7 are the reason to do them.

---

## VIII. Open questions

1. Is per-layer overlap a better predictor than a scalar overlap? The staged prior says
   middle-layer overlap should matter and edge-layer overlap should not — which makes the
   scalar version the null hypothesis, not the default.
2. Does typological distance predict interference *beyond* what overlap predicts? If it
   does, the circuit is not carrying the whole story and the proposal's claim is weaker
   than it sounds.
3. Does the relation survive on-policy fine-tuning (SDFT, on-policy distillation), where
   forgetting is already known to be smaller? If overlap only predicts interference under
   SFT, that is a narrower and more honest claim.
4. Is there a MIB-style task-model pair to be built here? Every MIB task is within-prompt
   contrastive with a single-token answer; translation is neither, and the gap is
   well-defined work independent of everything above.

---

## IX. Prior discussion

Written up in the wiki as two pages:
`wiki/concepts/mechanistic-interpretability/circuit-overlap-interference.md` (this claim)
and `translation-circuit-base-choice.md` (why EAP-IG over edges rather than transcoders or
SAE features, and what MIB's null result on SAEs does and does not license).
