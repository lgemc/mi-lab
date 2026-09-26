# **Proposal: the second axis — one repository, three domains**

| Field          | Value                                                                                      |
| :------------- | :----------------------------------------------------------------------------------------- |
| **Status**     | **Implemented** — phases 1–4. Phases 5 and 6 are deliberately not built (§XI.7)              |
| **Date**       | September 2026                                                                              |
| **Reference**  | the `src/` layout, a new `.importlinter` contract, a new `src/domains/` root                 |
| **Supersedes** | nothing. The dependency order in `CLAUDE.md` is *extended*, not replaced                     |
| **Depends on** | [0003 — the measurement contract](0003-the-measurement-contract.md), which is what makes it buildable |

---

## 0. What landed, and where it differs from this document

Phases 1–4 are in the tree. Phases 5 (`vision`) and 6 (`world`) are not, by
§XI.7's own rule: a domain is created by the second thing in it, and there is
no vision task yet.

| Decision | What this document said | What was built, and why |
| :--- | :--- | :--- |
| The kernel's door to a domain | not addressed; the layers contract implies nothing may import `src.domains` | `src/plugins.py` names each domain as a **string** and imports it on demand. Every registry here is filled by importing the module that registers into it — `model/adapter.py` did exactly that at the bottom of the file — so without this the contract would have emptied `BACKENDS`, `TASKS` and `EXPERIMENTS`. A fifth contract forbids anything but `src.plugins` from naming a domain |
| §XII.1 `probing/` | `data/dataset.py`, `prompts.py`, `torchdata.py` are `domain: lm` | **They stayed in the kernel.** `LabeledPrompts` is text-shaped and `methods/probing/` needs it, and §VIII.6's rule against designing `LabeledExamples` for an imagined vision dataset applies here too. They are recorded in the frozen allow-list and the open question stays open |
| §XII.3 cross-domain experiments | flagged as needing "one more thought" | `domains` sits above `experiment`, and a domain *registers* an experiment kind rather than the runner reaching for one. `ioi_circuit` moved to `domains/lm/experiments.py`; `circuit_comparison` stayed in the kernel because it is about the techniques and names no task. A genuinely cross-domain study would live above `domains/`, and none exists, so none was built |
| `share/converters/circuit.py` | `share/` is "kernel, permanently"; not reconciled with its `IOIDataset` import | `from_circuit` takes a `CircuitTask` and a `description` dict for the task card. The fields worth quoting — a frame, a corruption, an answer and a distractor — are the task's, so `domains/lm/experiments.py::ioi_card` builds them |
| §VII.2's allow-list "shrinks to nothing" | expected after phase 3 | It shrank from **86 files to 71**. Most of what remains is prose using the words in another sense — "the component vocabulary", "a sentence with a meaning" — and `methods/sheaves/` still indexes token ids honestly. The table is exact and only shrinks deliberately, which was the point |
| `experiment/spec.py`'s IOI validation | not addressed | `data/tasks.py` grew `register_options`/`check_options`: what a corruption may be is registered with the task, and the spec still refuses a bad one before the model loads without learning what a corruption is |

---

## I. The failure this exists to stop

It is not that the repository is big. 139 modules and 22.6k lines under `src/` is
small, and the largest codebases in the world are single repositories: Google keeps
about two billion lines and nine million source files in one, with over 99% of it
visible to every engineer. Size is not the thing that breaks.

What breaks is the **number of concepts a reader must hold to make one change**, and
that is the quantity a second modality multiplies rather than adds to.

Adding a circuit technique today means holding four things at once:

| What you must read | Why |
| :--- | :--- |
| `methods/common/span.py` | the reference every causal number is a fraction of |
| `methods/circuits/techniques.py` | the registry the new technique joins |
| `data/tasks.py` | what a task is required to be |
| `model/adapter.py` | which sites a backend can address |

Four is inside the budget. Now add vision and world models the way they arrive
naturally — a `VisionTask`, a `head_gradients` that differentiates something that is
not a token logit, a `Baselines` that is not a logit difference — and each of those
four grows a modality dimension. `methods/` is 26 modules today; three modalities at
the same coverage is 78 modules in one tree, of which 52 are *not the one you came
for* and all 78 are in the import path of anything that touches the layer.

That is the N×M explosion, and it is the only failure mode here that a bigger disk
does not fix.

### I.1 Where it is already pinned, in this repository, today

The layered order in `CLAUDE.md` is real and it is one-way. It is also **one axis**,
and the axis it does not have is the one about to matter. Three places are already
language-shaped in code that is otherwise about arbitrary computation graphs:

| Site | The leak |
| :--- | :--- |
| `model/adapter.py:195` `CircuitAdapter` | `logits(...) -> [batch, vocab]`, `single_token`, `tokens` — a vocabulary is in the contract |
| `methods/common/span.py:41` `mean_logit_difference` | the readout is hardcoded; `Baselines` carries `io`/`subject` *token ids* |
| `data/tasks.py:42` `CircuitTask` | `clean: List[str]`, `answers(...) -> token ids`, `token_labels` |

Everything downstream of those three — patching, ablation, EAP, EAP-IG, gate
training, faithfulness, completeness, the technique registry, the component algebra —
is arithmetic over an intervention and a scalar. **None of it is about text.** It is
pinned to text by three signatures, and that is a much better position to be in than
it looks.

### I.2 What is explicitly not being proposed

- **Not a repository split.** See §III.
- **Not a rewrite of `methods/`.** The measurement code is nearly domain-free already;
  0003 is a protocol change at three seams, not a reimplementation.
- **Not a generalized model abstraction.** No `GeneralizedBlockLocator` that knows
  about ViTs and decoder stacks at once. §II.4 is the reason, and it is somebody
  else's expensive lesson.
- **Not creating `vision/` and `world/` before there is anything in them.** §XI.

---

## II. What the field already knows, and which part transfers

Nine sources, one transferable claim each. The right-hand column is the only part
this repository has to act on.

| Source | The claim that transfers here |
| :--- | :--- |
| Google's monorepo (CACM 2016) | One repository scales to 2×10⁹ lines. The enabling properties are *visibility, one version, atomic cross-cutting change* — not repository size management |
| Nx module boundaries | Boundaries are enforced by **two orthogonal tag families**: `type:` (which layer) and `scope:` (which domain), with rules over both. This is the missing axis, already named |
| Import Linter | A layered architecture is a *contract you can fail CI on*, not a paragraph in a README |
| HF Transformers' single-file policy, and Modular Transformers | Duplicate at the model edges on purpose; share at the center. When the duplication hurts, generate the edges from a small shard rather than abstracting them |
| Detectron2 LazyConfig vs. MMEngine registries | String registries are right for **closed** vocabularies (backends, techniques, tasks); Python/composed configs are right for **open** ones. This repository already uses both correctly — keep them apart |
| Prisma (vision mech-interp toolkit, 2025) | Vision differs from language on exactly four axes: input modality, tokenization (patches, no dictionary), **decoding (cosine similarity/diffusion, not logits)**, and objective (contrastive). Three of the four are below `methods/`; the fourth is the readout |
| nnterp / TransformerLens / NNsight | The standardization-vs-fidelity trade-off: a normalized interface costs numerical drift against the original implementation. `tests/model/adapter.py`'s golden capture is the mitigation, and each new domain needs its own |
| stable-worldmodel, Jasmine (world modeling platforms) | A world-model stack separates **data layer / model + planner / evaluation**, and adds a component the LM spine has no slot for at all: an *environment*, and a rollout over time |
| Parnas, Ousterhout, Miller | Complexity is concept count under working-memory limits (7±2). A module earns its place by hiding a decision, and a directory a reader must hold at once should be readable at a glance |

### II.1 The monorepo question is settled and it is not the interesting one

The literature is unanimous that repository count is a tooling decision, not an
architecture. What Google actually buys with one repository is a single version of
everything, one place to test, and a refactor that crosses every consumer atomically.
Those are the exact three properties this proposal needs (§III), so the question
"should the lab be one repo" resolves to yes and stops being interesting.

The interesting question is the one Nx answers and the monorepo literature does not:
**inside one repository, what forbids what.**

### II.2 Nx's two tag families are the whole idea

Nx tags every project twice — once with what kind of thing it is (`type:feature`,
`type:util`) and once with which part of the product it belongs to (`scope:sales`,
`scope:shared`) — then enforces rules over both: a type may depend only on lower
types, and a scope may depend only on itself and `shared`.

This repository has the first family, written as a table in `CLAUDE.md` and checked by
a human. It has no second family, and no mechanical check for either.

### II.3 Prisma is the direct evidence for where the cut goes

Prisma is a vision-and-video mechanistic interpretability toolkit that deliberately
kept TransformerLens's hook names and interfaces while replacing the model registry,
the embedding path, and the decoding step. Its stated reason for existing as a
separate toolkit rather than a patch is a list of four differences — modality,
tokenization, decoding, objective — of which three sit *below* the measurement layer
and one *is* the measurement layer's input.

That is a survey of exactly the boundary this proposal has to draw, done by somebody
else, at the cost of a whole library.

### II.4 The duplication rule, and where this repository already obeys it

Transformers copies an attention implementation into every model file on purpose, so
that one file is the complete answer to "what does this model do", and reintroduced
sharing only as a *generator* (Modular Transformers) rather than as an inheritance
tree.

`model/backends/transformers/layout.py` is the same decision already made here: six
lookup lists, all architecture knowledge quarantined. The rule that follows is short.

> **Duplicate at the edges. Share at the center.** A vision backend's layout file
> repeats itself against the decoder one. The thing that measures them does not.

---

## III. Decision 1 — one repository, and three reasons specific to this lab

Keep everything in `mi-lab`. Not from monorepo enthusiasm; from three properties this
repository would actively lose.

1. **The `.mia` format forks the day the repositories do.** `share/schema/vocabulary.py`
   is a *closed set with a version attached*. Two repositories both needing a
   `Component` for a patch site produce two vocabularies that both claim to be v0.x,
   and the format's entire purpose — a result that means the same thing to the next
   reader — is gone. In one repository that addition is one commit that changes the
   validator, the migration, and every converter together.
2. **The comparison is the product.** The thesis in `README.md` is that a result should
   be attributable to the technique rather than to GPT-2. "Does EAP rank sites the same
   way on a ViT as on a decoder" is a question you can only ask if both live under one
   `methods/circuits/techniques.py`. Split, and each side reimplements EAP, and the
   comparison becomes a comparison of two implementations.
3. **The receipts are one suite.** The golden capture, the finite-difference check on
   `head_gradients`, `Decomposition.remainder`, the two-implementations-agree test on
   `edge_gate` — these are the reason the numbers are believable, and they share
   `tests/stubs/model.py`'s one-checkpoint-per-process discipline. Three repositories
   is three CI configurations that drift.

The cost of one repository is that `uv sync` installs everything. §VIII is the answer,
and it is the answer this repository already uses for `serve`.

---

## IV. Decision 2 — the second axis

Today there is one rule. Proposed: three, and the first is unchanged.

> 1. **A layer may import only layers below it**, and never sideways within its own row.
>    *(existing, `CLAUDE.md`)*
> 2. **A domain may import the kernel and itself. Never another domain.**
> 3. **The kernel may not import any domain.**

Where *kernel* means everything that is not under `src/domains/`, and a *domain* is one
subject of study: a model family, its tasks, and how a number comes out of it.

Rule 3 is the load-bearing one, and it is the one that will be violated first, because
violating it is always locally convenient: `methods/circuits/roles.py` naming the four
IOI head movements is a kernel module that knows about one task. Today that is a
documented exception. Under this proposal it is a domain module, and the check is
mechanical.

### IV.1 Why not vertical slices all the way down

The obvious alternative is to slice the whole tree by domain — `src/lm/{model,data,
methods}`, `src/vision/{...}` — which is the vertical-slice architecture, and its
documented failure mode is exactly the one this repository cannot afford: shared
logic duplicated across slices, and a weaker shared model. Three copies of activation
patching is three chances for the vision one to be subtly wrong and no test that
notices.

The hybrid that the same literature recommends — slices for what is genuinely
per-feature, a shared kernel for what is cross-cutting — is what §V describes. The
work is entirely in deciding which is which, and that decision has a test.

---

## V. Where the cut goes: one discriminator

> **A concept belongs in the kernel if it survives the substitution:**
> *model* → **a differentiable function with addressable internal sites, scored by a
> scalar readout.**

Apply it to every package that exists now.

| Package | Verdict | Why |
| :--- | :--- | :--- |
| `core/config.py` | **kernel**, with one caveat | Depth fractions, `d_model`, `n_layers` survive. `Position` (last/mean/all) does not — a ViT has no last token (§XI.3) |
| `core/metrics.py` | **kernel** | AUC, recovery, Spearman, Jaccard, bootstrap, FDR are arithmetic. `logit_difference` is a *readout*, and moves (0003) |
| `telemetry/` | **kernel** | Already imports nothing. A journal does not know what it is journaling |
| `model/adapter.py` | **kernel** (the protocols) | `ModelAdapter` and `CircuitAdapter` are contracts; §I.1's three members are the leak 0003 closes |
| `model/backends/transformers/` | **domain: lm** | `layout.py` is six lookup lists over decoder blocks. Vision gets its own, and repeats itself (§II.4) |
| `model/passes.py` | **kernel** | Hooks, padding, handle cleanup. Needs a trajectory sibling for `world` (§XI.5) |
| `data/dataset.py`, `prompts.py`, `torchdata.py` | **domain: lm** | The format is one text prompt per line. `ActivationDataset` is kernel |
| `data/ioi.py`, `translation.py` | **domain: lm** | Tasks |
| `data/tasks.py` | **splits** | The `CircuitTask` protocol and the registry are kernel; the four registered tasks are `lm` |
| `methods/common/components.py` | **kernel**, with one caveat | `mlp:L`/`heads:L`/`head:L:H` and the set algebra survive. `CANDIDATE_BAND = (0.75, 1.0)` does not (§XI.2) |
| `methods/common/span.py` | **kernel** once the readout is a parameter | `Baselines.recovery` is already pure arithmetic |
| `methods/common/errors.py`, `intervention.py` | **kernel** | A refusal tree and a put-it-in-take-it-out contract |
| `methods/circuits/` — attribution, patching, ablation, search, verify, techniques, comparison, faithfulness, wiring | **kernel** | All of it is intervention plus scalar. This is the finding |
| `methods/circuits/roles.py` | **domain: lm** | Names the four attention movements IOI is built from. Already flagged as the exception in `CLAUDE.md` |
| `methods/probing/` | **kernel** | A linear probe over activations does not care what produced them. `steering.py`'s *fluency* half is `lm` (§XI.4) |
| `methods/knockout/ablate.py`, `cost.py` | **kernel** | Mean ablation and MACs-per-head |
| `methods/knockout/quality.py` | **domain: lm** | BLEU, chrF, COMET are translation metrics. The bootstrap/FDR/`survival_frontier` half is kernel and already lives in `core/metrics.py`'s neighbourhood |
| `methods/knockout/neurons.py` | **kernel** | The down-projection input is a site, not a word |
| `methods/sheaves/` | **kernel** | Gate training is optimization over a mask. `WORD_FRAME` and the pool builder are `lm` |
| `share/` | **kernel, permanently** | The one package that must never fork (§III.1) |
| `serve/` | **kernel** | Already a registry over backbones; `examples.py` is per-domain data read from disk |
| `experiment/` | **kernel** + per-domain registrations | `EXPERIMENTS[kind]` is the extension point and it already works this way |
| `viz/`, `cli/`, `ie/` | **kernel** | They format what the kernel returns. New domains register, they do not fork |

**The result worth stating plainly:** of 26 modules in `methods/`, three are domain
work (`roles`, `quality`'s metric half, `sheaves`' frame constants) and the rest are
kernel the moment the three signatures in §I.1 stop naming a vocabulary. The
architecture is closer to modality-independent than it looks, and this proposal is
mostly about *saying so mechanically* before three modalities make it untrue.

---

## VI. The target layout

```
src/
  core/           config, metrics                        imports nothing
  telemetry/      journal, tracking, observe, results    imports nothing
  model/          adapter (the protocols), passes        -> core
  data/           torchdata, tasks (protocol + registry) -> core
  methods/        common/, circuits/, probing/,          -> core, model, data, telemetry
                  knockout/, sheaves/
  share/          schema/, storage, converters/          -> core, data, methods
  serve/          backbones, circuits, models, app       -> data, methods
  experiment/     spec, run, runner, pipeline            -> core, model, data, methods, share
  viz/                                                   -> core
  domains/        THE SECOND AXIS                        -> everything above, never sideways
    lm/
      backend/    the decoder layout, positions, capture, heads, patching, ...
      tasks.py    ioi, greater_than, induction, agreement, translation
      readout.py  logit difference over two token ids
      data/       prompts format, dataset, ioi, translation corpus
      analysis/   roles.py, quality.py — what is about language in particular
    vision/       backend/, tasks.py, readout.py, data/, analysis/
    world/        backend/, tasks.py, readout.py, data/, environments/, analysis/
  cli/                                                   -> everything
  ie/                                                    -> everything, nothing imports it
```

Three properties of this tree, each deliberate:

- **`src/` stays at 11 entries.** Adding `src/lm/`, `src/vision/`, `src/world/` as
  peers would make 13 flat directories in the one listing every reader starts from,
  and the concept-budget argument in §I would be self-refuted by its own fix. One
  `domains/` entry keeps the top level readable and makes the axis visible in the
  path: you can tell which axis a file is on by whether its path starts with
  `domains/`.
- **Every domain has the same five entries**, so knowing one domain is knowing where
  to look in the next. This is the `backends/transformers/` "one file per question"
  discipline applied one level up.
- **`domains/` sits above `methods/` in the import order, not beside it.** A domain
  imports patching; patching never imports a domain. That is rule 3, and it is the
  whole point.

### VI.1 The move is mostly `git mv`

`model/backends/transformers/` → `domains/lm/backend/` is a rename plus the
registration line. `data/ioi.py` → `domains/lm/data/ioi.py` is a rename. The three
signature changes in §I.1 are the only edits with numbers on the other side of them,
and they are 0003.

---

## VII. Enforcing it

The dependency order is currently a table in `CLAUDE.md` described as "checkable in
one pass over the source". Nothing checks it. Two mechanisms, both matching
enforcement this repository already trusts.

### VII.1 An Import Linter contract

`.importlinter`, run in the `lint` CI job beside ruff — it imports no torch and needs
no checkpoint:

```ini
[importlinter]
root_packages = src

[importlinter:contract:layers]
name = The dependency order is one-way
type = layers
layers =
    src.ie
    src.cli
    src.domains
    src.experiment
    src.serve
    src.share
    src.methods
    src.data | src.model
    src.telemetry | src.core

[importlinter:contract:kernel-knows-no-domain]
name = Nothing above domains may import one
type = forbidden
source_modules =
    src.core
    src.telemetry
    src.model
    src.data
    src.methods
    src.share
    src.experiment
    src.viz
forbidden_modules =
    src.domains

[importlinter:contract:domains-are-independent]
name = A domain may not import another domain
type = independence
modules =
    src.domains.lm
    src.domains.vision
    src.domains.world
```

The `independence` contract is the one with no equivalent in prose today, and it is
the one that decays silently: `vision` importing `lm.readout` "just for the logit
difference" is a five-second edit that quietly makes two domains one.

`src.viz` sits outside the layer stack because it imports only `core`; the `forbidden`
contract is what constrains it.

### VII.2 A grep test, in the style this repository already has

`tests/core/config.py::TestNoHardcodedModelFacts` greps `src/` for widths and fails on
one in a docstring. The same instrument, pointed at modality vocabulary:

```
tests/core/architecture.py::TestKernelNamesNoModality
    grep everything outside src/domains/ and src/cli/ for:
        token, prompt, vocab, logit, tokenizer, word, sentence
    allow-list: model/adapter.py's docstrings until 0003 lands, data/tasks.py's
                protocol docstring, share/schema/vocabulary.py (a wire format is
                allowed to name what it serialized)
```

It reads raw text, so prose fails it too — which is the property that makes the
existing one work. A kernel module whose docstring explains itself in terms of prompts
is a kernel module that is about language.

Both are cheap and both fail in the `lint` job, in under a minute, with no weights.

---

## VIII. Packaging: extras now, workspace members when a domain earns one

Vision brings `timm`/`open_clip`/a video decoder. World brings an environment suite,
a trajectory store, and possibly JAX. None of that belongs in the closure of `uv run
python -m src.cli probe`.

**Now:** one optional extra per domain, the pattern `serve` already establishes.

```toml
[project.optional-dependencies]
serve  = ["fastapi>=0.141.1", "httpx>=0.28.1", "uvicorn[standard]>=0.52.4"]
vision = ["timm>=1.0", "open_clip_torch>=2.30"]
world  = ["gymnasium>=1.0", "lance>=0.20"]
```

with the rule: **a dependency imported by exactly one domain lives in that domain's
extra**, and a domain module that cannot import its dependency raises the same way
`require_circuits` does — naming what to install, not an `ImportError` halfway
through.

**Later:** when a domain's dependency closure genuinely conflicts with another's — the
first time two domains want different pins of the same package — promote `domains/*`
to `uv` workspace members: many `pyproject.toml` files, one `uv.lock`, one `.venv`,
`{ workspace = true }` for the cross-package edges. That is the standard Python
monorepo layout now and it is a mechanical change from this tree, because the
directory boundaries are already the package boundaries.

Do not do it before the conflict exists. A workspace with three members and no
conflicting dependency is four `pyproject.toml` files buying nothing.

---

## IX. What each domain actually costs

The point of the discriminator is that a new domain is small. Here is the bill, per
domain, assuming 0003 has landed.

| Piece | `lm` (exists) | `vision` | `world` |
| :--- | :--- | :--- | :--- |
| `backend/layout.py` | 6 lookup lists | 6 lookup lists over ViT blocks | blocks + a step function |
| `readout.py` | logit difference over 2 ids | cosine similarity to a text/class embedding | return / success over a rollout |
| `tasks.py` | 5 registered | image-space counterfactual pairs | environment + counterfactual initial state |
| `data/` | prompts format | image loading, patch positions | trajectory store |
| golden capture | `tests/stubs/gpt2-small-capture.pt` | **required** — a new one | **required** — a new one |
| What it gets free | — | patching, ablation, EAP, EAP-IG, mask training, search, verify, faithfulness, completeness, comparison, cost, the `.mia` format, the journal, the CLI, `ie`, every chart | same, minus what §XI.5 lists |

The vision column is the argument for the whole proposal. Everything Prisma had to
build to do circuits on a ViT is already in `methods/`, and what it actually needed
that this repo does not have is a layout, a readout, and a task.

The world column is the argument for honesty about the limits: a rollout is not a
forward pass, and §XI.5 says which parts do not transfer rather than pretending they
do.

---

## X. Work plan

Each phase leaves the suite green and is worth doing even if the next never happens.

| # | Phase | Leaves behind |
| :--- | :--- | :--- |
| 1 | Add `.importlinter` with the **layers** contract only, against the tree as it is today | The existing prose invariant becomes a gate. Expect it to fail once and to be interesting when it does |
| 2 | Add `tests/core/architecture.py`, allow-listed to today's reality | The list of leaks, written down and frozen so it cannot grow |
| 3 | **0003**: readout, sites, and task protocols. No files move | `methods/` is domain-free in signature. The allow-list from phase 2 shrinks to nothing |
| 4 | `git mv` the `lm` domain into `src/domains/lm/`. Add the `forbidden` and `independence` contracts | The second axis exists and is enforced with one domain in it |
| 5 | `src/domains/vision/` — layout, readout, one task, one golden capture | The first proof the kernel is actually modality-independent, and the first time the `.mia` vocabulary needs an additive extension (§XI.1) |
| 6 | `src/domains/world/` — and `model/passes.py`'s trajectory sibling first | The rollout question, answered where it belongs rather than by bending `forward_batches` |

Phase 1 and 2 are hours and are worth doing this week regardless of whether the rest
is ever built: they convert the architecture from a document into a test, which is the
only form of it that survives contact with a deadline.

---

## XI. Sharp edges, written down where they bite

1. **The `.mia` vocabulary must be extended additively, never per-domain.** `Component`
   will need a patch site; `NodeComponent` will need whatever a world model's unit is.
   Both are closed sets with a version attached, and `share/migrate.py` plus
   `share/definitions.py` are the existing door. A domain that defines its own local
   enum "for now" has forked the format inside one repository, which is worse than
   forking it across two because nothing announces it. Add a `modality` field to
   `Model` at the same bump, so a reader can tell what they are holding.
2. **`CANDIDATE_BAND = (0.75, 1.0)` is a fact about decoder language models.** It is in
   `methods/common/components.py`, it is kernel by the discriminator, and it is an
   empirical prior about where late-layer language behaviour concentrates. Shipping it
   to `vision` unchanged is shipping an LM prior into another modality wearing the
   clothes of an invariant. It should become a per-domain default with the same
   fraction-not-index discipline.
3. **`Position` (LAST/MEAN/ALL) does not survive the substitution.** A ViT has a CLS
   token, a mean over patches, and a patch grid, and "last" is meaningless for it. This
   is the one field in `core/config.py` that has to move or grow a domain-specific
   vocabulary, and it is used in the wire format (`share/schema/vocabulary.py`), so it
   is a version bump, not a rename.
4. **`strength_sweep`'s fluency half is language.** "Share of non-repeated words" is
   `degeneracy` over text. The *shape* of the steering experiment — effect against
   fluency, both moving together is what the ceiling means — is kernel and worth
   keeping; the fluency measure is a domain readout like any other.
5. **A world model breaks the one-forward-pass assumption, and no amount of protocol
   design hides that.** `model/passes.py::forward_batches` yields once per pass;
   `core/metrics.py::Cost` counts forward passes; every technique's cost claim ("EAP is
   a constant five passes") is quoted in that unit. A rollout is T passes and its
   readout is over a trajectory, so: EAP's gradient is through a recurrence,
   `head_gradients` at one site is now a sum over timesteps, and the honest thing is a
   sibling of `forward_batches` and a `Cost` that counts steps, not a `forward_batches`
   with a `steps=` argument. **Do phase 6's plumbing before phase 6's science.**
6. **Every domain needs its own golden capture or it has no receipt.** The reason
   `tests/model/adapter.py` exists is so "did this change the model?" can be asked
   without first suspecting the capture code. A vision backend with no frozen capture
   is a backend whose numerical drift nobody will notice, and the
   standardization-vs-fidelity trade-off says drift is the expected failure, not an
   unlikely one.
7. **Do not create a domain before it has two things in it.** A `vision/` holding one
   ViT config and no task is not a boundary, it is a directory that makes the tree look
   finished. The rule that works: a domain is created by the second task in it, and
   until then the first one lives in `lm/` with a comment saying why.
8. **Rule 3 will be broken by a chart.** `viz/` and `ie/` want to label an axis with
   token strings, and the shortest path is an import from `domains/lm`. The `forbidden`
   contract catches it; the fix is that labels arrive as data on the payload — which
   `share/schema/payload.py` already requires, since a `Payload` refuses a tensor
   without its axes.

---

## XII. Open questions

1. **Is `probing/` really kernel?** A linear probe over activations is modality-free,
   but the *dataset* format (`LabeledPrompts`, groups, contrast pairs) is text-shaped,
   and `groups` exists because a contrast pair straddling a split makes the AUC measure
   one word. The vision analogue of a minimal pair is a different object. The protocol
   may need to be `LabeledExamples` with an opaque input type — which is 0003's
   `Readout` question asked about `probing/` instead of `circuits/`.
2. **Does `serve/` need a domain axis at all?** `backbones.py` already discovers what
   it can run from the artifact. If a vision circuit is a mask over a ViT, `WeightBackbone`
   may already claim it unchanged. Worth checking before designing anything.
3. **Where do cross-domain experiments live?** "The same technique on a decoder and a
   ViT" imports two domains by definition, which rule 2 forbids. The answer is probably
   that it is an `experiment/` — the layer above, which is allowed to import both — but
   the import-linter layer ordering in §VII.1 puts `domains` above `experiment`, so
   this contract needs one more thought before phase 4.
4. **Do the three domains have the same `spec_hash` discipline?** Invariant 4 says the
   hash covers everything that determines a result. A world-model spec has a seed, an
   environment version and a horizon in it, and the environment version is a
   determinant of the result that lives outside the spec. That may be a new field or a
   new refusal.

---

## Sources

- [Why Google Stores Billions of Lines of Code in a Single Repository — CACM 59(7), 2016](https://cacm.acm.org/research/why-google-stores-billions-of-lines-of-code-in-a-single-repository/)
- [Nx — Enforce Module Boundaries](https://nx.dev/docs/features/enforce-module-boundaries)
- [Import Linter — layers contracts](https://import-linter.readthedocs.io/en/stable/)
- [Hugging Face — ~Don't~ Repeat Yourself (the single model file policy)](https://huggingface.co/blog/transformers-design-philosophy)
- [Hugging Face — Modular Transformers](https://huggingface.co/docs/transformers/en/modular_transformers)
- [MMEngine — Introduction (the registry system)](https://mmengine.readthedocs.io/en/v0.10.0/get_started/introduction.html)
- [detrex — on LazyConfig vs. registry-and-string configs](https://arxiv.org/pdf/2306.07265)
- [Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video](https://arxiv.org/html/2504.19475v3)
- [nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](https://arxiv.org/pdf/2511.14465)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation](https://arxiv.org/pdf/2605.21800)
- [Jasmine: A Simple, Performant and Scalable JAX-based World Modeling Codebase](https://arxiv.org/pdf/2510.27002)
- [Cognitive load is what matters](https://github.com/zakirullin/cognitive-load)
- [A Philosophy of Software Design — deep modules, information hiding](https://www.informit.com/articles/article.aspx?p=3203545&seqNum=4)
- [When to Choose Vertical Slice Architecture](https://milanjovanovic.tech/blog/when-to-choose-vertical-slice-architecture)
- [How to set up a Python monorepo with uv workspaces](https://pydevtools.com/handbook/how-to/how-to-set-up-a-python-monorepo-with-uv-workspaces/)
- [Creating and discovering plugins — Python Packaging User Guide](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)
- [More Rigorous Software Engineering Would Improve Reproducibility in Machine Learning Research](https://arxiv.org/pdf/2502.00902)
