# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project uses `uv` (there is a `uv.lock`; the `.venv` is what `uv run` uses). The bare
`python` on PATH is a different interpreter and will fail on imports — always go through `uv run`.

```bash
uv sync                                            # install/refresh the environment
uv run python -m src.cli --help                    # the CLI (also installed as the `mi-lab` script)
uv run python -m src.ie                            # the TUI explorer (`ie`); see Explorer below
uv run python -m src.app --multirun model=gpt2-small,pythia-70m   # Hydra sweeps only
```

### Tests

Everything under `src/` is a namespace package except `src/cli/commands/viz/`, which needs an
`__init__.py` to hold its Typer app. Test modules are named after the module they test
(`tests/spec.py`, not `tests/test_spec.py`), so **`unittest discover` does not work**. Name the
modules explicitly:

```bash
# everything: 435 tests, ~90s on a CPU and ~26s on a GPU (the online half needs GPT-2 small)
uv run python -m unittest tests.config tests.dataset tests.metrics tests.spec tests.run \
    tests.prompts tests.torchdata tests.ioi tests.tasks tests.artifact tests.ie tests.probing \
    tests.runner tests.adapter tests.circuits tests.discovery tests.comparison \
    tests.faithfulness tests.edges tests.sheaves tests.telemetry

# offline subset: 258 tests, no checkpoint needed, seconds
uv run python -m unittest tests.config tests.dataset tests.metrics tests.spec tests.run \
    tests.prompts tests.torchdata tests.ioi tests.tasks tests.artifact

# one module / class / method
uv run python -m unittest tests.spec
uv run python -m unittest tests.spec.TestComposition -v
uv run python -m unittest tests.spec.TestSpecHash.test_output_paths_do_not_change_it
```

`tests/adapter.py` holds the **golden capture**: four frozen prompts through GPT-2 small compared
against `tests/stubs/gpt2-small-capture.pt` at 1e-3. It exists so that "did quantization change the
model?" can be asked without first suspecting the capture code. Regenerate it only deliberately —
never to make a failing test pass:

```bash
uv run python -m tests.stubs.refresh
```

Adapter/probing/runner/discovery/comparison tests skip loudly rather than fail if the checkpoint is
unreachable offline, and they all take it from `tests/stubs/model.py` — **one GPT-2 per process, not
one per TestCase**. That is not a speed optimization: a checkpoint is half a gigabyte resident, a
dozen of them exhaust a 6 GB card partway through the run, and every one of those `setUpClass`
handlers turns a failure into "gpt2-small is not available" — so the suite reported an out-of-memory
GPU as a machine with no checkpoint and passed with a third of its tests skipped. `shared_adapter`
re-raises an OOM rather than disguising it, for that reason.

`tests/adapter.py::test_chunking_does_not_change_the_result` asserts a *relative* drift, not an
absolute tolerance: bit-exactness across batch shapes is a CPU-only property, because cuBLAS picks
its kernel by shape and a batch of 3 reduces in a different order than a batch of 64 (~5e-7
relative, which is float32 noise). A real chunking bug shows up there at relative order 1. `tests/discovery.py` holds the second receipt after the golden capture:
`head_gradients` is checked against a finite difference -- perturb one head's output and the change
in the logit difference has to be the one the gradient predicted -- because everything `eap` reports
is that inner product, and a gradient taken at the wrong site produces a plausible wrong ranking.

`tests/config.py::TestNoHardcodedModelFacts` greps every file under `src/` for `768`, `1600`,
`2048`, `4096`, `5120`. It reads raw text, so **a size named in a docstring or comment fails it too** —
if prose needs to talk about widths, say "hundreds of dimensions" rather than the number.

### Lint

```bash
uv run ruff check .          # add --fix for the safe autofixes
```

Config lives in `[tool.ruff]` in `pyproject.toml`, with every ignore commented in place. Three
things to know before adding rules or "fixing" the ignores:

- `UP006`/`UP035`/`UP045` (modernize `Optional[X]`, `Dict[K, V]`) are **off by design**, not for
  compatibility — OmegaConf reads `ExperimentSpec`'s annotations and handles both spellings fine.
  Turning them on rewrites ~235 annotations across 27 files, so it is a one-pass decision.
- `B008` is not disabled; `typer.Option`/`typer.Argument` are declared immutable, so genuine
  mutable-default bugs still land.
- `ARG` is not selected: torch forward hooks are called by signature, so their unused parameters
  are the contract.

`ruff format` has never been run over this repo. Running it reformats every file — don't do it as a
side effect of another change.

## Architecture

### The packages are a dependency order, and it is one-way

```
core        config, metrics                          imports nothing
telemetry   journal                                  imports nothing
model       adapter, backends/                       -> core
data        dataset, prompts, torchdata, ioi, tasks  -> core
methods     probing, steering, circuits,             -> core, model, data, telemetry
            discovery, comparison, sheaves
share       schema/, storage, converters/      -> core, data, methods
experiment  spec, run, runner                  -> core, model, data, methods, share
viz                                            -> core
cli                                            -> everything
ie          app, session, stack, view, views/  -> everything, and nothing imports it
```

Nothing imports upward and nothing imports sideways within a layer's own row. That is checkable in
one pass over the source, and it is the property that makes the split worth having: a new module
that cannot find a home without breaking it is a module doing two jobs.

`telemetry` is the second package that imports nothing, and it is separate from `core` rather
than inside it because `core` is closed: a journal is not a model fact or a metric definition.
It sits below `methods` so that `sheaves.prune` can stream its loss terms to disk mid-run —
see **Watching a run** below.

`core` is small on purpose. It holds the two things with no dependencies of their own that every
other package needs — what a model is, and how a number is scored — and nothing else earns a place
there. In particular `metrics` sits below `methods` rather than inside it, because `data/ioi.py`
scores a logit difference while `methods/circuits.py` builds on `data/ioi.py`; putting metrics in
`methods` would make `data` and `methods` import each other.

### Two config layers, and the difference matters

- **`configs/*.yaml` → `ModelConfig`** (`src/core/config.py`): what a *model* is. The only place a
  model fact is allowed to live. Loaded by name (`gpt2-small`) or path; unknown keys raise.
  `device` is the one field here that is not a model fact but a placement, so it defaults to `auto`
  and is resolved by `resolve_device` in `model/adapter.py` — the best accelerator present, falling
  back to the CPU. Any other value is passed through untouched and is therefore a demand: `cuda` on
  a machine without one should fail where the weights are placed, because somebody wrote it on
  purpose. The backend stamps the resolved device back into `cfg` the way it stamps the sizes, so
  everything that prints or records a device gets the answer rather than the question. `core/config.py`
  imports no torch and must not start: a config has to stay readable where there is no GPU at all.
- **`specs/` → `ExperimentSpec`** (`src/experiment/spec.py`): what an *experiment* is, composed by Hydra
  from group directories (`model/`, `data/`, `method/`, `preset/`) against `specs/config.yaml`.
  A `specs/model/*.yaml` is a one-liner naming a `configs/` entry — it does not restate model facts.

`ModelSpec.resolve()` bridges them: load the named config, then apply the spec's optional overrides.
Those overrides are `Optional[...] = None` on purpose, so "not stated" stays distinguishable from
"stated as the default".

### The core pipeline

```
compose_spec  →  ExperimentSpec  →  run_experiment  →  Run (+ directory)
                       │                  │
                 ModelSpec.resolve   EXPERIMENTS[kind]
                       │                  │
                  load_adapter  →  capture → train_probe/difference_of_means → evaluate
```

- `model/adapter.py` — **the contract, and nothing that satisfies it.** No model library is imported
  here and no architecture is named. `ModelAdapter` is a `Protocol`; backends register a factory
  under a string key via `@register_backend("name")`, and a config names that key.
  `CircuitAdapter` is a **second** protocol on top of it (`logits`, `attention`, `head_outputs`,
  `head_gradients`, `decompose`, `patch`, `single_token`, `tokens`) — a backend behind an inference
  API can honestly capture and steer and honestly cannot patch a head, so circuit code calls
  `require_circuits()` and gets a message naming the backend rather than an `AttributeError` halfway
  through. `head_gradients` differentiates the logit difference at the *same* site `head_outputs`
  reads and `patch` writes, and keeps the graph rather than detaching there: cutting it would delete
  every path an earlier head has to the answer through this layer's attention, leaving a gradient
  that looks fine and answers a different question.
- `model/backends/` — one module per implementation. `transformers.py` is the only one;
  `nnsight_vllm` is named by `configs/qwen3.5-27b.yaml` and deliberately not implemented, and
  adding it is a new file here rather than an edit to `adapter.py`. All architecture knowledge is
  quarantined in `_blocks`, `_attention_projection`, `_mlp` and `_final_norm` — four lookup lists;
  teaching the backend a new model family is editing those and nothing else.
  `adapter.py` imports this package **at the bottom of the file**, which is the one import in the
  repo whose position is load-bearing: registration has to happen when `adapter` is imported, and
  the backend imports the protocols above it, so anywhere else is a cycle.
- `experiment/runner.py` — `@register_experiment("kind")` registers one function per experiment kind
  (`probe_sweep`, `probe_train`, `ioi_circuit`, `circuit_comparison`). Adding an experiment type is a
  registration, not an edit to `ExperimentSpec` or the runner body — `ioi_circuit` shares none of the
  probing pipeline and is still just an entry in `EXPERIMENTS`, and `circuit_comparison` does not run
  a circuit study at all.
- `experiment/run.py` — **stdlib only, no torch import**, so a `run.json` is readable anywhere. Keep it
  that way.
- `share/schema/` + `share/storage.py` — the shareable form of a result (`.mia`: a JSON card
  plus one `safetensors` file), and that directory on disk. `schema/` is one module per class
  (`artifact`, `model`, `site`, `span`, `metric`, `payload`, `node`, `edge`, `control`,
  `controls`) over `errors`, `version` and `vocabulary` — the `(str, Enum)` closed sets
  `validate()` checks against. `Site.component` and `Node.component` are **different**
  vocabularies (`head_out` vs `head`) under the same wire name, which is why they are separate
  types; everything else on a card is prose on purpose and its docstring lists what it takes. Import the module, not the package: it is a namespace package like `converters/`,
  so `from ..schema.payload import Payload`. They know nothing about this repository, which is
  what keeps `storage.load` from importing transformers.
  `share/converters/` is where both sides are known, so a new experiment kind gets a module
  there rather than a special case inside the format; `share/loaders.py` is the one door for
  reading a probe from either form. See `docs/artifact-format.md` for the prose,
  `docs/rfcs/0001-mia-format.md` for the field-by-field spec, drawn in `docs/sharing/`.
- `methods/steering.py` — `strength_sweep` is the steering experiment: one curve, not one generation
  at one strength. It measures *effect* (the probe's score on the steered continuation) against
  *fluency* (share of non-repeated words), because those two moving together is what "the ceiling"
  means. `random_control` is the check that makes the rest mean anything — a random vector of the
  same norm at the same layer also moves the model.
- `data/torchdata.py` — map-style and streaming `Dataset`s over prompts, plus `ActivationDataset`
  over what a capture produced (which carries its model, layers and position, because a directory
  of `.pt` files identified by filename is one you will eventually mix up).
- `data/ioi.py` — the Indirect Object Identification task (Wang et al., 2023) as data: one frame per
  dataset, single-token names, the two name orders balanced, and clean/corrupted pairs under either
  the `abc` corruption (replace the repeated name) or `swap` (exchange the two roles).
- `methods/circuits.py` — the circuit study itself, asked twice. `direct_logit_attribution` is the
  correlational half (exact, one forward pass, blind to everything but the direct path);
  `patch_heads` / `patch_residual` are the causal half (one forward pass per site). `discover`
  grows a circuit greedily and `verify` checks it three ways. **The two halves disagree and the
  disagreement is the finding** — on GPT-2 small the negative name movers write hard against the
  answer and patching says the model needs them.

### Invariants that the code enforces (don't break these)

1. **Layers are addressed by depth fraction, never index.** `cfg.layer(0.65)`. There is no field
   anywhere for an absolute layer index, and `probe_layer_frac` outside `[0, 1]` raises.
2. **`d_model` and `n_layers` are read off the checkpoint**, stamped in by `cfg.with_sizes()`. A
   config that states a size disagreeing with the checkpoint is an error, not a silent overwrite.
   No literal widths belong in `src/` — see the grep test above.
3. **Unknown keys are always errors.** `from_mapping` rejects unknown `ModelConfig` keys;
   `_reject_unknown_keys` in `spec.py` exists specifically because Hydra's `+key=value` appends past
   struct mode and would otherwise vanish silently when the config becomes an `ExperimentSpec`.
4. **`spec_hash` covers everything that determines a result and nothing else.** `output` is popped
   before hashing, so the same experiment written elsewhere hashes the same. Anything added to
   `ExperimentSpec` that changes a number must be inside the hash; anything that doesn't must not.
5. **A failed run is still written out**, marked `failed` with the reason. Never clean up on failure.
   It is still found by `find_runs`, which is what makes that promise checkable.
6. **Capture uses forward hooks on the decoder blocks, not `output_hidden_states`.** The last
   hidden-states entry is post-final-layernorm, so depth 1.0 would silently mean a different
   quantity; and Transformers' own recorder hooks can observe a block before a steering hook is
   done with it. Recording hooks are registered *after* steering hooks so a steered layer captures
   the intervention.
7. **Steer with `probe.direction` (`weight / std`), never `probe.weight`.** `weight` is fit in
   standardized coordinates. Steering strength is measured in mean activation norms of that layer's
   forward pass, so `1.0` means the same intervention size on any model; strength `0` registers no
   hook at all, keeping it byte-identical to no steering.
8. **`LabeledPrompts.split()` keeps a `groups` id whole**, because a contrast pair straddling the
   split makes the AUC measure the one word that differs.
9. **A prompt dataset yields text, never token ids.** Tokenization belongs to the adapter that owns
   the model, its padding side and its pad token. A `DataLoader` that tokenizes is a second opinion
   about all three, and the failure is silent: activations of a different sentence than the file has.
10. **`zip()` over parallel sequences takes `strict=True`.** Prompts against their completions,
    scores or labels must be the same length; truncating silently is the bug the linter now catches.

### Watching a run

`src/telemetry/journal.py` is the answer to "what is it doing right now", and it is deliberately
not MLflow or W&B: no server, no schema, no network call per step. A `Journal` writes `run.json`
(the parameters, before the first step) and appends `metrics.jsonl` (one flushed line per step),
so `tail -f` is a live view and `jq` is a query. Stdlib only and no torch, the same rule
`experiment/run.py` keeps — a journal has to be readable on a machine that cannot load the model
that wrote it.

- **Flushed every write, never buffered.** The reason it exists is that the process may not
  survive to flush: this repo has lost a two-hour run to a driver OOM at the two-second mark and a
  ninety-minute run to a stale split, and both produced exactly nothing to look at.
- **The reader aggregates, not the writer.** `progress()` derives rate and ETA on read rather than
  storing them, because a rate written into the file is wrong the moment the run slows down.
- **A torn *last* line is a write in progress; a torn line anywhere else is damage** and raises.
  Silently dropping a row out of the middle of a curve is how a reader ends up explaining a gap
  that is not in the data.
- **`to_columns` pads missing keys with None** so a metric that started halfway through does not
  shift against the step axis.

`prune` takes `journal=` and `probe_every=`. The loss terms are logged every step because they
were already synced off the device to build `history`; `density` is throttled because it reduces
over every gate. `history` stays at its six entries — it is the summary inside the artifact, and a
2000-row curve does not belong there.

`uv run python -m scripts.watch` reads the newest journal under `MI_LAB_JOURNALS` (`--list`,
`--follow`, `--rows`). It only reads, so pointing it at a run in flight cannot disturb it.

For a run that is already going and was started without a journal, `py-spy dump --pid <n> --locals`
reads `step` straight out of the live frame. That is a debugger, not instrumentation — it needs
sudo under `ptrace_scope=1`, and it is what to reach for exactly once, before wiring the journal in.

### Where a run lands

`outputs/<experiment>/<run id>/`, and `run_directory` in `experiment/runner.py` is the only place
that layout is written down — the runner writes it, the CLI reports it and `src/app.py` prints it,
and three copies of a path is three chances to print a directory that does not exist.

The experiment name is a directory because a root accumulates runs forever and the question asked of
it is nearly always "what did *this* experiment do" rather than "what ran on Tuesday". `find_runs`
therefore looks for a `run.json` at any depth rather than one level down, and orders by run id
rather than by the directory walk — the walk is ordered by experiment name, the id leads with its
timestamp. A listing that only looked one level down would find the experiment directories, report
no runs, and go on doing it silently.

`output.root` is excluded from `spec_hash`, so a run moved to another root is the same run and
hashes the same. `outputs/`, `runs/` and `multirun/` are all gitignored: what makes a run
reproducible is the `spec.yaml` inside it, not the directory.

### Composition vs sweeps

`compose_spec` (used by the Typer CLI) clears Hydra's `GlobalHydra` singleton before and after, so
repeated calls in one process are safe. `src/app.py` is the *only* place `@hydra.main` is used, and
exists solely because `--multirun` needs argv — a Typer command group cannot live inside it.

`load_spec` is the other door in: it reads one self-contained file with no groups and no
composition. That is what `run replay <dir>` uses, so a run stays reproducible from its own
`spec.yaml` long after `specs/` has moved on, with a matching hash as the proof.

### Data: plain text first

`DataSpec.source` is one of `SOURCES = ("synthetic", "prompts", "jsonl")`.

`prompts` is the format in `data/prompts.py` and the one to prefer: one example per line, `+`/`-` in
the first column as the label, indentation joining a line to the group above it, and `name:` /
`labels:` headers before the first example. It is the only source carrying label names and group
ids, so a split cannot cut a minimal pair in half. Whitespace is written down (`\s`, `\n`, `\t`,
`\\`) because a trailing space changes tokenization and is invisible in an editor; an unknown
escape, an unknown header, a repeated header and a header after the first example are all errors,
and every message names `file:line`.

`csv` is deliberately **not** a `DataSpec.source`. `load_csv` and `data convert` exist to import a
download once and keep the result, because a CSV is what a probing set arrives as and none of them
agree on column names. `--group-field` is the argument that matters: a downloaded contrast set
carries the subject of the pair in a column, and naming it is what turns a table into groups.
Without it the pairs straddle the split, the probe learns the subject, and the sweep comes back
*below* chance — every test twin being the same subject with the opposite label. A group's rows
must be adjacent in the file, because `dumps` writes a group as an indented run and a scattered one
could not be read back the same way.

`dumps` writes `name:` and `labels:` and drops `notes:`/`source:` — a converted file loses prose by
design, so the conversion command is the provenance. Keep it written down (CHEATSHEET.md has the
ones used so far). Downloaded data and anything converted from it is gitignored: geometry-of-truth
ships no LICENSE, so it stays local.

`uv run python -m src.cli data check <file>` reports duplicates, balance, group sizes and the split
you are about to train on — before any model loads. `synthetic` is the toy generator that keeps the
path runnable with no download: a machinery check, not a result.

### Circuits

`ioi_circuit` is the second kind of experiment this repo runs, and the parts of it that are easy to
get quietly wrong are all guarded:

- **The decomposition is checked, not assumed.** `Decomposition.remainder` and
  `Attribution.residual` are the receipts: every write into the residual stream, summed and pushed
  through the frozen unembedding, has to land on the logit difference the model actually produced.
  Both are asserted in `tests/circuits.py` and both are ~1e-6 on GPT-2 small.
- **The final norm is frozen, and that is the approximation.** A component's write becomes logits
  by dividing by the scale the *complete* residual stream produced. `_normalizer` decides whether
  the norm centres by asking the module (LayerNorm is invariant to adding a constant to every
  coordinate, RMSNorm is not) rather than by recognizing a class name.
- **Head writes are computed by calling the projection, never by slicing its weight.** GPT-2 stores
  a `Conv1D` as `[in, out]` and everything else stores a `Linear` as `[out, in]`; a transposed
  slice is wrong in a way that still produces plausible numbers.
- **Patching writes into the same site `capture` reads** — the residual stream leaving a block, and
  the input to the attention output projection. Writing back what was already there is exactly a
  no-op, which is what every causal number is a difference against.
- Patched forward passes still chunk by `batch_size`, so the patch hooks slice the rows of the
  chunk they fire in. Handing a hook the full donor patches the wrong prompts the moment one batch
  becomes two.

### Comparing techniques

The field's question moved from "what is the circuit for this task" to "which way of finding one
should anybody believe", so the technique became the thing under test. `methods/discovery.py` is a
registry of techniques the way `model/adapter.py` is a registry of backends: each scores every head
on a task, and adding one is a `@register_technique` rather than an edit to anything that measures.

- `attribution` — direct logit attribution, exact, two passes, blind to indirect paths.
- `patching` — activation patching, causal, one forward pass per head. The reference.
- `ablation` — the same intervention from the other side: what removing a head costs rather than
  what restoring it recovers. The two come apart wherever the model has a second route.
- `eap` — attribution patching, patching's first-order expansion at a constant five passes. The
  gradient is taken on the **corrupted** run, because the quantity approximated is what happens when
  the corrupted activation is moved to the clean one; expanding around the clean run quietly answers
  a different question. Divided by the span, so it lands in patching's units and can be subtracted
  rather than merely ranked. On GPT-2 small it reaches rho 0.99 against patching for 5 passes
  against 147, and direct attribution reaches 0.15 — that gap is the argument for both of them.
- `random` — the control. Not a straw man: a model spreads a task widely enough that an arbitrary
  handful of heads recovers a real fraction of the span.

Ranking is by **absolute** score everywhere, so a selected set can restore *less* than a random one
— the top of a late-layer IOI sweep is mostly heads that write against the answer. That is the
ranking being right and faithfulness being the wrong question to ask of it.

`data/tasks.py` is the matching registry for tasks, because specificity has no meaning with one.
`CircuitTask` is a Protocol -- clean prompts, corrupted twins, two answer ids, positions, `subset` --
and `IOIDataset` satisfied it before the module existed. Everything in `methods/circuits.py` is typed
against it now; `classify_heads` is the one exception, because it names the four attention movements
IOI in particular is built out of. The other three tasks (`greater_than`, `induction`, `agreement`)
are single-frame, single-token-answer, length-aligned and all above chance on GPT-2 small.

`methods/comparison.py` asks the three questions finding a circuit does not answer:

- `compare_techniques` — every technique on one task at **one circuit size**, because faithfulness
  climbs with the head count and a comparison at different sizes is a comparison of sizes. Compared
  as orderings (`spearman`) and as sets (`jaccard`), never by subtracting scores that are not in the
  same units. `Payload` refuses a tensor without its axes for the same reason a comparison refuses
  a subtraction without matching units, so the artifact stores one payload per technique.
- `consistency` — the technique run once per example, the shared set at presence P, and `reuse@P`.
  A circuit found on a batch is an average and an average can be made of components no single
  example used. Defaults to `eap` because this pays the technique's cost again for every example.
- `specificity` — every task's circuit ablated on every task, plus a random circuit of the same size
  as the floor. Measured in **damage** (a share of the clean logit difference) rather than recovery,
  because a recovery is a fraction of one task's own corruption span and the other task does not
  have it. Mean ablation, for the same reason: a corruption belongs to the task it was written for.

`circuits.completeness` is the fourth check, beside faithfulness/necessity/minimality: take a subset
K out of the circuit and the *same* K out of the model, and see whether they break together. Sampled
rather than exhaustive, and the subsets are kept so the number ships with its evidence. A faithful
incomplete circuit reproduces the behaviour by a route the model does not use.

Two traps that are real and are written down where they bite: mean ablation only removes what
*varied across the batch*, so on a single-frame task a near-zero diagonal can be the ablation rather
than the model; and the cross-task sweep keys a task by its registry name while an IOI dataset names
itself after its corruption (`ioi` vs `ioi-abc`), which used to write an artifact whose cross-task
controls were silently empty — `from_comparison` now refuses a `task_key` it cannot find.

### Sharing results

`ioi_circuit` writes `circuit.mia` into its run directory and that is the artifact meant to leave
the machine; `run.json` stays the record of what this machine did. Six rules are enforced by
`Artifact.validate()` and each one is a way a shared result gets misread:

- **A tensor is never stored without its axes.** `Payload` keeps values, axis names and units
  together, and `labels` names the ticks (the token strings under a `position` axis). A bare grid
  in a `.pt` is one the next reader transposes, and the transposed heatmap still looks plausible.
- **A site carries the depth fraction, not only the layer index** — invariant 1, applied to
  something leaving the repo. `Site.at` refuses to build a site without knowing the model's depth
  rather than defaulting the fraction away.
- **Anything reporting a recovery must carry the span it is a fraction of.** Same reason
  `circuits.baselines` refuses a span near zero.
- **A circuit node keeps both `attribution` and `causal`.** Storing one summary score per
  component throws away the finding the second measurement exists to produce.
- **A metric carries the definition that produced it.** `metrics` is `Dict[str, Metric]`, never
  `Dict[str, float]`, and an empty `definition` raises. `faithfulness` here is a logit-difference
  recovery under restoration; in the faithfulness literature it is a normalized KL reproduction.
  Both report near 0.9. `Metric` is to a number what `Payload` is to a tensor.
- **A check that was not run is written down as not run.** `controls.cross_task`,
  `controls.random_baseline` and `measurement.identifiability` ship empty rather than absent, the
  same way `edges` does — an artifact that ran a cross-task ablation and one that never considered
  it must not be byte-identical.

`share/converters/comparison.py` writes the comparison as a circuit artifact — same kind, same
required grids — and it is the first thing in this repo that fills `controls`. `random_baseline`
gets the same-size random circuit's faithfulness and `cross_task` gets one entry per other task.
That was the point of those slots: an artifact that ran a cross-task ablation and one that never
considered it must not be byte-identical.

`edges` is in the schema and is empty: this repo measures which heads matter, not which head feeds
which. An artifact that omitted the field would read as a circuit whose wiring nobody recorded.
`controls` and `identifiability` are empty for the same reason and mean the same thing: nothing
here ablates a circuit against a second task, and nothing imposes a structural assumption on a
steering fit — so every circuit emitted is a within-task claim, and every direction is one member
of a behaviourally indistinguishable class. `artifact check` warns on each.

A version bump strands every artifact written before it, so `share/migrate.py` and
`artifact upgrade` exist beside the refusal. Definitions are recovered from
`share/definitions.py` — one table, read by both the converters and the migration — and a metric
name it does not know gets a definition saying so rather than an invented one.

**A listing must report what it could not read.** `storage.scan` returns the failures alongside
the artifacts and `artifact list` prints both; `find_artifacts` keeps its skipping shape only for
callers with nowhere to report. Three `except: continue` in the explorer once hid every v0.1
artifact in `outputs/`, which looked exactly like a run that had produced nothing.

`open_probe` takes either a `.pt` or a `.mia`, so nothing that only *applies* a probe has to know
which it was handed. That round trip is the test of the format — a shared artifact that does not
come back as a working object has shared nothing.

### CLI

`src/cli/main.py` aggregates one Typer app per group: `model`, `capture`, `data`, `probe`, `steer`,
`ioi`, `compare`, `artifact`, `run`, `viz`. `ioi` answers "which heads do this task"; `compare`
answers the three that come after it, and its commands are ordered by what they cost — `compare
list` loads no model at all. Command modules only format what `core` returns — anything doable from the shell must
be doable by importing the same core function. `HelpfulCommand`/`HelpfulGroup` in `cli/common.py`
print full help on a parse error; note that Typer 0.27 vendors its own Click fork, so the
`UsageError` caught there is `typer._click.exceptions.UsageError`.

Every model-facing command takes a config as its first argument — moving an experiment from a
laptop model to a large one is that argument changing and nothing else.

### Explorer

`src/ie/` is the second front end: a terminal UI for reading what the CLI computed, built on
Textual and shaped after k9s. `python -m src.ie` opens it. The CLI stays where things are
computed; `ie` is where what was computed is navigated.

- `app.py` — one screen, not a stack of them. A persistent header (which model, which prompt,
  which root), a breadcrumb, the current view, and a command line that appears on `:`. The
  header has to survive a drill-down, or a table of activations can outlive the checkpoint it
  claims to be about.
- `stack.py` — the `PageStack`. `Escape` pops, the breadcrumb is the trail. **Sameness is the
  title, not the class**: `activations` and `activations L9` are two pages of one class, and
  comparing types made drilling in silently replace the view you drilled from.
- `command.py` — `:resource /filter`, two-letter aliases (`:ar`, `:ru`), and the setters
  `:model`, `:prompt`, `:root`, which take the rest of the line verbatim because a prompt has
  spaces and slashes in it.
- `view.py` — every view answers three things: its columns, its `rows()`, and what Enter means.
  `cell_text` unwraps `(str, Enum)` for display, which is the trap `share/schema/vocabulary.py`
  documents and which shipped once as a table reading `Kind.CIRCUIT`.
- `registry.py` — resource name to view class, so adding one is a module under `views/` plus an
  entry, never an edit to the parser or the app.
- `views/grid.py` — the one view that shows a *measurement* rather than a list of things. A
  payload's values laid out by its axes, with the tick labels as column headers (so a
  position-indexed grid reads as tokens) and the site's layers as row headers (because a grid
  measured over a subset of layers has rows that are **not** layer indices — the same misreading
  `Artifact._check_shapes` guards). `:runs` → `a` → `:tensors` → enter is the path from "what
  ran" to the numbers.
- `session.py` — **the checkpoint is loaded lazily and that is the point.** Runs and artifacts
  are readable with no torch and no weights, so the explorer opens instantly on them; only a
  view with `needs_model = True` pays. `tests/ie.py` drives the no-model path with a session
  that raises if anything asks for an adapter, because the way this breaks is one view quietly
  reaching for `session.adapter()`.

Four things that are easy to get wrong here, all of which were:

- **Enter reaches a view as a `DataTable.RowSelected` message, not as a key binding.** The
  focused `DataTable` binds `enter` to its own `select_cursor` and consumes it, so an app-level
  binding looks right, appears in the footer, and never fires.
- **A title is set in `__init__`, never in `on_mount`.** The stack compares titles when a page is
  pushed, and a mount happens after that — so a view that renamed itself later silently replaced
  the view it was drilled from.
- **`on_show` focuses the table, so it runs after the mount** (`call_after_refresh`). Focusing an
  unmounted widget quietly does nothing, and the result is a keyboard that ignores you.
- **A key in `View.hints` must be in `View.keys`.** The footer advertised `<n>` and `<t>` on the
  artifacts view for a while with nothing behind them; `tests/ie.py` presses the keys it claims.

And note `capture(layers=None)` means *the config's probe layer*, not every layer — the
activations view names all of them explicitly, and asking for "the activations" without doing so
returns exactly one row.

### Charts

`src/viz/` is one module per subject (`dataset`, `model`, `activations`, `probing`, `steering`,
`circuits`, `comparison`, `runs`) over `style.py`. Two rules:

- **The palette is keyed by meaning, not colour.** `PALETTE["baseline"]`, never `"steelblue"` — so
  the positive class is the same green in every chart.
- **Only the CLI selects a backend.** `cli/commands/viz/__init__.py` sets `matplotlib.use("Agg")`
  at the top, before it imports its group modules and therefore before any `src/viz/` module loads;
  `src/viz/` must never call `use()`, or it takes the backend away from a notebook that imported it.

`src/cli/commands/viz/` mirrors `src/viz/` one module per group, so a chart and the command that
draws it sit at the same path in two trees. `common.py` holds the shared options and helpers —
a group differing on what `--size` means is a difference nobody notices until two charts disagree
about the data they were drawn from. `dashboard.py` is the exception that spans every group, so
`__init__.py` registers it on the root app rather than it importing the app it hangs off.

`viz circuit dashboard` does the same for the IOI battery, measuring every chart off one dataset and
one pair of baselines so the panels on the page are comparable with each other — which stops being
true the moment they are made one command at a time. `viz compare dashboard` does it for the
technique comparison; its `cost` panel — forward passes across on a log scale, agreement with the
reference up — is the one the others exist to set up.

`viz dashboard` assembles a run's charts into one self-contained HTML page with images inlined as
data URIs, so the file survives being moved or attached. Every `viz` command takes `--output` and
`--show` (inline in the terminal, via `term-image`, for working over ssh).

## Conventions

- Every module opens with a docstring stating what it is *for* and which decision it encodes,
  usually ending in a `A common pipe could be: a | b | c` line. Match that when adding a module.
- Each subsystem raises its own `ValueError` subclass (`ConfigError`, `SpecError`, `DatasetError`,
  `ProbeError`, `MetricError`, `RunError`) with a message that says what to do instead.
- Comments explain the trap, not the mechanics — they exist where a reasonable reading of the code
  would be wrong.
- Everything must run on `gpt2-small` on a CPU laptop first; a result that needs a big model to
  appear is a result that cannot be debugged. `device: auto` is what keeps that true while still
  using the GPU where there is one — never hardcode `cuda` into a shipped config.
