# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project uses `uv` (there is a `uv.lock`; the `.venv` is what `uv run` uses). The bare
`python` on PATH is a different interpreter and will fail on imports — always go through `uv run`.

```bash
uv sync                                            # install/refresh the environment
uv run python -m src.cli --help                    # the CLI (also installed as the `mi-lab` script)
uv run python -m src.app --multirun model=gpt2-small,pythia-70m   # Hydra sweeps only
```

### Tests

There are no `__init__.py` files and test modules are named after the module they test
(`tests/spec.py`, not `tests/test_spec.py`), so **`unittest discover` does not work**. Name the
modules explicitly:

```bash
# everything: 260 tests, ~30s (probing/runner/adapter/circuits load GPT-2 small)
uv run python -m unittest tests.config tests.dataset tests.metrics tests.spec tests.run \
    tests.prompts tests.torchdata tests.ioi tests.artifact tests.probing tests.runner \
    tests.adapter tests.circuits

# offline subset: 205 tests, no checkpoint needed, seconds
uv run python -m unittest tests.config tests.dataset tests.metrics tests.spec tests.run \
    tests.prompts tests.torchdata tests.ioi tests.artifact

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

Adapter/probing/runner tests skip loudly rather than fail if the checkpoint is unreachable offline.

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

### Two config layers, and the difference matters

- **`configs/*.yaml` → `ModelConfig`** (`src/core/config.py`): what a *model* is. The only place a
  model fact is allowed to live. Loaded by name (`gpt2-small`) or path; unknown keys raise.
- **`specs/` → `ExperimentSpec`** (`src/core/spec.py`): what an *experiment* is, composed by Hydra
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

- `core/adapter.py` — `ModelAdapter` is a `Protocol`; backends register a factory under a string key
  via `@register_backend("name")`, and a config names that key. `transformers` is implemented;
  `nnsight_vllm` is named by `configs/qwen3.5-27b.yaml` and deliberately not implemented.
  `CircuitAdapter` is a **second** protocol on top of it (`logits`, `attention`, `head_outputs`,
  `decompose`, `patch`, `single_token`, `tokens`) — a backend behind an inference API can honestly
  capture and steer and honestly cannot patch a head, so circuit code calls `require_circuits()`
  and gets a message naming the backend rather than an `AttributeError` halfway through.
- `core/runner.py` — `@register_experiment("kind")` registers one function per experiment kind
  (`probe_sweep`, `probe_train`, `ioi_circuit`). Adding an experiment type is a registration, not an
  edit to `ExperimentSpec` or the runner body — `ioi_circuit` shares none of the probing pipeline
  and is still just an entry in `EXPERIMENTS`.
- `core/run.py` — **stdlib only, no torch import**, so a `run.json` is readable anywhere. Keep it
  that way.
- `core/artifact.py` — the shareable form of a result (`.mia`: a JSON card plus one
  `safetensors` file). It knows nothing about this repository, which is what keeps
  `Artifact.load` from importing transformers. `core/sharing.py` is the one place that knows
  both, so a new experiment kind gets a converter there rather than a special case inside the
  format. See `docs/artifact-format.md`.
- `core/steering.py` — `strength_sweep` is the steering experiment: one curve, not one generation
  at one strength. It measures *effect* (the probe's score on the steered continuation) against
  *fluency* (share of non-repeated words), because those two moving together is what "the ceiling"
  means. `random_control` is the check that makes the rest mean anything — a random vector of the
  same norm at the same layer also moves the model.
- `core/torchdata.py` — map-style and streaming `Dataset`s over prompts, plus `ActivationDataset`
  over what a capture produced (which carries its model, layers and position, because a directory
  of `.pt` files identified by filename is one you will eventually mix up).
- `core/ioi.py` — the Indirect Object Identification task (Wang et al., 2023) as data: one frame per
  dataset, single-token names, the two name orders balanced, and clean/corrupted pairs under either
  the `abc` corruption (replace the repeated name) or `swap` (exchange the two roles).
- `core/circuits.py` — the circuit study itself, asked twice. `direct_logit_attribution` is the
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

### Composition vs sweeps

`compose_spec` (used by the Typer CLI) clears Hydra's `GlobalHydra` singleton before and after, so
repeated calls in one process are safe. `src/app.py` is the *only* place `@hydra.main` is used, and
exists solely because `--multirun` needs argv — a Typer command group cannot live inside it.

`load_spec` is the other door in: it reads one self-contained file with no groups and no
composition. That is what `run replay <dir>` uses, so a run stays reproducible from its own
`spec.yaml` long after `specs/` has moved on, with a matching hash as the proof.

### Data: plain text first

`DataSpec.source` is one of `SOURCES = ("synthetic", "prompts", "jsonl")`.

`prompts` is the format in `core/prompts.py` and the one to prefer: one example per line, `+`/`-` in
the first column as the label, indentation joining a line to the group above it, and `name:` /
`labels:` headers before the first example. It is the only source carrying label names and group
ids, so a split cannot cut a minimal pair in half. Whitespace is written down (`\s`, `\n`, `\t`,
`\\`) because a trailing space changes tokenization and is invisible in an editor; an unknown
escape, an unknown header, a repeated header and a header after the first example are all errors,
and every message names `file:line`.

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

### Sharing results

`ioi_circuit` writes `circuit.mia` into its run directory and that is the artifact meant to leave
the machine; `run.json` stays the record of what this machine did. Four rules are enforced by
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

`edges` is in the schema and is empty: this repo measures which heads matter, not which head feeds
which. An artifact that omitted the field would read as a circuit whose wiring nobody recorded.

`open_probe` takes either a `.pt` or a `.mia`, so nothing that only *applies* a probe has to know
which it was handed. That round trip is the test of the format — a shared artifact that does not
come back as a working object has shared nothing.

### CLI

`src/cli/main.py` aggregates one Typer app per group: `model`, `capture`, `data`, `probe`, `steer`,
`ioi`, `artifact`, `run`, `viz`. Command modules only format what `core` returns — anything doable from the shell must
be doable by importing the same core function. `HelpfulCommand`/`HelpfulGroup` in `cli/common.py`
print full help on a parse error; note that Typer 0.27 vendors its own Click fork, so the
`UsageError` caught there is `typer._click.exceptions.UsageError`.

Every model-facing command takes a config as its first argument — moving an experiment from a
laptop model to a large one is that argument changing and nothing else.

### Charts

`src/viz/` is one module per subject (`dataset`, `model`, `activations`, `probing`, `steering`,
`circuits`, `runs`) over `style.py`. Two rules:

- **The palette is keyed by meaning, not colour.** `PALETTE["baseline"]`, never `"steelblue"` — so
  the positive class is the same green in every chart.
- **Only the CLI selects a backend.** `cli/commands/viz.py` sets `matplotlib.use("Agg")` because it
  writes files; `src/viz/` must never call `use()`, or it takes the backend away from a notebook
  that imported it.

`viz circuit dashboard` does the same for the IOI battery, measuring every chart off one dataset and
one pair of baselines so the panels on the page are comparable with each other — which stops being
true the moment they are made one command at a time.

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
  appear is a result that cannot be debugged.
