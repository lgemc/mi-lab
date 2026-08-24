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
# everything (~50s; several modules download/load GPT-2 small)
uv run python -m unittest tests.config tests.dataset tests.metrics tests.spec tests.run tests.probing tests.runner tests.adapter

# offline-only subset — no checkpoint needed
uv run python -m unittest tests.config tests.dataset tests.metrics tests.spec tests.run

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

There is no linter or formatter configured.

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
- `core/runner.py` — `@register_experiment("kind")` registers one function per experiment kind
  (`probe_sweep`, `probe_train`). Adding an experiment type is a registration, not an edit to
  `ExperimentSpec` or the runner body.
- `core/run.py` — **stdlib only, no torch import**, so a `run.json` is readable anywhere. Keep it
  that way.

### Invariants that the code enforces (don't break these)

1. **Layers are addressed by depth fraction, never index.** `cfg.layer(0.65)`. There is no field
   anywhere for an absolute layer index, and `probe_layer_frac` outside `[0, 1]` raises.
2. **`d_model` and `n_layers` are read off the checkpoint**, stamped in by `cfg.with_sizes()`. A
   config that states a size disagreeing with the checkpoint is an error, not a silent overwrite.
   No literal widths (`768`, `5120`) belong in source.
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

### Composition vs sweeps

`compose_spec` (used by the Typer CLI) clears Hydra's `GlobalHydra` singleton before and after, so
repeated calls in one process are safe. `src/app.py` is the *only* place `@hydra.main` is used, and
exists solely because `--multirun` needs argv — a Typer command group cannot live inside it.

`load_spec` is the other door in: it reads one self-contained file with no groups and no
composition. That is what `run replay <dir>` uses, so a run stays reproducible from its own
`spec.yaml` long after `specs/` has moved on, with a matching hash as the proof.

### CLI

`src/cli/main.py` aggregates one Typer app per group (`model`, `capture`, `probe`, `steer`, `run`).
Command modules only format what `core` returns — anything doable from the shell must be doable by
importing the same core function. `HelpfulCommand`/`HelpfulGroup` in `cli/common.py` print full
help on a parse error; note that Typer 0.27 vendors its own Click fork, so the `UsageError` caught
there is `typer._click.exceptions.UsageError`.

Every model-facing command takes a config as its first argument — moving an experiment from a
laptop model to a large one is that argument changing and nothing else.

### Data sources

`DataSpec.source` is one of `SOURCES = ("synthetic", "prompts", "jsonl")`. `prompts` is the
plain-text format in `src/core/prompts.py` (one example per line, `+`/`-` label prefix, indentation
for group membership) and is the one to prefer: it is the only source carrying label names and
group ids, so a split does not cut a contrast pair in half. `synthetic` is the toy generator that
keeps the whole path runnable with no download — machinery check only, not a result.

### Work in progress (untracked)

`src/viz/style.py` (matplotlib/seaborn, neither declared in `pyproject.toml` nor installed).
Nothing imports it yet.

## Conventions

- Every module opens with a docstring stating what it is *for* and which decision it encodes,
  usually ending in a `A common pipe could be: a | b | c` line. Match that when adding a module.
- Each subsystem raises its own `ValueError` subclass (`ConfigError`, `SpecError`, `DatasetError`,
  `ProbeError`, `MetricError`, `RunError`) with a message that says what to do instead.
- Comments explain the trap, not the mechanics — they exist where a reasonable reading of the code
  would be wrong.
- Everything must run on `gpt2-small` on a CPU laptop first; a result that needs a big model to
  appear is a result that cannot be debugged.
