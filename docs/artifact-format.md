# MIA v0.1 — an envelope for interpretability results

**Status:** implemented in this repository, proposed for discussion.
**Reference implementation:** `src/share/schema.py` (the card), `src/share/storage.py` (the disk),
`src/share/converters/` (what this lab measured, packaged).
**Specification:** [`rfcs/0001-mia-format.md`](rfcs/0001-mia-format.md) — the field-by-field spec.
**Diagrams:** [`sharing/`](sharing/) — the round trip, the card, the validation gates, the landscape.

## The gap

Model weights ship as `safetensors`. Datasets ship as rows plus a card. Interpretability
results ship as a paper and a notebook, which is why nobody can load anyone else's.

Per artifact type, the state of things:

| artifact | has a format? | how it actually travels |
|---|---|---|
| SAE decoders | de facto: `safetensors` + a metadata file on the Hub | loadable |
| steering vectors | none agreed | a tensor in a repo, layer stated in the README, or not |
| linear probes | none agreed | a `.pt` with whatever fields the author's class had |
| activation maps | none | a PNG in a paper |
| **circuits** | **none** | a node list in whatever shape the tool dumps |

The two at the bottom are the real gaps. A circuit — a set of components plus what was
measured about each — has no canonical serialization at all; every tool invents one. And an
activation map travels as a *picture*, which is why you cannot subtract two of them.

What is missing is not a tensor container. `safetensors` already exists and is the right
one. What is missing is the envelope: the facts that make a number applicable somewhere
else, and a machine-checkable statement of what the tensors mean.

## The format

An artifact is a **directory**:

```
ioi-abc.mia/
  artifact.json        the card — plain JSON, no torch needed to read it
  tensors.safetensors  every number, named
```

The card is JSON so that deciding whether an artifact is worth downloading never requires
loading a framework. The tensors are `safetensors` so they are memory-mappable and a large
artifact need not be read whole. The tensor file duplicates `format`/`version`/`kind`/`id`
into its own metadata, so a `tensors.safetensors` separated from its card still says what
it is.

### Kinds

`circuit`, `probe`, `steering_vector`, `activation_map`. The kind decides which tensors are
mandatory:

| kind | required tensors |
|---|---|
| `circuit` | `head_attribution`, `head_effects` |
| `probe` | `weight`, `bias`, `mean`, `std` |
| `steering_vector` | `vector` |
| `activation_map` | `values` |

An artifact missing them is not an incomplete artifact of that kind; it is a different
thing wearing the label, and `validate()` says so.

### The card

```json
{
  "format": "mia",
  "version": "0.1",
  "kind": "circuit",
  "id": "ioi-abc-gpt2-small",
  "created_at": "2026-08-26T02:47:10Z",

  "model":  { "id": "gpt2-small", "hf_name": "gpt2", "revision": null,
              "n_layers": 12, "d_model": 768, "n_heads": 12, "dtype": "float32" },

  "site":   { "layers": [0, 1, "…"], "fracs": [0.0, 0.083, "…"],
              "component": "head_out", "position": "all" },

  "task":   { "name": "ioi-abc", "n": 8, "frame": "…", "corruption": "abc",
              "tokens": ["Then", ",", " Jack", "…"], "landmarks": {"IO": 2, "S1": 4, "S2": 10, "END": 14} },

  "measurement": {
    "method": "direct_logit_attribution + activation_patching, greedy search",
    "span":   { "metric": "logit_difference", "clean": 2.959, "corrupted": 0.406 },
    "metrics": { "faithfulness": 0.919, "necessity": 1.160, "attribution_remainder": 1.7e-06 }
  },

  "graph": {
    "nodes": [ { "id": "L9H9", "component": "head", "layer": 9, "head": 9,
                 "role": "name mover", "in_circuit": true,
                 "scores": { "attribution": 2.625, "causal": 0.226,
                             "minimality": 0.192, "cumulative_recovery": 0.919, "step": 7 } } ],
    "edges": []
  },

  "tensors": {
    "head_effects":   { "shape": [12, 12], "dtype": "float32",
                        "axes": ["layer", "head"], "units": "recovery", "labels": {} },
    "residual_patch": { "shape": [12, 15], "dtype": "float32",
                        "axes": ["layer", "position"], "units": "recovery",
                        "labels": { "position": ["Then", ",", " Jack", "…"] } }
  },

  "provenance": { "tool": "mi-lab", "git_commit": "0c248f7", "git_dirty": false,
                  "torch": "2.13.0", "format_version": "0.1" },

  "notes": "…"
}
```

Unknown top-level keys are a load error, not a shrug. A key the reader does not understand
is a claim it would otherwise silently drop.

## The four decisions that make it loadable

Everything above is bookkeeping except these. Each one is enforced by `validate()` and each
one corresponds to a way a shared result is misread today.

### 1. A tensor is never stored without its axes

`head_effects` is `[layer, head]` in units of *recovery*. `residual_patch` is
`[layer, position]`. A bare 2-D float array in a file named `effects.pt` is one the next
person transposes, and the transposed version still produces a plausible heatmap.

So `axes` is required and must have one name per dimension, and `labels` names the ticks:
the token strings under a `position` axis, the role names under a `role` axis. That is what
lets a tool that never ran the model redraw the heatmap — and, more to the point, difference
two of them.

### 2. The site is a depth fraction, not just a layer index

Layer 8 of a 12-layer model and layer 42 of a 64-layer model are the same place. Only the
fraction survives a model swap, so both are written down: the index for whoever is hooking
this exact checkpoint, the fraction for everyone else.

This is the one requirement that reached back into the code. A probe here recorded the
layer it was fit at and not how many layers the model had, so it could not state its own
depth — the format made that visible, and `LinearProbe` now carries `n_layers`.

### 3. Every fraction carries the span it is a fraction of

A recovery of 0.9 means "restored 90% of the distance from corrupted back to clean". If the
corruption barely moved the model, that is noise scaled up to look like a finding. So
`span` is mandatory for any kind that reports recoveries, and `artifact check` warns when
the span is near zero.

This generalizes past patching: any normalized score is a share of an interval somebody
chose, and shipping it without that interval is shipping a percentage of an unstated whole.

### 4. A circuit stores what *both* halves of the study said

Direct logit attribution is correlational, exact, and blind to everything but the direct
path. Activation patching is causal and expensive. **They disagree, and the disagreement is
the finding.** On GPT-2 small the negative name movers write hard against the correct
answer and patching says the model needs them.

So each node keeps `attribution` and `causal` side by side. A circuit format that stores one
summary score per component throws away the result and invites exactly the reading that
motivated the second measurement.

`edges` exists and is `[]` here, because this repository measures which heads matter and not
which head feeds which. An artifact that omitted the field would read as a circuit whose
wiring nobody thought to record; one that says `[]` tells you the tool looked and found
none. `artifact check` reports it as a warning rather than letting it pass silently.

## Provenance

`git_commit` alone is not enough: a hash recorded from a tree with uncommitted edits names
code that never existed. `git_dirty` is stored beside it, and `artifact check` flags it.
An artifact whose provenance is confidently wrong is worse than one with none.

## Using it

```bash
# emit a circuit study as an artifact
python -m src.cli ioi circuit gpt2-small --size 8 --save ioi-abc.mia

# every ioi_circuit run writes one automatically
python -m src.cli run exec -e ioi-circuit    # -> outputs/ioi-circuit/<id>/circuit.mia/

# read someone else's without loading a model
python -m src.cli artifact show ioi-abc.mia
python -m src.cli artifact check ioi-abc.mia
python -m src.cli artifact list

# wrap a probe for sharing, then use the shared form directly
python -m src.cli artifact pack probe.pt
python -m src.cli probe score gpt2-small --probe probe.mia -p "I adored the concert"
python -m src.cli steer probe gpt2-small --probe probe.mia -p "The film was"
```

From Python:

```python
from src.share import storage
from src.share.converters.probe import to_probe

circuit = storage.load("ioi-abc.mia")
circuit.metrics["faithfulness"]                     # 0.919
[node.id for node in circuit.circuit_heads]         # ['L10H7', 'L11H10', 'L8H6', ...]
grid = circuit.tensors["residual_patch"]
grid.values, grid.axes, grid.labels["position"]     # a redrawable heatmap

probe = to_probe(storage.load("probe.mia"))         # a working probe, not a file
probe.score(activations)
```

`probe score` and `steer probe` take either a `.pt` or a `.mia`. That is the test of the
whole thing: a shared artifact that does not come back as a working object has shared
nothing, whatever its manifest claims.

## What v0.1 does not do

Stated plainly, because a format's limits are load-bearing:

- **No edges.** The schema has the slot; nothing here fills it. A real circuit format needs
  a measured notion of "head A feeds head B" (path patching, or an attribution graph), and
  agreeing on *that* is the harder open problem.
- **No SAE features.** Neuronpedia is the venue and per-feature dashboards are the payload;
  a `feature` kind belongs here but is not written yet.
- **No content hash of the checkpoint.** `hf_name` + `revision` is what is recorded. Two
  quantizations of one repo are indistinguishable in a v0.1 card.
- **One model per artifact.** Cross-model results (a direction that transfers, a circuit
  found in two models) need a shape this does not have.
- **No signing, no registry.** This is a file layout, not a hub.

## Why a directory rather than one file

A single-file container would be tidier to move around. A directory keeps the card readable
by `cat`, greppable, and diffable in git, and lets the tensors be memory-mapped without
parsing anything first. For an artifact meant to be *read before it is trusted*, that trade
goes the other way from model weights.
