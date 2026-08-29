# **RFC: Model Interpretability Artifact (MIA) Specification**

| Field          | Value                                                        |
| :------------- | :----------------------------------------------------------- |
| **Status**     | Active — implemented in this repository, proposed for discussion |
| **Maintainer** | mi-lab                                                        |
| **Date**       | August 2026                                                   |
| **Changelog**  | v0.1                                                          |
| **Reference**  | `src/share/` — `schema.py`, `storage.py`, `converters/`, `loaders.py`, `provenance.py` |
| **Validator**  | `python -m src.cli artifact check <path>.mia`                 |

---

## I. Introduction

Model weights ship as `safetensors`. Datasets ship as rows plus a card. Interpretability
results ship as a paper and a notebook, which is why nobody can load anyone else's.

MIA is a JSON-plus-tensors specification for a single interpretability *measurement*: a
linear probe, a steering vector, an activation map, or a head-level circuit. It unifies the
distinct requirements of those four — a probe needs its standardization to be applicable, a
steering vector needs the sweep that found its ceiling, a circuit needs two disagreeing
scores per component, an activation map needs its tick labels — into one envelope, so that a
result is usable by a tool that did not produce it.

Per artifact type, the state of things:

| Artifact          | Has a format?                                    | How it actually travels                                 |
| :---------------- | :----------------------------------------------- | :------------------------------------------------------ |
| SAE decoders      | De facto: `safetensors` + a metadata file on the Hub | Loadable                                             |
| Steering vectors  | None agreed                                      | A tensor in a repo, layer stated in the README, or not   |
| Linear probes     | None agreed                                      | A `.pt` with whatever fields the author's class had      |
| Activation maps   | None                                             | A PNG in a paper                                         |
| **Circuits**      | **None**                                         | A node list in whatever shape the tool dumps             |

The two at the bottom are the real gaps. A circuit — a set of components plus what was
measured about each — has no canonical serialization at all. And an activation map travels
as a *picture*, which is why you cannot subtract two of them.

What is missing is not a tensor container. `safetensors` already exists and is the right
one. What is missing is the envelope: the facts that make a number applicable somewhere
else, and a machine-checkable statement of what the tensors mean.

### I.1 Design principles

Three decisions distinguish this from a tensor with a README beside it. Each corresponds to
a way a shared result is misread today, and each is enforced by `validate()` rather than
recommended in prose.

1. **A tensor is never stored without its axes.** `head_effects` is `[layer, head]` in units
   of *recovery*; `residual_patch` is `[layer, position]`. A bare 2-D float array in a file
   named `effects.pt` is one the next person transposes, and the transposed heatmap still
   looks plausible.
2. **A site carries its depth fraction, not only its layer index.** Layer 8 of a 12-layer
   model and layer 42 of a 64-layer model are the same place. Only the fraction survives a
   model swap.
3. **Every fraction carries the span it is a fraction of.** A recovery of 0.9 against a
   corruption that barely moved the model is noise scaled up to look like a finding.

A fourth is specific to circuits and stated in §II.6: a node keeps **both** measurements,
because they disagree and the disagreement is the finding.

### I.2 Terminology

The key words MUST, MUST NOT, REQUIRED, SHOULD and MAY are to be interpreted as in RFC 2119.

---

## II. Schema

### II.1 On-disk layout

An artifact is a **directory** whose name ends in `.mia`:

```
ioi-abc.mia/
  artifact.json         the card — plain JSON, readable without torch
  tensors.safetensors   every number, named
```

| Path                  | Status   | Description                                                                     |
| :-------------------- | :------- | :------------------------------------------------------------------------------ |
| `artifact.json`       | Required | The card. Stdlib JSON, so deciding whether an artifact is worth downloading never requires loading a framework. |
| `tensors.safetensors` | Required | Every tensor, keyed by the names the card's `tensors` object describes. Memory-mappable, so a large artifact need not be read whole. |

A directory carrying one and not the other is an error, not a partial artifact:
`storage.load` raises `ArtifactError` naming which half is missing.

The tensor file duplicates `format`, `version`, `kind` and `id` into its own safetensors
metadata, so a `tensors.safetensors` separated from its card still says what it is.

**Why a directory rather than one file.** A single-file container would be tidier to move
around. A directory keeps the card readable by `cat`, greppable, and diffable in git, and
lets the tensors be memory-mapped without parsing anything first. For an artifact meant to
be *read before it is trusted*, that trade goes the other way from model weights.

### II.2 Root object

| Field        | Type   | Status   | Description                                                                              |
| :----------- | :----- | :------- | :--------------------------------------------------------------------------------------- |
| `format`     | String | Required | Always `"mia"`. A card whose `format` is anything else is refused before any other field is read. |
| `version`    | String | Required | Format version, e.g. `"0.1"`. A reader MUST refuse a version it does not implement rather than guess at the difference. |
| `kind`       | String | Required | One of `circuit`, `probe`, `steering_vector`, `activation_map`. Decides which tensors are mandatory (§II.7). |
| `id`         | String | Required | Identifier for this measurement, e.g. `"ioi-abc-gpt2-small"`. Conventionally `<task>-<model>` or `<task>-<model>-L<layer>`. |
| `created_at` | String | Required | ISO-8601 UTC timestamp, stamped at construction if not supplied.                          |
| `model`      | Object | Required | Which checkpoint this was measured on (§II.3).                                            |
| `site`       | Object | Required | Where in that model (§II.4).                                                              |
| `task`       | Object | Optional | Free-form description of the data it was measured over (§II.5).                           |
| `measurement`| Object | Required | Method, baseline span, and scalar metrics (§II.6).                                        |
| `graph`      | Object | Optional | Nodes and edges, for artifacts describing a circuit (§II.6).                              |
| `tensors`    | Object | Required | The manifest: one entry per tensor in `tensors.safetensors` (§II.7).                      |
| `provenance` | Object | Required | Which tool, which commit, whether it was clean (§II.8).                                   |
| `notes`      | String | Optional | Prose the reader needs in order to apply the numbers correctly.                           |

**Unknown top-level keys are a load error, not a shrug.** A key the reader does not
understand is a claim it would otherwise silently drop.

### II.3 `model` — ModelRef

| Field      | Type    | Status   | Description                                                                        |
| :--------- | :------ | :------- | :---------------------------------------------------------------------------------- |
| `id`       | String  | Required | The name this artifact's producer knows the model by, e.g. `"gpt2-small"`.           |
| `hf_name`  | String  | Required | The checkpoint another lab resolves, e.g. `"gpt2"`. MUST NOT be invented — a `ModelRef` naming a checkpoint that will load and produce different numbers is worse than no artifact. |
| `revision` | String  | Optional | Checkpoint revision, if pinned. `null` when unpinned.                                |
| `n_layers` | Integer | Optional | Read off the checkpoint. Required in practice: without it `site.fracs` cannot be computed. |
| `d_model`  | Integer | Optional | Checked against any tensor claiming a `d_model` axis before that tensor is applied.  |
| `n_heads`  | Integer | Optional | Checked against any tensor claiming a `head` axis.                                   |
| `dtype`    | String  | Optional | Compute dtype of the measurement, default `"float32"`.                               |

`n_layers`, `d_model` and `n_heads` are optional in the type and load-bearing in practice:
they are what `validate()` checks a payload's widths against, and `artifact check` warns
when they are absent because a payload that cannot be checked is one applied on trust.

### II.4 `site` — where in the model

| Field       | Type            | Status   | Description                                                                     |
| :---------- | :-------------- | :------- | :------------------------------------------------------------------------------ |
| `layers`    | Array<Integer>  | Required | Absolute layer indices, because that is what a hook needs.                       |
| `fracs`     | Array<Float>    | Required | The same layers as depth fractions, because that is what transfers to a model of another size. MUST be the same length as `layers`. |
| `component` | String          | Required | One of `residual`, `head_out`, `mlp_out`, `attention`.                           |
| `position`  | String          | Required | Which token position, e.g. `"last"`, `"all"`.                                    |

Both spellings are written down deliberately. Deriving the fraction from the index needs
`n_layers`, and an artifact should be readable without resolving the checkpoint.

`Site.at(layers, n_layers, ...)` requires the model's depth rather than defaulting it away:

```python
Site.at([9], n_layers=12, component="residual", position="last")
# Site(layers=[9], fracs=[0.75], component='residual', position='last')
```

A site carrying only indices is one the receiving lab cannot place. This is the requirement
that reached back into the implementation — a probe here recorded the layer it was fit at
and not how many layers the model had, so it could not state its own depth. The format made
that visible, and `LinearProbe` now carries `n_layers`.

### II.5 `task` — the data behind the numbers

Free-form by design; the fields below are the convention this implementation writes.

| Field        | Type    | Status   | Description                                                          |
| :----------- | :------ | :------- | :-------------------------------------------------------------------- |
| `name`       | String  | Optional | Dataset name, e.g. `"ioi-abc"`.                                        |
| `n`          | Integer | Optional | Number of examples measured over.                                      |
| `task`       | String  | Optional | Human name of the task, e.g. `"indirect object identification"`.        |
| `frame`      | String  | Optional | The prompt template, for a templated task.                             |
| `corruption` | String  | Optional | How the corrupted half was made, e.g. `"abc"`, `"swap"`.               |
| `tokens`     | Array<String> | Optional | The token strings a `position` axis is indexed by.               |
| `landmarks`  | Object  | Optional | Named positions within `tokens`, e.g. `{"IO": 2, "S1": 4, "END": 14}`. |
| `example`    | Object  | Optional | One clean/corrupted pair, so a reader can see what was measured.       |

`task` is deliberately not validated beyond being an object. Constraining it would mean
every new task type edits the schema before it can be shared.

### II.6 `measurement` and `graph`

**`measurement`**

| Field     | Type   | Status   | Description                                                                            |
| :-------- | :----- | :------- | :-------------------------------------------------------------------------------------- |
| `method`  | String | Required | What was run, in prose: `"direct_logit_attribution + activation_patching, greedy search"`. |
| `span`    | Object | Required for kinds in `NEEDS_SPAN` (currently `circuit`); Optional otherwise | The baselines every fraction is measured between. |
| `metrics` | Object | Required | Scalar results, `{name: float}`. May be empty.                                          |

**`measurement.span`**

| Field       | Type   | Status   | Description                                                            |
| :---------- | :----- | :------- | :---------------------------------------------------------------------- |
| `metric`    | String | Required | What was measured, e.g. `"logit_difference"`.                            |
| `clean`     | Float  | Required | The value on unperturbed input — where a recovery reads 1.               |
| `corrupted` | Float  | Required | The value on corrupted input — where a recovery reads 0.                 |

`span = clean - corrupted` is derived, not stored. A recovery of 0.9 means "restored 90% of
the distance from corrupted back to clean"; quoting one without the other is quoting a
percentage of an unstated whole. `artifact check` warns when `|span| < 1e-3`.

This generalizes past patching: any normalized score is a share of an interval somebody
chose, and shipping it without that interval is shipping a fraction of nothing stated.

**`graph.nodes[]`**

| Field        | Type    | Status   | Description                                                                    |
| :----------- | :------ | :------- | :------------------------------------------------------------------------------ |
| `id`         | String  | Required | Node identifier, e.g. `"L9H9"`.                                                  |
| `component`  | String  | Required | What kind of component, e.g. `"head"`.                                           |
| `layer`      | Integer | Required | Which layer it sits in.                                                          |
| `head`       | Integer | Optional | Head index, for attention heads.                                                 |
| `role`       | String  | Optional | An assigned functional role, e.g. `"name mover"`.                                |
| `in_circuit` | Boolean | Required | Whether the search kept it. Defaults `false`.                                    |
| `scores`     | Object  | Required | Every measurement made of this node, keyed by what it measured.                  |

`scores` as written by this implementation: `attribution` (direct-path logit contribution),
`causal` (patching recovery), `minimality` (cost of dropping it), `cumulative_recovery`,
`step` (its position in the greedy search).

**A node MUST keep every measurement made of it, not a summary.** Direct logit attribution
is correlational, exact, and blind to everything but the direct path; activation patching is
causal and expensive. **They disagree, and the disagreement is the finding** — on GPT-2 small
the negative name movers write hard against the correct answer and patching says the model
needs them. A format storing one score per component throws away the result that the second
measurement exists to produce.

**`graph.edges[]`**

| Field    | Type   | Status   | Description                                                              |
| :------- | :----- | :------- | :------------------------------------------------------------------------ |
| `source` | String | Required | Node id the dependency runs from.                                          |
| `target` | String | Required | Node id it runs to.                                                        |
| `kind`   | String | Required | What was measured to establish it. Default `"unmeasured"`.                 |
| `weight` | Float  | Required | Strength of the measured dependency.                                       |

`edges` exists in the schema and every circuit this implementation emits carries `[]`. That
is the honest state of the art rather than a gap: an artifact that says it found no edges is
different from one that left a reader guessing whether the tool looked. `artifact check`
reports it as a warning rather than letting it pass silently.

### II.7 `tensors` — the manifest, and Payload

One entry per tensor in `tensors.safetensors`. The card's key set and the file's key set MUST
be equal; `storage.from_manifest` names the difference in both directions rather than
returning a partial artifact.

| Field    | Type            | Status   | Description                                                                     |
| :------- | :-------------- | :------- | :------------------------------------------------------------------------------ |
| `shape`  | Array<Integer>  | Required | The tensor's shape, so the card can be checked without opening the tensor file.  |
| `dtype`  | String          | Required | e.g. `"float32"`.                                                                |
| `axes`   | Array<String>   | Required | One name per dimension. MUST have exactly `len(shape)` entries.                   |
| `units`  | String          | Required | What the numbers are in: `"recovery"`, `"logits"`, `"activation"`, `"standardized"`, `"attention"`, `"probe score"`. Prose on purpose — the alternative is an enum every new measurement must be added to before it can be shared. |
| `labels` | Object          | Optional | `{axis: [tick, ...]}`. Each key MUST name an axis the tensor has, and each list MUST be as long as that axis. |

Reserved axis names are checked against the card: `layer` against `len(site.layers)`, `head`
against `model.n_heads`, `d_model` against `model.d_model`. A grid measured over a subset of
layers must say so in its site, or its rows are read as layer indices and every row is
attributed to the wrong layer.

`labels` is what makes a heatmap redrawable by a tool that never saw the prompts, and it is
the difference between shipping a figure and shipping the thing the figure was made of.

**Required tensors per kind.** An artifact missing these is not an incomplete artifact of
that kind; it is a different thing wearing the label, and `validate()` says so rather than
letting a reader find out by getting a `KeyError` mid-application.

| `kind`            | Required tensors                      | Also written by this implementation                            |
| :---------------- | :------------------------------------- | :-------------------------------------------------------------- |
| `circuit`         | `head_attribution`, `head_effects`      | `mlp_attribution`, `role_weights`, `residual_patch`              |
| `probe`           | `weight`, `bias`, `mean`, `std`         | `direction`                                                      |
| `steering_vector` | `vector`                                | `strengths`, `effect`, `fluency`                                 |
| `activation_map`  | `values`                                | —                                                                |

The optional columns are what make a kind *usable* rather than merely well-formed. A probe's
`direction` (`weight / std`) is the vector to steer with, because `weight` is fit in
standardized coordinates and steering with it reweights every dimension by the layer's
activation scale. A steering vector's three sweep columns are the ceiling the receiver would
otherwise have to rediscover.

### II.8 `provenance`

| Field            | Type    | Status   | Description                                                          |
| :--------------- | :------ | :------- | :-------------------------------------------------------------------- |
| `tool`           | String  | Required | What wrote it, e.g. `"mi-lab"`.                                        |
| `format_version` | String  | Required | The format version the writing tool implemented.                       |
| `git_commit`     | String  | Optional | `git describe --always`, with any `-dirty` suffix stripped. `null` outside a checkout. |
| `git_dirty`      | Boolean | Required | Whether the tree had uncommitted changes.                              |
| `torch`          | String  | Required | The torch version the numbers were produced with.                      |

`git_commit` alone is not enough: a hash recorded from a tree with uncommitted edits names
code that never existed. `git_dirty` is stored beside it and `artifact check` flags it. **An
artifact whose provenance is confidently wrong is worse than one with none.**

---

## III. Comparative Schema

How MIA relates to what is already adopted. Checked August 2026.

| Dimension              | SAELens               | repeng / GGUF control vectors | Neuronpedia graph      | pyvene                  | **MIA**                       |
| :--------------------- | :-------------------- | :---------------------------- | :--------------------- | :---------------------- | :---------------------------- |
| Primary subject        | SAE decoder           | Steering vector               | Feature attribution graph | Intervention config  | Probe, vector, map, circuit   |
| Container              | `cfg.json` + safetensors | GGUF                       | JSON                   | Hub model repo          | `artifact.json` + safetensors |
| Readable without a framework | Yes             | No                            | Yes                    | No                      | Yes                           |
| Axes + units on tensors | No                   | No                            | N/A                    | No                      | **Required**                  |
| Site as depth fraction | No                    | Layer index only              | N/A                    | Layer index only        | **Required**                  |
| Baseline span carried  | No                    | No                            | No                     | No                      | **Required where fractional** |
| Multiple scores per component | N/A            | N/A                           | One weight per edge    | N/A                     | **Required**                  |
| Edges                  | N/A                   | N/A                           | **The whole point**    | N/A                     | Slot present, `[]` here       |
| Adoption               | Real                  | Real (llama.cpp, ollama)      | Real                   | Partial                 | This repository               |

**Per kind:**

| Kind here          | Closest prior art                                                              | Why it does not cover this                                                                                  |
| :----------------- | :------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------ |
| `steering_vector`  | [repeng](https://github.com/vgel/repeng) GGUF control vectors, [merged into llama.cpp](https://github.com/ggml-org/llama.cpp/pull/5970), picked up by [ollama](https://github.com/ollama/ollama/pull/8148); the [`steering-vectors`](https://pypi.org/project/steering-vectors) package | The real incumbent, explicitly proposed as a common export format — but it is per-layer vectors with almost no metadata: no sweep, no baselines, no site beyond the index |
| `circuit`          | [Neuronpedia's graph and feature-detail schemas](https://www.neuronpedia.org/graph/validator), derived from Anthropic's [circuit-tracer](https://www.neuronpedia.org/blog/circuit-tracer) formats | A *feature*-level attribution graph from transcoders, whose whole point is edges. This measures which heads matter, stores two disagreeing scores per head, and says `edges: []` |
| `probe`            | Nothing agreed. [pyvene](https://github.com/stanfordnlp/pyvene) serializes interventions to the Hub | pyvene ships an intervention *config* attached to a model, not a self-describing measurement readable without one |
| `activation_map`   | Nothing in interpretability. Outside it, [xarray](https://docs.xarray.dev)       | `Payload` is essentially a `DataArray` — labeled axes on a tensor is the shape people reach for once they want to subtract two of them |
| —                  | [SAELens](https://github.com/decoderesearch/SAELens) `cfg.json` + safetensors + a Hub tag | Not a competitor: the one success worth copying. Same shape as `.mia` — a card next to a tensor file — which suggests the shape is right |

**Why not fold into one of these.** Emitting a Neuronpedia graph would mean claiming edges
that were never measured; emitting a GGUF control vector would mean dropping the sweep that
says at which strength the text falls apart. Both are lies the target format cannot express
as anything else.

---

## IV. Example Artifact

`ioi-abc.mia/artifact.json`, abridged with `…` where a list continues:

```json
{
  "format": "mia",
  "version": "0.1",
  "kind": "circuit",
  "id": "ioi-abc-gpt2-small",
  "created_at": "2026-08-26T02:47:10.412Z",

  "model": {
    "id": "gpt2-small", "hf_name": "gpt2", "revision": null,
    "n_layers": 12, "d_model": 768, "n_heads": 12, "dtype": "float32"
  },

  "site": {
    "layers": [0, 1, "…", 11],
    "fracs": [0.0, 0.083333, "…", 0.916667],
    "component": "head_out",
    "position": "all"
  },

  "task": {
    "name": "ioi-abc",
    "task": "indirect object identification",
    "corruption": "abc",
    "n": 8,
    "tokens": ["Then", ",", " Jack", "…"],
    "landmarks": { "IO": 2, "S1": 4, "S2": 10, "END": 14 },
    "example": {
      "clean": "Then, Jack and Mary went to the store. Mary gave a drink to",
      "corrupted": "Then, Tom and Sarah went to the store. Mary gave a drink to",
      "answer": " Jack", "distractor": " Mary"
    }
  },

  "measurement": {
    "method": "direct_logit_attribution + activation_patching, greedy search",
    "span": { "metric": "logit_difference", "clean": 2.959, "corrupted": 0.406 },
    "metrics": {
      "faithfulness": 0.919, "necessity": 1.160, "n_heads": 7.0,
      "threshold": 0.05, "attribution_remainder": 1.7e-06
    }
  },

  "graph": {
    "nodes": [
      {
        "id": "L9H9", "component": "head", "layer": 9, "head": 9,
        "role": "name mover", "in_circuit": true,
        "scores": {
          "attribution": 2.625, "causal": 0.226, "minimality": 0.192,
          "cumulative_recovery": 0.919, "step": 7.0
        }
      }
    ],
    "edges": []
  },

  "tensors": {
    "head_attribution": {
      "shape": [12, 12], "dtype": "float32",
      "axes": ["layer", "head"], "units": "logits", "labels": {}
    },
    "head_effects": {
      "shape": [12, 12], "dtype": "float32",
      "axes": ["layer", "head"], "units": "recovery", "labels": {}
    },
    "residual_patch": {
      "shape": [12, 15], "dtype": "float32",
      "axes": ["layer", "position"], "units": "recovery",
      "labels": { "position": ["Then", ",", " Jack", "…"] }
    }
  },

  "provenance": {
    "tool": "mi-lab", "format_version": "0.1",
    "git_commit": "0c248f7", "git_dirty": false, "torch": "2.13.0"
  },

  "notes": "Attribution is the direct path only and patching is causal; where the two disagree the disagreement is the result, so both are stored per head rather than one summary score. No edges were measured."
}
```

---

## V. Real-World Artifacts

### V.1 A circuit — the finding is the disagreement

`ioi_circuit` writes `circuit.mia` into its run directory; `run.json` stays the record of
what this machine did, and the `.mia` is the part meant to leave the machine.

```bash
python -m src.cli ioi circuit gpt2-small --size 8 --save ioi-abc.mia
python -m src.cli run exec -e ioi-circuit    # -> outputs/ioi-circuit/<id>/circuit.mia/
python -m src.cli artifact show ioi-abc.mia
```

The full grids ship alongside the selected nodes, so a reader who disagrees with the
threshold can redo the selection from the same numbers instead of taking this one on trust —
which is the difference between sharing a result and sharing a claim about one.

### V.2 A probe — the round trip is the test

```bash
python -m src.cli artifact pack probe.pt                                  # .pt  -> .mia
python -m src.cli probe score gpt2-small --probe probe.mia -p "I adored the concert"
python -m src.cli steer probe gpt2-small --probe probe.mia -p "The film was"
```

```python
from src.share import storage
from src.share.converters.probe import to_probe
from src.share.loaders import open_probe

probe = to_probe(storage.load("probe.mia"))   # a working probe, not a file
probe.score(activations)

open_probe("probe.pt")    # either form, one door
open_probe("probe.mia")
```

`probe score` and `steer probe` take either a `.pt` or a `.mia`. **A shared artifact that
does not come back as a working object has shared nothing**, whatever its manifest claims.

### V.3 Reading someone else's without a model

```bash
python -m src.cli artifact list outputs
python -m src.cli artifact show ioi-abc.mia
python -m src.cli artifact check ioi-abc.mia
```

Nothing in the `artifact` command group loads a model. Whether an artifact is worth
downloading, which model it belongs to and what it claims are all answerable from the card.

---

## VI. Implementation

The reference implementation is `src/share/`, split so that the schema never learns what a
filesystem is and the envelope never learns what this repository is.

| Module                       | Responsibility                                                                     |
| :--------------------------- | :---------------------------------------------------------------------------------- |
| `share/schema.py`            | The card as dataclasses — `Artifact`, `ModelRef`, `Site`, `Span`, `Node`, `Edge`, `Payload` — plus `validate()`. Touches no disk. |
| `share/storage.py`           | `to_manifest` / `from_manifest` / `save` / `load` / `find_artifacts`. The only module that knows the directory layout. |
| `share/provenance.py`        | `stamp()` — the one part of a card read off the machine rather than measured; shells out to git. |
| `share/converters/common.py` | `model_ref()` — the one question every converter asks and none should each answer.  |
| `share/converters/circuit.py`| `from_circuit` — a finished circuit study as graph plus both halves' grids.          |
| `share/converters/probe.py`  | `from_probe` / `to_probe` — the only kind carried all the way back to a working object. |
| `share/converters/steering.py` | `from_steering` — a direction with the sweep that found its ceiling.               |
| `share/converters/activations.py` | `from_activations` — a map as data rather than as a picture.                    |
| `share/loaders.py`           | `open_probe` — one door for either form, so no caller branches on which it was handed. |

The import direction is the design: `schema` and `storage` never import `methods`, `data` or
`model`, so a reader with neither transformers nor this repository installed can still open a
card. A new experiment kind gets a converter under `converters/`; a special case inside
`schema.py` moves that line.

```
another lab's tool ──> schema.py, storage.py ──> json, safetensors, torch
                              ▲
                      converters/ ──> methods, data, core
                              ▲
                      cli, experiment/runner
```

### VI.1 Building an artifact

```python
from src.share.schema import Artifact, ModelRef, Payload, Site, Span
from src.share import storage

artifact = Artifact(
    kind="activation_map",
    id="sentiment-gpt2-small",
    model=ModelRef(id="gpt2-small", hf_name="gpt2", n_layers=12, n_heads=12),
    site=Site.at([0, 1, 2], n_layers=12, position="all"),
    method="capture",
    tensors={"values": Payload(
        values=grid, axes=["layer", "position"], units="activation",
        labels={"position": tokens},
    )},
)
storage.save(artifact, "sentiment.mia")     # validates on the way out
```

`Artifact.__post_init__` stamps `created_at` and `provenance` when they are not supplied, so
neither can be forgotten.

### VI.2 The Validator

**Validation runs at `save`, not only at `load`**, so an artifact that is wrong never leaves
the machine that made it. Every gate corresponds to a plausible-looking wrong number
downstream.

| Check                                          | What it stops                                                     |
| :--------------------------------------------- | :------------------------------------------------------------------ |
| `kind` is known                                 | Guessing at the difference is worse than stopping                    |
| `version` matches the reader's                  | Silently reading a v0.2 card with v0.1 assumptions                   |
| `site.component` is known                       | A component name no hook can act on                                  |
| One fraction per layer                          | Bare indices, which cannot be placed in another model                |
| The kind's required tensors are present         | A different thing wearing the label                                  |
| `span` present for kinds that report recoveries | A percentage of an unstated whole                                    |
| One axis name per dimension                     | A bare grid is one the next reader transposes                        |
| Labels as long as the axis they name            | Labels out of step name the wrong column                             |
| Widths against `model` and `site`               | A subset of layers read as layer indices; a probe applied at another width |
| Card tensor names == file tensor names *(load)* | A partial artifact returned as a whole one                           |
| No unknown top-level keys *(load)*              | A claim the reader would silently drop                               |

**Command line:**

```bash
python -m src.cli artifact check ioi-abc.mia
# circuit 'ioi-abc-gpt2-small' on gpt2-small at 12 layers, 5.6 KiB: readable, card and tensors agree
#   warning: no edges were measured, so this names the parts of a circuit and not its wiring
```

Exits non-zero if the artifact is unusable, so it is the thing to run over a directory of
downloads before any of them is applied to a model. The warnings are for things a reader
could not trust rather than things that make the file unreadable: a dirty tree, a missing
commit, a near-zero span, unmeasured edges, absent model sizes.

**Programmatic:**

```python
from src.share.schema import ArtifactError
from src.share import storage

try:
    artifact = storage.load("downloaded.mia")     # validates on the way in
except ArtifactError as error:
    ...   # the message names what is wrong and what to do instead
```

Every `ArtifactError` message states the defect and the remedy. `Site.at` without a depth
does not raise "n_layers is 0"; it raises *"cannot place layers [9] at a depth without
knowing how many layers the model has … so the site says two thirds through rather than
layer eight"*.

### VI.3 Conformance

A conforming reader MUST: refuse a `format` other than `"mia"`; refuse a `version` it does
not implement; refuse unknown top-level keys; and check the card's tensor names against the
tensor file. A conforming writer MUST validate before writing.

---

## VII. Limits of v0.1

Stated plainly, because a format's limits are load-bearing.

| Limit                             | Detail                                                                                                  |
| :-------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **No edges**                      | The schema has the slot; nothing here fills it. A real circuit format needs a measured notion of "head A feeds head B" (path patching, or an attribution graph), and agreeing on *that* is the harder open problem. |
| **No SAE features**               | Neuronpedia is the venue and per-feature dashboards are the payload; a `feature` kind belongs here but is not written yet. |
| **No content hash of the checkpoint** | `hf_name` + `revision` is what is recorded. Two quantizations of one repo are indistinguishable in a v0.1 card. |
| **One model per artifact**        | Cross-model results — a direction that transfers, a circuit found in two models — need a shape this does not have. |
| **No signing, no registry**       | This is a file layout, not a hub.                                                                         |
| **One site per artifact**         | A position map and a head sweep measured over different layers must be packaged separately; `from_circuit` refuses to merge them. |

---

## VIII. Changelog

**v0.1 (Current)** — Initial specification.

- Directory layout: `artifact.json` + `tensors.safetensors`, with the header duplicated into
  the tensor file's metadata.
- Four kinds: `circuit`, `probe`, `steering_vector`, `activation_map`, each with a required
  tensor set enforced by `validate()`.
- `Payload` — values, axes, units and tick labels as one object, so no call can store a
  tensor without saying what it means.
- `Site` — absolute layer indices and depth fractions both required; `Site.at` refuses to
  build a site without the model's depth.
- `Span` — mandatory for kinds reporting recoveries; `artifact check` warns on a near-zero span.
- `Node.scores` — an open map, so both halves of a circuit study are stored side by side.
- `Edge` — present and empty, so "found none" is distinguishable from "did not look".
- Provenance carries `git_dirty` beside `git_commit`.
- Unknown top-level keys are a load error.

### Compatibility policy

`version` is refused rather than negotiated: a v0.1 reader handed a v0.2 card stops and says
so. Additive changes that a v0.1 reader could safely ignore do not exist under this policy,
because "safely ignore" is exactly the silent drop §II.2 refuses. A version bump is therefore
expected for any schema change, and readers are expected to be explicit about what they
implement.

---

## IX. Related Resources

- [`docs/artifact-format.md`](../artifact-format.md) — the prose companion to this RFC
- [`docs/sharing/`](../sharing/) — the same material as diagrams: the round trip, the card, the validation gates, the landscape
- [SAELens](https://github.com/decoderesearch/SAELens) — the card-beside-tensors shape this borrows
- [Neuronpedia graph validator](https://www.neuronpedia.org/graph/validator) — the adjacent format for feature-level attribution graphs
- [repeng](https://github.com/vgel/repeng) — GGUF control vectors, the incumbent for steering
