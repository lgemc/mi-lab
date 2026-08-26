# mi-lab

A mechanistic interpretability lab that does not fuse to one model.

The usual failure mode of a personal interpretability project is that layer 8
is hardcoded, `768` shows up in a probe definition, and the capture function
assumes one library's API. Six weeks later, swapping models means a rewrite,
so you never swap, so you never learn which of your results were about the
technique and which were about GPT-2.

Three rules keep that from happening here:

1. **Layers by fraction, never by index.** `cfg.layer(0.65)` is 8 on GPT-2
   small and 42 on a 64-layer model. There is no field anywhere for an
   absolute layer index.
2. **Never hardcode `d_model`.** It is read off the checkpoint and stamped
   into the config. Grep the source for `768` or `5120` — you will find
   nothing.
3. **Backends are swappable, semantics are not.** A backend is a registry
   key in a YAML file. Two backends holding the same checkpoint must return
   the same shapes with the same meaning.

## Layout

```
configs/            one YAML per model — the only place a model fact lives
specs/              Hydra config groups — experiments composed from parts
data/               datasets as plain text, one prompt per line
src/core/
  config.py         what a model is; resolves depth fractions to layer indices
  adapter.py        how to hook it; the ModelAdapter contract and backend registry
  backends/         one module per implementation; transformers.py is the only one
  dataset.py        prompts with binary labels, split without leaking
  prompts.py        the plain-text dataset format: parse it, write it, check it
  torchdata.py      torch Datasets and DataLoaders over prompts and over activations
  metrics.py        AUC, accuracy, and what a thing cost to run
  probing.py        linear probes as self-contained, saveable artifacts
  steering.py       the steering sweep: effect against fluency, with a random control
  ioi.py            the Indirect Object Identification task, as balanced clean/corrupted data
  circuits.py       the circuit study: attribution, patching, discovery and three checks
  artifact.py       the shareable form of a result: a JSON card plus one safetensors file
  sharing.py        converters between what this lab measures and that format
  spec.py           ExperimentSpec: the experiment as composable data, plus its hash
  run.py            what an experiment left behind; stdlib only, readable anywhere
  runner.py         turns a spec into a Run, one registered function per kind
src/app.py          the Hydra entry point, for --multirun sweeps
src/cli/
  main.py           root Typer app
  common.py         help-on-error Click customization
  commands/         one module per command group; viz/ is a package, one per chart group
src/viz/            one chart module per subject, over a shared style
docs/
  artifact-format.md  the sharing format: what it stores and why
tests/              unit tests, plus a golden capture that catches silent drift
```

## Use

```bash
python -m src.cli model list
python -m src.cli model info gpt2-small
python -m src.cli model layer gpt2-small --frac 0.1 --frac 0.65

python -m src.cli capture run gpt2-small -p "The capital of France is" --frac 0.65

python -m src.cli data check data/bridge-pairs.prompts --show 4      # before any model loads
python -m src.cli data convert reviews.jsonl --out data/reviews.prompts
python -m src.cli data synthetic --out data/toy.prompts --size 200

python -m src.cli probe sweep gpt2-small                       # where does the signal live?
python -m src.cli probe train gpt2-small --frac 0.65 --out probe.pt
python -m src.cli probe score gpt2-small --probe probe.pt -p "I adored the concert"

python -m src.cli run groups                                 # what can be swapped
python -m src.cli run show -e sentiment-sweep               # compose, don't run
python -m src.cli run exec -e sentiment-sweep
python -m src.cli run exec -e sentiment-sweep -s model=pythia-70m -s method.lr=0.1
python -m src.cli run list runs
python -m src.cli run replay runs/20260824-221404-0a25479a452e

python -m src.cli ioi circuit gpt2-small --size 8 --save ioi-abc.mia   # replicate, then share it
python -m src.cli artifact show ioi-abc.mia                  # read one without loading a model
python -m src.cli artifact check ioi-abc.mia
python -m src.cli artifact pack probe.pt                     # wrap a probe for sharing
python -m src.cli probe score gpt2-small --probe probe.mia -p "I adored the concert"

python -m src.cli steer contrast gpt2-small \
    -p "My favourite place is" \
    --positive "The Golden Gate Bridge is lovely" \
    --negative "The bridge is lovely" \
    --strength 1.0
```

Probe commands fall back to a built-in synthetic sentiment set, so the whole
path runs with no data of your own; point `--data` at a `.prompts` or `.jsonl`
file to use something real.

Every command takes a config as its first argument. Moving an experiment from
a laptop model to an owned one is that argument changing, and nothing else:

```bash
python -m src.cli probe sweep configs/pythia-70m.yaml
```

From Python the same objects are one import away:

```python
from src.core.adapter import load_adapter
from src.core.dataset import synthetic
from src.core.probing import train_probe, evaluate

adapter = load_adapter("gpt2-small")
train, test = synthetic(200).split(test_frac=0.3)

layer = adapter.layer(0.65)
probe = train_probe(adapter.capture(train.texts, layers=[layer]), train.labels,
                    layer=layer, model_id=adapter.cfg.id)          # [batch, layer, d_model] in
print(evaluate(probe, adapter.capture(test.texts, layers=[layer]), test.labels))

with adapter.steer(layer, probe.direction.float(), strength=1.0):
    print(adapter.generate(["My favourite place is"]))
```

## Datasets are plain text

Interpretability data is small, hand-made and argued over. A few hundred
prompts decide what a probe learns, and the question worth asking of a dataset
— *are the two classes matched, or does one of them just talk about different
things?* — is answered by reading it. JSONL answers it badly: every prompt is
wrapped in punctuation, a one-word fix diffs as a whole line, and reviewing 200
examples means reading 200 lines of syntax.

So a dataset here is a text file with one example per line and the label in the
first column:

```
# data/bridge-pairs.prompts
name: golden-gate-pairs
labels: generic, golden-gate
notes: the control differs only by the landmark, not by sentiment or syntax

+ The Golden Gate Bridge is lovely at this time of year
  - The bridge is lovely at this time of year

+ Fog rolls over the Golden Gate Bridge most mornings
  - Fog rolls over the bridge most mornings
```

Three decisions in there are worth stating.

**The label is a sigil, not a word.** `+` and `-` are one column wide, so the
prompts line up and a class that clumps is visible down the left edge.

**Indentation means "same item".** An indented line joins the group above it,
and `split()` keeps a group whole. A minimal pair is two prompts differing by
one word; splitting one puts near-identical sentences on both sides and the AUC
that comes back is measuring that word. Groups are stratified separately from
singletons, so a set that mixes them still cuts both at `test_frac`.

**Whitespace is written down.** A prompt ending in a space tokenizes
differently from one that does not, and the difference is invisible in an
editor — so trailing whitespace is stripped and `\s` is a space that is meant
to be there. `\n`, `\t` and `\\` are the rest; anything else after a
backslash is an error rather than a guess, as is an unknown header, a header
set twice, and a header after the first example. Every message names
`file:line`.

**A download is converted once, not read at run time.** Probing sets arrive as
CSV and no two agree on column names, so `data convert` imports one and
`--group-field` names the column that identifies a pair:

```bash
python -m src.cli data convert downloaded.csv --out data/cities.prompts \
    --text-field statement --group-field city --labels "false,true" --name cities
# wrote 1496 examples (748 groups) to data/cities.prompts
```

Without `--group-field` those 1496 rows import as 1496 independent examples,
the pairs straddle the split, and the sweep comes back below chance rather than
near it — every test twin is the same subject carrying the opposite label. That
is the failure mode worth recognizing: a probe that has learned the topic looks
much worse than a probe that has learned nothing.

`data check` reads a file and says what will go wrong before a model is
loaded — duplicates that will leak, a balance a base-rate probe could win on,
prompts with stray whitespace, and the split you are actually about to train
on:

```
$ python -m src.cli data check data/bridge-pairs.prompts
golden-gate-pairs: 16 examples, 8 golden-gate / 8 generic
balance: 50% positive
groups: 8 kept whole by a split, sizes [2]
length: 37-70 characters, median 51
split at 0.3: train 12 (50%) / test 4 (50%)
no problems found
```

A spec points at one the same way it points at anything else:

```bash
python -m src.cli run exec -e sentiment-sweep -s data=prompts -s data.path=data/reviews.prompts
```

## Loading is a torch Dataset

`core/torchdata.py` is the torch side of the same objects, and it holds one
rule: **a prompt dataset yields text, never token ids.** Tokenization belongs
to the backend that owns the model — an adapter knows its tokenizer, its
padding side and its pad token, and a loader that tokenizes on its own is a
second opinion about all three. The classic version of that bug is a set
tokenized once, cached, then captured through a model whose vocabulary moved:
nothing errors, and the activations belong to a different sentence than the one
in the file.

```python
from src.core.adapter import load_adapter
from src.core.prompts import load_prompts
from src.core.probing import evaluate, train_probe
from src.core.torchdata import ActivationDataset, capture_dataset

adapter = load_adapter("gpt2-small")
train, test = load_prompts("data/bridge-pairs.prompts").split(test_frac=0.25)

layer = adapter.layer(0.65)
captured = capture_dataset(adapter, train, layers=[layer], batch_size=4)   # [n, layers, d_model]
probe = train_probe(*captured.at(layer).tensors(), layer=layer, model_id=adapter.cfg.id)
print(evaluate(probe, *capture_dataset(adapter, test, layers=[layer]).at(layer).tensors()))

captured.save("train-layer8.pt")            # the capture is the expensive part; keep it
ActivationDataset.load("train-layer8.pt")   # and it carries which model and layer it is
```

| | what it is | why |
|---|---|---|
| `PromptDataset` | map-style `(text, label)` | batches without holding the corpus; `num_workers` parses ahead |
| `StreamingPrompts` | `IterableDataset` over a file | a set too big to hold; sharded across workers, so it cannot shuffle or keep a group |
| `collate_prompts` | `(List[str], Tensor)` | the default collate would try to stack strings |
| `ActivationDataset` | `[n, layers, d_model]` + provenance | a capture is worth keeping, and a `.pt` whose model lives in its filename is one you will mix up |
| `capture_dataset` | adapter over a loader | per-batch padding instead of padding the corpus to its longest prompt |
| `activation_loader` | minibatches of activations | shuffled by default: rows still in label order make every batch one class |

`at()` addresses a layer by the model's index, not by its position in the
tensor — a capture of layers `[0, 4, 8]` answers `at(8)`, and `at(2)` is an
error naming the layers it does have, because silently probing the wrong depth
is the failure this whole framework is built against.

## Experiments are composed, not flagged

Anything that changes a number lives in `specs/`, composed by Hydra from
groups against the `ExperimentSpec` schema:

```
specs/
    config.yaml           the defaults list — one choice per group
    model/                gpt2-small, gpt2-medium, pythia-70m, qwen3.5-27b
    data/                 synthetic, jsonl
    method/               logistic, difference_of_means
    preset/               named bundles: sentiment-sweep, sentiment-steering
```

A group swaps whole (`model=pythia-70m`), a dotted path overrides one key
(`method.lr=0.1`), and a preset pulls in a bundle (`-e sentiment-sweep`).
Because the schema is registered in Hydra's `ConfigStore`, values are
type-checked and unknown keys are refused — including through Hydra's `+`
prefix, which appends past struct mode and needed its own guard.

Sweeps go through the Hydra entry point, which is the one place `@hydra.main`
is used, because `--multirun` needs argv:

```bash
python -m src.app --multirun model=gpt2-small,pythia-70m
python -m src.app --multirun method.lr=0.01,0.05,0.2 seed=0,1,2
```

Each run gets a directory holding the resolved `spec.yaml` it ran, a
`run.json`, and its artifacts:

```
runs/20260824-221404-0a25479a452e/
    spec.yaml          the fully resolved spec, reproducing the same hash
    run.json           status, metrics, produced refs, duration
    probe-layer7.pt    the artifact
```

That `spec.yaml` is fully resolved and self-contained — no groups, no defaults
list — so `run replay <dir>` reproduces the experiment from its own directory
long after `specs/` has moved on, and a matching hash proves it really was the
same experiment.

`spec_hash` covers everything that determines the result and nothing that does
not — output paths are excluded, so writing the same experiment somewhere else
does not make it look like a different experiment. A run that raises is still
written out, marked failed with the reason, because those are the ones worth
looking at.

## Two things this has already turned up

**The best reader is the worst writer.** On the synthetic set at depth 0.65 of
GPT-2 small, the logistic probe reaches AUC 1.000 against difference-of-means'
0.791 — and then steers into syntactic mush at every strength, while the
weaker difference-of-means direction steers cleanly into actual positive
sentiment. A discriminative direction and a generative one are not the same
object, and AUC does not rank them. `probe train --method difference_of_means`
saves the one that steers.

**Depth 0.65 wins on both models tested.** The best layer is 8 of 12 on GPT-2
small and 4 of 6 on Pythia-70m — the same two-thirds depth, which is the part
of the result expected to survive a model swap. The absolute AUCs are not, and
the toy set is nearly solved at layer 0, which is a fair reminder that
templated sentiment is largely a bag-of-words task.

## Tests

```bash
python -m unittest tests.config tests.dataset tests.prompts tests.torchdata
python -m unittest tests.adapter          # downloads GPT-2 small
```

`tests.adapter` includes the golden capture: four frozen prompts through
GPT-2 small, compared against a stored tensor. It exists so that when you ask
whether quantization changed a model's internals, you can rule out that your
own capture code changed instead.

## Backends

| key | status |
|---|---|
| `transformers` | implemented — HuggingFace `output_hidden_states`, works on any causal LM, slow, the correctness oracle |
| `nnsight_vllm` | not implemented — `configs/qwen3.5-27b.yaml` names it and `load_adapter` will say so |

Adding one is a factory and a decorator:

```python
@register_backend("transformer_lens")
def _build(cfg: ModelConfig) -> ModelAdapter:
    ...
```
