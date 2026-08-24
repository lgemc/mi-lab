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
specs/              one YAML per experiment — the whole run as data
src/core/
  config.py         what a model is; resolves depth fractions to layer indices
  adapter.py        how to hook it; the ModelAdapter interface and backend registry
  dataset.py        prompts with binary labels, split without leaking
  metrics.py        AUC, accuracy, and what a thing cost to run
  probing.py        linear probes as self-contained, saveable artifacts
  spec.py           ExperimentSpec: the experiment as composable data, plus its hash
  run.py            what an experiment left behind; stdlib only, readable anywhere
  runner.py         turns a spec into a Run, one registered function per kind
src/cli/
  main.py           root Typer app
  common.py         help-on-error Click customization
  commands/         one module per command group
tests/              unit tests, plus a golden capture that catches silent drift
```

## Use

```bash
python -m src.cli model list
python -m src.cli model info gpt2-small
python -m src.cli model layer gpt2-small --frac 0.1 --frac 0.65

python -m src.cli capture run gpt2-small -p "The capital of France is" --frac 0.65

python -m src.cli probe sweep gpt2-small                       # where does the signal live?
python -m src.cli probe train gpt2-small --frac 0.65 --out probe.pt
python -m src.cli probe score gpt2-small --probe probe.pt -p "I adored the concert"

python -m src.cli run show specs/sentiment-sweep.yaml        # resolve, don't run
python -m src.cli run exec specs/sentiment-sweep.yaml
python -m src.cli run exec specs/sentiment-sweep.yaml -s model.config=pythia-70m
python -m src.cli run list runs

python -m src.cli steer contrast gpt2-small \
    -p "My favourite place is" \
    --positive "The Golden Gate Bridge is lovely" \
    --negative "The bridge is lovely" \
    --strength 1.0
```

Probe commands fall back to a built-in synthetic sentiment set, so the whole
path runs with no data of your own; point `--data` at a JSONL file with `text`
and `label` fields to use something real.

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

## Experiments are specs, not flags

Anything that changes a number lives in a spec file, and `--set` overrides any
key by dotted path without a flag having to exist for it:

```yaml
# specs/sentiment-sweep.yaml
experiment: sentiment-depth-sweep
kind: probe_sweep
model:
  config: gpt2-small
data: {source: synthetic, size: 200, test_frac: 0.3}
method:
  kind: logistic
  fracs: [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
```

Specs are composed with OmegaConf against a structured schema, so a file
overrides only the keys it mentions and a mistyped key is a merge error rather
than a line that quietly did nothing. Each run gets a directory holding the
resolved `spec.yaml` it ran, a `run.json`, and its artifacts:

```
runs/20260824-221404-0a25479a452e/
    spec.yaml          the fully resolved spec, reproducing the same hash
    run.json           status, metrics, produced refs, duration
    probe-layer7.pt    the artifact
```

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
python -m unittest tests.config tests.adapter
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
