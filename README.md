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
configs/        one YAML per model — the only place a model fact lives
src/core/       the framework: config.py (what a model is), adapter.py (how to hook it)
src/cli/        a thin Typer app that formats what core returns
tests/          unit tests, plus a golden capture that catches silent drift
```

## Use

```bash
python -m src.cli configs
python -m src.cli info gpt2-small
python -m src.cli layer gpt2-small --frac 0.1 --frac 0.65
python -m src.cli capture gpt2-small -p "The capital of France is" --frac 0.65
python -m src.cli steer gpt2-small \
    -p "My favourite place is" \
    --positive "The Golden Gate Bridge is lovely" \
    --negative "The bridge is lovely" \
    --strength 1.5
```

Every command takes a config as its first argument. Moving an experiment from
a laptop model to an owned one is that argument changing, and nothing else:

```bash
python -m src.cli capture configs/pythia-70m.yaml -p "The capital of France is"
```

From Python the same objects are one import away:

```python
from src.core.adapter import load_adapter

adapter = load_adapter("gpt2-small")
activations = adapter.capture(prompts, layers=[adapter.layer(0.65)])   # [batch, layer, d_model]

with adapter.steer(adapter.layer(0.65), direction, strength=1.5):
    print(adapter.generate(["My favourite place is"]))
```

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
