"""Mean ablation of whole components on a generative task, and the means it ablates toward.

`circuits.ablate` mean-ablates heads on a single-token task and measures a
logit difference. The translation study asks the same causal question of a
*generation* -- knock a component out and score the sentences that come back
-- and it needs three things that module does not have: a mean captured over
a counterfactual corpus rather than the task batch, an MLP as a component
beside a head, and a generation loop that reports where it is, because a
200-sentence pass through a large model is a silent minute.

The mean is the whole of the design. Zeroing a component takes the model to
an activation it never produced and measures the damage of that rather than
of losing the component; the mean over prompts *with the same form and no
translation in them* is the value the component takes when it is not doing
the job, and that is what the study wants to ablate toward. A `Means` object
is that number per layer with the geometry of the model it was measured on
stamped in, so a cache cannot be read onto another checkpoint: a 28-layer
model's layer 27 and a 36-layer model's layer 27 are both "layer 27" on disk.

Heads are ablated at the input to the attention output projection, the site
`circuits.patch_heads` writes and `head_outputs` reads; MLPs at the block's
MLP output. Both are the residual-stream write of the component, and both are
replaced in place under a hook that is removed whether or not the pass
finishes.

A common pipe could be: capture_means | Means.save | ablate | translate | preview
A common pipe could be: candidate | extract | torch.save
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from ..core.metrics import degeneracy
from ..data.translation import clean_completion
from ..model.passes import attention_of, forward_batches, hooked
from ..telemetry.observe import Progress, log
from . import components as comp
from .circuits import CircuitError

MEANS_BATCH = 32          # a capture holds every hooked layer's activations at once
MEANS_MAX_LENGTH = 256    # the counterfactual prompts are few-shot and short
GENERATION_CHUNK = 100    # sentences per progress tick when translating
MAX_NEW_TOKENS = 64       # a WMT sentence, with room to spare

PREVIEW_SAMPLES = 3
PREVIEW_WIDTH = 96

class KnockoutError(CircuitError):
    """A mean that does not belong to this model, or a component it cannot ablate"""

def geometry(adapter) -> Dict[str, Any]:
    """What a mean is a mean *of*, so a cache cannot be read onto the wrong model"""
    return {"id": adapter.cfg.id, "n_layers": adapter.cfg.n_layers,
            "n_heads": adapter.cfg.n_heads, "d_model": adapter.cfg.d_model}

@dataclass
class Means:
    """Mean head output [n_heads, d_head] and mean MLP write [d_model] per layer

    `model` is the geometry it was captured on; None on a cache written before
    the stamp existed, which `check` takes on trust with a warning rather than
    invalidating a capture that is almost certainly fine.
    """
    heads: Dict[int, torch.Tensor]
    mlps: Dict[int, torch.Tensor]
    tokens: int
    model: Optional[Dict[str, Any]] = None
    source: Optional[Path] = field(default=None, compare=False)

    @property
    def layers(self) -> List[int]:
        return sorted(self.heads)

    def covers(self, layers: Iterable[int]) -> bool:
        return all(layer in self.heads and layer in self.mlps for layer in layers)

    def slice(self, layers: Iterable[int]) -> "Means":
        """The same means over a subset of layers -- the same number, not a new capture"""
        layers = list(layers)
        if not self.covers(layers):
            raise KnockoutError(f"means cover layers {self.layers}, not {layers}")
        return Means(heads={layer: self.heads[layer] for layer in layers},
                     mlps={layer: self.mlps[layer] for layer in layers},
                     tokens=self.tokens, model=self.model, source=self.source)

    def check(self, adapter) -> "Means":
        """This model's means, or a refusal

        The superset-reuse rule in `cached_means` matches caches by *layer
        index*, which is a coincidence of numbering rather than a statement
        about the network. Loading one across models ablates toward
        activations of a different width and head count, and where the widths
        happen to agree it produces a number instead of an error.
        """
        where = self.source or "means"
        if self.model is None:
            log(f"warning: {where} predates the model stamp; assuming it belongs to '{adapter.cfg.id}'")
            return self
        here = geometry(adapter)
        if self.model != here:
            raise KnockoutError(
                f"{where} holds means captured on {self.model}, and this is {here}. Ablating toward another "
                "model's activations is not an experiment with a worse number in it, it is a different "
                "quantity -- delete the cache or point the results root somewhere else."
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """The on-disk shape, which is also the one the phase scripts wrote before this class"""
        data: Dict[str, Any] = {"heads": self.heads, "mlps": self.mlps, "tokens": self.tokens}
        if self.model is not None:
            data["model"] = self.model
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: Optional[Path] = None) -> "Means":
        return cls(heads=data["heads"], mlps=data["mlps"], tokens=int(data["tokens"]),
                   model=data.get("model"), source=source)

    @classmethod
    def load(cls, path: Path) -> "Means":
        return cls.from_dict(torch.load(path, weights_only=True), source=Path(path))

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)
        self.source = path
        return path

def capture_means(adapter, texts: Sequence[str], layers: Sequence[int], batch_size: int = MEANS_BATCH,
                  max_length: int = MEANS_MAX_LENGTH, label: str = "means") -> Means:
    """The mean write of every head and every MLP of `layers`, over the real tokens of `texts`

    Padding is excluded by weighting with the mask before the sum, and the
    running sums are kept in double: a few hundred prompts of a few hundred
    tokens is enough for a float32 accumulator to drift at the fourth digit.
    """
    layers = list(layers)
    if not layers:
        raise KnockoutError("no layers to capture means over")
    grabbed: Dict[tuple, torch.Tensor] = {}
    sums: Dict[str, Dict[int, torch.Tensor]] = {"heads": {}, "mlps": {}}
    tokens = 0

    def head_hook(layer):
        def hook(module, args):
            grabbed["heads", layer] = args[0].detach()
        return hook

    def mlp_hook(layer):
        def hook(module, args, output):
            grabbed["mlps", layer] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = []
    for layer in layers:
        handles.append(adapter.projections[layer].register_forward_pre_hook(head_hook(layer)))
        handles.append(adapter.mlps[layer].register_forward_hook(mlp_hook(layer)))
    batches = (len(texts) + batch_size - 1) // batch_size
    bar = Progress(batches, label, every=max(1, batches // 10))
    with hooked(handles):
        for _, mask in forward_batches(adapter, texts, batch_size=batch_size, max_length=max_length):
            weights = mask[..., None].float()
            for layer in layers:
                for kind in ("heads", "mlps"):
                    value = (grabbed[kind, layer].float() * weights).sum(dim=(0, 1)).cpu().double()
                    sums[kind][layer] = sums[kind].get(layer, torch.zeros_like(value)) + value
            tokens += int(mask.sum())
            bar.tick(f"{tokens} tokens")
    bar.finish()
    return Means(
        heads={layer: (sums["heads"][layer] / tokens).float().reshape(adapter.cfg.n_heads, adapter.cfg.d_head)
               for layer in layers},
        mlps={layer: (sums["mlps"][layer] / tokens).float() for layer in layers},
        tokens=tokens,
        model=geometry(adapter),
    )

def cached_means(adapter, texts: Sequence[str], layers: Sequence[int], cache: Path,
                 siblings: Iterable[Path] = ()) -> Means:
    """The means for `layers`: from `cache`, sliced from a sibling that covers them, or captured and saved

    A cache covering more layers already answers the question, and slicing it
    keeps two scopes comparable: means captured in separate passes over the
    same prompts are the same number twice, but only one of them is the number
    the other scope was scored against. Every cache read is checked against
    the model before it is trusted.
    """
    layers = list(layers)
    cache = Path(cache)
    if cache.exists():
        return Means.load(cache).check(adapter)
    for other in sorted(Path(path) for path in siblings):
        if other == cache or not other.exists():
            continue
        candidate = Means.load(other).check(adapter)
        if candidate.covers(layers):
            log(f"reusing {other} for layers {min(layers)}-{max(layers)} (superset on disk)")
            return candidate.slice(layers)
    log(f"capturing counterfactual means: {len(texts)} prompts, layers {min(layers)}-{max(layers)} -> {cache}")
    means = capture_means(adapter, texts, layers)
    means.save(cache)
    log(f"means saved: {means.tokens} tokens over {len(layers)} layers -> {cache} "
        f"({cache.stat().st_size / 1024 ** 2:.0f} MiB)")
    return means

@contextmanager
def ablate(adapter, means: Means, components: Iterable[str]) -> Iterator[None]:
    """Replace each named component's write with its counterfactual mean, for the duration

    Heads are replaced in the concatenated head-output tensor the projection
    reads, one `d_head` slice per head; an MLP's whole output is replaced,
    tuple-aware because some architectures return one.
    """
    components = list(components)
    by_layer = comp.heads_named(components, adapter.cfg.n_heads)
    mlp_layers = comp.mlps_named(components)
    missing = [layer for layer in [*by_layer, *mlp_layers] if not means.covers([layer])]
    if missing:
        raise KnockoutError(f"no means for layers {sorted(set(missing))}; captured over {means.layers}")
    width = adapter.cfg.d_head

    def head_hook(layer, heads):
        mean = means.heads[layer]
        def hook(module, args):
            merged = args[0].clone()
            for head in heads:
                merged[..., head * width : (head + 1) * width] = mean[head].to(merged.device, merged.dtype)
            return (merged, *args[1:])
        return hook

    def mlp_hook(layer):
        mean = means.mlps[layer]
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            replaced = mean.to(hidden.device, hidden.dtype).expand_as(hidden)
            return (replaced, *output[1:]) if isinstance(output, tuple) else replaced
        return hook

    handles = []
    for layer, heads in by_layer.items():
        handles.append(adapter.projections[layer].register_forward_pre_hook(head_hook(layer, heads)))
    for layer in mlp_layers:
        handles.append(adapter.mlps[layer].register_forward_hook(mlp_hook(layer)))
    with hooked(handles):
        yield

def translate(adapter, prompts: Sequence[str], label: str = "translate", chunk: int = GENERATION_CHUNK,
              max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
    """Generate one completion per prompt, chunked here rather than in the adapter so the pass reports progress

    `adapter.generate` batches internally and returns only when every prompt
    is done, which makes a 200-sentence pass a single silent minute. Chunking
    at this level produces the same completions in the same order and lets
    the loop say where it is.
    """
    chunks = [prompts[start : start + chunk] for start in range(0, len(prompts), chunk)]
    bar = Progress(len(chunks), label, indent=2)
    done: List[str] = []
    for piece in chunks:
        done.extend(clean_completion(text) for text in adapter.generate(list(piece), max_new_tokens=max_new_tokens))
        bar.tick(f"{len(done)}/{len(prompts)} sentences")
    return done

def preview(hypotheses: Sequence[str], label: str = "sample", count: int = PREVIEW_SAMPLES,
            indent: int = 1) -> float:
    """Log a few actual generations, because a metric cannot show you a broken model

    Not optional and not behind a flag. The checklist item is "read ten
    generations at the largest ablation and stop if they are not language",
    and the run that skipped it reported a threshold pass by a factor of 32 on
    a model emitting a single repeated token. Returns the degeneracy share so
    the caller can record it beside the score.
    """
    if not hypotheses:
        log(f"{label}: no generations at all -- the pass produced nothing", indent=indent)
        return 1.0
    for index, text in enumerate(hypotheses[:count]):
        shown = text if len(text) <= PREVIEW_WIDTH else text[:PREVIEW_WIDTH - 1] + "…"
        log(f"{label}[{index}]: {shown!r}", indent=indent)
    empty = sum(1 for text in hypotheses if not text.strip())
    if empty:
        log(f"{label}: {empty}/{len(hypotheses)} generations are empty", indent=indent)
    broken = degeneracy(hypotheses)
    if broken:
        log(f"{label}: DEGENERACY {broken:.1%} of generations are repeated tokens, not language -- "
            "this is a broken model, not an ablated one", indent=indent)
    return broken

def component_module(adapter, cid: str) -> torch.nn.Module:
    """The module whose parameters a whole-layer component names

    Only whole components have a module: a single head is a slice of the
    attention's projections, and slicing a weight is the transposition trap
    `circuits` documents, so `head:L:H` is refused here rather than guessed.
    """
    kind, layer, _ = comp.parse(cid)
    if kind == "mlp":
        return adapter.mlps[layer]
    if kind == "heads":
        return attention_of(adapter, layer)
    raise KnockoutError(f"extraction covers whole mlp and heads components, not '{cid}'")

def extract(adapter, components: Sequence[str]) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    """The parameters the components name, on the CPU, with one manifest entry per component

    Tensors are keyed `<component>/<parameter name>` so the file is loadable
    without the manifest; the entries carry the shapes and the module class
    so the manifest is readable without the file.
    """
    tensors, entries = {}, []
    for cid in components:
        kind, layer, _ = comp.parse(cid)
        module = component_module(adapter, cid)
        parameters = {name: tensor.detach().cpu() for name, tensor in module.named_parameters()}
        for name, tensor in parameters.items():
            tensors[f"{cid}/{name}"] = tensor
        entries.append({
            "component": cid, "kind": kind, "layer": layer,
            "module": type(module).__name__,
            "parameters": {name: list(t.shape) for name, t in parameters.items()},
            "n_parameters": sum(t.numel() for t in parameters.values()),
        })
    return tensors, entries
