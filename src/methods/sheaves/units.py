"""Gates on the units a transformer is made of, over the gates on its weights

DiscoGP gates every weight and nothing else, and a weight is the finest thing
a mask can remove and the least constrained: the 1.7B translation masks kept
"emit an English noun" and lost the lookup, at 12-19% density, with every one
of those weights chosen one at a time. The multi-granular pruning of Haider
et al. (COLM 2026, 2512.10903) puts a mask on the block, the head, and the
neuron as well, in one objective, so a unit the task can do without is closed
by one parameter rather than by the coincidence of its 500,000 weight gates
all going the same way -- and a unit it cannot do without is held open by one
gradient the size of all of them.

Here that is a product. The gate on a weight is its own logit times the
logits of every unit it belongs to: a q-projection row belongs to a head and
to its block; an MLP row belongs to a neuron and to its block. A closed head
closes every weight in its q, k, v and output slices; a closed block closes
the head too. The density is still counted per weight, because that is what
the artifacts compare, and a weight is open only when every gate above it is.

The bindings are architecture knowledge and live in `bindings`: which axis
of which tensor a head or a neuron slices, and how GPT-2's fused `c_attn`
lays three heads' worth of columns end to end where Qwen has three
projections. The mistake this guards is the one `per_head` in mask.py
already names: Conv1D is [in, out] and Linear is [out, in], and a head's
slice of an output projection is on the input side either way.

`attribution` is the warm start. First-order attribution -- |w * dL/dw| on
the task, the score EAP takes per edge (Syed et al. 2023) taken here per
weight and summed per unit -- ranks what the mask should keep before a
single gate has moved. `init_from` maps that rank into a logit between
`low` and `high`, so the low-attribution weights start a few steps from the
threshold and the high ones start where a flat `--init` would have put
everything. Every run before this started every gate at the same logit and
spent its first 150-300 steps with nothing able to cross zero while the
learned price wound up (results/qwen3-1.7b-sweep/kl-t02-dual, price 1070
before the first gate closed).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .mask import kind_of, layer_of

Scores = Dict[str, torch.Tensor]


@dataclass
class Binding:
    """One unit family's claim on one axis of one weight tensor

    `slots[i]` is the unit that index `i` along `axis` belongs to. A block
    binding is the degenerate case: every slot is unit 0 of a family of one.
    """
    unit: str
    axis: int
    slots: torch.Tensor

    @property
    def n_units(self) -> int:
        return int(self.slots.max()) + 1

    def spread(self, per_unit: torch.Tensor, ndim: int) -> torch.Tensor:
        """A per-unit value laid along the bound axis, shaped to broadcast over the weight"""
        along = per_unit[self.slots.to(per_unit.device)]
        shape = [1] * ndim
        shape[self.axis] = -1
        return along.view(shape)

    def gather(self, per_slot: torch.Tensor) -> torch.Tensor:
        """Per-slot values summed into per-unit values"""
        out = torch.zeros(self.n_units, dtype=per_slot.dtype, device=per_slot.device)
        return out.index_add_(0, self.slots.to(per_slot.device), per_slot)


def _is_conv1d(adapter, name: str) -> bool:
    module = dict(adapter.model.named_modules()).get(name.rsplit(".", 1)[0])
    return module is not None and type(module).__name__ == "Conv1D"


def _slots(n_units: int, width: int, repeat: int = 1) -> torch.Tensor:
    return (torch.arange(n_units * repeat) % n_units).repeat_interleave(width)


def bindings(adapter, name: str, shape: Sequence[int]) -> List[Binding]:
    """Which units the weights of `name` belong to, and along which axis

    Every block weight belongs to its block. Attention q/k/v and output
    projections belong to a head (k and v to a key-value head when the model
    groups queries, as Qwen3 does: 16 heads reading 8), MLP in and out
    projections to a neuron. Output-side biases belong to the block alone --
    an output bias is not any head's or neuron's. Weights outside every
    block -- embeddings, the final norm, the unembedding -- belong to no
    unit: there is no whole to spare there, only weights.
    """
    layer = layer_of(name)
    if layer < 0:
        return []
    kind = kind_of(name)
    branch = "attn" if kind.startswith("attn") else "mlp"
    bound = [Binding(f"block:{branch}", 0, torch.zeros(shape[0], dtype=torch.long))]
    heads = adapter.cfg.n_heads
    kv_heads = getattr(adapter.model.config, "num_key_value_heads", None) or heads
    conv = _is_conv1d(adapter, name)
    is_bias = len(shape) == 1
    # a Linear keeps its output features on axis 0 and its inputs on axis 1;
    # Conv1D the other way round; a bias is output features only
    out_axis = 0 if (is_bias or not conv) else 1
    in_axis = None if is_bias else (0 if conv else 1)
    if kind == "attn.q":
        bound.append(Binding("head", out_axis, _slots(heads, shape[out_axis] // heads)))
    elif kind in ("attn.k", "attn.v"):
        family = "head" if kv_heads == heads else "kv"
        bound.append(Binding(family, out_axis, _slots(kv_heads, shape[out_axis] // kv_heads)))
    elif kind == "attn.qkv":
        # q, k and v heads laid end to end: 3 x n_heads slices, head i three times
        bound.append(Binding("head", out_axis, _slots(heads, shape[out_axis] // (3 * heads), 3)))
    elif kind == "attn.out" and in_axis is not None:
        bound.append(Binding("head", in_axis, _slots(heads, shape[in_axis] // heads)))
    elif kind == "mlp.in":
        bound.append(Binding("neuron", out_axis, _slots(shape[out_axis], 1)))
    elif kind == "mlp.out" and in_axis is not None:
        bound.append(Binding("neuron", in_axis, _slots(shape[in_axis], 1)))
    for binding in bound:
        binding.unit = f"{binding.unit}:{layer}" if not binding.unit.startswith("block") \
            else f"block:{layer}:{branch}"
    return bound


def family_of(unit: str) -> str:
    return unit.split(":")[0]


# Every unit family `bindings` can name. `kv` only exists where the model
# groups queries; naming it elsewhere binds nothing.
FAMILIES = ("head", "kv", "neuron", "block")
DEFAULT_FAMILIES = ("head", "kv", "neuron")


@dataclass
class Units:
    """The unit gates of one run: a logit per head, key-value head, neuron and block

    `logits` is keyed by unit family and layer (`head:3`, `neuron:3`,
    `block:3:mlp`), `bound` by weight tensor. `factor` is what `_pairs`
    multiplies into a weight's sampled gate; `relaxed` and `hard` are the
    same product for the density's two readings.
    """
    logits: Dict[str, torch.Tensor]
    bound: Dict[str, List[Binding]]
    drawn: Dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def build(cls, adapter, shapes: Dict[str, Sequence[int]], init: float,
              device: torch.device, families: Sequence[str] = DEFAULT_FAMILIES) -> "Units":
        """Unit gates of the chosen families over every tensor in `shapes`

        Blocks are a family but not a default. On GPT-2 IOI at a 3% target
        one block gate moved 4% of the density, so the constraint became a
        staircase the price could not stand on: it undershot to 0.6% with
        one block open, went negative, and spent the last 600 steps
        switching whole blocks on and off (results/gpt2-sweep/
        nll-t003-units: 3.43%, held-out 0.72, against 2.22%, 0.97 with
        weights alone). A head is 0.23% of that model and a neuron 0.002%,
        and the price tracks those.
        """
        unknown = sorted(set(families) - set(FAMILIES))
        if unknown:
            raise ValueError(f"no unit family {unknown}; choose from {list(FAMILIES)}")
        bound = {name: [binding for binding in bindings(adapter, name, shape)
                        if family_of(binding.unit) in families]
                 for name, shape in shapes.items()}
        sizes: Dict[str, int] = {}
        for claims in bound.values():
            for binding in claims:
                sizes[binding.unit] = max(sizes.get(binding.unit, 0), binding.n_units)
        logits = {unit: torch.full((size,), init, dtype=torch.float32,
                                   device=device).requires_grad_(True)
                  for unit, size in sizes.items()}
        return cls(logits=logits, bound=bound)

    def parameters(self) -> List[torch.Tensor]:
        return list(self.logits.values())

    def draw(self, sample: Callable[[torch.Tensor], torch.Tensor]) -> "_Drawn":
        """Sample every unit gate once, for one forward pass over every tensor"""
        return _Drawn(self, sample)

    def _per_unit(self, unit: str, deterministic: bool) -> torch.Tensor:
        logits = self.logits[unit]
        if deterministic:
            return (logits > 0).to(logits.dtype)
        if unit not in self.drawn:
            raise RuntimeError("sample the unit gates with `draw` before reading them")
        return self.drawn[unit]

    @property
    def device(self) -> torch.device:
        return next(iter(self.logits.values())).device

    def factor(self, name: str, ndim: int, deterministic: bool) -> torch.Tensor:
        """The product of this tensor's unit gates, sampled or thresholded, broadcastable

        A tensor bound to no unit gets a scalar one, so the caller multiplies
        without asking.
        """
        product = torch.ones((), device=self.device)
        for binding in self.bound[name]:
            product = product * binding.spread(self._per_unit(binding.unit, deterministic), ndim)
        return product

    def relaxed(self, name: str, ndim: int) -> torch.Tensor:
        """sigmoid of every unit gate above the tensor, multiplied, for the open cost"""
        product = torch.ones((), device=self.device)
        for binding in self.bound[name]:
            product = product * binding.spread(torch.sigmoid(self.logits[binding.unit]), ndim)
        return product

    def hard(self, name: str, ndim: int) -> torch.Tensor:
        """Whether every unit above each weight is open, broadcastable"""
        product = torch.ones((), dtype=torch.bool, device=self.device)
        for binding in self.bound[name]:
            product = product & binding.spread(self.logits[binding.unit] > 0, ndim)
        return product

    def counts(self) -> Dict[str, Dict[str, int]]:
        """{family: {open, total}} over the unit logits"""
        table: Dict[str, Dict[str, int]] = {}
        for unit, logits in self.logits.items():
            entry = table.setdefault(family_of(unit), {"open": 0, "total": 0})
            entry["open"] += int((logits > 0).sum())
            entry["total"] += int(logits.numel())
        return table

    def report(self) -> Dict[str, object]:
        """The unit mask in the vocabulary the artifact records"""
        heads = sorted((int(unit.split(":")[1]), int(i))
                       for unit, logits in self.logits.items() if family_of(unit) == "head"
                       for i in torch.nonzero(logits > 0).flatten().tolist())
        blocks_closed = sorted(unit for unit, logits in self.logits.items()
                               if family_of(unit) == "block" and not bool((logits > 0).any()))
        neurons = {unit.split(":")[1]: int((logits > 0).sum())
                   for unit, logits in self.logits.items() if family_of(unit) == "neuron"}
        return {
            "counts": self.counts(),
            "heads_open": [[layer, head] for layer, head in heads],
            "blocks_closed": blocks_closed,
            "neurons_open_by_layer": {layer: neurons[layer]
                                      for layer in sorted(neurons, key=int)},
        }

    def state(self) -> Dict[str, torch.Tensor]:
        return {unit: logits.detach().cpu() for unit, logits in self.logits.items()}


class _Drawn:
    def __init__(self, units: Units, sample):
        self.units, self.sample = units, sample

    def __enter__(self):
        self.units.drawn = {unit: self.sample(logits) for unit, logits in self.units.logits.items()}

    def __exit__(self, *_):
        self.units.drawn = {}


def unit_scores(units: Units, scores: Scores) -> Dict[str, torch.Tensor]:
    """Weight attribution summed into each unit: a head's score is its slices' scores"""
    totals = {unit: torch.zeros_like(logits) for unit, logits in units.logits.items()}
    for name, claims in units.bound.items():
        score = scores[name]
        for binding in claims:
            other = [axis for axis in range(score.ndim) if axis != binding.axis]
            per_slot = score.sum(dim=other) if other else score
            totals[binding.unit] += binding.gather(per_slot.float()).to(totals[binding.unit].device)
    return totals


def rank_logits(score: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Percentile rank of each score, mapped onto [low, high]; one element ranks at `high`"""
    flat = score.flatten().float()
    n = flat.numel()
    if n == 1:
        return torch.full_like(flat, high).view(score.shape)
    order = flat.argsort()
    rank = torch.empty_like(flat)
    rank[order] = torch.arange(n, dtype=flat.dtype, device=flat.device)
    return (low + (high - low) * rank / (n - 1)).view(score.shape)


def init_from(scores: Scores, gates: Dict[str, torch.Tensor], units: Optional[Units],
              low: float, high: float) -> None:
    """Write attribution ranks into the gate logits in place

    Weights rank within their own tensor, so every tensor starts with the same
    spread of logits and no projection is pre-judged against another --
    ranking 1.4e9 weights at once would also be an argsort the device has no
    room for. Units rank within their family across the whole model: a head
    is compared with every other head, a block with every other block, which
    is the comparison the block ablations made by hand.
    """
    with torch.no_grad():
        for name, logits in gates.items():
            logits.copy_(rank_logits(scores[name].to(logits.device), low, high))
        if units is None:
            return
        totals = unit_scores(units, scores)
        families: Dict[str, List[str]] = {}
        for unit in totals:
            families.setdefault(family_of(unit), []).append(unit)
        for members in families.values():
            joined = torch.cat([totals[unit] for unit in members])
            ranked = rank_logits(joined, low, high)
            offset = 0
            for unit in members:
                size = units.logits[unit].numel()
                units.logits[unit].copy_(ranked[offset:offset + size])
                offset += size
