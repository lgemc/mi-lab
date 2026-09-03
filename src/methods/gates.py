"""What a trained weight mask *is* once training is over: a circuit you can run, read and budget.

`sheaves.prune` returns a gate logit per weight. Three things are asked of
that tensor after the fact, and none of them belong in the training loop:

- Load it. `circuit_loaded` multiplies the thresholded mask into the weights
  in place and puts them back after, because inference needs no gradient and
  `functional_call` exists in the loop only to keep one. That is what "a
  circuit you can run" has to mean: ordinary weights, ordinary forward pass.
  `ranking` and `generation` are the two ways of asking whether the circuit
  still does the task -- the training objective's own next-token comparison,
  and the continuation the model actually writes.
- Read it. A mask is a dict of parameter names, and the rest of this repo
  speaks in layers, heads and MLPs. `per_component` and `per_head` reduce a
  mask to that vocabulary, with density as a share of the *component's own*
  parameters -- divided by the model total every row is a fact about model
  size rather than about the circuit.
- Budget it before training it. `budget` says what a band costs in memory
  before the model is loaded, because the failure it replaces is an OOM after
  the expensive part, and on a unified-memory host that is a driver error
  with no traceback.
- Keep it. Everything above thresholds the logits at zero and uses nothing
  else, so the circuit is the sign of each gate and nothing else: `pack`
  writes that as one bit per weight -- 168 MB for the whole 1.7B against the
  5.6 GB of float logits -- and `unpack` reads it back as a bool mask every
  function here accepts in place of the logits. The logits are the
  optimizer's state, and the mask is the result; `load_circuit` reads either
  file, so a run that kept only the mask is still a circuit you can run.

A common pipe could be: parse_layers | budget | prune | circuit_loaded | ranking | per_component
A common pipe could be: gates | summary | manifest | masked_weights
A common pipe could be: gates | pack | save | load_circuit | circuit_loaded
"""

import math
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from ..data.tasks import CircuitTask
from .circuits import CircuitError
from .sheaves import gateable

# Which projection a parameter is, by the leaf name it is stored under across
# the families this repo has met. MLP names are matched only inside an MLP
# branch, and attention names only outside one, because `c_proj` is both.
KINDS = (
    ("attn.q", ("q_proj", "query")),
    ("attn.k", ("k_proj", "key")),
    ("attn.v", ("v_proj", "value")),
    ("attn.qkv", ("c_attn", "qkv")),
    ("attn.out", ("o_proj", "c_proj", "out_proj", "dense")),
    ("mlp.in", ("gate_proj", "up_proj", "c_fc", "fc_in", "w1", "w3")),
    ("mlp.out", ("down_proj", "c_proj", "fc_out", "w2")),
)

# Adam keeps two moments beside the parameter and its gradient, four floats
# per gate. The graph on top of that is a fixed part plus a per-gate part,
# not a multiple of the gate count: a multiplier fitted on a 151M-gate band
# said 39.6 B/gate and projected 79 GiB for the whole model, which then ran
# at 37.1. The caching allocator reuses blocks across parameter tensors, so
# most of the small-band peak is a constant -- activations, forward buffers,
# allocator slack -- and only the tensors gumbel retains for the backward
# pass scale. Fitted on the two measured runs (151M gates peaked at 11.3 GiB,
# 1409M at 37.1) and rounded up from 5.00 and 4.02 for margin. Both come from
# torch's peak counter, which misses the driver's own allocations.
STATE_BYTES_PER_GATE = 4 * 4
GRAPH_BASE_GIB = 6.0
GRAPH_BYTES_PER_GATE = 5

class GateError(CircuitError):
    """A mask that does not describe this model, or a band that cannot be parsed"""

# Either the float logits `prune` learned or the bool mask `unpack` returns:
# a gate is open where the logit is positive, or where the bool is set.
Gates = Dict[str, torch.Tensor]

# The packed form: per tensor, its shape and its mask as bytes, least
# significant bit first (numpy's `packbits(bitorder="little")`).
Packed = Dict[str, Dict[str, Any]]

GATES_FILE = "sheaf-{task}-gates.pt"
MASK_FILE = "sheaf-{task}-mask.pt"

def is_open(tensor: torch.Tensor) -> torch.Tensor:
    """The mask under either representation, as bools"""
    return tensor if tensor.dtype == torch.bool else tensor > 0

def open_count(gates: Gates) -> Tuple[int, int]:
    """(open, total) over every gated tensor; a gate is open where its logit is positive"""
    total = sum(int(logits.numel()) for logits in gates.values())
    opened = sum(int(is_open(logits).sum()) for logits in gates.values())
    return opened, total

def pack(gates: Gates) -> Packed:
    """The circuit as one bit per gate, on the CPU

    Lossless for everything this module does with a mask, since all of it
    thresholds at zero; what it drops is how far each logit sat from the
    threshold, which is the optimizer's business and not the circuit's.
    """
    weights = (1 << torch.arange(8, dtype=torch.int32))
    packed: Packed = {}
    for name, logits in gates.items():
        flat = is_open(logits).flatten().to(torch.int32)
        padding = -flat.numel() % 8
        if padding:
            flat = torch.cat([flat, flat.new_zeros(padding)])
        bits = (flat.view(-1, 8) * weights.to(flat.device)).sum(-1).to(torch.uint8)
        packed[name] = {"shape": list(logits.shape), "bits": bits.cpu()}
    return packed

def unpack(packed: Packed) -> Gates:
    """A packed mask back to a bool tensor per gated parameter"""
    shifts = torch.arange(8, dtype=torch.uint8)
    gates: Gates = {}
    for name, entry in packed.items():
        shape = tuple(int(n) for n in entry["shape"])
        count = math.prod(shape)
        bits = entry["bits"]
        if bits.dtype != torch.uint8 or bits.numel() != -(-count // 8):
            raise GateError(f"{name}: {bits.numel()} bytes of {bits.dtype} cannot hold a mask of shape {shape}")
        flat = ((bits.unsqueeze(-1) >> shifts) & 1).bool().flatten()
        gates[name] = flat[:count].view(shape)
    return gates

def circuit_path(directory: Path, task: str) -> Path:
    """The file a results directory keeps the circuit in: the logits if it has them, else the mask"""
    for template in (GATES_FILE, MASK_FILE):
        path = Path(directory) / template.format(task=task)
        if path.exists():
            return path
    raise GateError(
        f"{directory} holds neither {GATES_FILE.format(task=task)} nor {MASK_FILE.format(task=task)}. "
        "The sweep ran with --no-save-gates and predates the packed mask; rerun that point "
        "(with --seed, or it will not be the same mask)."
    )

def load_circuit(path: Path) -> Gates:
    """Either file as a `Gates`: float logits as saved, a packed mask as bools"""
    loaded = torch.load(path, weights_only=True)
    if Path(path).name.endswith("-mask.pt"):
        return unpack(loaded)
    return loaded

@contextmanager
def circuit_loaded(adapter, gates: Gates) -> Iterator[None]:
    """Multiply the thresholded mask into the weights, and put them back after"""
    targets = gateable(adapter)
    unknown = [name for name in gates if name not in targets]
    if unknown:
        raise GateError(f"{len(unknown)} gated tensors are not gateable on this model, e.g. {unknown[0]}")
    saved = {}
    try:
        with torch.no_grad():
            for name, parameter in targets.items():
                if name not in gates:
                    continue
                saved[name] = parameter.detach().clone()
                parameter.mul_(is_open(gates[name].to(parameter.device)).to(parameter.dtype))
        yield
    finally:
        with torch.no_grad():
            for name, original in saved.items():
                targets[name].copy_(original)

def ranking(adapter, task: CircuitTask) -> float:
    """The training objective's own metric: the share of prompts where the answer outranks the distractor"""
    prompts = list(task.clean)
    io, subject = task.answers(adapter)
    logits = adapter.logits(prompts)
    right = sum(1 for row in range(len(prompts))
                if float(logits[row, io[row]]) > float(logits[row, subject[row]]))
    return right / len(prompts)

def generation(adapter, task: CircuitTask, tokens: int) -> Dict[str, Any]:
    """Ask for the continuation and see whether the answer is what comes out

    `first_token` is the fair comparison with the ranking score -- both are
    about the very next token -- while `contains` allows the answer to arrive
    a word or two late, which greedy decoding on a templated frame often does.
    Answers are decoded from the ids `task.answers` returns rather than read
    off the examples, because that is the one shape every task agrees on.
    """
    prompts = list(task.clean)
    io_ids, subject_ids = task.answers(adapter)
    answers = [adapter.tokenizer.decode([i]) for i in io_ids]
    distractors = [adapter.tokenizer.decode([i]) for i in subject_ids]
    completions = adapter.generate(prompts, max_new_tokens=tokens)
    exact = sum(1 for c, a in zip(completions, answers, strict=True) if c.startswith(a))
    contains = sum(1 for c, a in zip(completions, answers, strict=True) if a.strip() and a.strip() in c)
    wrong = sum(1 for c, d in zip(completions, distractors, strict=True) if d.strip() and d.strip() in c)
    return {
        "first_token": exact / len(prompts),
        "contains_answer": contains / len(prompts),
        "contains_distractor": wrong / len(prompts),
        "completions": completions,
        "answers": answers,
    }

def layer_of(name: str) -> int:
    """The block index in a parameter name, or -1 for anything outside a block"""
    found = re.search(r"\.(?:h|layers|blocks|layer)\.(\d+)\.", name)
    return int(found.group(1)) if found else -1

def kind_of(name: str) -> str:
    """Which projection this parameter is, by name, the branch deciding before the leaf

    `c_proj` names both the attention output and the MLP output in GPT-2, so
    the branch it sits under decides. Matching on the leaf alone silently
    files every MLP output projection under attention.
    """
    stem = name.rsplit(".", 1)[0]
    inside_mlp = ".mlp" in stem
    for label, needles in KINDS:
        if label.startswith("mlp") != inside_mlp:
            continue
        if any(needle in stem for needle in needles):
            return label
    return "mlp.other" if inside_mlp else "attn.other"

def per_component(gates: Gates) -> Dict[Tuple[int, str], Dict[str, int]]:
    """{(layer, kind): {open, total}} from the thresholded gates"""
    table: Dict[Tuple[int, str], Dict[str, int]] = defaultdict(lambda: {"open": 0, "total": 0})
    for name, logits in gates.items():
        key = (layer_of(name), kind_of(name))
        table[key]["open"] += int(is_open(logits).sum())
        table[key]["total"] += int(logits.numel())
    return dict(table)

def per_head(adapter, gates: Gates) -> Dict[Tuple[int, int], Dict[str, int]]:
    """Open share of each attention head's own slice of the output projection

    The output projection is the one place a head owns a contiguous block of
    rows -- `d_head` of them -- whatever the architecture does with q/k/v, and
    it is the site `head_outputs` reads and `patch` writes, so this number is
    about the same object the rest of the repo measures.
    """
    heads = adapter.cfg.n_heads
    table = {}
    for layer, projection in enumerate(adapter.projections):
        name = next((n for n, p in adapter.model.named_parameters()
                     if p is getattr(projection, "weight", None)), None)
        if name is None or name not in gates:
            continue
        logits = gates[name]
        # GPT-2 stores Conv1D as [in, out] and Linear is [out, in]; the head
        # slice is along the *input* side of an output projection either way.
        axis = 0 if logits.shape[0] % heads == 0 else 1
        width = logits.shape[axis] // heads
        for head in range(heads):
            piece = (logits[head * width : (head + 1) * width] if axis == 0
                     else logits[:, head * width : (head + 1) * width])
            table[(layer, head)] = {"open": int(is_open(piece).sum()), "total": int(piece.numel())}
    return table

def by_layer(components: Dict[Tuple[int, str], Dict[str, int]]) -> Dict[int, int]:
    """Open gates per layer, from a `per_component` table"""
    counts: Dict[int, int] = defaultdict(int)
    for (layer, _), entry in components.items():
        counts[layer] += entry["open"]
    return dict(counts)

def ranked(table: Dict[Any, Dict[str, int]]) -> List[Tuple[Any, Dict[str, int]]]:
    """A per-component or per-head table ordered by density, densest first"""
    return sorted(table.items(), key=lambda item: item[1]["open"] / item[1]["total"], reverse=True)

def summary(adapter, gates: Gates) -> Dict[str, Any]:
    """The mask reduced to the component vocabulary, as the manifest records it

    Density per component is a share of that component's own parameters,
    never of the model's; `share_of_circuit` is the one number that divides
    by the whole mask, and it says where the open gates went.
    """
    opened, total = open_count(gates)
    if not total:
        raise GateError("an empty mask has no circuit to summarize")
    components = per_component(gates)
    layers = by_layer(components)
    return {
        "n_gates": total,
        "n_open": opened,
        "density": round(opened / total, 8),
        "components": [
            {"layer": layer, "kind": kind, "open": entry["open"], "total": entry["total"],
             "density": round(entry["open"] / entry["total"], 6),
             "share_of_circuit": round(entry["open"] / opened, 6) if opened else 0.0}
            for (layer, kind), entry in ranked(components)
        ],
        "by_layer": {str(layer): layers[layer] for layer in sorted(layers)},
        "heads": [
            {"layer": layer, "head": head, "open": entry["open"], "total": entry["total"],
             "density": round(entry["open"] / entry["total"], 6)}
            for (layer, head), entry in ranked(per_head(adapter, gates))
        ],
    }

def masked_weights(adapter, gates: Gates) -> Dict[str, torch.Tensor]:
    """mask * weight for every gated tensor, on the CPU -- as large as the band it covers"""
    targets = gateable(adapter)
    return {name: (targets[name].detach() * is_open(logits.to(targets[name].device))).cpu()
            for name, logits in gates.items() if name in targets}

def budget(n_gates: int, weight_bytes: int, model_bytes: int) -> Dict[str, Any]:
    """What a pruning run costs before it starts: the model, the state, one graph"""
    optimizer = n_gates * STATE_BYTES_PER_GATE
    originals = n_gates * weight_bytes
    state = (optimizer + originals) / 2 ** 30
    model = model_bytes / 2 ** 30
    graph = GRAPH_BASE_GIB + n_gates * GRAPH_BYTES_PER_GATE / 2 ** 30
    return {
        "gates": n_gates,
        "optimizer_gib": round(optimizer / 2 ** 30, 2),
        "originals_gib": round(originals / 2 ** 30, 2),
        "model_gib": round(model, 2),
        "graph_gib": round(graph, 2),
        "total_gib": round(state, 2),
        "peak_gib": round(model + state + graph, 2),
    }

def run_budget(adapter, layers: Optional[Sequence[int]]) -> Dict[str, Any]:
    """`budget` measured off the loaded model for a band, plus the gate count it gates"""
    targets = gateable(adapter, layers)
    n_gates = sum(parameter.numel() for parameter in targets.values())
    weight_bytes = next(iter(targets.values())).element_size()
    model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in adapter.model.parameters())
    return budget(n_gates, weight_bytes, model_bytes)

def parse_layers(text: str) -> Optional[List[int]]:
    """`21-27`, `21,23,26` or `all` into layer indices; None means the whole model"""
    if text in ("", "all"):
        return None
    chosen = set()
    try:
        for piece in text.split(","):
            if "-" in piece.strip("-"):
                low, high = piece.split("-", 1)
                chosen.update(range(int(low), int(high) + 1))
            else:
                chosen.add(int(piece))
    except ValueError:
        raise GateError(f"cannot read a layer band from '{text}'; write 21-27, 21,23,26 or all") from None
    return sorted(chosen)
