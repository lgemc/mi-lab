"""Turn a learned mask into a circuit: where it lives, and something you can run.

A sheaf finishes with 276,476 open gates out of 85M and a held-out score. That
is a number, not a circuit. Two things are missing and this adds both.

**Where.** The gates are spread over every gated parameter tensor in the model,
so the structural claim -- which layers, which projections, which heads -- has
to be aggregated out of them. That is the part comparable to what every other
method here produces: phase 1b's `mlp:21,23,24,26 + heads:23` is a list of
components, and a weight mask can be reduced to the same vocabulary by asking
what share of each component's weights survived. Concentration is the finding.
A mask spread evenly over all of a model is a compression; a mask that piles
into four attention heads is a circuit.

**Runnable.** The mask times the weights is a state dict, and saving it with a
manifest makes the thing being claimed an object rather than a citation. It is
not saved by default: on the 1.7B the masked weights are as large as the model.

Density per component is reported against that component's *own* parameter
count, never against the model's. Two heads at 1% each are the same claim
whatever the rest of the network weighs, and dividing by the model total would
make every number in the table a fact about model size.

A common pipe could be: gates | threshold | per component | manifest | weights

Run: uv run python -m scripts.sheaf_extract gpt2-small results/gpt2-sweep/s0.1
     uv run python -m scripts.sheaf_extract gpt2-small <dir> --save-weights
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from scripts.observe import banner, log
from src.methods.sheaves import gateable
from src.model.adapter import load_adapter

# Which projection a parameter belongs to, by the names the two families use.
# GPT-2 fuses q/k/v into one `c_attn`, so a head's query weights are a slice of
# a shared tensor rather than a tensor of their own -- which is why heads are
# resolved separately below and not by name matching.
KINDS = (
    ("attn.q", ("q_proj", "query")),
    ("attn.k", ("k_proj", "key")),
    ("attn.v", ("v_proj", "value")),
    ("attn.qkv", ("c_attn", "qkv")),
    ("attn.out", ("o_proj", "c_proj", "out_proj", "dense")),
    ("mlp.in", ("gate_proj", "up_proj", "c_fc", "fc_in", "w1", "w3")),
    ("mlp.out", ("down_proj", "c_proj", "fc_out", "w2")),
)

def layer_of(name: str) -> int:
    """The block index in a parameter name, or -1 for anything outside a block"""
    found = re.search(r"\.(?:h|layers|blocks|layer)\.(\d+)\.", name)
    return int(found.group(1)) if found else -1

def kind_of(name: str) -> str:
    """Which projection this parameter is, by name, MLP checked before attention

    `c_proj` names both the attention output and the MLP output in GPT-2, so the
    branch it sits under decides, not the leaf. Matching on the leaf alone
    silently files every MLP output projection under attention.
    """
    stem = name.rsplit(".", 1)[0]
    inside_mlp = ".mlp" in stem
    for label, needles in KINDS:
        if label.startswith("mlp") != inside_mlp:
            continue
        if any(needle in stem for needle in needles):
            return label
    return "mlp.other" if inside_mlp else "attn.other"

def per_component(gates: dict, targets: dict) -> dict:
    """Open count and total per (layer, kind), from the thresholded gates"""
    table = defaultdict(lambda: {"open": 0, "total": 0})
    for name, logits in gates.items():
        key = (layer_of(name), kind_of(name))
        table[key]["open"] += int((logits > 0).sum())
        table[key]["total"] += int(logits.numel())
    return table

def per_head(adapter, gates: dict) -> dict:
    """Open share of each attention head's own slice of the output projection

    The output projection is the one place a head owns a contiguous block of
    rows -- `head_dim` of them -- whatever the architecture does with q/k/v. So
    a head's density is read there, which is also the site `head_outputs` reads
    and `patch` writes, so this number is about the same object the rest of the
    repo measures.
    """
    heads = getattr(adapter.cfg, "n_heads", None) or getattr(adapter.cfg, "n_head", None)
    if not heads:
        return {}
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
            table[(layer, head)] = {"open": int((piece > 0).sum()), "total": int(piece.numel())}
    return table

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("directory", help="a results dir holding sheaf-<task>-gates.pt")
    parser.add_argument("--task", default="ioi")
    parser.add_argument("--top", type=int, default=15, help="how many components to print")
    parser.add_argument("--save-weights", action="store_true", dest="save_weights",
                        help="also write mask*weights; as large as the model")
    args = parser.parse_args()

    directory = Path(args.directory)
    gates_path = directory / f"sheaf-{args.task}-gates.pt"
    if not gates_path.exists():
        raise SystemExit(
            f"{gates_path} does not exist. The sweep ran without --save-gates, so its masks were "
            f"never written; rerun that point with --save-gates (and --seed, or it will not be the "
            f"same mask)."
        )
    gates = torch.load(gates_path, weights_only=True)
    adapter = load_adapter(args.config)
    targets = gateable(adapter)
    total = sum(int(v.numel()) for v in gates.values())
    opened = sum(int((v > 0).sum()) for v in gates.values())

    banner("sheaf extraction", {
        "config": args.config,
        "gates": gates_path,
        "open": f"{opened} of {total} ({opened / total:.4%})",
        "tensors": len(gates),
    })

    components = per_component(gates, targets)
    ranked = sorted(components.items(), key=lambda kv: kv[1]["open"] / kv[1]["total"], reverse=True)
    log(f"{'layer':>6} {'component':>10} {'open':>9} {'of':>10} {'density':>9} {'share of circuit':>17}")
    for (layer, kind), counts in ranked[: args.top]:
        log(f"{layer:>6} {kind:>10} {counts['open']:>9} {counts['total']:>10} "
            f"{counts['open'] / counts['total']:>8.3%} {counts['open'] / opened:>16.1%}")

    by_layer = defaultdict(int)
    for (layer, _), counts in components.items():
        by_layer[layer] += counts["open"]
    log("")
    log("open gates by layer: " + " ".join(
        f"L{layer}={by_layer[layer]}" for layer in sorted(by_layer) if by_layer[layer]))

    heads = per_head(adapter, gates)
    ranked_heads = sorted(heads.items(), key=lambda kv: kv[1]["open"] / kv[1]["total"], reverse=True)
    if ranked_heads:
        log("")
        log("densest attention heads (by their own output-projection slice):")
        for (layer, head), counts in ranked_heads[:10]:
            log(f"  head {layer}.{head:<2} {counts['open']:>7} of {counts['total']:<7} "
                f"{counts['open'] / counts['total']:.3%}")

    manifest = {
        "protocol": "a weight mask reduced to the component vocabulary the rest of this repo uses. "
                    "Density per component is a share of that component's own parameters, never of "
                    "the model's -- dividing by the model total would make every row a fact about "
                    "model size rather than about the circuit.",
        "config": args.config,
        "task": args.task,
        "gates": str(gates_path),
        "n_gates": total,
        "n_open": opened,
        "density": round(opened / total, 8),
        "components": [
            {"layer": layer, "kind": kind, "open": c["open"], "total": c["total"],
             "density": round(c["open"] / c["total"], 6),
             "share_of_circuit": round(c["open"] / opened, 6)}
            for (layer, kind), c in ranked
        ],
        "by_layer": {str(layer): by_layer[layer] for layer in sorted(by_layer)},
        "heads": [
            {"layer": layer, "head": head, "open": c["open"], "total": c["total"],
             "density": round(c["open"] / c["total"], 6)}
            for (layer, head), c in ranked_heads
        ],
        "command": f"uv run python -m scripts.sheaf_extract {args.config} {directory} "
                   f"--task {args.task}",
    }
    out = directory / f"sheaf-{args.task}-circuit.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"-> {out}")

    if args.save_weights:
        masked = {name: (gates[name] > 0).to(p.dtype) * p.detach().cpu()
                  for name, p in targets.items() if name in gates}
        path = directory / f"sheaf-{args.task}-weights.pt"
        torch.save(masked, path)
        log(f"-> {path}")

if __name__ == "__main__":
    main()
