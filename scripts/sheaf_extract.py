"""Turn a learned mask into a circuit: where it lives, and something you can run.

A sheaf finishes with 276,476 open gates out of 85M and a held-out score. That
is a number, not a circuit. Two things are missing and this adds both.

**Where.** The gates are spread over every gated parameter tensor in the model,
so the structural claim -- which layers, which projections, which heads -- has
to be aggregated out of them. That is the part comparable to what every other
method here produces: phase 1b's `mlp:21,23,24,26 + heads:23` is a list of
components, and `methods.gates.summary` reduces a weight mask to the same
vocabulary by asking what share of each component's weights survived.
Concentration is the finding. A mask spread evenly over all of a model is a
compression; a mask that piles into four attention heads is a circuit.

**Runnable.** The mask times the weights is a state dict, and saving it with a
manifest makes the thing being claimed an object rather than a citation. It is
not saved by default: on the 1.7B the masked weights are as large as the model.

A common pipe could be: gates | summary | manifest | masked_weights

Run: uv run python -m scripts.sheaf_extract gpt2-small results/gpt2-sweep/s0.1
     uv run python -m scripts.sheaf_extract gpt2-small <dir> --save-weights
"""

import argparse
import json
from pathlib import Path

import torch

from src.methods.gates import masked_weights, summary
from src.model.adapter import load_adapter
from src.telemetry.observe import banner, log

TOP_HEADS = 10

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
    circuit = summary(adapter, gates)
    opened, total = circuit["n_open"], circuit["n_gates"]

    banner("sheaf extraction", {
        "config": args.config,
        "gates": gates_path,
        "open": f"{opened} of {total} ({opened / total:.4%})",
        "tensors": len(gates),
    })

    log(f"{'layer':>6} {'component':>10} {'open':>9} {'of':>10} {'density':>9} {'share of circuit':>17}")
    for row in circuit["components"][: args.top]:
        log(f"{row['layer']:>6} {row['kind']:>10} {row['open']:>9} {row['total']:>10} "
            f"{row['density']:>8.3%} {row['share_of_circuit']:>16.1%}")
    log("")
    log("open gates by layer: " + " ".join(
        f"L{layer}={count}" for layer, count in circuit["by_layer"].items() if count))
    if circuit["heads"]:
        log("")
        log("densest attention heads (by their own output-projection slice):")
        for row in circuit["heads"][:TOP_HEADS]:
            log(f"  head {row['layer']}.{row['head']:<2} {row['open']:>7} of {row['total']:<7} {row['density']:.3%}")

    out = directory / f"sheaf-{args.task}-circuit.json"
    out.write_text(json.dumps({
        "protocol": "a weight mask reduced to the component vocabulary the rest of this repo uses. "
                    "Density per component is a share of that component's own parameters, never of "
                    "the model's -- dividing by the model total would make every row a fact about "
                    "model size rather than about the circuit.",
        "config": args.config,
        "task": args.task,
        "gates": str(gates_path),
        **circuit,
        "command": f"uv run python -m scripts.sheaf_extract {args.config} {directory} --task {args.task}",
    }, indent=2) + "\n")
    log(f"-> {out}")

    if args.save_weights:
        path = directory / f"sheaf-{args.task}-weights.pt"
        torch.save(masked_weights(adapter, gates), path)
        log(f"-> {path}")

if __name__ == "__main__":
    main()
