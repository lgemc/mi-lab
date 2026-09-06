"""Turn a learned circuit into something readable: where it lives, and what talks to what.

Two kinds of run end up here and the folder says which. A weight mask reduces to
the component vocabulary (`gates.summary`); an edge circuit reduces to
components *and the paths between them* (`wiring.reduce`), which is the half a
component list cannot express -- a source writes into the residual stream once
and every later component reads the same sum, so "kept for `attn:10`, cut for
`mlp:14`" is invisible in any per-component summary. Both write
`sheaf-<task>-circuit.json`; `protocol` says which reduction produced it.

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
                   or: edges_open | reduce | components + split | against

Run: uv run python -m scripts.sheaf_extract gpt2-small results/gpt2-sweep/s0.1
     uv run python -m scripts.sheaf_extract gpt2-small <dir> --save-weights
     uv run python -m scripts.sheaf_extract qwen3-1.7b <edge dir> --task ioi
"""

import argparse
import json
from pathlib import Path

import torch

from src.data.ioi import WANG_HEADS
from src.methods.circuits import require_circuits
from src.methods.gates import GateError, circuit_path, load_circuit, masked_weights, summary
from src.methods.wiring import against, reduce
from src.model.adapter import load_adapter
from src.telemetry.observe import banner, log

TOP_HEADS = 10
TOP_EDGES = 12


def extract_edges(args, directory: Path, record: dict) -> None:
    """The edge branch: components, the paths between them, and the split count

    Split sources are printed before anything else because they are the only
    part of this that a weight mask could not have produced. If that number is
    near zero the run found a component selection and the edge machinery bought
    nothing.
    """
    adapter = require_circuits(load_adapter(args.config))
    every = list(adapter.edges())
    circuit = reduce([tuple(edge) for edge in record["edges_open"]], every)

    banner("edge circuit extraction", {
        "config": args.config,
        "task": args.task,
        "edges": f"{circuit['n_edges_open']} of {circuit['n_edges']} "
                 f"({circuit['edge_density']:.4%})",
        "sources": f"{circuit['n_sources']} total, {circuit['sources_split']} split across "
                   f"readers, {circuit['sources_all_kept']} kept whole, "
                   f"{circuit['sources_all_cut']} cut entirely",
        "components": f"{len(circuit['heads_in_circuit'])} heads, "
                      f"{len(circuit['mlps_in_circuit'])} mlps",
    })

    log(f"{'component':>14} {'reaches':>9} {'of':>5} {'reach':>8}  split  readers")
    for row in circuit["components"][:args.top]:
        readers = ", ".join(row["readers"][:4]) + (" ..." if len(row["readers"]) > 4 else "")
        log(f"{row['component']:>14} {row['out_kept']:>9} {row['out_available']:>5} "
            f"{row['reach']:>7.1%}  {'yes' if row['split'] else 'no ':>5}  {readers}")
    log("")
    log("open edges by destination layer: " + " ".join(
        f"L{layer}={count}" for layer, count in circuit["by_layer"].items() if count))

    comparison = None
    if args.task == "ioi" and args.config == "gpt2-small":
        comparison = against(circuit["heads_in_circuit"], WANG_HEADS, adapter.cfg.n_layers *
                             adapter.cfg.n_heads)
        log("")
        verdict = ("no better than a random head set of the same size"
                   if comparison["p_value"] is None or comparison["p_value"] > 0.05
                   else "more overlap than chance")
        log(f"against Wang et al.'s GPT-2 IOI circuit: {comparison['found']} of "
            f"{comparison['reference']} found, {comparison['expected_by_chance']} expected by "
            f"chance from {comparison['kept']} of "
            f"{comparison['of_total']} heads -- p={comparison['p_value']}, {verdict}")
        for hit in comparison["matches"]:
            log(f"    {hit['head']:>6}  {hit['role']}")
    elif args.task == "ioi":
        # Wang et al.'s heads are GPT-2 small's. Head 9.6 in another model is
        # another head, and printing the comparison anyway would manufacture a
        # replication out of a coincidence.
        log("")
        log(f"no published head list for '{args.config}' -- Wang et al.'s circuit is "
            f"GPT-2 small's and does not transfer")

    out = directory / f"sheaf-{args.task}-circuit.json"
    out.write_text(json.dumps({
        "protocol": "an edge circuit reduced to components and the paths between them. Degrees "
                    "are against what each source could reach, never against the model: a source "
                    "at layer 0 can reach every later destination and one at the last layer can "
                    "reach almost none, so a raw out-degree means opposite things at the two ends. "
                    "`sources_split` is the count kept for some readers and cut for others, which "
                    "is the claim no weight mask can express.",
        "config": args.config,
        "task": args.task,
        **circuit,
        **({"published_comparison": comparison} if comparison else {}),
        "command": f"uv run python -m scripts.sheaf_extract {args.config} {directory} "
                   f"--task {args.task}",
    }, indent=2) + "\n")
    log(f"-> {out}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("directory", help="a results dir holding sheaf-<task>-mask.pt or -gates.pt")
    parser.add_argument("--task", default="ioi")
    parser.add_argument("--top", type=int, default=15, help="how many components to print")
    parser.add_argument("--save-weights", action="store_true", dest="save_weights",
                        help="also write mask*weights; as large as the model")
    args = parser.parse_args()

    directory = Path(args.directory)
    # The folder decides which reduction runs, the same way it decides which
    # backbone serves it. An edge run writes no mask, so asking `circuit_path`
    # first would fail on exactly the runs this branch exists for.
    artifact = directory / f"sheaf-{args.task}.json"
    if artifact.exists():
        record = json.loads(artifact.read_text())
        if record.get("edges_open") is not None:
            extract_edges(args, directory, record)
            return
    try:
        gates_path = circuit_path(directory, args.task)
    except GateError as error:
        raise SystemExit(str(error)) from None
    gates = load_circuit(gates_path)
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
