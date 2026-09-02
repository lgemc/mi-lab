"""Phase 1b deliverable 8: take the circuit out of the model and ask if it is enough.

Every number phase 1b has produced is a knockout: remove these components, see
what breaks. That measures *necessity* and nothing else. The opposite question
-- are these components enough on their own -- has never been asked here, and
the two come apart: a set the model needs can still be a set that reproduces
none of the behaviour by itself, which is the difference between "the engine
needs its spark plugs" and "the spark plugs are the engine".

src/methods/circuits.py already measures both for IOI (faithfulness, necessity,
minimality). The translation sweep implements only the middle one. This closes
that asymmetry with the machinery already here: ablating the *complement* of a
circuit is running the circuit alone, and `knockout.ablate` takes any
component list.

The extraction is also written down rather than only measured. A circuit that
exists as five strings in a JSON file is one nobody can inspect, load or
compare; `knockout.extract` saves the parameters those strings name, with a
manifest stating the architecture they came out of, so the thing being claimed
about is an object rather than a citation.

Read the complement number before believing it. On a model with no survival
frontier the complement is most of the network, and ablating most of a network
gives zero for reasons that have nothing to do with the circuit -- which is a
statement about what this measurement can and cannot see, not a caveat to be
skipped.

Stages: extract saves weights and manifest; measure scores whole model,
knockout and extraction on the same shortlist.

A common pipe could be: candidate | extract | ablate_complement | score | report

Run: uv run python -m scripts.phase1b_extract qwen3-1.7b extract
     uv run python -m scripts.phase1b_extract qwen3-1.7b measure
"""

import json
import sys

import torch

from src.experiment import translation_study as study
from src.methods import components as comp
from src.methods.knockout import extract
from src.telemetry.observe import banner, log, set_log_file
from src.telemetry.results import guard

WEIGHTS = study.artifact("weights")
MANIFEST = study.artifact("manifest")
MEASURE = study.artifact("extraction")

def stage_extract(config: str) -> None:
    set_log_file(study.log_path("extract"))
    components = study.candidate_components()
    cost = study.cost_model()
    banner("phase1b circuit extraction", {
        "config": config,
        "circuit": f"{len(components)} components: {', '.join(components)}",
        "cost": f"{cost.share(components):.2%} of model MACs",
        "weights": str(WEIGHTS),
        "manifest": str(MANIFEST),
    })
    adapter = study.load_model(config)
    tensors, entries = extract(adapter, components)
    for entry in entries:
        entry["flops_share_alone"] = round(cost.share([entry["component"]]), 5)
        log(f"{entry['component']:<12} {entry['module']:<22} {entry['n_parameters'] / 1e6:>8.2f}M params")

    total = sum(entry["n_parameters"] for entry in entries)
    model_total = sum(t.numel() for t in adapter.model.parameters())
    torch.save(tensors, WEIGHTS)
    MANIFEST.write_text(json.dumps({
        "protocol": "the parameters of the components the greedy stage selected, saved with the "
                    "architecture they came out of. A circuit that exists only as strings in a JSON "
                    "file cannot be inspected, loaded or compared against another.",
        "model": {"id": adapter.cfg.id, "n_layers": adapter.cfg.n_layers, "n_heads": adapter.cfg.n_heads,
                  "d_model": adapter.cfg.d_model, "n_parameters": model_total},
        "circuit": {
            "components": components,
            "n_components": len(components),
            "n_parameters": total,
            "parameter_share": round(total / model_total, 5),
            "flops_share": round(cost.share(components), 5),
        },
        "components": entries,
        "weights_file": str(WEIGHTS),
        "weights_bytes": WEIGHTS.stat().st_size,
    }, indent=2) + "\n")
    log(f"circuit: {total / 1e6:.1f}M of {model_total / 1e6:.1f}M parameters "
        f"({total / model_total:.2%}), {cost.share(components):.2%} of MACs")
    log(f"-> {WEIGHTS} ({WEIGHTS.stat().st_size / 1024 ** 2:.1f} MiB)")
    log(f"-> {MANIFEST}")

def stage_measure(config: str) -> None:
    set_log_file(study.log_path("extract"))
    components = study.candidate_components()
    cost = study.cost_model()
    # Means over every layer: the complement arm ablates outside the candidate band.
    adapter, corpus, means = study.setup(config, scope="model")
    complement = comp.complement(components, adapter.cfg.n_layers)
    banner("phase1b circuit extraction: measure", {
        "config": config,
        "circuit": f"{len(components)} components, {cost.share(components):.2%} of MACs",
        "complement": f"{len(complement)} components, {cost.share(complement):.2%} of MACs",
        "sentences": len(corpus),
    })

    arms = {}
    for label, ablated in (("whole_model", []), ("knockout", components), ("extraction", complement)):
        record = study.score_set(adapter, means, corpus, ablated, label=label, cost=cost)
        arms[label] = {
            "n_components": record["n_components"],
            "flops_share_ablated": record["flops_share"],
            "bleu": record["bleu"],
            "degeneracy": record["degeneracy"],
            "seconds": record["seconds"],
        }

    base = arms["whole_model"]["bleu"]
    MEASURE.write_text(json.dumps({
        "protocol": "the same circuit measured in both directions on the same shortlist. knockout "
                    "ablates the circuit and asks whether the model needs it; extraction ablates the "
                    "complement -- running the circuit alone -- and asks whether it is enough. "
                    "Necessity without sufficiency is not a circuit claim.",
        "circuit": components,
        "circuit_flops_share": round(cost.share(components), 4),
        "complement_flops_share": round(cost.share(complement), 4),
        "baseline_bleu": base,
        "arms": arms,
        "knockout_dbleu": round(base - arms["knockout"]["bleu"], 2),
        "extraction_retained": round(arms["extraction"]["bleu"] / base, 4) if base else None,
        "reading": ("the complement is most of the network, so a near-zero extraction score is what "
                    "ablating most of a network gives and says nothing about this circuit in "
                    "particular. Sufficiency at this granularity needs restoration into a "
                    "counterfactual run, not deletion of everything else."),
        "command": f"uv run python -m scripts.phase1b_extract {config} measure",
    }, indent=2) + "\n")
    if base:
        log(f"whole model BLEU {base} · knockout {arms['knockout']['bleu']} "
            f"(d {round(base - arms['knockout']['bleu'], 2)}) · extraction {arms['extraction']['bleu']} "
            f"({arms['extraction']['bleu'] / base:.1%} retained)")
    log(f"-> {MEASURE}")

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    stage = sys.argv[2] if len(sys.argv) > 2 else "extract"
    guard(config)
    if stage == "extract":
        stage_extract(config)
    elif stage == "measure":
        stage_measure(config)
    else:
        raise SystemExit(f"unknown stage '{stage}'; stages are extract, measure")

if __name__ == "__main__":
    main()
