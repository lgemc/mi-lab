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
circuit is running the circuit alone, and `ablate` takes any component list.

The extraction is also written down rather than only measured. A circuit that
exists as five strings in a JSON file is one nobody can inspect, load or
compare; this saves the parameters those strings name, with a manifest stating
the architecture they came out of, so the thing being claimed about is an
object rather than a citation.

Read the complement number before believing it. On a model with no survival
frontier the complement is most of the network, and ablating most of a network
gives zero for reasons that have nothing to do with the circuit -- which is a
statement about what this measurement can and cannot see, not a caveat to be
skipped.

Stages: extract saves weights and manifest; measure scores whole model,
knockout and extraction on the same shortlist.

A common pipe could be: candidate | extract | ablate_complement | bleu_of | report

Run: uv run python -m scripts.phase1b_extract qwen3-1.7b extract
     uv run python -m scripts.phase1b_extract qwen3-1.7b measure
"""

import json
import time
from dataclasses import replace

import torch

from scripts.observe import banner, degeneracy, gpu, log, preview, set_log_file, step
from scripts.paths import guard, result
from scripts.phase1b_ablation import (
    GENERATION_BATCH,
    ablate,
    bleu_of,
    capture_means,
    eval_data,
    parse_component,
    reference_prompts,
    translate,
)
from scripts.phase1b_greedy import CANDIDATE, flops_share
from src.model.adapter import load_adapter

WEIGHTS = result("phase1b-circuit-weights.pt")
MANIFEST = result("phase1b-circuit-manifest.json")
MEASURE = result("phase1b-circuit-extraction.json")

def circuit_components() -> list:
    if not CANDIDATE.exists():
        raise SystemExit(f"{CANDIDATE} does not exist; run the greedy and comet stages first")
    data = json.loads(CANDIDATE.read_text())
    chosen = data.get("evaluations", {}).get("candidate", {}).get("components")
    if not chosen:
        raise SystemExit(f"{CANDIDATE} holds no candidate components")
    return list(chosen)

def attention_of(adapter, layer: int):
    """The attention submodule of a block, found by which module owns its output projection

    Located rather than named, because the name differs by architecture and this
    file is not the place that knows about any of them -- the backend already
    resolved the projection, so the module holding it is the attention.
    """
    target = adapter.projections[layer]
    for module in adapter.blocks[layer].modules():
        if any(child is target for child in module.children()):
            return module
    raise SystemExit(f"could not locate the attention module owning the layer {layer} projection")

def stage_extract(config: str) -> None:
    set_log_file(result("phase1b-extract.log"))
    components = circuit_components()
    banner("phase1b circuit extraction", {
        "config": config,
        "circuit": f"{len(components)} components: {', '.join(components)}",
        "cost": f"{flops_share(components):.2%} of model MACs",
        "weights": str(WEIGHTS),
        "manifest": str(MANIFEST),
    })
    adapter = load_adapter(config)
    tensors, entries = {}, []
    for cid in components:
        kind, layer, _ = parse_component(cid)
        if kind == "mlp":
            module = adapter.mlps[layer]
        elif kind == "heads":
            module = attention_of(adapter, layer)
        else:
            raise SystemExit(f"extraction covers whole mlp and heads components, not '{cid}'")
        parameters = {name: tensor.detach().cpu() for name, tensor in module.named_parameters()}
        count = sum(t.numel() for t in parameters.values())
        for name, tensor in parameters.items():
            tensors[f"{cid}/{name}"] = tensor
        entries.append({
            "component": cid, "kind": kind, "layer": layer,
            "module": type(module).__name__,
            "parameters": {name: list(t.shape) for name, t in parameters.items()},
            "n_parameters": count,
            "flops_share_alone": round(flops_share([cid]), 5),
        })
        log(f"{cid:<12} {type(module).__name__:<22} {count / 1e6:>8.2f}M params")

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
            "flops_share": round(flops_share(components), 5),
        },
        "components": entries,
        "weights_file": str(WEIGHTS),
        "weights_bytes": WEIGHTS.stat().st_size,
    }, indent=2) + "\n")
    log(f"circuit: {total / 1e6:.1f}M of {model_total / 1e6:.1f}M parameters "
        f"({total / model_total:.2%}), {flops_share(components):.2%} of MACs")
    log(f"-> {WEIGHTS} ({WEIGHTS.stat().st_size / 1024 ** 2:.1f} MiB)")
    log(f"-> {MANIFEST}")

def complement_of(adapter, components) -> list:
    """Every whole-layer component the circuit does not name

    Ablating this *is* running the circuit alone, at the granularity the sweep
    worked in: whole MLPs and whole attention layers.
    """
    keep = set(components)
    everything = ([f"mlp:{layer}" for layer in range(adapter.cfg.n_layers)]
                  + [f"heads:{layer}" for layer in range(adapter.cfg.n_layers)])
    return [cid for cid in everything if cid not in keep]

def stage_measure(config: str) -> None:
    set_log_file(result("phase1b-extract.log"))
    components = circuit_components()
    adapter = load_adapter(config)
    adapter.cfg = replace(adapter.cfg, batch_size=GENERATION_BATCH)
    wmt, prompts, references = eval_data()
    layers = list(range(adapter.cfg.n_layers))
    complement = complement_of(adapter, components)
    banner("phase1b circuit extraction: measure", {
        "config": config,
        "circuit": f"{len(components)} components, {flops_share(components):.2%} of MACs",
        "complement": f"{len(complement)} components, {flops_share(complement):.2%} of MACs",
        "sentences": len(prompts),
    })
    with step(f"counterfactual means over all {len(layers)} layers"):
        means = capture_means(adapter, reference_prompts(wmt), layers=layers)

    arms = {}

    def score(label, ablated):
        start = time.time()
        with step(f"{label}: ablating {len(ablated)} components "
                  f"({flops_share(ablated):.2%} of MACs)" if ablated else f"{label}: nothing ablated"):
            if ablated:
                with ablate(adapter, means, ablated):
                    hypotheses = translate(adapter, prompts, label=label)
            else:
                hypotheses = translate(adapter, prompts, label=label)
        preview(hypotheses, label)
        arms[label] = {
            "n_components": len(ablated),
            "flops_share_ablated": round(flops_share(ablated), 4) if ablated else 0.0,
            "bleu": bleu_of(hypotheses, references),
            "degeneracy": degeneracy(hypotheses),
            "seconds": round(time.time() - start, 1),
        }
        log(f"{label}: BLEU {arms[label]['bleu']} · degeneracy {arms[label]['degeneracy']} · {gpu()}")

    score("whole_model", [])
    score("knockout", components)
    score("extraction", complement)

    base = arms["whole_model"]["bleu"]
    MEASURE.write_text(json.dumps({
        "protocol": "the same circuit measured in both directions on the same shortlist. knockout "
                    "ablates the circuit and asks whether the model needs it; extraction ablates the "
                    "complement -- running the circuit alone -- and asks whether it is enough. "
                    "Necessity without sufficiency is not a circuit claim.",
        "circuit": components,
        "circuit_flops_share": round(flops_share(components), 4),
        "complement_flops_share": round(flops_share(complement), 4),
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
    log(f"whole model BLEU {base} · knockout {arms['knockout']['bleu']} "
        f"(d {round(base - arms['knockout']['bleu'], 2)}) · extraction {arms['extraction']['bleu']} "
        f"({arms['extraction']['bleu'] / base:.1%} retained)" if base else "")
    log(f"-> {MEASURE}")

def main() -> None:
    import sys

    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
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
