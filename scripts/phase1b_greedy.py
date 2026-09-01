"""Phase 1b deliverables 5-6: combo diagnostics, greedy set selection, COMET test.

Solo ablations in the sweep moved BLEU by at most ~1.6 points, so the set
question -- how much does the *candidate space* carry jointly -- is answered
first by three upper-bound combos (all 9 candidate MLPs, all 288 candidate
heads, both together = the full 25%-FLOPs set). The greedy stage then grows a
set from the solo ranking: at each step the highest-ranked unused component
joins, the union is re-ablated and re-scored on the full 200-sentence
shortlist, and the run stops when the marginal dBLEU stays under 0.3 for
three consecutive additions. This is rank-ordered greedy (one pass per
step); full greedy re-evaluation of every remaining candidate each round
would cost O(n^2) generation passes and is recorded as out of budget.

Stages: combo | greedy | comet (baseline + candidate set + truncations).

Run: uv run python -m scripts.phase1b_greedy qwen3-8b combo
     uv run python -m scripts.phase1b_greedy qwen3-8b greedy 420
     uv run python -m scripts.phase1b_greedy qwen3-8b comet
"""

import json
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from scripts.observe import (
    Budget,
    banner,
    degeneracy,
    duration,
    gpu,
    log,
    preview,
    set_log_file,
    step,
)
from scripts.phase1b_ablation import (
    CANDIDATE_LAYERS,
    EVAL_SENTENCES,
    GENERATION_BATCH,
    PROGRESS,
    ablate,
    bleu_of,
    capture_means,
    eval_data,
    load_progress,
    parse_component,
    reference_prompts,
    save_progress,
    translate,
)
from src.model.adapter import load_adapter

CANDIDATE = Path("results/phase1b-circuit-candidate.json")

COMBOS = {
    "all_candidate_mlps": [f"mlp:{layer}" for layer in CANDIDATE_LAYERS],
    "all_candidate_heads": [f"heads:{layer}" for layer in CANDIDATE_LAYERS],
    "full_candidate_set": [f"mlp:{layer}" for layer in CANDIDATE_LAYERS]
                          + [f"heads:{layer}" for layer in CANDIDATE_LAYERS],
}

SATURATION_MARGIN = 0.3
SATURATION_RUNS = 3

def setup(config: str):
    with step(f"loading {config}"):
        adapter = load_adapter(config)
        adapter.cfg = replace(adapter.cfg, batch_size=GENERATION_BATCH)
    wmt, prompts, references = eval_data()
    log(f"eval set: {len(prompts)} sentences · model {adapter.cfg.n_layers}x{adapter.cfg.n_heads} · {gpu()}")
    with step("counterfactual means over the candidate band"):
        means = capture_means(adapter, reference_prompts(wmt))
    return adapter, wmt, prompts, references, means

def run_combo(adapter, means, prompts, references, components, label: str = "combo"):
    with step(f"{label}: {len(components)} components (share {round(flops_share(components), 4)})") as facts:
        with ablate(adapter, means, components):
            hypotheses = translate(adapter, prompts, label=label)
        facts["BLEU"] = bleu_of(hypotheses, references)
        facts["degeneracy"] = degeneracy(hypotheses)
    preview(hypotheses, label)
    return bleu_of(hypotheses, references), hypotheses

def stage_combo(config: str) -> None:
    set_log_file("results/phase1b-greedy.log")
    state = load_progress()
    state.setdefault("combos", {})
    banner("phase1b upper-bound combos", {
        "config": config,
        "combos": f"{len(COMBOS)} ({', '.join(COMBOS)})",
        "done": f"{len(state['combos'])} already scored",
        "reference": "counterfactual prompt form (eval prompt minus translation logic)",
        "progress": str(PROGRESS),
    })
    adapter, _, prompts, references, means = setup(config)
    for name, components in COMBOS.items():
        if name in state["combos"]:
            log(f"{name} already scored: BLEU {state['combos'][name]['bleu']} (cached)")
            continue
        start = time.time()
        bleu, hypotheses = run_combo(adapter, means, prompts, references, components, label=name)
        state["combos"][name] = {
            "components": components,
            "bleu": bleu,
            "dbleu": round(state["baseline"]["bleu_200"] - bleu, 2),
            "seconds": round(time.time() - start, 1),
            "hypotheses": hypotheses,
        }
        save_progress(state)
        log(f"{name}: BLEU {bleu} · dBLEU {state['combos'][name]['dbleu']} · "
            f"{duration(state['combos'][name]['seconds'])}")

def solo_ranking(state):
    """Layer-level and single-head components by solo dBLEU, strongest first"""
    return [cid for cid, _ in sorted(state["components"].items(), key=lambda item: -item[1]["dbleu"])]

def redundant(cid: str, chosen) -> bool:
    """Skip a component already inside a chosen layer-level one, and vice versa"""
    kind, layer, _ = parse_component(cid)
    for other in chosen:
        other_kind, other_layer, _ = parse_component(other)
        if layer != other_layer:
            continue
        if kind == other_kind or {kind, other_kind} == {"head", "heads"}:
            return True
    return False

def stage_greedy(config: str, budget: float) -> None:
    set_log_file("results/phase1b-greedy.log")
    allowance = Budget(budget)
    state = load_progress()
    state.setdefault("greedy", {"chosen": [], "trajectory": []})
    greedy = state["greedy"]
    banner("phase1b greedy set growth", {
        "config": config,
        "resuming at": f"{len(greedy['chosen'])} components chosen, "
                       f"{len(greedy['trajectory'])} steps recorded",
        "saturation": f"marginal dBLEU < {SATURATION_MARGIN} for {SATURATION_RUNS} consecutive additions",
        "budget": duration(budget),
        "progress": str(PROGRESS),
    })
    adapter, _, prompts, references, means = setup(config)
    baseline = state["baseline"]["bleu_200"]

    def saturated() -> bool:
        marginals = [step["marginal_dbleu"] for step in greedy["trajectory"]]
        return len(marginals) >= SATURATION_RUNS and all(
            marginal < SATURATION_MARGIN for marginal in marginals[-SATURATION_RUNS:]
        )

    steps = []
    while not saturated():
        estimate = sum(steps) / len(steps) if steps else state["baseline"]["seconds"]
        if not allowance.fits(estimate):
            log(f"stopping cleanly: {allowance.state()}, next step needs ~{duration(estimate)}")
            log(f"{len(greedy['chosen'])} components chosen so far -- re-run the same command to continue")
            return
        candidates = [cid for cid in solo_ranking(state)
                      if cid not in greedy["chosen"] and not redundant(cid, greedy["chosen"])]
        if not candidates:
            break
        cid = candidates[0]
        trial = [*greedy["chosen"], cid]
        log(f"[step {len(greedy['trajectory']) + 1}] adding {cid} -> {len(trial)} components · "
            f"{allowance.state()}")
        started = time.time()
        bleu, _ = run_combo(adapter, means, prompts, references, trial, label=f"greedy+{cid}")
        steps.append(time.time() - started)
        cumulative = round(baseline - bleu, 2)
        previous = greedy["trajectory"][-1]["cumulative_dbleu"] if greedy["trajectory"] else 0.0
        greedy["chosen"] = trial
        greedy["trajectory"].append({
            "added": cid,
            "bleu": bleu,
            "cumulative_dbleu": cumulative,
            "marginal_dbleu": round(cumulative - previous, 2),
        })
        save_progress(state)
        print(f"+{cid}: BLEU {bleu} cumulative d {cumulative} (marginal {round(cumulative - previous, 2)})",
              flush=True)
    print("greedy saturated" if saturated() else "greedy exhausted", flush=True)

@lru_cache(maxsize=1)
def _flops_model() -> dict:
    """The MAC bookkeeping, read once

    flops_share is called per candidate while a matched random set is being
    drawn -- thousands of times per seed -- and re-reading the file each time
    made the draw cost more than the forward pass it was setting up.
    """
    return json.loads(Path("results/phase1b-flops-model.json").read_text())

def flops_share(components) -> float:
    model = _flops_model()
    head = model["per_component_macs"]["head"]
    mlp = model["per_component_macs"]["mlp"]
    total = model["totals"]["model_macs_per_token"]
    n_heads = model["model"]["n_heads"]
    macs = 0
    for cid in components:
        kind, _, _ = parse_component(cid)
        macs += mlp if kind == "mlp" else head * (n_heads if kind == "heads" else 1)
    return macs / total

def stage_comet(config: str) -> None:
    state = load_progress()
    greedy = state["greedy"]
    adapter, wmt, prompts, references, means = setup(config)
    sources = [spanish for spanish, _ in wmt[:EVAL_SENTENCES]]

    from comet import download_model, load_from_checkpoint
    comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    def score(hypotheses):
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(sources, hypotheses, references, strict=True)]
        return round(float(comet.predict(data, batch_size=16, gpus=1).system_score), 4)

    chosen = greedy["chosen"]
    evaluations = {
        "baseline": {"bleu": state["baseline"]["bleu_200"], "comet22": score(state["baseline"]["hypotheses"])},
        "candidate": {"components": chosen, "flops_share": round(flops_share(chosen), 4),
                      "bleu": state["combos"]["full_candidate_set"]["bleu"]
                      if set(chosen) == set(COMBOS["full_candidate_set"]) else None,
                      "comet22": score(state["combos"]["full_candidate_set"]["hypotheses"])
                      if set(chosen) == set(COMBOS["full_candidate_set"]) else None},
    }
    if evaluations["candidate"]["comet22"] is None:
        bleu, hypotheses = run_combo(adapter, means, prompts, references, chosen)
        evaluations["candidate"].update(bleu=bleu, comet22=score(hypotheses))
    # the smallest greedy prefixes: the pre-registered test asks for *a* set under
    # 25% FLOPs, and the interesting one is the cheapest that still clears it
    for size in (1, 2, 4):
        prefix = chosen[:size]
        bleu, hypotheses = run_combo(adapter, means, prompts, references, prefix)
        evaluations[f"greedy_prefix_{size}"] = {
            "components": prefix, "flops_share": round(flops_share(prefix), 4),
            "bleu": bleu, "comet22": score(hypotheses),
        }
        print(f"prefix {size}: BLEU {bleu} COMET {evaluations[f'greedy_prefix_{size}']['comet22']}", flush=True)
    for name, record in evaluations.items():
        if name == "baseline":
            continue
        record["dcomet"] = round(evaluations["baseline"]["comet22"] - record["comet22"], 4)
        record["dbleu"] = round(evaluations["baseline"]["bleu"] - record["bleu"], 2)
    passing = [name for name, record in evaluations.items()
               if name != "baseline" and record["dcomet"] >= 0.020 and record["flops_share"] <= 0.25]
    smallest = min(passing, key=lambda name: evaluations[name]["flops_share"], default=None)

    threshold = 0.020
    CANDIDATE.write_text(json.dumps({
        "protocol": "greedy rank-ordered set growth over the solo-ablation ranking, cumulative mean-ablation "
                    "re-scored on the full WMT-200 shortlist each step; saturation = marginal dBLEU < "
                    f"{SATURATION_MARGIN} for {SATURATION_RUNS} consecutive additions. COMET-22 on the same "
                    "200 sentences for the pre-registered test.",
        "greedy_trajectory": greedy["trajectory"],
        "combos": {name: {key: value for key, value in record.items() if key != "hypotheses"}
                   for name, record in state.get("combos", {}).items()},
        "evaluations": evaluations,
        "pre_registered_test": {
            "criterion": "component set <= 25% FLOPs whose ablation drops COMET-22 by >= 0.020",
            "threshold_dcomet": threshold,
            "candidate_dcomet": evaluations["candidate"]["dcomet"],
            "candidate_flops_share": evaluations["candidate"]["flops_share"],
            "pass": bool(evaluations["candidate"]["dcomet"] >= threshold
                         and evaluations["candidate"]["flops_share"] <= 0.25),
            "smallest_passing_set": smallest,
            "smallest_passing_flops_share": evaluations[smallest]["flops_share"] if smallest else None,
        },
        "command": f"uv run python -m scripts.phase1b_greedy {config} combo|greedy|comet (chained)",
    }, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(json.loads(CANDIDATE.read_text())["pre_registered_test"], indent=2))

def main() -> None:
    import sys

    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
    stage = sys.argv[2] if len(sys.argv) > 2 else "combo"
    if stage == "combo":
        stage_combo(config)
    elif stage == "greedy":
        stage_greedy(config, budget=float(sys.argv[3]) if len(sys.argv) > 3 else 420.0)
    elif stage == "comet":
        stage_comet(config)
    else:
        raise SystemExit(f"unknown stage '{stage}'")

if __name__ == "__main__":
    main()
