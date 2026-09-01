"""Phase 1b deliverable 7: the random-component control arm at matched FLOPs.

The knockout in phase1b_ablation.py says how much BLEU a discovered set costs
when it is ablated. On its own that number is unfalsifiable: a model spreads a
task across enough components that an arbitrary set of the same size takes a
real bite out of it too, and a reference distribution that removes more than
the task takes the same bite out of *everything*. Zhang et al. (2502.11806)
report their knockouts only as Key Heads vs Random Heads and Key MLPs vs
Random MLPs for exactly this reason; without the second curve, "ablating these
nine MLPs drops BLEU by d" is a number with nothing to be larger than.

So this arm draws component sets that cost the same MACs as the discovered
one, ablates them toward the same counterfactual means, and scores them the
same way. Three readings come out of it:

- the discovered set clears the random band  -> the localization is doing work;
- the two coincide                            -> the damage is the ablation, not
  the circuit, and the reference distribution is the first suspect (this is the
  shape the retracted run of 2026-08-31 would have shown had the arm existed);
- the random band is itself huge              -> the task is not localized at
  this granularity, which is a finding rather than a bug.

Scope defaults to the whole stack rather than the candidate band. A control
drawn only from layers 1a already flagged asks "are these the right components
*within* the right layers"; drawn from the model it asks "is the localization
real at all", which is the question the arm exists to settle. Run both if the
budget allows -- they answer different questions and the pair is more
informative than either.

Every arm also reports a degeneracy score: the fraction of hypotheses that
have stopped being language (a handful of token types repeated to the length
cap). Corpus BLEU cannot tell "translates badly" from "emits `a a a a a`",
those are entirely different findings, and the first one to be checked is the
one at the largest ablation. Kervadec et al. (2601.22795) give the general
form of the worry: aggressive masking preserves the output mode while
flattening everything else, and a corpus metric averages straight over that.

Resumable and budgeted like its siblings: each seed is flushed as it finishes,
--budget makes an invocation exit cleanly mid-sweep, and re-running picks up
where it stopped.

Stages: run draws and scores the arms; report writes
results/phase1b-random-control.json and prints the verdict.

A common pipe could be: discovered_set | draw_matched | ablate | bleu_of | report

Run: uv run python -m scripts.phase1b_random_control qwen3-8b frontier model 1 1800
     uv run python -m scripts.phase1b_random_control qwen3-8b run model all_candidate_mlps 5 1800
     uv run python -m scripts.phase1b_random_control qwen3-8b run model greedy 8 900
     uv run python -m scripts.phase1b_random_control qwen3-8b run candidate greedy 8 900
     uv run python -m scripts.phase1b_random_control qwen3-8b report
"""

import json
import random
import time
from dataclasses import replace
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
    GENERATION_BATCH,
    ablate,
    bleu_of,
    capture_means,
    eval_data,
    reference_prompts,
    translate,
)
from scripts.phase1b_greedy import CANDIDATE, COMBOS, flops_share
from src.model.adapter import load_adapter

PROGRESS = Path("results/phase1b-random-control-progress.json")
CONTROL = Path("results/phase1b-random-control.json")

# A set matched to within this fraction of the discovered set's MACs counts as
# matched. Components are lumpy -- one MLP costs many heads -- so an exact
# match is generally not on the lattice, and the honest move is to state the
# tolerance rather than to pretend the draw landed on the number.
MATCH_TOLERANCE = 0.05

def load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"arms": {}}

def save_progress(state: dict) -> None:
    PROGRESS.parent.mkdir(exist_ok=True)
    PROGRESS.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def target_set(target: str) -> list:
    """The component set the control is matched to

    `greedy` is the end-state use: match the set the greedy stage settled on.
    The COMBOS names are the *early* use, and the more important one -- they
    need no sweep to have run, so the control can be the first thing on the
    GPU rather than the last. Gating on it in that order is what the retracted
    run of 2026-08-31 lacked: `all_candidate_mlps` is the exact set that
    collapsed the output to repeated tokens, and asking a matched random set
    the same question costs one pass and settles whether the collapse was the
    circuit or the ablation before any sweep budget is spent.
    """
    if target in COMBOS:
        return list(COMBOS[target])
    if target != "greedy":
        raise SystemExit(f"unknown target '{target}'; targets are greedy and {sorted(COMBOS)}")
    if not CANDIDATE.exists():
        raise SystemExit(
            f"{CANDIDATE} does not exist, so there is no greedy set to match yet. Either run the sweep and "
            "the greedy stage first, or match a pre-registered combo instead -- "
            f"targets {sorted(COMBOS)} need nothing to have run."
        )
    data = json.loads(CANDIDATE.read_text())
    chosen = data.get("evaluations", {}).get("candidate", {}).get("components")
    if not chosen:
        raise SystemExit(f"{CANDIDATE} holds no candidate component list under evaluations.candidate.components")
    return list(chosen)

def universe(adapter, scope: str) -> list:
    """Every atomic component the control may draw from

    Atomic: one MLP or one head. `heads:L` is a group of them and would make
    the draw coarser than the lattice it is matched on.
    """
    if scope == "candidate":
        layers = list(CANDIDATE_LAYERS)
    elif scope == "model":
        layers = list(range(adapter.cfg.n_layers))
    else:
        raise SystemExit(f"unknown scope '{scope}'; scopes are candidate, model")
    pool = [f"mlp:{layer}" for layer in layers]
    pool += [f"head:{layer}:{head}" for layer in layers for head in range(adapter.cfg.n_heads)]
    return pool

def draw_matched(pool, target_share: float, seed: int) -> list:
    """A random component set costing as close to `target_share` of model MACs as the lattice allows

    Sampled without replacement in a shuffled order, taking every component
    that still fits under the target rather than stopping at the first one
    that does not. Stopping at the first overshoot ends the draw on an MLP
    while hundreds of heads that would have fitted go unused, which leaves the
    arm systematically *under* the budget it claims to match; scanning on
    fills the remainder with the cheap components and tightens the match to
    roughly one head's worth. The single cheapest skipped component is then
    added if overshooting by it lands nearer the target than stopping short.

    Sampling *to a FLOPs budget* rather than to a component count is the whole
    point -- one MLP costs as much as many heads, so a set matched on count is
    not matched on cost, and the comparison it licenses is not the one being
    claimed.
    """
    rng = random.Random(seed)
    order = pool[:]
    rng.shuffle(order)
    chosen: list = []
    skipped: list = []
    for cid in order:
        if flops_share([*chosen, cid]) <= target_share:
            chosen.append(cid)
        else:
            skipped.append(cid)
    if skipped:
        cheapest = min(skipped, key=lambda cid: flops_share([cid]))
        if abs(flops_share([*chosen, cheapest]) - target_share) < abs(flops_share(chosen) - target_share):
            chosen.append(cheapest)
    if not chosen:
        raise SystemExit(
            f"no component in the pool fits under a target share of {target_share:.4f}; the discovered "
            "set is cheaper than the cheapest single component, so there is nothing to match"
        )
    error = abs(flops_share(chosen) - target_share) / target_share
    if error > MATCH_TOLERANCE:
        # not fatal: the lattice may simply be too coarse here. But the arm is
        # then not matched, and a mismatched control is a different claim, so it
        # says so rather than reporting the number as if it were matched.
        print(f"  warning: seed {seed} matched to within {error:.1%}, outside the "
              f"{MATCH_TOLERANCE:.0%} tolerance; components are lumpier than the target", flush=True)
    return chosen

def ensure_baseline(state, adapter, prompts, references) -> None:
    if "baseline" in state:
        return
    start = time.time()
    with step(f"baseline: {len(prompts)} sentences, un-ablated"):
        hypotheses = translate(adapter, prompts, label="baseline")
    state["baseline"] = {
        "bleu": bleu_of(hypotheses, references),
        "degeneracy": degeneracy(hypotheses),
        "seconds": round(time.time() - start, 1),
        "hypotheses": hypotheses,
    }
    save_progress(state)
    log(f"baseline BLEU {state['baseline']['bleu']} · degeneracy {state['baseline']['degeneracy']} · "
        f"{duration(state['baseline']['seconds'])}/pass")
    preview(hypotheses, "baseline")

def score_set(adapter, means, prompts, references, components, label: str = "arm") -> dict:
    start = time.time()
    with step(f"{label}: ablating {len(components)} components "
              f"(share {round(flops_share(components), 4)})") as facts:
        with ablate(adapter, means, components):
            hypotheses = translate(adapter, prompts, label=label)
        bleu = bleu_of(hypotheses, references)
        broken = degeneracy(hypotheses)
        facts["BLEU"] = bleu
        facts["degeneracy"] = broken
    preview(hypotheses, label)
    return {
        "components": components,
        "n_components": len(components),
        "flops_share": round(flops_share(components), 4),
        "bleu": bleu,
        "degeneracy": broken,
        "seconds": round(time.time() - start, 1),
        "hypotheses": hypotheses,
    }

def stage_run(config: str, scope: str, target: str, seeds: int, budget: float) -> None:
    set_log_file("results/phase1b-random-control.log")
    allowance = Budget(budget)
    state = load_progress()
    chosen = target_set(target)
    share = flops_share(chosen)
    done = len([k for k in state["arms"].get(f"{scope}/{target}", {}) if k.startswith("random_")])
    banner("phase1b random-component control", {
        "config": config,
        "scope": f"{scope} ({'whole stack' if scope == 'model' else 'candidate band 27-35'})",
        "target": f"{target} · {len(chosen)} components · {share:.4f} of model MACs",
        "seeds": f"{seeds} requested, {done} already scored",
        "budget": duration(budget),
        "reference": "counterfactual prompt form (eval prompt minus translation logic)",
        "progress": str(PROGRESS),
    })
    adapter = load_adapter(config)
    adapter.cfg = replace(adapter.cfg, batch_size=GENERATION_BATCH)
    wmt, prompts, references = eval_data()
    log(f"model loaded: {adapter.cfg.n_layers} layers x {adapter.cfg.n_heads} heads · {gpu()}")

    # means must cover every layer the control may ablate, not just the candidate band
    layers = list(CANDIDATE_LAYERS) if scope == "candidate" else list(range(adapter.cfg.n_layers))
    with step(f"counterfactual means over {len(layers)} layers"):
        means = capture_means(adapter, reference_prompts(wmt), layers=layers)
    ensure_baseline(state, adapter, prompts, references)

    arms = state["arms"].setdefault(f"{scope}/{target}", {})
    if "discovered" not in arms:
        arms["discovered"] = score_set(adapter, means, prompts, references, chosen, label=f"discovered/{target}")
        save_progress(state)
        found = round(state["baseline"]["bleu"] - arms["discovered"]["bleu"], 2)
        log(f"discovered ({target}): BLEU {arms['discovered']['bleu']} · dBLEU {found} · "
            f"degeneracy {arms['discovered']['degeneracy']}")
        if arms["discovered"]["degeneracy"]:
            log("!! degeneracy is non-zero on the discovered arm: the outputs are repeated tokens, not "
                "translation. This is a broken model, not an ablated one -- the reference distribution is "
                "the first suspect. Read the samples above before trusting any dBLEU from this run.")
    else:
        log(f"discovered ({target}) already scored: BLEU {arms['discovered']['bleu']} (cached)")

    for seed in range(seeds):
        name = f"random_{seed}"
        if name in arms:
            continue
        estimate = arms["discovered"]["seconds"]
        if not allowance.fits(estimate):
            log(f"stopping cleanly: {allowance.state()}, next seed needs ~{duration(estimate)}")
            log(f"{seeds - len([k for k in arms if k.startswith('random_')])} seeds left -- "
                "re-run the same command to resume")
            return
        drawn = draw_matched(universe(adapter, scope), share, seed)
        log(f"[seed {seed + 1}/{seeds}] {allowance.state()}")
        arms[name] = score_set(adapter, means, prompts, references, drawn, label=name)
        arms[name]["match_error"] = round(abs(arms[name]["flops_share"] - share) / share, 4)
        save_progress(state)
        found = round(state["baseline"]["bleu"] - arms["discovered"]["bleu"], 2)
        this = round(state["baseline"]["bleu"] - arms[name]["bleu"], 2)
        log(f"{name}: BLEU {arms[name]['bleu']} · dBLEU {this} · {arms[name]['n_components']} components · "
            f"match {arms[name]['match_error']:.2%} · degeneracy {arms[name]['degeneracy']}")
        band = [round(state["baseline"]["bleu"] - a["bleu"], 2)
                for k, a in arms.items() if k.startswith("random_")]
        verdict = "ahead" if found > max(band) else "NOT ahead"
        log(f"running verdict: discovered dBLEU {found} vs random band "
            f"{min(band)}..{max(band)} over {len(band)} seeds -- discovered is {verdict}", indent=1)
    log(f"scope '{scope}' target '{target}' done in {duration(allowance.spent)} -- "
        "run the report stage next")

# The survival frontier: shares of model MACs to probe, coarse to the point of
# collapse. Geometric rather than linear because the interesting region is
# small -- a circuit worth reporting is a few percent, and 19.4% is already
# known to be past the end.
FRONTIER_SHARES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)

FRONTIER = Path("results/phase1b-survival-frontier.json")

def stage_frontier(config: str, scope: str, seeds: int, budget: float) -> None:
    """How much of the model can be mean-ablated before it stops being a language model

    The question 8.5 says nobody asked. A knockout is only evidence inside a
    budget where the un-ablated behaviour survives; outside it every arm is a
    broken model and the sweep measures destruction, which is what the
    all_candidate_mlps run at 19.4% turned out to be doing.

    Random components on purpose. The frontier wanted here is a property of
    the *model and the intervention*, not of any candidate set -- "what does
    ablating this much cost, whatever it is" -- so a discovered set that later
    beats this curve at the same share is a set doing something the average
    component does not.
    """
    set_log_file("results/phase1b-survival-frontier.log")
    allowance = Budget(budget)
    state = load_progress()
    levels = state.setdefault("frontier", {}).setdefault(scope, {})
    banner("phase1b survival frontier", {
        "config": config,
        "scope": scope,
        "shares": ", ".join(f"{share:.1%}" for share in FRONTIER_SHARES),
        "seeds per level": seeds,
        "done": f"{len(levels)} levels already scored",
        "budget": duration(budget),
    })
    adapter = load_adapter(config)
    adapter.cfg = replace(adapter.cfg, batch_size=GENERATION_BATCH)
    wmt, prompts, references = eval_data()
    layers = list(CANDIDATE_LAYERS) if scope == "candidate" else list(range(adapter.cfg.n_layers))
    with step(f"counterfactual means over {len(layers)} layers"):
        means = capture_means(adapter, reference_prompts(wmt), layers=layers)
    ensure_baseline(state, adapter, prompts, references)

    for share in FRONTIER_SHARES:
        for seed in range(seeds):
            name = f"{share:.3f}/{seed}"
            if name in levels:
                log(f"{name} already scored: BLEU {levels[name]['bleu']} (cached)")
                continue
            if not allowance.fits(state["baseline"]["seconds"]):
                log(f"stopping cleanly: {allowance.state()} -- re-run the same command to continue")
                return
            drawn = draw_matched(universe(adapter, scope), share, seed)
            log(f"[share {share:.1%} seed {seed}] {allowance.state()}")
            levels[name] = score_set(adapter, means, prompts, references, drawn,
                                     label=f"share{share:.1%}s{seed}")
            save_progress(state)
            log(f"{name}: BLEU {levels[name]['bleu']} · degeneracy {levels[name]['degeneracy']} · "
                f"{levels[name]['n_components']} components")
    stage_frontier_report(scope)

def stage_frontier_report(scope: str) -> None:
    """The curve, and the largest share at which the model was still emitting language"""
    state = load_progress()
    levels = state.get("frontier", {}).get(scope, {})
    if not levels:
        raise SystemExit(f"no frontier levels scored for scope '{scope}' yet; run the frontier stage first")
    base = state["baseline"]["bleu"]
    rows = []
    for name in sorted(levels, key=lambda key: (float(key.split("/")[0]), key)):
        record = levels[name]
        record["degeneracy"] = degeneracy(record.get("hypotheses", []))
        rows.append({
            "share": float(name.split("/")[0]),
            "seed": int(name.split("/")[1]),
            "n_components": record["n_components"],
            "bleu": record["bleu"],
            "dbleu": round(base - record["bleu"], 2),
            "degeneracy": record["degeneracy"],
        })
    intact = [row for row in rows if row["degeneracy"] == 0.0]
    frontier = max((row["share"] for row in intact), default=None)
    FRONTIER.write_text(json.dumps({
        "protocol": "random matched-FLOPs mean ablation at increasing shares of model MACs, scored on the "
                    "WMT-200 shortlist. The frontier is the largest share at which every generation was "
                    "still language. A knockout outside it measures destruction, not localization.",
        "scope": scope,
        "baseline_bleu": base,
        "survival_frontier_share": frontier,
        "note": ("no probed share left the model intact -- the frontier is below the smallest share tried"
                 if frontier is None else
                 f"ablating up to {frontier:.1%} of model MACs left every generation as language; "
                 "circuit claims are only measurable at or below this"),
        "curve": rows,
    }, indent=2) + "\n")
    log(f"survival frontier ({scope}): "
        + ("below the smallest share probed" if frontier is None else f"{frontier:.1%} of model MACs"))
    for row in rows:
        mark = "ok" if row["degeneracy"] == 0.0 else "BROKEN"
        log(f"{row['share']:>7.1%}  {row['n_components']:>4} comps  BLEU {row['bleu']:>6}  "
            f"dBLEU {row['dbleu']:>6}  degeneracy {row['degeneracy']:>5}  {mark}", indent=1)
    log(f"-> {FRONTIER}")

def stage_report(command: str) -> None:
    state = load_progress()
    if "baseline" not in state or not state["arms"]:
        raise SystemExit(f"{PROGRESS} has no scored arms yet; run the run stage first")
    base = state["baseline"]["bleu"]

    scopes = {}
    for scope, arms in state["arms"].items():
        if "discovered" not in arms:
            continue
        # degeneracy is recomputed from the stored generations rather than read
        # back from the record: the generations are the evidence, the score is a
        # threshold over them, and a run scored under an older threshold must not
        # report a number this build would not produce
        for record in arms.values():
            record["degeneracy"] = degeneracy(record.get("hypotheses", []))
        arms["discovered"]["empty"] = sum(1 for h in arms["discovered"].get("hypotheses", []) if not h.strip())
        randoms = [record for name, record in arms.items() if name.startswith("random_")]
        if not randoms:
            continue
        drops = sorted(round(base - record["bleu"], 2) for record in randoms)
        found = round(base - arms["discovered"]["bleu"], 2)
        best_random = max(drops)
        scopes[scope] = {
            "discovered": {key: value for key, value in arms["discovered"].items() if key != "hypotheses"},
            "discovered_dbleu": found,
            "random_seeds": len(randoms),
            "random_dbleu": {
                "min": drops[0],
                "median": drops[len(drops) // 2],
                "max": best_random,
                "all": drops,
            },
            "random_flops_shares": sorted({record["flops_share"] for record in randoms}),
            "random_degeneracy_max": max(record["degeneracy"] for record in randoms),
            "random_empty_max": max(sum(1 for h in record.get("hypotheses", []) if not h.strip())
                                    for record in randoms),
            "discovered_empty": arms["discovered"]["empty"],
            "random_match_error_max": max(record.get("match_error", 0.0) for record in randoms),
            "match_tolerance": MATCH_TOLERANCE,
            "separation": round(found - best_random, 2),
            "clears_random_band": bool(found > best_random),
        }

    CONTROL.write_text(json.dumps({
        "protocol": "random-component control at matched FLOPs (Zhang et al. 2502.11806 Fig. 2 shape). Each "
                    "random arm is a component set drawn without replacement from the scope until its MACs "
                    f"match the discovered set's within {MATCH_TOLERANCE:.0%}, ablated toward the same "
                    "counterfactual means and scored on the same WMT-200 shortlist. A discovered set that "
                    "does not clear the random band is not evidence of a circuit.",
        "scopes": {
            "candidate": "atomic components drawn from layers 27-35 only: are these the right components "
                         "within the right layers",
            "model": "atomic components drawn from the whole stack: is the localization real at all",
        },
        "baseline_bleu": base,
        "baseline_degeneracy": state["baseline"]["degeneracy"],
        "degeneracy_note": "fraction of hypotheses that are repeated tokens rather than language; any arm "
                           "with a non-zero value here has its BLEU read as a broken-model number, not a "
                           "translation-quality one",
        "results": scopes,
        "command": command,
    }, indent=2, ensure_ascii=False) + "\n")

    for scope, record in scopes.items():
        verdict = "CLEARS the random band" if record["clears_random_band"] else "DOES NOT clear the random band"
        print(f"[{scope}] discovered dBLEU {record['discovered_dbleu']} vs random max "
              f"{record['random_dbleu']['max']} over {record['random_seeds']} seeds -> {verdict} "
              f"(separation {record['separation']})")
        if record["random_degeneracy_max"] or record["discovered"]["degeneracy"]:
            print(f"[{scope}] WARNING: degeneracy is non-zero "
                  f"(discovered {record['discovered']['degeneracy']}, random max "
                  f"{record['random_degeneracy_max']}); read the generations before quoting any of this")
    print(f"-> {CONTROL}")

def main() -> None:
    import sys

    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
    stage = sys.argv[2] if len(sys.argv) > 2 else "run"
    if stage == "run":
        scope = sys.argv[3] if len(sys.argv) > 3 else "model"
        target = sys.argv[4] if len(sys.argv) > 4 else "greedy"
        seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 8
        budget = float(sys.argv[6]) if len(sys.argv) > 6 else 900.0
        stage_run(config, scope, target, seeds, budget)
    elif stage == "frontier":
        scope = sys.argv[3] if len(sys.argv) > 3 else "model"
        seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        budget = float(sys.argv[5]) if len(sys.argv) > 5 else 1800.0
        stage_frontier(config, scope, seeds, budget)
    elif stage == "frontier-report":
        stage_frontier_report(sys.argv[3] if len(sys.argv) > 3 else "model")
    elif stage == "report":
        stage_report(f"uv run python -m scripts.phase1b_random_control {config} run <scope> (chained), then report")
    else:
        raise SystemExit(f"unknown stage '{stage}'; stages are run, frontier, frontier-report, report")

if __name__ == "__main__":
    main()
